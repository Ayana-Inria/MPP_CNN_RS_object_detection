from typing import Union, List
import os
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

FONT_PATH = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'Minitel.ttf')


def make_image_from_bunch(ndarray, nrow=8, padding=2, pad_value=0.0):
    if ndarray.ndim == 3:
        ndarray = np.expand_dims(ndarray, -1)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = np.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = np.full((height * ymaps + padding, width * xmaps + padding, num_channels), pad_value).astype(np.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width] = ndarray[k]
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # ndarr = np.clip(grid * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return grid


def add_header(image_array, text: str, pad_value=0.0, draw_value=1.0, padding=2):
    height = 3 * padding + 10 + 1

    num_channels = image_array.shape[2]

    header_arr = np.full((height, image_array.shape[1], num_channels), pad_value).astype(np.float32)

    header_arr[-padding, padding:-padding] = draw_value

    int_draw_value = int(draw_value * 255)

    img = Image.fromarray(np.uint8(header_arr * 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 8)
    draw.text((padding, padding), text, (int_draw_value, int_draw_value, int_draw_value), font=font)

    header_arr = np.array(img) / 255

    return np.concatenate((header_arr, image_array), axis=0), height


def _legend(texts, padding, num_channels, legend_width, header_size, pad_value, draw_value):
    width = 2 * padding + 10 + 1

    all_legends = []
    unit_width = (legend_width - header_size) // len(texts)
    for t in texts[::-1]:
        legend_arr = np.full((width, unit_width, num_channels), pad_value).astype(np.float32)

        int_draw_value = int(draw_value * 255)

        img = Image.fromarray(np.uint8(legend_arr * 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_PATH, 8)
        draw.text((padding, padding), t, (int_draw_value, int_draw_value, int_draw_value), font=font, align='center')
        legend_arr = np.array(img) / 255
        all_legends.append(legend_arr)

    all_legends = np.concatenate(all_legends, axis=1)
    return all_legends, width, unit_width


def add_left_legend(image_array, texts: List[str], padding=2, pad_value=0.0, draw_value=1.0, header_size=0):
    num_channels = image_array.shape[2]
    all_legends, width, unit_width = _legend(texts, padding, num_channels, image_array.shape[0],
                                             header_size, pad_value, draw_value)

    all_legends = np.rot90(all_legends, 1)
    filler = np.full((image_array.shape[0] - unit_width * len(texts), width, num_channels), pad_value)
    all_legends = np.concatenate([filler, all_legends], axis=0)
    right_fill = np.full((all_legends.shape[0], 2 * padding, num_channels), pad_value)
    right_fill[:, -padding] = draw_value
    all_legends = np.concatenate([all_legends, right_fill], axis=1)

    return np.concatenate((all_legends, image_array), axis=1), all_legends.shape[1]


def add_top_legend(image_array, texts: List[str], padding=2, pad_value=0.0, draw_value=1.0):
    num_channels = image_array.shape[2]
    all_legends, width, unit_width = _legend(texts[::-1], padding, num_channels, image_array.shape[1], 0, pad_value,
                                             draw_value)

    filler = np.full((width, image_array.shape[1] - unit_width * len(texts), num_channels), pad_value)
    all_legends = np.concatenate([filler, all_legends], axis=1)
    bot_fille = np.full((2 * padding, all_legends.shape[1], num_channels), pad_value)
    bot_fille[-padding, :] = draw_value
    all_legends = np.concatenate([all_legends, bot_fille], axis=0)

    return np.concatenate((all_legends, image_array), axis=0), all_legends.shape[0]


def hist_image(size: int, distribution: np.ndarray, vmax: Union[float, str] = 1, vmin=0, pad_value=0.0,
               support_value=1.0,
               plot_color=0.5, plot_cmap=None, gt=None, gt_color=(0, 0.5, 0)):
    """

    :param size: size of image
    :param distribution: hist to show
    :param vmax:
    :param vmin:
    :param pad_value:
    :param support_value:
    :param plot_color:
    :return:
    """
    if vmax == 'auto':
        vmax = np.max(distribution)
    if vmin == 'auto':
        vmin = np.min(distribution)
    distribution = np.clip(distribution, vmin, vmax)

    plot = np.full((size, size, 3), pad_value)
    d_size = distribution.shape[0]
    assert size >= d_size
    bar_width = size // d_size
    support_len = d_size * bar_width
    pad_left_support = (size - support_len) // 2
    # plot[-1, pad_left_support:pad_left_support + support_len] = support_value
    bar_range = size - 1

    bar_height = (bar_range * (distribution - vmin) / (vmax - vmin)).astype(int)
    norm_value = bar_height / np.sum(bar_height)

    for k, h in enumerate(bar_height):
        if gt is not None and k == gt:
            sv = gt_color
        else:
            sv = support_value
        plot[-1:, pad_left_support + k * bar_width] = sv

        if plot_cmap is not None:
            plot_color = plot_cmap(norm_value[k])[:3]

        plot[bar_range - h:-2, pad_left_support + k * bar_width] = plot_color

    return plot


def multi_hist_image(size: int, distribution: np.ndarray, vmax: Union[float, str] = 1, vmin=0, pad_value=0.0,
                     support_value=1.0,
                     plot_color=0.5, plot_cmap=None, gt=None, gt_color=(0, 0.5, 0), min_plot_size=5):
    if vmax == 'auto':
        vmax = np.max(distribution)
    if vmin == 'auto':
        vmin = np.min(distribution)
    distribution = np.clip(distribution, vmin, vmax)

    plot = np.full((size, size, 3), pad_value)

    n_dist = len(distribution)
    plot_height = size // n_dist
    if plot_height < min_plot_size:
        n_dist = size // min_plot_size
        plot_height = size // n_dist
        distribution = distribution[:n_dist]

    for i, d in enumerate(distribution):

        v_offset = i * plot_height

        d_size = d.shape[0]
        assert size >= d_size
        bar_width = size // d_size
        support_len = d_size * bar_width
        pad_left_support = (size - support_len) // 2
        # plot[-1, pad_left_support:pad_left_support + support_len] = support_value
        bar_range = plot_height - 4

        bar_height = np.ceil((bar_range * (d - vmin) / (vmax - vmin))).astype(int)
        assert np.all(bar_height < plot_height)
        norm_value = bar_height / np.sum(bar_height)

        for k, h in enumerate(bar_height):
            if gt is not None and k == gt[i]:
                sv = gt_color
            else:
                sv = support_value
            plot[size - v_offset - 1, pad_left_support + k * bar_width:pad_left_support + (k + 1) * bar_width - 1] = sv

            if plot_cmap is not None:
                plot_color = plot_cmap(norm_value[k])[:3]

            plot[size - v_offset - 2 - h: size - v_offset - 2,
            pad_left_support + k * bar_width:pad_left_support + (k + 1) * bar_width - 1] = plot_color

    return plot


def distrib_pixel(size: int, distributions: np.ndarray, vmax=1, vmin=0, pad_value=0.0, support_value=1.0, cmap=None):
    distributions = np.clip(distributions, vmin, vmax)
    plot = np.full((size, size, 3), pad_value)
    n_dist = distributions.shape[0]
    d_size = distributions.shape[1]
    assert size >= d_size
    bar_width = size // d_size
    support_len = d_size * bar_width
    pad_left_support = (size - support_len) // 2
    # plot[-1, pad_left_support:pad_left_support + support_len] = support_value

    values = (distributions - vmin) / (vmax - vmin)

    height_per_d = (size - 3) // n_dist

    for k in range(d_size):
        plot[0, pad_left_support + k * bar_width] = support_value
        plot[height_per_d * n_dist + 2, pad_left_support + k * bar_width] = support_value

        for d in range(n_dist):
            if cmap is not None:
                v = cmap(values[d, k])[:3]
            else:
                v = values[d, k]
            plot[2 + d * height_per_d:2 + (d + 1) * height_per_d - 1, pad_left_support + k * bar_width] = v

    return plot
