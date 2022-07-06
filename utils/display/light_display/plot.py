import numpy as np
from matplotlib.colors import ColorConverter
from skimage import draw


def plot_stuff(arr: np.ndarray, h: int, w: int, pad_value=0.0, support_value=1.0, color='tab:blue'):
    image = np.zeros((h, w, 3))

    if not len(arr.shape) == 2:
        arr = np.expand_dims(arr, 0)
        color = [color]
    n_pts = arr.shape[1]
    n_series = arr.shape[0]

    pad = 2
    left_limit, right_limit = pad, w - pad - 1

    ticks = np.linspace(left_limit, right_limit, n_pts, dtype=int)

    image[-pad - 1, pad:-pad] = support_value
    image[-pad, ticks] = support_value

    vmax = np.max(arr)
    vmin = np.min(arr)

    plot_up = pad
    plot_down = h - 2 *pad - 1
    plot_height = plot_down - plot_up

    arr_norm = (arr - vmin) / (vmax - vmin)
    h_plot = ((1 - arr_norm) * plot_height).astype(int) + plot_up

    for j in range(n_series):
        c = ColorConverter.to_rgb(color[j])
        for i in range(n_pts - 1):
            v0 = h_plot[j, i]
            v1 = h_plot[j, i + 1]
            t0 = ticks[i]
            t1 = ticks[i + 1]
            coor = draw.line(v0, t0, v1, t1)
            image[coor] = c

    return image
