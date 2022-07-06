import matplotlib.pyplot as plt
import numpy as np
import torch

from models.position_net.torch_div import torch_divergence
from utils.math_utils import divergence


def test_torch_div():
    a = (np.random.random((20, 20, 2)) - 0.5) * 2
    a[10:, :, :] = 0
    a[:, 5:7, 0] = 0
    a[:, 5:7, 1] = 1
    a[:, 7:9, 0] = 0
    a[:, 7:9, 1] = -1

    a_t = torch.tensor(a).permute((2, 0, 1))
    a_t = torch.unsqueeze(a_t, 0)

    div_np = divergence(np.moveaxis(a, 2, 0), sp=None, indexing='ij')

    div_t = torch_divergence(a_t, 'ij')
    assert tuple(div_t.shape) == (1,1,20,20)

    vec_col = np.zeros(a.shape[:2] + (3,))
    vec_col[..., 0] = a[..., 0]
    vec_col[..., 1] = a[..., 1]
    vec_col = 0.5 * vec_col / np.max(np.abs(vec_col)) + 0.5

    fig, axs = plt.subplots(1, 4)

    axs[0].imshow(vec_col)
    axs[0].set_title('vector field')
    axs[1].imshow(div_np)
    axs[1].set_title('numpy div')
    axs[2].imshow(np.squeeze(div_t.cpu().numpy()))
    axs[2].set_title('torch div')
    error = np.abs(div_np - np.squeeze(div_t.cpu().numpy()))
    axs[3].imshow(error, vmin=0)
    axs[3].set_title('pixel error')
    fig.show()

    mean_error = np.mean(error)
    print(f"mean error: {mean_error}")

    assert mean_error < 1e-8


if __name__ == '__main__':
    test_torch_div()
