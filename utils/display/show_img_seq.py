
from typing import List, Dict, Union

import numpy as np
import matplotlib.pyplot as plt

class ImageStackDisplay(object):
    def __init__(self, axs: Union[plt.Axes, np.ndarray], display_method,
                 plot_data_list: List[Dict]):

        self.axs = axs
        self.data = plot_data_list
        self.n_frames = len(plot_data_list)
        self.ind = 0
        self.display_method = display_method

        self.update()

    def key(self, event):
        if event.key == 'right':
            if self.ind < self.n_frames - 1:
                print("next frame")
                self.ind += 1
            else:
                print("NO next frame")
        elif event.key == 'left':
            if self.ind > 0:
                print("previous frame")
                self.ind -= 1
            else:
                print("NO previous frame")
        # elif event.key == 'e':
        #     if self.save_path is not None:
        #         if type(self.axs) is np.ndarray:
        #             fig = self.axs[0].figure
        #         else:
        #             fig = self.axs.figure
        #         fig.tight_layout()
        #         savepath = os.path.join(self.save_path, f"{self.save_prefix}_{self.ind:03}.png")
        #         fig.savefig(savepath)
        #         print(f"saved file at {savepath}")

        self.update()

    def update(self):

        if type(self.axs) is np.ndarray:
            [ax.clear() for ax in self.axs.ravel()]
            self.display_method(self.ind, self.axs, self.data)
            [ax.figure.canvas.draw() for ax in self.axs.ravel()]
        else:
            self.axs.clear()
            self.display_method(self.ind, self.axs, self.data)
            self.axs.figure.canvas.draw()