from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import List


@dataclass
class Plotter:
    @staticmethod
    def make_plot(index: int, xs: List[float], ys: List[float], title: str) -> None:
        plt.subplot(2, 2, index)
        plt.title(title)
        plt.yticks(np.arange(0, 110, 10))
        plt.grid()
        plt.plot(np.array(xs), np.array(ys))

    @staticmethod
    def show() -> None:
        plt.show()
