import matplotlib.pyplot as plt
import numpy as np


class DrawPlot:
    @staticmethod
    def plot(
        i: int,
        xs: np.ndarray[np.float64] | list[int],
        ys: np.ndarray[np.float64] | list[int],
        title: str,
    ) -> None:
        plt.subplot(2, 1, i)
        plt.title(title)

        if i > 1:
            plt.xlabel("x")
        plt.ylabel("y")

        plt.yticks(np.arange(0, 110, 10))
        plt.grid()
        plt.plot(np.array(xs), np.array(ys))
