from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from human_model import HumanModel


class BaseAnalyzer(metaclass=ABCMeta):
    """
    解析器
    """

    SAVE_PATH: str = "analysis"
    """
    保存するパス
    """

    @abstractmethod
    def export_heatmap(self, train_humans: list, test_humans: list, train_results: np.ndarray, test_results: np.ndarray):
        """
        ヒートマップを出力
        """

        raise NotImplementedError()

    @abstractmethod
    def export_log_graph(self):
        """
        ロググラフを出力
        """

        raise NotImplementedError()

    def export_age_histogram(self, humans: list):
        """
        年齢のヒストグラムを表示
        """

        age_list = [human.age for human in humans]

        plt.figure()
        plt.hist(age_list, bins=90)
        plt.xlabel("Age")
        plt.savefig(f"{self.SAVE_PATH}/age.png")
