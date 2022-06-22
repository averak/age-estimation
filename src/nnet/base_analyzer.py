from abc import ABCMeta, abstractmethod

import numpy as np


class BaseAnalyzer(metaclass=ABCMeta):
    """
    解析器
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
