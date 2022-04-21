from abc import ABCMeta, abstractmethod

import numpy as np
from tensorflow.keras import Sequential


class BaseNNet(metaclass=ABCMeta):
    """
    ニューラルネットワーク
    """

    model: Sequential
    """
    モデル
    """

    INPUT_SHAPE = (200, 200, 3)
    """
    入力形状
    """

    def __init__(self):
        self.make_model()

    @abstractmethod
    def make_model(self) -> None:
        """
        NNモデルを作成
        """

        raise NotImplementedError()

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        学習
        """

        raise NotImplementedError()
