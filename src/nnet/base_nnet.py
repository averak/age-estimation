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

    EPOCHS: int = 30
    """
    エポック数
    """

    BATCH_SIZE: int = 256
    """
    バッチサイズ
    """

    VALIDATION_SPLIT_RATE: float = 0.1
    """
    検証用データの割合
    """

    def __init__(self):
        self.make_model()

    @abstractmethod
    def make_model(self) -> None:
        """
        NNモデルを作成
        """

        raise NotImplementedError()

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        学習
        """

        # TODO: save checkpoint
        self.model.fit(
            x,
            y,
            epochs=3,
            batch_size=256,
            validation_split=0.1
        )
