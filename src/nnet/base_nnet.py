from abc import ABCMeta, abstractmethod
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, metrics, backend


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

    CHECKPOINT_PATH: str = "ckpt"
    """
    チェックポイントの保存パス
    """

    def __init__(self):
        self.make_model()

    @abstractmethod
    def make_model(self) -> None:
        """
        NNモデルを作成
        """

        raise NotImplementedError()

    def show_model_summary(self) -> None:
        """
        モデルのサマリを表示
        """

        self.model.summary()

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        学習
        """

        for step in range(self.EPOCHS):
            self.model.fit(
                x,
                y,
                initial_epoch=step,
                epochs=step + 1,
                batch_size=self.BATCH_SIZE,
                validation_split=self.VALIDATION_SPLIT_RATE
            )
            self.model.save_weights('%s/%d_%d.h5' % (self.CHECKPOINT_PATH, step + 1, time.time()))

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        損失関数
        """

        y_true = tf.cast(y_true, y_pred.dtype)

        # theta: 推定年齢θ
        # sigma: 残差標準偏差σ
        theta_true = y_true[:, 0]
        theta_pred = y_pred[:, 0]
        sigma_pred = y_pred[:, 1]

        return backend.mean(tf.math.log(2 * np.pi * (sigma_pred ** 2)) + ((theta_true - theta_pred) ** 2) / (sigma_pred ** 2))

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        評価関数
        """

        theta_true = y_true[:, 0]
        theta_pred = y_pred[:, 0]

        # 推定年齢θの平均絶対誤差
        return metrics.mean_absolute_error(theta_true, theta_pred)
