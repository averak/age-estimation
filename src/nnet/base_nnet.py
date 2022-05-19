from abc import ABCMeta, abstractmethod

import numpy as np
import sklearn.preprocessing
import tensorflow as tf
from tensorflow.keras import Model, backend, metrics
from tensorflow.keras.callbacks import ModelCheckpoint


class BaseNNet(metaclass=ABCMeta):
    """
    ニューラルネットワーク
    """

    model: Model
    """
    モデル
    """

    INPUT_SHAPE = (200, 200, 3)
    """
    入力形状
    """

    EPOCHS: int = 100
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

    MIN_AGE = 1.0
    """
    年齢の最小値
    """

    MAX_AGE = 116.0
    """
    年齢の最大値
    """

    def __init__(self):
        self.make_model()

        self.model.compile(
            optimizer="adam",
            loss=self.loss,
            metrics=[self.theta_metric, self.sigma_metric]
        )

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

    def load_weights(self, file_name: str) -> None:
        """
        学習済みモデルを読み込む
        """

        self.model.load_weights(file_name)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        学習
        """

        y_train = sklearn.preprocessing.minmax_scale(y_train)
        y_test = sklearn.preprocessing.minmax_scale(y_test)

        # チェックポイントを保存するコールパックを定義
        checkpoint_file = "%s/cp-{epoch}.h5" % self.CHECKPOINT_PATH
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_file,
            verbose=1,
            save_weights_only=True
        )
        self.model.save_weights(checkpoint_file.format(epoch=0))

        # 学習
        self.model.fit(
            x_train,
            y_train,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=[checkpoint_callback]
        )

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

    def theta_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        年齢θの評価関数
        """

        max_age = tf.constant(self.MAX_AGE)
        min_age = tf.constant(self.MIN_AGE)

        theta_true = y_true[:, 0] * (max_age - min_age) + min_age
        theta_pred = y_pred[:, 0] * (max_age - min_age) + min_age

        return metrics.mean_absolute_error(theta_true, theta_pred)

    def sigma_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        残差標準偏差σの評価関数
        """

        max_age = tf.constant(self.MAX_AGE)
        min_age = tf.constant(self.MIN_AGE)

        sigma_true = backend.abs(y_true[:, 0] - y_pred[:, 0]) * (max_age - min_age) + min_age
        sigma_pred = y_pred[:, 1] * (max_age - min_age) + min_age

        return metrics.mean_absolute_error(sigma_true, sigma_pred)

    def activation(self, x: np.ndarray):
        """
        活性化関数
        """

        return tf.math.log(x ** 2)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        年齢θと残差標準偏差σを推定
        """

        results = self.model.predict(x)
        results[:, 0] = results[:, 0] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE
        results[:, 1] = results[:, 1] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE

        return results
