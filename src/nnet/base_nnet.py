from abc import ABCMeta, abstractmethod

import numpy as np
import sklearn.preprocessing
import tensorflow as tf
from tensorflow.keras import Model, metrics
import tensorflow.keras.backend as K
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
            # metrics=[self.P_M_metric, self.θ_metric, self.σ_metric]
            metrics=[self.P_M_metric]
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
        y_test[:, 0] = sklearn.preprocessing.minmax_scale(y_test[:, 0])

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

        # y: 年齢
        # s: 性別
        y = y_true[:, 0]
        s = y_true[:, 1]

        # P_M: 男性である確率
        # θ: 推定年齢
        # σ: 残差標準偏差
        P_M = y_pred[:, 0]
        P_F = K.constant(1.0) - P_M
        θ_M = y_pred[:, 1]
        θ_F = y_pred[:, 2]
        σ_M = y_pred[:, 3]
        σ_F = y_pred[:, 4]

        epsilon = K.constant(K.epsilon())

        # 男性の場合はp_M、女性の場合はp_Fの尤度を最大化する
        L_M = K.log(2 * np.pi * σ_M ** 2) + ((y - θ_M) ** 2) / (σ_M ** 2 + epsilon) - K.log(P_M + epsilon) * 2
        L_F = K.log(2 * np.pi * σ_F ** 2) + ((y - θ_F) ** 2) / (σ_F ** 2 + epsilon) - K.log(P_F + epsilon) * 2

        return K.mean(K.switch(s == 0, L_M, L_F))

    def P_M_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        男性である確率P_Mの評価関数
        """

        s = y_true[:, 1]
        P_M = y_pred[:, 0]

        return metrics.mean_absolute_error(K.constant(1.0) - s, P_M)

    def θ_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        年齢θの評価関数
        """

        max_age = tf.constant(self.MAX_AGE)
        min_age = tf.constant(self.MIN_AGE)

        y = y_true[:, 0] * (max_age - min_age) + min_age
        s = y_true[:, 1]

        θ_M = y_pred[:, 1] * (max_age - min_age) + min_age
        θ_F = y_pred[:, 2] * (max_age - min_age) + min_age

        return K.switch(
            K.greater_equal(K.constant(0), s),
            metrics.mean_absolute_error(y, θ_M),
            metrics.mean_absolute_error(y, θ_F),
        )

    def σ_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        残差標準偏差σの評価関数
        """

        max_age = tf.constant(self.MAX_AGE)
        min_age = tf.constant(self.MIN_AGE)

        y = y_true[:, 0] * (max_age - min_age) + min_age
        s = y_true[:, 1]

        θ_M = y_pred[:, 1] * (max_age - min_age) + min_age
        θ_F = y_pred[:, 2] * (max_age - min_age) + min_age
        σ_M = y_pred[:, 3] * (max_age - min_age) + min_age
        σ_F = y_pred[:, 4] * (max_age - min_age) + min_age

        return K.switch(
            s == 0,
            metrics.mean_absolute_error(K.abs(y - θ_M), σ_M),
            metrics.mean_absolute_error(K.abs(y - θ_F), σ_F)
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        年齢θと残差標準偏差σを推定
        """

        results = self.model.predict(x)
        results[:, 0] = results[:, 0] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE
        results[:, 1] = np.sqrt(np.exp(results[:, 1])) * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE

        return results
