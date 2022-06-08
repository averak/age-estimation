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
            metrics=[self.P_M_metric, self.θ_metric, self.σ_metric]
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
        # P_F: 女性である確率
        # P_M = exp(q_M) / (exp(q_M) + exp(q_F))
        # P_F = exp(q_F) / (exp(q_M) + exp(q_F))
        # θ: 推定年齢
        # σ: 残差標準偏差
        # ρ: ρ = log(σ^2)
        q_M = y_pred[:, 0]
        q_F = y_pred[:, 1]
        θ_M = y_pred[:, 2]
        θ_F = y_pred[:, 3]

        # ρ = log(σ^2)として変数変換
        ρ_M = y_pred[:, 4]
        ρ_F = y_pred[:, 5]

        # 男性の場合はL_M、女性の場合はL_Fを最小化する
        L_M = ρ_M + ((y - θ_M) ** 2) * K.exp(-ρ_M) - 2 * q_M + 2 * K.log(K.exp(q_M) + K.exp(q_F))
        L_F = ρ_F + ((y - θ_F) ** 2) * K.exp(-ρ_F) - 2 * q_F + 2 * K.log(K.exp(q_M) + K.exp(q_F))

        return K.mean(K.switch(K.equal(s, 0.0), L_M, L_F))

    def P_M_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        男性である確率P_Mの評価関数
        """

        s = y_true[:, 1]
        q_M = y_pred[:, 0]
        q_F = y_pred[:, 1]

        P_M = K.exp(q_M) / (K.exp(q_M) + K.exp(q_F))

        return metrics.mean_absolute_error(K.constant(1.0) - s, P_M)

    def θ_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        年齢θの評価関数
        """

        max_age = tf.constant(self.MAX_AGE)
        min_age = tf.constant(self.MIN_AGE)

        y = y_true[:, 0] * (max_age - min_age) + min_age
        s = y_true[:, 1]

        θ_M = y_pred[:, 2] * (max_age - min_age) + min_age
        θ_F = y_pred[:, 3] * (max_age - min_age) + min_age

        θ = K.switch(
            K.equal(s, 0.0),
            θ_M,
            θ_F,
        )

        return metrics.mean_absolute_error(y, θ)

    def σ_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        残差標準偏差σの評価関数
        """

        max_age = tf.constant(self.MAX_AGE)
        min_age = tf.constant(self.MIN_AGE)

        y = y_true[:, 0] * (max_age - min_age) + min_age
        s = y_true[:, 1]

        θ_M = y_pred[:, 2] * (max_age - min_age) + min_age
        θ_F = y_pred[:, 3] * (max_age - min_age) + min_age
        σ_M = K.sqrt(K.exp(y_pred[:, 4])) * (max_age - min_age) + min_age
        σ_F = K.sqrt(K.exp(y_pred[:, 5])) * (max_age - min_age) + min_age

        θ = K.switch(
            K.equal(s, 0.0),
            θ_M,
            θ_F,
        )
        σ = K.switch(
            K.equal(s, 0.0),
            σ_M,
            σ_F,
        )

        return metrics.mean_absolute_error(K.abs(y - θ), σ)

    def activation(self, y_pred: np.ndarray):
        """
        活性化関数
        """

        q_M = y_pred[:, 0]
        q_F = y_pred[:, 1]
        θ_M = K.sigmoid(y_pred[:, 2])
        θ_F = K.sigmoid(y_pred[:, 3])
        ρ_M = y_pred[:, 4]
        ρ_F = y_pred[:, 5]

        return tf.stack([q_M, q_F, θ_M, θ_F, ρ_M, ρ_F], 1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        [P_M, θ_M, θ_F, σ_M, σ_F]を推定
        """

        results = self.model.predict(x)
        q_M = results[:, 0]
        q_F = results[:, 1]

        results[:, 0] = K.exp(q_M) / (K.exp(q_M) + K.exp(q_F))
        results[:, 1] = K.exp(q_F) / (K.exp(q_M) + K.exp(q_F))
        results[:, 2] = results[:, 2] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE
        results[:, 3] = results[:, 3] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE
        results[:, 4] = np.sqrt(np.exp(results[:, 4])) * (self.MAX_AGE - self.MIN_AGE)
        results[:, 5] = np.sqrt(np.exp(results[:, 5])) * (self.MAX_AGE - self.MIN_AGE)

        return results
