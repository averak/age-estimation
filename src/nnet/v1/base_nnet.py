import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
import tensorflow.keras.backend as K

from nnet.base_nnet import BaseNNet


class BaseNNet_V1(BaseNNet):
    """
    ニューラルネットワーク
    """

    def compile_model(self):
        """
        NNモデルをコンパイル
        """

        # adam = optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer='adam',
            loss=self.loss,
            metrics=[self.θ_metric, self.σ_metric]
        )
        self.model.summary()

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        損失関数
        """

        y_true = tf.cast(y_true, y_pred.dtype)

        # y: 年齢
        y = y_true[:, 0]

        # θ: 推定年齢
        # σ: 残差標準偏差
        θ = y_pred[:, 0]
        σ = y_pred[:, 1]

        return K.mean(K.log(2 * np.pi * (σ ** 2)) + ((y - θ) ** 2) / (σ ** 2 + self.EPSILON))

    def θ_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        年齢θの評価関数
        """

        y = y_true[:, 0]

        θ = y_pred[:, 0]

        if self.IS_NORMALIZED:
            y = y * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR
            θ = θ * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR

        return metrics.mean_absolute_error(y, θ)

    def σ_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        残差標準偏差σの評価関数
        """

        y = y_true[:, 0]

        θ = y_pred[:, 0]
        σ = y_pred[:, 1]

        if self.IS_NORMALIZED:
            y = y * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR
            θ = θ * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR
            σ = σ * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR)

        return metrics.mean_absolute_error(K.abs(y - θ), σ)

    def activation(self, y_pred: np.ndarray):
        """
        活性化関数
        """

        θ = y_pred[:, 0]
        σ = y_pred[:, 1]

        if self.IS_NORMALIZED:
            θ = K.sigmoid(θ)
            σ = K.sigmoid(σ)

        return tf.stack([θ, σ], 1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        [θ, σ]を推定
        """

        results = self.model.predict(x)

        if self.IS_NORMALIZED:
            results[:, 0] = results[:, 0] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE
            results[:, 1] = results[:, 1] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE

        return results
