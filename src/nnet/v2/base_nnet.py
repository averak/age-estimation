import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
import tensorflow.keras.backend as K

from nnet.base_nnet import BaseNNet


class BaseNNet_V2(BaseNNet):
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
            metrics=[self.P_M_metric, self.θ_metric, self.σ_metric]
        )
        self.model.summary()

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

        return K.mean(K.switch(s == 0, L_M, L_F))

    def P_M_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        男性である確率P_Mの評価関数
        """

        s = y_true[:, 1]

        q_M = y_pred[:, 0]
        q_F = y_pred[:, 1]

        P_M = K.exp(q_M) / (K.exp(q_M) + K.exp(q_F))

        return metrics.binary_accuracy(K.constant(1.0) - s, P_M)

    def θ_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        年齢θの評価関数
        """

        y = y_true[:, 0]
        s = y_true[:, 1]

        θ_M = y_pred[:, 2]
        θ_F = y_pred[:, 3]

        if self.IS_NORMALIZE:
            y = y * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR
            θ_M = θ_M * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR
            θ_F = θ_F * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR

        θ = K.switch(
            s == 0,
            θ_M,
            θ_F,
        )

        return metrics.mean_absolute_error(y, θ)

    def σ_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        残差標準偏差σの評価関数
        """

        y = y_true[:, 0]
        s = y_true[:, 1]

        θ_M = y_pred[:, 2]
        θ_F = y_pred[:, 3]
        σ_M = K.sqrt(K.exp(y_pred[:, 4]))
        σ_F = K.sqrt(K.exp(y_pred[:, 5]))

        if self.IS_NORMALIZE:
            y = y * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR
            θ_M = θ_M * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR
            θ_F = θ_F * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR
            σ_M = σ_M * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR)
            σ_F = σ_F * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR)

        θ = K.switch(
            s == 0,
            θ_M,
            θ_F,
        )
        σ = K.switch(
            s == 0,
            σ_M,
            σ_F,
        )

        return metrics.mean_absolute_error(K.abs(y - θ), σ)

    def loss_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        lossの評価関数
        """

        y = y_true[:, 0]
        s = y_true[:, 1]

        q_M = y_pred[:, 0]
        q_F = y_pred[:, 1]
        P_M = K.exp(q_M) / (K.exp(q_M) + K.exp(q_F))
        P_F = K.exp(q_F) / (K.exp(q_M) + K.exp(q_F))
        θ_M = y_pred[:, 2]
        θ_F = y_pred[:, 3]
        σ_M = K.sqrt(K.exp(y_pred[:, 4]))
        σ_F = K.sqrt(K.exp(y_pred[:, 5]))

        L_M = K.log(2 * np.pi * (σ_M ** 2)) + ((y - θ_M) ** 2) / (σ_M ** 2 + self.EPSILON) - K.log(P_M + self.EPSILON) * 2
        L_F = K.log(2 * np.pi * (σ_F ** 2)) + ((y - θ_F) ** 2) / (σ_F ** 2 + self.EPSILON) - K.log(P_F + self.EPSILON) * 2

        return K.mean(K.switch(s == 0, L_M, L_F))

    def activation(self, y_pred: np.ndarray):
        """
        活性化関数
        """

        q_M = y_pred[:, 0]
        q_F = y_pred[:, 1]
        θ_M = y_pred[:, 2]
        θ_F = y_pred[:, 3]
        ρ_M = y_pred[:, 4]
        ρ_F = y_pred[:, 5]

        if self.IS_NORMALIZE:
            θ_M = K.sigmoid(θ_M)
            θ_F = K.sigmoid(θ_F)

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
        results[:, 2] = results[:, 2]
        results[:, 3] = results[:, 3]
        results[:, 4] = np.sqrt(np.exp(results[:, 4]))
        results[:, 5] = np.sqrt(np.exp(results[:, 5]))

        if self.IS_NORMALIZE:
            results[:, 2] = results[:, 2] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE
            results[:, 3] = results[:, 3] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE
            results[:, 4] = results[:, 4] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE
            results[:, 5] = results[:, 5] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE

        return results
