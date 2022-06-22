import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
import tensorflow.keras.backend as K

from nnet.base_nnet import BaseNNet
from nnet.v1.analyzer import Analyzer


class BaseNNet_V1(BaseNNet):
    """
    ニューラルネットワーク
    """

    analyzer: Analyzer = Analyzer()
    """
    解析器
    """

    def compile_model(self):
        """
        NNモデルをコンパイル
        """

        self.model.compile(
            optimizer=self.OPTIMIZER,
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
        ρ = y_pred[:, 1]

        return K.mean(ρ + ((y - θ) ** 2) * K.exp(-ρ))

    def θ_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        年齢θの評価関数
        """

        y = y_true[:, 0]

        θ = y_pred[:, 0]

        if self.IS_NORMALIZE:
            y = y * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR
            θ = θ * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR

        return metrics.mean_absolute_error(y, θ)

    def σ_metric(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        残差標準偏差σの評価関数
        """

        y = y_true[:, 0]

        θ = y_pred[:, 0]
        σ = K.sqrt(K.exp(y_pred[:, 1]))

        if self.IS_NORMALIZE:
            y = y * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR
            θ = θ * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR) + self.MIN_AGE_TENSOR
            σ = σ * (self.MAX_AGE_TENSOR - self.MIN_AGE_TENSOR)

        return metrics.mean_absolute_error(K.abs(y - θ), σ)

    def activation(self, y_pred: np.ndarray):
        """
        活性化関数
        """

        θ = y_pred[:, 0]
        ρ = y_pred[:, 1]

        if self.IS_NORMALIZE:
            θ = K.sigmoid(θ)

        return tf.stack([θ, ρ], 1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        [θ, σ]を推定
        """

        results = self.model.predict(x)

        results[:, 0] = results[:, 0]
        results[:, 1] = np.sqrt(np.exp(results[:, 1]))

        if self.IS_NORMALIZE:
            results[:, 0] = results[:, 0] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE
            results[:, 1] = results[:, 1] * (self.MAX_AGE - self.MIN_AGE) + self.MIN_AGE

        return results

    def export_heatmap(self, train_humans: list, test_humans: list, train_results: np.ndarray, test_results: np.ndarray):
        """
        ヒートマップを出力
        """

        self.analyzer.export_heatmap(train_humans, test_humans, train_results, test_results)

    def export_log_graph(self):
        """
        ロググラフを出力
        """

        self.analyzer.export_log_graph()
