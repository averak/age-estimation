import numpy as np
import sklearn.preprocessing
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

from nnet.callback import Callback


class BaseNNet:
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

    EPOCHS: int = 500
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

    EPSILON = K.constant(K.epsilon())
    """
    ε
    """

    MIN_AGE = 1.0
    """
    年齢の最小値
    """

    MAX_AGE = 116.0
    """
    年齢の最大値
    """

    MIN_AGE_TENSOR = K.constant(MIN_AGE)
    """
    最小年齢のTensor
    """

    MAX_AGE_TENSOR = K.constant(MAX_AGE)
    """
    最大年齢のTensor
    """

    IS_NORMALIZED = True
    """
    正規化するか
    """

    def __init__(self):
        self.make_model()
        self.compile_model()

    def make_model(self) -> None:
        """
        NNモデルを作成
        """

        raise NotImplementedError()

    def compile_model(self) -> None:
        """
        NNモデルをコンパイル
        """

        raise NotImplementedError()

    def load_weights(self, file_name: str) -> None:
        """
        学習済みモデルを読み込む
        """

        self.model.load_weights(file_name)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        学習
        """

        if self.IS_NORMALIZED:
            y_train[:, 0] = sklearn.preprocessing.minmax_scale(y_train[:, 0])
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
            shuffle=False,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=[checkpoint_callback, Callback()]
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        推定
        """

        raise NotImplementedError()
