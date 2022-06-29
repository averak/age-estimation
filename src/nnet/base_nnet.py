import numpy as np
from tensorflow.keras import Model, optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

from messages import Messages
from nnet.callback import Callback
from nnet.data_generator import DataGenerator


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

    EPOCHS: int = 250
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

    IS_NORMALIZE = True
    """
    正規化するか
    """

    IS_CALLBACK: bool = True
    """
    コールバックするか
    """

    OPTIMIZER = optimizers.Adam(learning_rate=0.001)
    """
    オプティマイザ
    """

    def __init__(self, normalize: bool, callback: bool, learning_rate: float):
        self.IS_NORMALIZE = normalize
        self.IS_CALLBACK = callback
        self.OPTIMIZER = optimizers.Adam(learning_rate=learning_rate)

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

        y_train = np.array(y_train, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)

        if self.IS_NORMALIZE:
            y_train[:, 0] = (y_train[:, 0] - self.MIN_AGE) / (self.MAX_AGE - self.MIN_AGE)
            y_test[:, 0] = (y_test[:, 0] - self.MIN_AGE) / (self.MAX_AGE - self.MIN_AGE)

        # チェックポイントを保存するコールパックを定義
        checkpoint_file = "%s/cp-{epoch}.h5" % self.CHECKPOINT_PATH
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_file,
            verbose=1,
            save_weights_only=True
        )
        self.model.save_weights(checkpoint_file.format(epoch=0))

        # CSVにロギングするコールバックを定義
        csv_logger = CSVLogger('analysis/log.csv', separator=',')

        # 監視する値の変化が停止した時に訓練を終了させるコールバックを定義
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

        # 学習
        callbacks = []
        if self.IS_CALLBACK:
            callbacks = [checkpoint_callback, csv_logger, early_stopping, Callback()]
        else:
            callbacks = [checkpoint_callback, csv_logger, early_stopping]
        self.model.fit_generator(
            DataGenerator(x_train, y_train, self.BATCH_SIZE, True),
            epochs=self.EPOCHS,
            validation_data=(x_test, y_test),
            callbacks=callbacks
        )

        # データ群AとBを切り替えて学習
        print(Messages.RESTART_TRAIN(early_stopping.stopped_epoch))
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
        self.model.fit_generator(
            DataGenerator(x_train, y_train, self.BATCH_SIZE, False),
            initial_epoch=early_stopping.stopped_epoch,
            epochs=self.EPOCHS,
            validation_data=(x_test, y_test),
            callbacks=callbacks
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        推定
        """

        raise NotImplementedError()

    def export_heatmap(self, train_humans: list, test_humans: list, train_results: np.ndarray, test_results: np.ndarray):
        """
        ヒートマップを出力
        """

        raise NotImplementedError()

    def export_log_graph(self):
        """
        ロググラフを出力
        """

        raise NotImplementedError()
