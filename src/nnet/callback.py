from tensorflow.keras import callbacks
from tensorflow.python.keras.engine.training import Model


class Callback(callbacks.Callback):
    """
    学習のコールバック
    """

    model: Model
    """
    モデル
    """

    def on_train_batch_begin(self, batch: int, logs: dict):
        """
        訓練データのバッチ開始時
        """

        number_of_layers = len(self.model.layers)

        for i in range(number_of_layers):
            self.model.layers[i].trainable = True

        # バッチ単位でモデルのフリーズ箇所を切り替える
        if batch % 2 == 0:
            # 出力層をフリーズ
            self.model.layers[number_of_layers - 1].trainable = False
        else:
            # 出力層手前までフリーズ
            for i in range(number_of_layers - 1):
                self.model.layers[i].trainable = False
