from tensorflow.python.keras.engine.training import Model

from tensorflow.keras import callbacks


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

        split_border = 3

        # バッチ単位でモデルのフリーズ箇所を切り替える
        if batch % 2 == 0:
            # 特徴抽出部をフリーズ
            for i in range(number_of_layers - split_border):
                self.model.layers[number_of_layers - i - 1].trainable = False
        else:
            # 識別部をフリーズ
            for i in range(split_border + 1):
                self.model.layers[i].trainable = False
