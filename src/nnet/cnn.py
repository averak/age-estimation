import numpy as np
from base_nnet import BaseNNet

from tensorflow.keras import layers
from tensorflow.keras import Sequential


class CNN(BaseNNet):
    """
    CNN
    """

    def make_model(self) -> None:
        """
        NNモデルを作成
        """

        self.model = Sequential()
        self.model.add(layers.Input(shape=self.INPUT_SHAPE))

        # convolution 1st layer
        self.model.add(layers.Conv2D(32, (3, 3), padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPool2D())
        self.model.add(layers.Dropout(0.5))

        # convolution 2st layer
        self.model.add(layers.Conv2D(32, (3, 3), padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPool2D())
        self.model.add(layers.Dropout(0.5))

        # fully connected 1st layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, use_bias=False))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.Dropout(0.5))

        # fully connected final layer
        self.model.add(layers.Dense(1))
        self.model.add(layers.Activation('sigmoid'))

        # self.model.summary()

        self.model.compile(
            optimizer="adam",
            loss="mse",
            metrics=['mae']
        )

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        学習
        """

        # TODO: 実装する
        pass
