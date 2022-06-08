from tensorflow.keras import Sequential, layers, regularizers

from nnet.base_nnet import BaseNNet


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
        # self.model.add(layers.Dropout(0.2))

        # convolution 1st layer
        self.model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D())
        # self.model.add(layers.Dropout(0.5))

        # convolution 1st layer
        self.model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D())
        # self.model.add(layers.Dropout(0.5))

        # fully connected 1st layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        # self.model.add(layers.Dropout(0.5))

        # fully connected final layer
        self.model.add(layers.Dense(6))
        self.model.add(layers.Activation(self.activation))
