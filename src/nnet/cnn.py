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

        # convolution 1st layer
        self.model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D())

        # fully connected 1st layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))

        # fully connected final layer
        self.model.add(layers.Dense(6))
        self.model.add(layers.Activation(self.activation))
