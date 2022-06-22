from tensorflow.keras import Sequential, layers

from nnet.v2.base_nnet import BaseNNet_V2


class CNN_V2(BaseNNet_V2):
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

        # convolution 2nd layer
        self.model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D())

        # fully connected 1st layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.BatchNormalization())

        # fully connected final layer
        self.model.add(layers.Dense(6, activation=self.activation))
