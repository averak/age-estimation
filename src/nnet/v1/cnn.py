from tensorflow.keras import Sequential, layers

from nnet.v1.base_nnet import BaseNNet_V1


class CNN_V1(BaseNNet_V1):
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
        self.model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D())

        # fully connected 1st layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.BatchNormalization())

        # fully connected final layer
        self.model.add(layers.Dense(2, activation=self.activation))
