from tensorflow.keras import Sequential, layers

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
        self.model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
        self.model.add(layers.AveragePooling2D(pool_size=(2, 2)))

        self.model.add(layers.GlobalAveragePooling2D())

        self.model.add(layers.Dense(132, activation='relu'))

        self.model.add(layers.Dense(2))
        self.model.add(layers.Activation('sigmoid'))

        self.model.add(layers.Input(shape=self.INPUT_SHAPE))

        self.model.compile(
            optimizer="adam",
            loss=self.loss,
            metrics=[self.metric]
        )
