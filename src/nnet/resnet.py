from tensorflow.keras import Model, layers
from tensorflow.keras.applications import ResNet50

from nnet.base_nnet import BaseNNet


class ResNet(BaseNNet):
    """
    ResNet
    """

    def make_model(self) -> None:
        """
        NNモデルを作成
        """

        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=self.INPUT_SHAPE, pooling="avg")
        prediction = layers.Dense(2, use_bias=False, activation="sigmoid")(base_model.output)
        self.model = Model(inputs=base_model.input, outputs=prediction)

        self.model.compile(
            optimizer="adam",
            loss=self.loss,
            metrics=[self.metric]
        )
