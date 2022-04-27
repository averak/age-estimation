import numpy as np

from human_repository import HumanRepository
from messages import Messages
from nnet.base_nnet import BaseNNet
import sklearn


class HumanService:
    """
    人間サービス
    """

    human_repository: HumanRepository
    """
    人間リポジトリ
    """

    nnet: BaseNNet
    """
    ニューラルネットワーク
    """

    def __init__(self, nnet: BaseNNet):
        self.human_repository = HumanRepository()
        self.nnet = nnet

    def train(self) -> None:
        """
        学習
        """

        # データセットを作成
        print(Messages.LOAD_ALL_DATA())
        humans = self.human_repository.select_all()
        x: np.ndarray = np.array([human.image for human in humans])
        y: np.ndarray = np.array([human.age for human in humans])

        # 学習
        print(Messages.START_TRAINING())
        self.nnet.train(x, y)
