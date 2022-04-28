import glob
import os

import numpy as np

from human_repository import HumanRepository
from messages import Messages
from nnet.base_nnet import BaseNNet


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
        # FIXME: データセットを削減しているので、全件に戻す
        humans = list(filter(lambda x: x.age == 10 or x.age == 30 or x.age == 60, humans))
        x: np.ndarray = np.array([human.image for human in humans])
        y: np.ndarray = np.array([human.age for human in humans])

        # 学習
        print(Messages.START_TRAINING())
        self.nnet.train(x, y)

    def clear_checkpoint(self) -> None:
        """
        チェックポイントを削除
        """

        file_names: list[str] = glob.glob(f"{self.nnet.CHECKPOINT_PATH}/*.h5")
        print(Messages.DELETE_FILES(len(file_names)))
        for file_name in file_names:
            os.remove(file_name)
