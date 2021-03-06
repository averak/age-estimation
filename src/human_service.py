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
        humans_train, humans_test = self.human_repository.split_train_test(humans)
        x_train: np.ndarray = np.array([human.image for human in humans_train])
        y_train: np.ndarray = np.array([[human.age, human.gender] for human in humans_train])
        x_test: np.ndarray = np.array([human.image for human in humans_test])
        y_test: np.ndarray = np.array([[human.age, human.gender] for human in humans_test])

        # 学習
        print(Messages.START_TRAINING())
        self.nnet.train(x_train, y_train, x_test, y_test)

    def estimate(self) -> None:
        """
        年齢を推定
        """

        self.nnet.load_weights(f"{self.nnet.CHECKPOINT_PATH}/cp-final.h5")
        humans = self.human_repository.select_all()
        humans_train, humans_test = self.human_repository.split_train_test(humans)

        x_train = [human.image for human in humans_train]
        x_test = [human.image for human in humans_test]
        results_train = self.nnet.predict(np.array(x_train))
        results_test = self.nnet.predict(np.array(x_test))

        self.nnet.export_heatmap(humans_train, humans_test, results_train, results_test)

    def analytics_log(self) -> None:
        """
        学習ログを分析
        """

        self.nnet.export_log_graph()

    def plot_age_histogram(self) -> None:
        """
        年齢のヒストグラムをプロット
        """

        print(Messages.LOAD_ALL_DATA())
        humans = self.human_repository.select_all()
        self.nnet.export_age_histogram(humans)

    def clear_checkpoint(self) -> None:
        """
        チェックポイントを削除
        """

        file_names: list[str] = glob.glob(f"{self.nnet.CHECKPOINT_PATH}/*.h5")
        print(Messages.DELETE_FILES(len(file_names)))
        for file_name in file_names:
            os.remove(file_name)
