import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing

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
        x: np.ndarray = np.array([human.image for human in humans])
        y: np.ndarray = sklearn.preprocessing.minmax_scale([human.age for human in humans])

        # 学習
        print(Messages.START_TRAINING())
        self.nnet.train(x, y)

    def estimate(self) -> None:
        """
        年齢を推定
        """

        self.nnet.load_weights(f"{self.nnet.CHECKPOINT_PATH}/cp-final.h5")
        humans = self.human_repository.select_all()

        human_images = np.array([human.image for human in humans])
        results = self.nnet.predict(human_images)

        theta_pred_list: list[float] = []
        theta_true_list: list[float] = []
        sigma_pred_list: list[float] = []
        sigma_true_list: list[float] = []

        for i in range(len(results)):
            human = humans[i]
            theta, sigma = results[i]
            theta_true_list.append(abs(human.age))
            theta_pred_list.append(abs(theta))
            sigma_true_list.append(abs(human.age - theta))
            sigma_pred_list.append(abs(sigma))

        # 推定年齢θのヒートマップを作成
        plt.hist2d(theta_pred_list, theta_true_list, bins=116)
        plt.xlabel("θ")
        plt.ylabel("Age")
        plt.savefig('analysis/theta.png')

        # 残差標準偏差σのヒートマップを作成
        plt.hist2d(sigma_pred_list, sigma_true_list, bins=116)
        plt.xlabel("σ")
        plt.ylabel("|V-θ|")
        plt.savefig('analysis/sigma.png')

    def clear_checkpoint(self) -> None:
        """
        チェックポイントを削除
        """

        file_names: list[str] = glob.glob(f"{self.nnet.CHECKPOINT_PATH}/*.h5")
        print(Messages.DELETE_FILES(len(file_names)))
        for file_name in file_names:
            os.remove(file_name)
