import glob
import os

import matplotlib.pyplot as plt
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
        y_train: np.ndarray = np.array([human.age for human in humans_train])
        x_test: np.ndarray = np.array([human.image for human in humans_test])
        y_test: np.ndarray = np.array([human.age for human in humans_test])

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

        theta_pred_list_train: list[float] = []
        theta_true_list_train: list[float] = []
        sigma_pred_list_train: list[float] = []
        sigma_true_list_train: list[float] = []
        theta_pred_list_test: list[float] = []
        theta_true_list_test: list[float] = []
        sigma_pred_list_test: list[float] = []
        sigma_true_list_test: list[float] = []

        for i in range(len(results_train)):
            human = humans_train[i]
            theta, sigma = results_train[i]
            theta_true_list_train.append(abs(human.age))
            theta_pred_list_train.append(abs(theta))
            sigma_true_list_train.append(abs(human.age - theta))
            sigma_pred_list_train.append(abs(sigma))

        for i in range(len(results_test)):
            human = humans_test[i]
            theta, sigma = results_test[i]
            theta_true_list_test.append(abs(human.age))
            theta_pred_list_test.append(abs(theta))
            sigma_true_list_test.append(abs(human.age - theta))
            sigma_pred_list_test.append(abs(sigma))

        # 推定年齢θのヒートマップを作成
        plt.figure()
        plt.hist2d(theta_pred_list_train, theta_true_list_train, bins=116)
        plt.xlabel("θ")
        plt.ylabel("Age")
        plt.savefig('analysis/theta_train.png')

        plt.figure()
        plt.hist2d(theta_pred_list_test, theta_true_list_test, bins=116)
        plt.xlabel("θ")
        plt.ylabel("Age")
        plt.savefig('analysis/theta_test.png')

        # 残差標準偏差σのヒートマップを作成
        plt.figure()
        plt.hist2d(sigma_pred_list_train, sigma_true_list_train, bins=116)
        plt.xlabel("σ")
        plt.ylabel("|V-θ|")
        plt.savefig('analysis/sigma_train.png')

        plt.figure()
        plt.hist2d(sigma_pred_list_test, sigma_true_list_test, bins=116)
        plt.xlabel("σ")
        plt.ylabel("|V-θ|")
        plt.savefig('analysis/sigma_test.png')

    def clear_checkpoint(self) -> None:
        """
        チェックポイントを削除
        """

        file_names: list[str] = glob.glob(f"{self.nnet.CHECKPOINT_PATH}/*.h5")
        print(Messages.DELETE_FILES(len(file_names)))
        for file_name in file_names:
            os.remove(file_name)
