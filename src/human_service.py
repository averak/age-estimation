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

        s_pred_list_train: list[float] = []
        s_true_list_train: list[float] = []
        θ_M_pred_list_train: list[float] = []
        θ_M_true_list_train: list[float] = []
        θ_F_pred_list_train: list[float] = []
        θ_F_true_list_train: list[float] = []
        σ_M_pred_list_train: list[float] = []
        σ_M_true_list_train: list[float] = []
        σ_F_pred_list_train: list[float] = []
        σ_F_true_list_train: list[float] = []
        s_pred_list_test: list[float] = []
        s_true_list_test: list[float] = []
        θ_M_pred_list_test: list[float] = []
        θ_M_true_list_test: list[float] = []
        θ_F_pred_list_test: list[float] = []
        θ_F_true_list_test: list[float] = []
        σ_M_pred_list_test: list[float] = []
        σ_M_true_list_test: list[float] = []
        σ_F_pred_list_test: list[float] = []
        σ_F_true_list_test: list[float] = []

        for i in range(len(results_train)):
            human = humans_train[i]
            P_M, θ_M, θ_F, σ_M, σ_F = results_train[i]
            s_pred_list_train.append(P_M)
            s_true_list_train.append(human.gender)
            if (human.gender == 0):
                θ_M_pred_list_train.append(θ_M)
                σ_M_pred_list_train.append(σ_M)
                θ_M_true_list_train.append(human.age)
                σ_M_true_list_train.append(abs(human.age - θ_M))
            else:
                θ_F_pred_list_train.append(θ_F)
                σ_F_pred_list_train.append(σ_F)
                θ_F_true_list_train.append(human.age)
                σ_F_true_list_train.append(abs(human.age - θ_F))

        for i in range(len(results_test)):
            human = humans_test[i]
            P_M, θ_M, θ_F, σ_M, σ_F = results_test[i]
            s_pred_list_test.append(P_M)
            s_true_list_test.append(human.gender)
            if (human.gender == 0):
                θ_M_pred_list_test.append(θ_M)
                σ_M_pred_list_test.append(σ_M)
                θ_M_true_list_test.append(human.age)
                σ_M_true_list_test.append(abs(human.age - θ_M))
            else:
                θ_F_pred_list_test.append(θ_F)
                σ_F_pred_list_test.append(σ_F)
                θ_F_true_list_test.append(human.age)
                σ_F_true_list_test.append(abs(human.age - θ_F))

        # 性別sのヒートマップを作成
        plt.figure()
        plt.hist2d(s_pred_list_train, s_true_list_train, bins=100, range=[(0, 1), (0, 1)])
        plt.xlabel("P_M")
        plt.ylabel("s")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig('analysis/θ_M_train.png')

        plt.figure()
        plt.hist2d(s_pred_list_test, s_true_list_test, bins=100, range=[(0, 1), (0, 1)])
        plt.xlabel("P_M")
        plt.ylabel("s")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig('analysis/θ_M_train.png')

        # 推定年齢θのヒートマップを作成
        plt.figure()
        plt.hist2d(θ_M_pred_list_train, θ_M_true_list_train, bins=116, range=[(0, 116), (0, 116)])
        plt.xlabel("θ_M")
        plt.ylabel("Age")
        plt.xlim(0, 116)
        plt.ylim(0, 116)
        plt.savefig('analysis/θ_M_train.png')

        plt.figure()
        plt.hist2d(θ_F_pred_list_train, θ_F_true_list_train, bins=116, range=[(0, 116), (0, 116)])
        plt.xlabel("θ_M")
        plt.ylabel("Age")
        plt.xlim(0, 116)
        plt.ylim(0, 116)
        plt.savefig('analysis/θ_F_train.png')

        plt.figure()
        plt.hist2d(θ_M_pred_list_test, θ_M_true_list_test, bins=116, range=[(0, 116), (0, 116)])
        plt.xlabel("θ_M")
        plt.ylabel("Age")
        plt.xlim(0, 116)
        plt.ylim(0, 116)
        plt.savefig('analysis/θ_M_train.png')

        plt.figure()
        plt.hist2d(θ_F_pred_list_test, θ_F_true_list_test, bins=116, range=[(0, 116), (0, 116)])
        plt.xlabel("θ_M")
        plt.ylabel("Age")
        plt.xlim(0, 116)
        plt.ylim(0, 116)
        plt.savefig('analysis/θ_F_train.png')

        # 残差標準偏差σのヒートマップを作成
        plt.figure()
        plt.hist2d(σ_M_pred_list_train, σ_M_true_list_train, bins=116, range=[(0, 20), (0, 50)])
        plt.xlabel("σ_M")
        plt.ylabel("|y-θ_M|")
        plt.xlim(0, 20)
        plt.ylim(0, 50)
        plt.savefig('analysis/σ_M_train.png')

        plt.figure()
        plt.hist2d(σ_F_pred_list_train, σ_F_true_list_train, bins=116, range=[(0, 20), (0, 50)])
        plt.xlabel("σ_F")
        plt.ylabel("|y-θ_F|")
        plt.xlim(0, 20)
        plt.ylim(0, 50)
        plt.savefig('analysis/σ_F_train.png')

        plt.figure()
        plt.hist2d(σ_M_pred_list_test, σ_M_true_list_test, bins=116, range=[(0, 20), (0, 50)])
        plt.xlabel("σ_M")
        plt.ylabel("|y-θ_M|")
        plt.xlim(0, 20)
        plt.ylim(0, 50)
        plt.savefig('analysis/σ_M_test.png')

        plt.figure()
        plt.hist2d(σ_F_pred_list_test, σ_F_true_list_test, bins=116, range=[(0, 20), (0, 50)])
        plt.xlabel("σ_F")
        plt.ylabel("|y-θ_F|")
        plt.xlim(0, 20)
        plt.ylim(0, 50)
        plt.savefig('analysis/σ_F_test.png')

    def clear_checkpoint(self) -> None:
        """
        チェックポイントを削除
        """

        file_names: list[str] = glob.glob(f"{self.nnet.CHECKPOINT_PATH}/*.h5")
        print(Messages.DELETE_FILES(len(file_names)))
        for file_name in file_names:
            os.remove(file_name)
