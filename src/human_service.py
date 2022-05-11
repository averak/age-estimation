import glob
import os
import random

import numpy as np
import sklearn.preprocessing

from human_model import HumanModel
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
        # y: np.ndarray = np.array([human.age / 116.0 for human in humans])
        y: np.ndarray = sklearn.preprocessing.minmax_scale([human.age for human in humans])

        # 学習
        print(Messages.START_TRAINING())
        self.nnet.train(x, y)

    def estimate(self) -> None:
        """
        年齢を推定
        """

        # TODO: 試しに30件だけ推定しているが、指定した年齢のみ推定など別の方法を考える
        self.nnet.load_weights(f"{self.nnet.CHECKPOINT_PATH}/cp-21.h5")
        humans = self.human_repository.select_all()
        humans = list(filter(lambda x: x.age % 30 == 0, humans))
        random.shuffle(humans)

        test_cases: list[HumanModel] = []
        test_cases.extend(list(filter(lambda x: x.age == 30, humans))[:100])
        test_cases.extend(list(filter(lambda x: x.age == 60, humans))[:100])
        test_cases.extend(list(filter(lambda x: x.age == 90, humans))[:100])

        class ResultModel:
            age: int
            theta: float
            sigma: float

            def __init__(self, age: int, theta: float, sigma: float):
                self.age = age
                self.theta = theta
                self.sigma = sigma

            def show_detail(self):
                print("%d歳: 推定年齢θ = %4.2f, 残差標準偏差σ = %4.2f" % (result.age, result.theta, result.sigma))

        results: list[ResultModel] = []
        for human in test_cases:
            theta, sigma = self.nnet.predict(np.array([human.image]))
            results.append(ResultModel(human.age, theta, sigma))

        results_30 = list(filter(lambda x: x.age == 30, results))
        results_30.sort(key=lambda x: x.theta)
        results_60 = list(filter(lambda x: x.age == 60, results))
        results_60.sort(key=lambda x: x.theta)
        results_90 = list(filter(lambda x: x.age == 90, results))
        results_90.sort(key=lambda x: x.theta)

        for result in results_30:
            result.show_detail()
        for result in results_60:
            result.show_detail()
        for result in results_90:
            result.show_detail()

    def clear_checkpoint(self) -> None:
        """
        チェックポイントを削除
        """

        file_names: list[str] = glob.glob(f"{self.nnet.CHECKPOINT_PATH}/*.h5")
        print(Messages.DELETE_FILES(len(file_names)))
        for file_name in file_names:
            os.remove(file_name)
