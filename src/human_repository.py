import glob
import os

import numpy as np
from numpy import ndarray
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tqdm

from human_model import HumanModel


class HumanRepository:
    """
    人間リポジトリ
    """

    SAVE_PATH: str = "data"
    """
    保存するパス
    """

    def __init__(self):
        # ディレクトリが存在しない場合は作成
        os.makedirs(self.SAVE_PATH, exist_ok=True)

    def select_all(self) -> list[HumanModel]:
        """
        人間リストを取得

        @return 人間リスト
        """

        humans: list[HumanModel] = []

        file_names: list[str] = glob.glob(f"{self.SAVE_PATH}/*.jpg")
        for file_name in tqdm.tqdm(file_names):
            humans.append(self.select_by_filename(file_name))

        return humans

    def select_by_filename(self, file_name: str) -> HumanModel:
        """
        ファイル名から人間を取得

        @return 人間
        """

        # ファイル名の命名規則は下記を参照
        # https://susanqq.github.io/UTKFace/
        age, gender, race, _ = os.path.basename(file_name).split("_")

        # 整形済みのコーパスを利用しているので、前処理は不要
        image: ndarray = img_to_array(load_img(file_name))

        return HumanModel(age=int(age), gender=int(gender), race=int(race), image=image, file_name=file_name)

    def split_train_test(self, humans: list[HumanModel], test_rate: float = 0.1) -> list[list[HumanModel]]:
        """
        学習用、検証用に人間データを分割

        @return 人間リスト
        """

        # 分割履歴を保存するファイル名
        train_file_name = f"{self.SAVE_PATH}/train_files.txt"
        test_file_name = f"{self.SAVE_PATH}/test_files.txt"

        # 分割するファイルリストを生成
        # 分割履歴が残っている場合は、それを再現する
        train_files: list[str] = []
        test_files: list[str] = []
        if os.path.exists(train_file_name) and os.path.exists(test_file_name):
            with open(train_file_name, "r") as f:
                train_files = [line.replace("\n", '') for line in f.readlines()]
            with open(test_file_name, "r") as f:
                test_files = [line.replace("\n", '') for line in f.readlines()]
        else:
            np.random.shuffle(humans)
            split_index = int(test_rate * len(humans))
            train_files = [human.file_name for human in humans[split_index:]]
            test_files = [human.file_name for human in humans[0:split_index]]

        # 学習用、検証用に分割
        humans_train = list(filter(lambda x: x.file_name in train_files, humans))
        humans_test = list(filter(lambda x: x.file_name in test_files, humans))

        # 分割結果を保存する
        with open(train_file_name, "w") as f:
            f.writelines([human.file_name + "\n" for human in humans_train])
        with open(test_file_name, "w") as f:
            f.writelines([human.file_name + "\n" for human in humans_test])

        return [humans_train, humans_test]
