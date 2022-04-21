import glob
import os

import tqdm
from numpy import ndarray
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from human_model import HumanModel
from message import Message


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

        print(Message.LOAD_ALL_DATA())
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
        image: ndarray = img_to_array(load_img(file_name))

        return HumanModel(age=int(age), gender=int(gender), race=int(race), image=image)
