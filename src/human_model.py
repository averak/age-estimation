import numpy as np
from pydantic import BaseModel


class HumanModel(BaseModel):
    """
    人間モデル
    """

    age: int
    """
    年齢
    """

    gender: int
    """
    性別(0: 男性、1: 女性)
    """

    race: int
    """
    人種: (0: 白、1: 黒、2: アジア、3: インド、4: その他)
    """

    image: np.ndarray
    """
    画像データ(200 * 200 * 3)
    """

    class Config:
        arbitrary_types_allowed = True
