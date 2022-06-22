import numpy as np
from sklearn import utils
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """
    バッチデータ生成器
    """

    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

        # データを2グループに分割
        self.x1, self.x2 = np.array_split(self.x, 2)
        self.y1, self.y2 = np.array_split(self.y, 2)

    def __len__(self):
        length = int(len(self.x) / self.batch_size)
        if length * self.batch_size < len(self.x):
            length += 1
        return length

    def __getitem__(self, batch: int):
        index = batch // 2
        if batch % 2 == 0:
            x = self.x1[index * self.batch_size:(index + 1) * self.batch_size]
            y = self.y1[index * self.batch_size:(index + 1) * self.batch_size]
        else:
            x = self.x2[index * self.batch_size:(index + 1) * self.batch_size]
            y = self.y2[index * self.batch_size:(index + 1) * self.batch_size]

        return x, y

    def on_epoch_end(self):
        self.x1, self.y1 = utils.shuffle(self.x1, self.y1)
        self.x2, self.y2 = utils.shuffle(self.x2, self.y2)
