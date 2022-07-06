import csv

import matplotlib.pyplot as plt
import numpy as np

from nnet.base_analyzer import BaseAnalyzer


class Analyzer(BaseAnalyzer):
    """
    解析器
    """

    def export_heatmap(self, train_humans: list, test_humans: list, train_results: np.ndarray, test_results: np.ndarray):
        """
        ヒートマップを出力
        """

        θ_pred_list_train: list[float] = []
        θ_true_list_train: list[float] = []
        σ_pred_list_train: list[float] = []
        σ_true_list_train: list[float] = []
        θ_pred_list_test: list[float] = []
        θ_true_list_test: list[float] = []
        σ_pred_list_test: list[float] = []
        σ_true_list_test: list[float] = []

        for i in range(len(train_results)):
            human = train_humans[i]
            θ, σ = train_results[i]
            θ_pred_list_train.append(θ)
            σ_pred_list_train.append(σ)
            θ_true_list_train.append(human.age)
            σ_true_list_train.append(abs(human.age - θ))

        for i in range(len(test_results)):
            human = test_humans[i]
            θ, σ = test_results[i]
            θ_pred_list_test.append(θ)
            σ_pred_list_test.append(σ)
            θ_true_list_test.append(human.age)
            σ_true_list_test.append(abs(human.age - θ))

        # 推定年齢θのヒートマップを作成
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(θ_pred_list_train, θ_true_list_train, bins=90, range=[(0, 90), (0, 90)], cmin=1)
        plt.colorbar()
        plt.xlabel("θ")
        plt.ylabel("Age")
        plt.xlim(0, 90)
        plt.ylim(0, 90)
        plt.savefig('analysis/heatmap/θ_train.png')

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(θ_pred_list_test, θ_true_list_test, bins=90, range=[(0, 90), (0, 90)], cmin=1)
        plt.colorbar()
        plt.xlabel("θ")
        plt.ylabel("Age")
        plt.xlim(0, 90)
        plt.ylim(0, 90)
        plt.savefig('analysis/heatmap/θ_test.png')

        # 残差標準偏差σのヒートマップを作成
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_pred_list_train, σ_true_list_train, bins=80, range=[(0, 50), (0, 50)], cmin=1)
        plt.colorbar()
        plt.xlabel("σ")
        plt.ylabel("|y - θ|")
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.savefig('analysis/heatmap/σ_train.png')

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_pred_list_test, σ_true_list_test, bins=80, range=[(0, 50), (0, 50)], cmin=1)
        plt.colorbar()
        plt.xlabel("σ")
        plt.ylabel("|y - θ|")
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.savefig('analysis/heatmap/σ_test.png')

    def export_log_graph(self):
        """
        ロググラフを出力
        """
        loss_list: list[float] = []
        val_loss_list: list[float] = []
        θ_metric_list: list[float] = []
        val_θ_metric_list: list[float] = []
        σ_metric_list: list[float] = []
        val_σ_metric_list: list[float] = []

        with open('analysis/log.csv', 'r') as f:
            # ヘッダーを読み飛ばす
            next(csv.reader(f))

            for row in csv.reader(f):
                columns = [float(column) for column in row]
                epoch, loss, val_loss, val_θ_metric, val_σ_metric, θ_metric, σ_metric = columns

                loss_list.append(loss)
                val_loss_list.append(val_loss)
                θ_metric_list.append(θ_metric)
                val_θ_metric_list.append(val_θ_metric)
                σ_metric_list.append(σ_metric)
                val_σ_metric_list.append(val_σ_metric)

        # show loss graph
        plt.figure()
        plt.plot(range(len(loss_list)), loss_list, label="train")
        plt.plot(range(len(loss_list)), val_loss_list, label="validation")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig('analysis/log/loss.png')

        # show θ graph
        plt.figure()
        plt.plot(range(len(loss_list)), θ_metric_list, label="train")
        plt.plot(range(len(loss_list)), val_θ_metric_list, label="validation")
        plt.xlabel("epoch")
        plt.ylabel("θ mae")
        plt.savefig('analysis/log/θ.png')

        # show σ graph
        plt.figure()
        plt.plot(range(len(loss_list)), σ_metric_list, label="train")
        plt.plot(range(len(loss_list)), val_σ_metric_list, label="validation")
        plt.xlabel("epoch")
        plt.ylabel("σ mae")
        plt.savefig('analysis/log/σ.png')
