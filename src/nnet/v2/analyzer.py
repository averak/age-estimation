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

        s_pred_list_train: list[float] = []
        s_true_list_train: list[float] = []
        s_collect_rate_train: float = 0.0
        s_collect_rate_test: float = 0.0
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

        for i in range(len(train_results)):
            human = train_humans[i]
            P_M, P_F, θ_M, θ_F, σ_M, σ_F = train_results[i]
            s_pred_list_train.append(P_M)
            s_true_list_train.append(1 - human.gender)
            if (round(P_M) == 1):
                θ_M_pred_list_train.append(θ_M)
                σ_M_pred_list_train.append(σ_M)
                θ_M_true_list_train.append(human.age)
                σ_M_true_list_train.append(abs(human.age - θ_M))
            else:
                θ_F_pred_list_train.append(θ_F)
                σ_F_pred_list_train.append(σ_F)
                θ_F_true_list_train.append(human.age)
                σ_F_true_list_train.append(abs(human.age - θ_F))

        for i in range(len(test_results)):
            human = test_humans[i]
            P_M, P_F, θ_M, θ_F, σ_M, σ_F = test_results[i]
            s_pred_list_test.append(P_M)
            s_true_list_test.append(1 - human.gender)
            if (round(P_M) == 1):
                θ_M_pred_list_test.append(θ_M)
                σ_M_pred_list_test.append(σ_M)
                θ_M_true_list_test.append(human.age)
                σ_M_true_list_test.append(abs(human.age - θ_M))
            else:
                θ_F_pred_list_test.append(θ_F)
                σ_F_pred_list_test.append(σ_F)
                θ_F_true_list_test.append(human.age)
                σ_F_true_list_test.append(abs(human.age - θ_F))

        s_collect_rate_train = len(list(filter(lambda i: (1 - train_humans[i].gender) == round(s_pred_list_train[i]), range(len(train_humans))))) / len(train_humans) * 100.0
        s_collect_rate_test = len(list(filter(lambda i: (1 - test_humans[i].gender) == round(s_pred_list_test[i]), range(len(test_humans))))) / len(test_humans) * 100.0
        print("学習用データの性別正解率: %3.2f%%" % s_collect_rate_train)
        print("検証用データの性別正解率: %3.2f%%" % s_collect_rate_test)

        # 性別sのヒートマップを作成
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(s_pred_list_train, s_true_list_train, bins=10, range=[(0, 1), (0, 1)])
        plt.colorbar()
        plt.xlabel("P_M")
        plt.ylabel("s")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig('analysis/heatmap/s_train.png')

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(s_pred_list_test, s_true_list_test, bins=10, range=[(0, 1), (0, 1)])
        plt.colorbar()
        plt.xlabel("P_M")
        plt.ylabel("s")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig('analysis/heatmap/s_test.png')

        # 推定年齢θのヒートマップを作成
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(θ_M_pred_list_train, θ_M_true_list_train, bins=116, range=[(0, 116), (0, 116)])
        plt.colorbar()
        plt.xlabel("θ_M")
        plt.ylabel("Age")
        plt.xlim(0, 116)
        plt.ylim(0, 116)
        plt.savefig('analysis/heatmap/θ_M_train.png')

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(θ_F_pred_list_train, θ_F_true_list_train, bins=116, range=[(0, 116), (0, 116)])
        plt.colorbar()
        plt.xlabel("θ_F")
        plt.ylabel("Age")
        plt.xlim(0, 116)
        plt.ylim(0, 116)
        plt.savefig('analysis/heatmap/θ_F_train.png')

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(θ_M_pred_list_test, θ_M_true_list_test, bins=116, range=[(0, 116), (0, 116)])
        plt.colorbar()
        plt.xlabel("θ_M")
        plt.ylabel("Age")
        plt.xlim(0, 116)
        plt.ylim(0, 116)
        plt.savefig('analysis/heatmap/θ_M_test.png')

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(θ_F_pred_list_test, θ_F_true_list_test, bins=116, range=[(0, 116), (0, 116)])
        plt.colorbar()
        plt.xlabel("θ_F")
        plt.ylabel("Age")
        plt.xlim(0, 116)
        plt.ylim(0, 116)
        plt.savefig('analysis/heatmap/θ_F_test.png')

        # 残差標準偏差σのヒートマップを作成
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_M_pred_list_train, σ_M_true_list_train, bins=80, range=[(0, 10), (0, 10)])
        plt.colorbar()
        plt.xlabel("σ_M")
        plt.ylabel("|y-θ_M|")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig('analysis/heatmap/σ_M_train.png')

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_F_pred_list_train, σ_F_true_list_train, bins=80, range=[(0, 10), (0, 10)])
        plt.colorbar()
        plt.xlabel("σ_F")
        plt.ylabel("|y-θ_F|")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig('analysis/heatmap/σ_F_train.png')

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_M_pred_list_test, σ_M_true_list_test, bins=80, range=[(0, 50), (0, 50)])
        plt.colorbar()
        plt.xlabel("σ_M")
        plt.ylabel("|y-θ_M|")
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.savefig('analysis/heatmap/σ_M_test.png')

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        plt.hist2d(σ_F_pred_list_test, σ_F_true_list_test, bins=80, range=[(0, 50), (0, 50)])
        plt.colorbar()
        plt.xlabel("σ_F")
        plt.ylabel("|y-θ_F|")
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        for i in range(len(train_results)):
            plt.savefig('analysis/heatmap/σ_F_test.png')

    def export_log_graph(self):
        """
        ロググラフを出力
        """
        loss_list: list[float] = []
        val_loss_list: list[float] = []
        P_M_metric_list: list[float] = []
        val_P_M_metric_list: list[float] = []
        θ_metric_list: list[float] = []
        val_θ_metric_list: list[float] = []
        σ_metric_list: list[float] = []
        val_σ_metric_list: list[float] = []

        with open('analysis/heatmap/log.csv', 'r') as f:
            # ヘッダーを読み飛ばす
            next(csv.reader(f))

            for row in csv.reader(f):
                columns = [float(column) for column in row]
                epoch, P_M_metric, loss, val_P_M_metric, val_loss, val_θ_metric, val_σ_metric, θ_metric, σ_metric = columns

                P_M_metric_list.append(P_M_metric)
                val_P_M_metric_list.append(val_P_M_metric)
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
        plt.savefig('analysis/heatmap/log/loss.png')

        # show P_M graph
        plt.figure()
        plt.plot(range(len(loss_list)), P_M_metric_list, label="train")
        plt.plot(range(len(loss_list)), val_P_M_metric_list, label="validation")
        plt.xlabel("epoch")
        plt.ylabel("binary accuracy")
        plt.savefig('analysis/heatmap/log/P_M.png')

        # show θ graph
        plt.figure()
        plt.plot(range(len(loss_list)), θ_metric_list, label="train")
        plt.plot(range(len(loss_list)), val_θ_metric_list, label="validation")
        plt.xlabel("epoch")
        plt.ylabel("θ mae")
        plt.savefig('analysis/heatmap/log/θ.png')

        # show σ graph
        plt.figure()
        plt.plot(range(len(loss_list)), σ_metric_list, label="train")
        plt.plot(range(len(loss_list)), val_σ_metric_list, label="validation")
        plt.xlabel("epoch")
        plt.ylabel("σ mae")
        plt.savefig('analysis/heatmap/log/σ.png')
