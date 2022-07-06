import argparse

from human_service import HumanService
from nnet.v1.cnn import CNN_V1
from nnet.v2.cnn import CNN_V2

# アプリケーションのオプションを定義
# --helpでヘルプを表示できます
argument_parser: argparse.ArgumentParser = argparse.ArgumentParser()
argument_parser.add_argument('-t', '--train',
                             help='学習',
                             action='store_true')
argument_parser.add_argument('-e', '--estimate',
                             help='推定',
                             action='store_true')
argument_parser.add_argument('-l', '--log_analytics',
                             help='学習ログ分析',
                             action='store_true')
argument_parser.add_argument('-a', '--age_histogram',
                             help='年齢のヒストグラムを表示',
                             action='store_true')
argument_parser.add_argument('-v', '--version', help='バージョン(1 or 2)', type=int, default=1)
argument_parser.add_argument('-n', '--normalize', help='正規化するか', action='store_true')
argument_parser.add_argument('-c', '--callback', help='コールバックするか', action='store_true')
argument_parser.add_argument('-sd', '--switch_data', help='データを切り替える', action='store_true')
argument_parser.add_argument('-lr', '--learning_rate', help='学習率', type=float, default=0.001)
arguments = argument_parser.parse_args()

# アプリケーションサービスを作成
if arguments.version == 1:
    nnet = CNN_V1(arguments.normalize, arguments.callback, arguments.switch_data, arguments.learning_rate)
else:
    nnet = CNN_V2(arguments.normalize, arguments.callback, arguments.switch_data, arguments.learning_rate)
human_service = HumanService(nnet)

if arguments.train:
    human_service.train()
elif arguments.estimate:
    human_service.estimate()
elif arguments.log_analytics:
    human_service.analytics_log()
elif arguments.age_histogram:
    human_service.plot_age_histogram()
else:
    argument_parser.print_help()
