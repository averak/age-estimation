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
argument_parser.add_argument('-c', '--clear',
                             help='チェックポイントを削除',
                             action='store_true')
argument_parser.add_argument('-v', '--version', help='バージョン(1 or 2)', type=int, default=1)
arguments = argument_parser.parse_args()

# アプリケーションサービスを作成
if arguments.version == 1:
    nnet = CNN_V1()
else:
    nnet = CNN_V2()
human_service = HumanService(nnet)

if arguments.train:
    human_service.train()
elif arguments.estimate:
    human_service.estimate()
elif arguments.clear:
    human_service.clear_checkpoint()
else:
    argument_parser.print_help()
