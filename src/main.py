import argparse

from human_service import HumanService
from nnet.cnn import CNN

# アプリケーションサービスを作成
human_service: HumanService = HumanService(CNN())

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
arguments = argument_parser.parse_args()

if arguments.train:
    human_service.train()
elif arguments.estimate:
    human_service.estimate()
elif arguments.clear:
    human_service.clear_checkpoint()
else:
    argument_parser.print_help()
