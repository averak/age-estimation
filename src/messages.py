class Messages:
    """
    表示するメッセージ
    """

    class FontColors:
        """
        標準出力の文字色
        """

        RED: str = '\033[31m'
        GREEN: str = '\033[32m'
        YELLOW: str = '\033[33m'
        RESET: str = '\033[0m'

    @classmethod
    def LOAD_ALL_DATA(cls) -> str:
        return "データを読み込み中..."

    @classmethod
    def START_TRAINING(cls) -> str:
        return "学習開始"

    @classmethod
    def DELETE_FILES(cls, file_size: int) -> str:
        return f"{file_size}ファイル削除しました"

    @classmethod
    def RESTART_TRAIN(cls, epoch: int) -> str:
        return f"{epoch}目から学習を再開します"
