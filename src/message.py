class Message:
    """
    表示するメッセージを管理
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
