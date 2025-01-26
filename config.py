from typing import List

class Config:
    def __init__(self):
        self.SYMBOL: str = "AAPL"
        self.TIMEFRAME: str = "1d"
        self.LEARNING_RATE: float = 0.001
        self.BATCH_SIZE: int = 32
        self.EPOCHS: int = 50
        self.SEQUENCE_LENGTH: int = 60
        self.FEATURES: List[str] = ["Open", "High", "Low", "Close", "Volume"]
        self.TECHNICAL_INDICATORS: List[str] = ["RSI", "MACD", "BBands"]
        self.MODEL_PATH: str = "models/lstm_model.h5"
        self.DATA_PATH: str = "data/"
        self.LOG_PATH: str = "logs/"