import numpy as np
import pandas as pd
import ta
from typing import Tuple
from multiprocessing import Process, Queue
import yfinance as yf
from config import Config
import time
import traceback
import os

class DataProcessor(Process):
    def __init__(self, config: Config, data_queue: Queue):
        super().__init__()
        self.config = config
        self.data_queue = data_queue
        
    def get_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if "RSI" in self.config.TECHNICAL_INDICATORS:
            # DataFrame'den direkt numpy array'e çevirip kullanıyoruz
            close_prices = df['Close'].values  # pandas Series'ten numpy array'e
            df['RSI'] = ta.momentum.RSIIndicator(close_prices).rsi()
        if "MACD" in self.config.TECHNICAL_INDICATORS:
            close_prices = df['Close'].values
            macd = ta.trend.MACD(close_prices)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
        if "BBands" in self.config.TECHNICAL_INDICATORS:
            close_prices = df['Close'].values
            bb = ta.volatility.BollingerBands(close_prices)
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
            df['BB_Mid'] = bb.bollinger_mavg()
        return df
       

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # Fix column names if MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Add technical indicators
            df = self.get_technical_indicators(df)
            
            # Create sequences
            sequences = []
            targets = []
            
            # Get base features
            base_data = df[self.config.FEATURES].values
            
            for i in range(len(df) - self.config.SEQUENCE_LENGTH):
                seq = base_data[i:(i + self.config.SEQUENCE_LENGTH)]
                target = df['Close'].iloc[i + self.config.SEQUENCE_LENGTH]
                sequences.append(seq)
                targets.append(target)
            
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            return sequences, targets
            
        except Exception as e:
            print(f"Error in preprocess_data: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            
            raise

    def save_data_sample(self, df):
            try:
                # Son 1000 veriyi kaydedelim
                sample_file = os.path.join(self.config.DATA_PATH, f"{self.config.SYMBOL}_latest.csv")
                os.makedirs(self.config.DATA_PATH, exist_ok=True)
                df.tail(1000).to_csv(sample_file)
                print(f"Data sample saved to: {sample_file}")
            except Exception as e:
                print(f"Error saving data: {e}")
    def run(self):
        while True:
            try:
                # Fetch data
                df = yf.download(self.config.SYMBOL, interval=self.config.TIMEFRAME)
                
                # Process data
                sequences, targets = self.preprocess_data(df)
                
                # Convert timestamp to string for JSON serialization
                current_time = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                
                # Send to model process
                self.data_queue.put((sequences, targets, current_time))
                
                time.sleep(1)  # Add small delay
                
            except Exception as e:
                print(f"Data processing error: {e}")
                print(f"Full traceback: {traceback.format_exc()}")

                time.sleep(1)
    def __init__(self, config: Config, data_queue: Queue):
        super().__init__()
        self.config = config
        self.data_queue = data_queue
        
    def get_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if "RSI" in self.config.TECHNICAL_INDICATORS:
            # pandas Series olarak gönderiyoruz
            df['RSI'] = ta.momentum.RSIIndicator(pd.Series(df['Close'])).rsi()
        if "MACD" in self.config.TECHNICAL_INDICATORS:
            macd = ta.trend.MACD(pd.Series(df['Close']))
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
        if "BBands" in self.config.TECHNICAL_INDICATORS:
            bb = ta.volatility.BollingerBands(pd.Series(df['Close']))
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
            df['BB_Mid'] = bb.bollinger_mavg()
        return df

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # Add technical indicators
            df = self.get_technical_indicators(df)
            
            # Select only the required features
            feature_data = df[self.config.FEATURES].values
            
            # Create sequences for LSTM
            sequences = []
            targets = []
            
            for i in range(len(df) - self.config.SEQUENCE_LENGTH):
                seq = feature_data[i:(i + self.config.SEQUENCE_LENGTH)]
                target = df['Close'].iloc[i + self.config.SEQUENCE_LENGTH]
                sequences.append(seq)
                targets.append(target)
            
            # Convert to numpy arrays with correct shapes
            sequences = np.array(sequences)  # Shape: (n_samples, sequence_length, n_features)
            targets = np.array(targets)      # Shape: (n_samples,)
            
            # Print shapes for debugging
            #print(f"Sequences shape: {sequences.shape}")
            #print(f"Targets shape: {targets.shape}")
            
            return sequences, targets
            
        except Exception as e:
            print(f"Error in preprocess_data: {str(e)}")
            raise

    def run(self):
        df = yf.download(self.config.SYMBOL, interval=self.config.TIMEFRAME)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df = pd.DataFrame({
            'Close': df['Close'].to_numpy().flatten(),
            'High': df['High'].to_numpy().flatten(),
            'Low': df['Low'].to_numpy().flatten(),
            'Open': df['Open'].to_numpy().flatten(),
            'Volume': df['Volume'].to_numpy().flatten()
        }, index=df.index)
        self.save_data_sample(df)
        while True:
            try:
                
                sequences, targets = self.preprocess_data(df)
                current_time = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                self.data_queue.put((sequences, targets, current_time))
                
            except Exception as e:
                print(f"Data processing error: {e}")
                print(f"traceback: {traceback.format_exc()}")
                time.sleep(1)