import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from multiprocessing import Process, Queue
import os
import numpy as np
import traceback
import time
from typing import Dict, Any
from config import Config

class ModelProcessor(Process):
    def __init__(self, config: Config, data_queue: Queue, signal_queue: Queue):
        super().__init__()
        self.config = config
        self.data_queue = data_queue
        self.signal_queue = signal_queue
        self.model = self._build_model()
        print("ModelProcessor initialized")
        
    def _build_model(self) -> Sequential:
        try:
            # Try to load existing model
            if os.path.exists(self.config.MODEL_PATH):
                print(f"Loading existing model from {self.config.MODEL_PATH}")
                return load_model(self.config.MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
        
        print("Building new model")
        model = Sequential([
            LSTM(50, return_sequences=True, 
                 input_shape=(self.config.SEQUENCE_LENGTH, len(self.config.FEATURES))),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='mse'
        )
        return model
        
    def save_model(self):
        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.MODEL_PATH), exist_ok=True)
            # Save the model
            self.model.save(self.config.MODEL_PATH)
            print(f"Model saved to: {self.config.MODEL_PATH}")
        except Exception as e:
            print(f"Error saving model: {e}")
            print(f"Traceback: {traceback.format_exc()}")

    def generate_signal(self, prediction: float, actual: float, threshold: float = 0.01) -> str:
        try:
            if prediction > actual * (1 + threshold):
                return 'BUY'
            elif prediction < actual * (1 - threshold):
                return 'SELL'
            return 'NEUTRAL'
        except Exception as e:
            print(f"Error generating signal: {e}")
            return 'NEUTRAL'  # Default to neutral on error

    def prepare_signal_data(self, prediction: float, actual: float, timestamp: str) -> Dict[str, Any]:
        try:
            signal = self.generate_signal(prediction, actual)
            confidence = abs(prediction - actual) / actual
            
            return {
                'timestamp': timestamp,
                'symbol': self.config.SYMBOL,
                'signal': signal,
                'confidence': float(confidence),
                'predicted_price': float(prediction),
                'current_price': float(actual)
            }
        except Exception as e:
            print(f"Error preparing signal data: {e}")
            return None

    def run(self):
        print("ModelProcessor started")
        model_save_counter = 0
        
        while True:
            try:
                # Get preprocessed data
                print("Waiting for data...")
                sequences, targets, timestamp = self.data_queue.get()
                #print(f"Received data batch - Sequences shape: {sequences.shape}, Targets shape: {targets.shape}")
                
                # Make prediction using the last sequence
                last_sequence = sequences[-1:]
                prediction = self.model.predict(last_sequence, verbose=0)
                print(f"Made prediction: {prediction[0][0]}")
                
                # Prepare and send signal
                signal_data = self.prepare_signal_data(prediction[0][0], targets[-1], timestamp)
                if signal_data:
                    print(f"Sending signal: {signal_data['signal']} with confidence: {signal_data['confidence']}")
                    self.signal_queue.put(signal_data)
                
                # Update model
                if len(sequences) > self.config.BATCH_SIZE:
                    print("Training model with new data")
                    self.model.fit(
                        sequences[-self.config.BATCH_SIZE:],
                        targets[-self.config.BATCH_SIZE:],
                        epochs=1,
                        verbose=0
                    )
                    
                    # Save model periodically
                    model_save_counter += 1
                    if model_save_counter % 100 == 0:
                        self.save_model()
                
                time.sleep(1)  # Prevent too rapid processing
                
            except Exception as e:
                print(f"Model processing error: {e}")
                print(f"Full traceback: {traceback.format_exc()}")
                time.sleep(5)  # Wait before retrying on error
                continue

class ModelProcessor2(Process):
    def __init__(self, config: Config, data_queue: Queue, signal_queue: Queue):
        super().__init__()
        self.config = config
        self.data_queue = data_queue
        self.signal_queue = signal_queue
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        model = Sequential([
            LSTM(50, return_sequences=True, 
                 input_shape=(self.config.SEQUENCE_LENGTH, len(self.config.FEATURES))),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE), 
                     loss='mse')
        return model
        
    def generate_signal(self, prediction: float, actual: float, threshold: float = 0.01) -> str:
        if prediction > actual * (1 + threshold):
            return 'BUY'
        elif prediction < actual * (1 - threshold):
            return 'SELL'
        return 'NEUTRAL'

    def run(self):
        while True:
            try:
                # Get preprocessed data
                sequences, targets, timestamp = self.data_queue.get()
                
                # Make prediction using the last sequence
                last_sequence = sequences[-1:] # Shape: (1, sequence_length, n_features)
                prediction = self.model.predict(last_sequence, verbose=0)
                
                # Get the last actual value
                last_actual = targets[-1]  # Shape: scalar
                
                # Generate signal
                signal = self.generate_signal(prediction[0][0], last_actual)
                
                # Send signal
                signal_data = {
                    'timestamp': timestamp,
                    'symbol': self.config.SYMBOL,
                    'signal': signal,
                    'confidence': abs(prediction[0][0] - last_actual) / last_actual,
                    'predicted_price': float(prediction[0][0]),
                    'current_price': float(last_actual)
                }
                self.signal_queue.put(signal_data)
                
                # Update model (simplified version)
                if len(sequences) > self.config.BATCH_SIZE:
                    self.model.fit(
                        sequences[-self.config.BATCH_SIZE:],
                        targets[-self.config.BATCH_SIZE:],
                        epochs=1,
                        verbose=0
                    )
                
            except Exception as e:
                print(f"Model processing error: {e}")
                print(f"Error details: {str(e)}")
                continue