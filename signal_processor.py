from multiprocessing import Process, Queue
import json
import logging
from datetime import datetime
from config import Config
import os
import time
import traceback

class SignalProcessor(Process):
    def __init__(self, config: Config, signal_queue: Queue):
        super().__init__()
        self.config = config
        self.signal_queue = signal_queue
        self.logger = None 
        
    def setup_logging(self):
        try:
            os.makedirs(self.config.LOG_PATH, exist_ok=True)
            
            log_file = os.path.join(self.config.LOG_PATH, 'signals.log')
            print(f"Setting up logging to file: {log_file}")

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            
            self.logger = logging.getLogger('SignalProcessor')
            self.logger.setLevel(logging.INFO)
            
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            
            print("Logging setup completed successfully")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
            print(f"Traceback: {traceback.format_exc()}")

        
    def run(self):
        self.setup_logging()
        print("SignalProcessor started")
        self.logger.info("SignalProcessor started")
        while True:
            try:
                signal_data = self.signal_queue.get()
                print(f"Received signal: {signal_data}")
                
                log_data = {
                    'timestamp': signal_data['timestamp'],
                    'symbol': signal_data['symbol'],
                    'signal': signal_data['signal'],
                    'confidence': float(signal_data['confidence']),
                    'predicted_price': float(signal_data['predicted_price']),
                    'current_price': float(signal_data['current_price'])
                }
                

                self.logger.info(json.dumps(log_data))
                print(f"Logged signal: {log_data['signal']} for {log_data['symbol']}")
                
            except Exception as e:
                print(f"Signal processing error: {e}")
                print(f"traceback: {traceback.format_exc()}")
                time.sleep(1)

  