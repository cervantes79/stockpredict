from config import Config
from data_processor import DataProcessor
from model_processor import ModelProcessor
from signal_processor import SignalProcessor
from multiprocessing import Queue
import os

def setup_environment():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for directory in ["logs", "data", "models"]:
        try:
            test_file = os.path.join(directory, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"{directory} directory is writable")
        except Exception as e:
            print(f"Warning: Cannot write to {directory}: {e}")


def main():
    setup_environment()
    config = Config()
    data_queue = Queue()
    signal_queue = Queue()
    
    data_processor = DataProcessor(config, data_queue)
    model_processor = ModelProcessor(config, data_queue, signal_queue)
    signal_processor = SignalProcessor(config, signal_queue)
    
    data_processor.start()
    model_processor.start()
    signal_processor.start()
    
    try:
        data_processor.join()
        model_processor.join()
        signal_processor.join()
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
        data_processor.terminate()
        model_processor.terminate()
        signal_processor.terminate()

if __name__ == "__main__":
    main()