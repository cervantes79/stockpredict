# Stock Price Prediction System with Real-time Signals

This project implements a real-time stock price prediction system using LSTM (Long Short-Term Memory) neural networks. The system continuously processes market data, generates predictions, and produces trading signals while incorporating self-learning capabilities.

## Features

- Real-time stock data processing using Yahoo Finance API
- Technical indicator calculations (RSI, MACD, Bollinger Bands)
- LSTM-based price prediction model
- Continuous learning and model adaptation
- Real-time trading signal generation (BUY/SELL/NEUTRAL)
- Multi-process architecture for parallel processing
- Comprehensive logging system

## System Requirements

- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- ta (Technical Analysis library)
- yfinance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cervantes79/stockpredict
cd stock-prediction-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p data models logs
```

## Project Structure

```
stock-prediction-system/
├── config.py           # Configuration settings
├── data_processor.py   # Data processing and technical indicators
├── model_processor.py  # LSTM model and predictions
├── signal_processor.py # Signal generation and logging
├── main.py            # Main execution file
├── data/              # Data storage
├── models/            # Saved models
└── logs/              # System logs
```

## Configuration

The system can be configured through `config.py`. Key parameters include:

- `SYMBOL`: Stock symbol to track (default: "AAPL")
- `TIMEFRAME`: Data timeframe (default: "1d")
- `SEQUENCE_LENGTH`: Number of time steps for LSTM (default: 60)
- `BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Model learning rate (default: 0.001)

## Usage

1. Start the system:
```bash
python main.py
```

2. Monitor signals in the logs:
```bash
tail -f logs/signals.log
```

3. To stop the system:
Press `Ctrl+C` in the terminal running main.py

## Model Architecture

The LSTM model consists of:
- Input layer with configurable sequence length
- Two LSTM layers with dropout
- Dense output layer for price prediction

## Signal Generation

Signals are generated based on:
- Predicted price movement
- Confidence score
- Technical indicator confirmation
- Current market conditions

## Logging

The system maintains detailed logs of:
- Generated signals with timestamps
- Model performance metrics
- System status and errors

## Error Handling

The system includes comprehensive error handling:
- Process recovery mechanisms
- Data validation
- Exception logging
- Graceful shutdown

## Contributing

1. Fork the repository
2. Create your feature branch
3. Implement your changes
4. Run tests
5. Submit a pull request

