# Advanced Stock Prediction System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

An advanced stock price prediction system combining LSTM neural networks and Random Forest models with technical indicators and sentiment analysis.

---

## Features

- Dual-model approach (LSTM + Random Forest)
- Technical indicator analysis (RSI, MACD, Bollinger Bands)
- News sentiment integration
- Ensemble predictions
- Comprehensive visualization
- Walk-forward validation and robust evaluation
- Modular, extensible codebase
- CLI interface for interactive use

---

## Installation

```bash
git clone https://github.com/yourusername/stock-prediction-system.git
cd stock-prediction-system
pip install -r requirements.txt
```

---

## Usage

```bash
python StockPricePredictor.py
```

You will be prompted for a stock ticker and prediction horizon.

---

## Project Structure

```
stock-prediction-system/
│
├── StockPricePredictor.py         # Main script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Files to ignore in git
├── LICENSE                        # Project license
├── data/                          # (Optional) Data storage
├── models/                        # (Optional) Saved models
├── tests/                         # (Optional) Unit tests
└── utils/                         # (Optional) Utility modules
```

---

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

---

## Example

```
Enter stock ticker (e.g., AAPL, MSFT) or 'quit' to exit: AAPL
Enter number of days to predict (1-90): 30
...
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance)
- [ta](https://github.com/bukosabino/ta)
- [scikit-learn](https://scikit-learn.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [NewsAPI](https://newsapi.org/)
- [Transformers](https://github.com/huggingface/transformers)

---
