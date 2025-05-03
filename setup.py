from setuptools import setup, find_packages

setup(
    name="stock_predictor",
    version="1.0.0",
    description="Advanced Stock Price Prediction System",
    author="AI Assistant",
    packages=find_packages(),
    install_requires=[
        "yfinance",
        "pandas",
        "numpy",
        "matplotlib",
        "mplfinance",
        "scikit-learn",
        "tensorflow",
        "keras",
        "keras-tuner",
        "ta",
        "seaborn",
        "newsapi-python",
        "transformers",
        "python-dotenv",
        "pandas_datareader"
    ],
    python_requires=">=3.8",
)
