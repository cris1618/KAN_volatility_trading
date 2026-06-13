# KAN-Based Volatility Forecasting and Paper-Traded Risk Strategy

The main idea of this project is to Forecast future realized volatility 
using classical models, ML models, and eventually Kolmogorov-Arnold Networks, 
then test whether the forecasts improve a simple volatility-controlled trading 
strategy.

## Main Research Question
Can KANs forecast future realized volatility better than classical models and standard neural networks, and can those forecasts improve a simple paper-traded risk-management strategy?

# Data & Model
For the data in the beginning, we will start with daily ETF/index data through yfinance, 
which is convenient for research and education but should not be treated as 
institutional-grade market data. The yfinance documentation explicitly says it is not 
affiliated with Yahoo and is intended for research and educational purposes. 
For KANs, we will use efficientKAN, publicly available on GitHub. 

# Assets
Initial assets:

- SPY
- QQQ
- IWM
- TLT
- GLD
- ^VIX