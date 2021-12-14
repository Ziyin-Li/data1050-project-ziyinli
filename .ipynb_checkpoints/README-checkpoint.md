# DS Web Application for Facebook Stock Prices and Performance Metrics
> DATA1050 Final Project

> Ziyin Li


This interactive dashboard uses two datasets to conduct analysis on Facebook stock prices and performance metrics. The stock data is fetched from [Yahoo Finance](https://finance.yahoo.com/quote/FB/history/). This dataset stores the historical stock prices records of 25 selected companies over 10 years exactly from today. The Facebook metrics dataset is downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Facebook+metrics). It stores several Facebook performance metrics of a renowned cosmetic's brand Facebook page.

The dashboard has three main tabs. In the Stock Prices tab, you can choose which other companies to compare Facebook stock prices with. In the Performance Metrics tab, you can analyze the distributions of each of the Facebook metrics. Particular interest is on how paying to advertise posts can boost posts visibility. In the Machine Learning tab, you can choose a company and then get to see the plots of ARIMA predictions on its stock prices. The results from two ML regression models (SVR and Random Forest Regressor) trained on the Facebook metrics dataset are also provided.

The stock data is daily updated and the ML models are trained in real time.

```python

```
