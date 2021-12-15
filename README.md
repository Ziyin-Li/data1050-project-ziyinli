# DS Web Application for Facebook Stock Prices and Performance Metrics
> DATA1050 Final Project - 
> Ziyin Li


The app has been launched on [Heroku](https://thawing-bastion-07816.herokuapp.com/).

It is an interactive dashboard using two datasets to conduct analysis on stock price trends and relationships between a Facebook post likes and its metrics. The stock data is fetched from [Yahoo Finance](https://finance.yahoo.com/quote/FB/history/). It stores the historical stock prices records of 6 selected companies over 10 years exactly from today. The Facebook metrics dataset is downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Facebook+metrics). It stores several Facebook performance metrics of a renowned cosmetic's brand Facebook page.

The dashboard has three main tabs. In the Stock Prices tab, you can choose which other companies to compare Facebook stock prices with. In the Facebook Metrics tab, you can analyze the distributions of each of the Facebook metrics. In the Predictions tab, you can choose a company and then get to see the plots of ARIMA predictions on its stock prices.

The stock data is daily updated and the ARIMA model is trained in real time.

```python

```
