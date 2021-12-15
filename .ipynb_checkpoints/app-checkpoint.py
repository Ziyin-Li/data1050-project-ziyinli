# +
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pandas_datareader as web
import datetime

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split,KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from statsmodels.tsa.arima_model import ARIMA

# +
external_scripts = ['./assets/style.css']

app = dash.Dash(__name__, external_scripts=external_scripts)
server = app.server
# -

stock_names=['AAPL','ADBE','ADP','AMAT','AMD','AMZN','AVGO','CRM','CSCO','EBAY','FB','GOOG','IBM','INTC','INTU','LRCX','MSFT','NFLX','NVDA','ORCL','PYPL','QCOM','TSLA','TWTR','TXN']

end = datetime.datetime.today() 
start = datetime.date(end.year-10,1,1)

for i in range(0,len(stock_names)):
    try:
        df = web.DataReader(stock_names[i], 'yahoo', start, end)
        df.insert(0,'Stock',stock_names[i])
        df = df.drop(['Adj Close'], axis=1)
        df.to_csv('./stock_data.csv',mode = 'a',header=False)
    except Exception:
        continue

df = pd.read_csv('./stock_data.csv')
df2 = pd.read_csv('./Facebook_metrics.csv',sep=';')

app.layout = html.Div([html.H1('DS Web Application for Facebook Stock Prices and Performance Metrics', style={'textAlign': 'center'}), 
                       dcc.Markdown('''This interactive dashboard uses two datasets to conduct analysis on Facebook stock prices and performance metrics. The stock data is fetched from [Yahoo Finance](https://finance.yahoo.com/quote/FB/history/). This dataset stores the historical stock prices records of 25 selected companies over 10 years exactly from today. The Facebook metrics dataset is downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Facebook+metrics). It stores several Facebook performance metrics of a renowned cosmetic's brand Facebook page.

The dashboard has three main tabs. In the Stock Prices tab, you can choose which other companies to compare Facebook stock prices with. In the Performance Metrics tab, you can analyze the distributions of each of the Facebook metrics. Particular interest is on how paying to advertise posts can boost posts visibility. In the Machine Learning tab, you can choose a company and then get to see the plots of ARIMA predictions on its stock prices.

The stock data is daily updated and the ML model is trained in real time.''') ,
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Stock Prices', children=[
html.Div([html.H1("Facebook Stocks High vs Low", style={'textAlign': 'center', 'padding-top': 5}),
    dcc.Dropdown(id='my-dropdown1',
                 options=[
                          {'label': 'Apple', 'value': 'AAPL'},
                          {'label': 'Adobe', 'value': 'ADBE'},
                          {'label': 'Automatic Data Processing', 'value': 'ADP'},
                          {'label': 'Applied Materials', 'value': 'AMAT'},
                          {'label': 'AMD', 'value': 'AMD'},
                          {'label': 'Amazon', 'value': 'AMZN'},
                          {'label': 'Broadcom', 'value': 'AVGO'},
                          {'label': 'Salesforce', 'value': 'CRM'},
                          {'label': 'Cisco', 'value': 'CSCO'},
                          {'label': 'Disney', 'value': 'DIS'},
                          {'label': 'eBay', 'value': 'EBAY'},
                          {'label': 'Facebook', 'value': 'FB'},
                          {'label': 'Google', 'value': 'GOOG'},
                          {'label': 'IBM', 'value': 'IBM'},
                          {'label': 'Intel', 'value': 'INTC'},
                          {'label': 'Intuit', 'value': 'INTU'},
                          {'label': 'Lam Research', 'value': 'LRCX'},
                          {'label': 'Microsoft', 'value': 'MSFT'},
                          {'label': 'Netflix', 'value': 'NFLX'},
                          {'label': 'NVIDIA', 'value': 'NVDA'},
                          {'label': 'Oracle', 'value': 'ORCL'},
                          {'label': 'PayPal', 'value': 'PYPL'},
                          {'label': 'QUALCOMM', 'value': 'QCOM'},
                          {'label': 'Tesla', 'value': 'TSLA'},
                          {'label': 'Twitter', 'value': 'TWTR'},
                          {'label': 'Texas Instruments', 'value': 'TXN'}
                         ],
        multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
    dcc.Graph(id='highlow'), 
    html.H1("Facebook Market Volume", style={'textAlign': 'center', 'padding-top': 5}),
    dcc.Dropdown(id='my-dropdown2',
                 options=[
                          {'label': 'Apple', 'value': 'AAPL'},
                          {'label': 'Adobe', 'value': 'ADBE'},
                          {'label': 'Automatic Data Processing', 'value': 'ADP'},
                          {'label': 'Applied Materials', 'value': 'AMAT'},
                          {'label': 'AMD', 'value': 'AMD'},
                          {'label': 'Amazon', 'value': 'AMZN'},
                          {'label': 'Broadcom', 'value': 'AVGO'},
                          {'label': 'Salesforce', 'value': 'CRM'},
                          {'label': 'Cisco', 'value': 'CSCO'},
                          {'label': 'Disney', 'value': 'DIS'},
                          {'label': 'eBay', 'value': 'EBAY'},
                          {'label': 'Facebook', 'value': 'FB'},
                          {'label': 'Google', 'value': 'GOOG'},
                          {'label': 'IBM', 'value': 'IBM'},
                          {'label': 'Intel', 'value': 'INTC'},
                          {'label': 'Intuit', 'value': 'INTU'},
                          {'label': 'Lam Research', 'value': 'LRCX'},
                          {'label': 'Microsoft', 'value': 'MSFT'},
                          {'label': 'Netflix', 'value': 'NFLX'},
                          {'label': 'NVIDIA', 'value': 'NVDA'},
                          {'label': 'Oracle', 'value': 'ORCL'},
                          {'label': 'PayPal', 'value': 'PYPL'},
                          {'label': 'QUALCOMM', 'value': 'QCOM'},
                          {'label': 'Tesla', 'value': 'TSLA'},
                          {'label': 'Twitter', 'value': 'TWTR'},
                          {'label': 'Texas Instruments', 'value': 'TXN'}
                     ],
        multi=True,value=['FB'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
    dcc.Graph(id='volume'),
    html.H1("Scatter Analysis", style={'textAlign': 'center', 'padding-top': -10}),
    dcc.Dropdown(id='my-dropdown3',
                 options=[
                          {'label': 'Apple', 'value': 'AAPL'},
                          {'label': 'Adobe', 'value': 'ADBE'},
                          {'label': 'Automatic Data Processing', 'value': 'ADP'},
                          {'label': 'Applied Materials', 'value': 'AMAT'},
                          {'label': 'AMD', 'value': 'AMD'},
                          {'label': 'Amazon', 'value': 'AMZN'},
                          {'label': 'Broadcom', 'value': 'AVGO'},
                          {'label': 'Salesforce', 'value': 'CRM'},
                          {'label': 'Cisco', 'value': 'CSCO'},
                          {'label': 'Disney', 'value': 'DIS'},
                          {'label': 'eBay', 'value': 'EBAY'},
                          {'label': 'Facebook', 'value': 'FB'},
                          {'label': 'Google', 'value': 'GOOG'},
                          {'label': 'IBM', 'value': 'IBM'},
                          {'label': 'Intel', 'value': 'INTC'},
                          {'label': 'Intuit', 'value': 'INTU'},
                          {'label': 'Lam Research', 'value': 'LRCX'},
                          {'label': 'Microsoft', 'value': 'MSFT'},
                          {'label': 'Netflix', 'value': 'NFLX'},
                          {'label': 'NVIDIA', 'value': 'NVDA'},
                          {'label': 'Oracle', 'value': 'ORCL'},
                          {'label': 'PayPal', 'value': 'PYPL'},
                          {'label': 'QUALCOMM', 'value': 'QCOM'},
                          {'label': 'Tesla', 'value': 'TSLA'},
                          {'label': 'Twitter', 'value': 'TWTR'},
                          {'label': 'Texas Instruments', 'value': 'TXN'}
                     ],
                 value= 'FB',
                 style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
    dcc.Dropdown(id='my-dropdown4',
                 options=[
                          {'label': 'Apple', 'value': 'AAPL'},
                          {'label': 'Adobe', 'value': 'ADBE'},
                          {'label': 'Automatic Data Processing', 'value': 'ADP'},
                          {'label': 'Applied Materials', 'value': 'AMAT'},
                          {'label': 'AMD', 'value': 'AMD'},
                          {'label': 'Amazon', 'value': 'AMZN'},
                          {'label': 'Broadcom', 'value': 'AVGO'},
                          {'label': 'Salesforce', 'value': 'CRM'},
                          {'label': 'Cisco', 'value': 'CSCO'},
                          {'label': 'Disney', 'value': 'DIS'},
                          {'label': 'eBay', 'value': 'EBAY'},
                          {'label': 'Facebook', 'value': 'FB'},
                          {'label': 'Google', 'value': 'GOOG'},
                          {'label': 'IBM', 'value': 'IBM'},
                          {'label': 'Intel', 'value': 'INTC'},
                          {'label': 'Intuit', 'value': 'INTU'},
                          {'label': 'Lam Research', 'value': 'LRCX'},
                          {'label': 'Microsoft', 'value': 'MSFT'},
                          {'label': 'Netflix', 'value': 'NFLX'},
                          {'label': 'NVIDIA', 'value': 'NVDA'},
                          {'label': 'Oracle', 'value': 'ORCL'},
                          {'label': 'PayPal', 'value': 'PYPL'},
                          {'label': 'QUALCOMM', 'value': 'QCOM'},
                          {'label': 'Tesla', 'value': 'TSLA'},
                          {'label': 'Twitter', 'value': 'TWTR'},
                          {'label': 'Texas Instruments', 'value': 'TXN'}
                     ],
                 value= 'AAPL',
                 style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
  dcc.RadioItems(id="radiob", value= "High", labelStyle={'display': 'inline-block', 'padding': 10},
                 options=[{'label': "High", 'value': "High"}, {'label': "Low", 'value': "Low"} , {'label': "Volume", 'value': "Volume"}],
 style={'textAlign': "center", }),
    dcc.Graph(id='scatter')
], className='container'),
]),
dcc.Tab(label='Facebook Metrics', children=[
html.Div([html.H1("Facebook Metrics Distributions", style={"textAlign": "center"}),
            html.Div([html.Div([dcc.Dropdown(id='feature-selected1',
                                             options=[{'label': i.title(), 'value': i} for i in
                                                      df2.columns.values[1:]],
                                             value="Type")],
                               style={"display": "block", "margin-left": "auto", "margin-right": "auto",
                                      "width": "80%"}),
                      ],),
            dcc.Graph(id='my-graph2'),
     html.H1('Paid vs Free Posts by Type', style={'textAlign': "center", 'padding-top': 5}),
     html.Div([
         dcc.RadioItems(id="select-survival", value=str(1), labelStyle={'display': 'inline-block', 'padding': 10},
                        options=[{'label': "Paid", 'value': str(1)}, {'label': "Free", 'value': str(0)}], )],
         style={'textAlign': "center", }),
     html.Div([html.Div([dcc.Graph(id="hist-graph", clear_on_unhover=True, )]), ]),
        ], className="container"),
]),
dcc.Tab(label='Machine Learning', children=[
html.Div([html.H1("Machine Learning", style={"textAlign": "center"}), 
          html.H2("ARIMA Time Series Prediction", style={"textAlign": "left"}),
          dcc.Dropdown(id='my-dropdowntest',
                 options=[
                          {'label': 'Apple', 'value': 'AAPL'},
                          {'label': 'Adobe', 'value': 'ADBE'},
                          {'label': 'Automatic Data Processing', 'value': 'ADP'},
                          {'label': 'Applied Materials', 'value': 'AMAT'},
                          {'label': 'AMD', 'value': 'AMD'},
                          {'label': 'Amazon', 'value': 'AMZN'},
                          {'label': 'Broadcom', 'value': 'AVGO'},
                          {'label': 'Salesforce', 'value': 'CRM'},
                          {'label': 'Cisco', 'value': 'CSCO'},
                          {'label': 'Disney', 'value': 'DIS'},
                          {'label': 'eBay', 'value': 'EBAY'},
                          {'label': 'Facebook', 'value': 'FB'},
                          {'label': 'Google', 'value': 'GOOG'},
                          {'label': 'IBM', 'value': 'IBM'},
                          {'label': 'Intel', 'value': 'INTC'},
                          {'label': 'Intuit', 'value': 'INTU'},
                          {'label': 'Lam Research', 'value': 'LRCX'},
                          {'label': 'Microsoft', 'value': 'MSFT'},
                          {'label': 'Netflix', 'value': 'NFLX'},
                          {'label': 'NVIDIA', 'value': 'NVDA'},
                          {'label': 'Oracle', 'value': 'ORCL'},
                          {'label': 'PayPal', 'value': 'PYPL'},
                          {'label': 'QUALCOMM', 'value': 'QCOM'},
                          {'label': 'Tesla', 'value': 'TSLA'},
                          {'label': 'Twitter', 'value': 'TWTR'},
                          {'label': 'Texas Instruments', 'value': 'TXN'}
                     ],
                style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "50%"}),
          dcc.RadioItems(id="radiopred", value="High", labelStyle={'display': 'inline-block', 'padding': 10},
                         options=[{'label': "High", 'value': "High"}, {'label': "Low", 'value': "Low"},
                                  {'label': "Volume", 'value': "Volume"}], style={'textAlign': "center", }),
        dcc.Graph(id='traintest'), dcc.Graph(id='preds'),
         ],)
], className='container')
])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown1', 'value')])
def update_graph(selected_dropdown):
    dropdown = {'AAPL':'Apple','ADBE':'Adobe','ADP':'Automatic Data Processing','AMAT':'Applied Materials','AMD':'AMD','AMZN':'Amazon',
            'AVGO':'Broadcom','CRM':'Salesforce','CSCO':'Cisco','DIS':'Disney','EBAY':'eBay','FB':'Facebook','GOOG':'Google',
           'IBM':'IBM','INTC':'Intel','INTU':'Intuit','LRCX':'Lam Research','MSFT':'Microsoft','NFLX':'Netflix','NVDA':'NVIDIA',
           'ORCL':'Oracle','PYPL':'PayPal','QCOM':'QUALCOMM','TSLA':'Tesla','TWTR':'Twitter','TXN':'Texas Instruments',}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["High"],mode='lines',
            opacity=0.7,name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["Low"],mode='lines',
            opacity=0.6,name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {'AAPL':'Apple','ADBE':'Adobe','ADP':'Automatic Data Processing','AMAT':'Applied Materials','AMD':'AMD','AMZN':'Amazon',
            'AVGO':'Broadcom','CRM':'Salesforce','CSCO':'Cisco','DIS':'Disney','EBAY':'eBay','FB':'Facebook','GOOG':'Google',
           'IBM':'IBM','INTC':'Intel','INTU':'Intuit','LRCX':'Lam Research','MSFT':'Microsoft','NFLX':'Netflix','NVDA':'NVIDIA',
           'ORCL':'Oracle','PYPL':'PayPal','QCOM':'QUALCOMM','TSLA':'Tesla','TWTR':'Twitter','TXN':'Texas Instruments',}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock]["Date"],y=df[df["Stock"] == stock]["Volume"],mode='lines',
            opacity=0.7,name=f'Volume {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Transactions Volume"},paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('scatter', 'figure'),
              [Input('my-dropdown3', 'value'), Input('my-dropdown4', 'value'), Input("radiob", "value"),])
def update_graph(stock, stock2, radioval):
    dropdown = {'AAPL':'Apple','ADBE':'Adobe','ADP':'Automatic Data Processing','AMAT':'Applied Materials','AMD':'AMD','AMZN':'Amazon',
            'AVGO':'Broadcom','CRM':'Salesforce','CSCO':'Cisco','DIS':'Disney','EBAY':'eBay','FB':'Facebook','GOOG':'Google',
           'IBM':'IBM','INTC':'Intel','INTU':'Intuit','LRCX':'Lam Research','MSFT':'Microsoft','NFLX':'Netflix','NVDA':'NVIDIA',
           'ORCL':'Oracle','PYPL':'PayPal','QCOM':'QUALCOMM','TSLA':'Tesla','TWTR':'Twitter','TXN':'Texas Instruments',}
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    trace1 = []
    if (stock == None) or (stock2 == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        trace1.append(go.Scatter(x=df[df["Stock"] == stock][radioval][-1000:], y=df[df["Stock"] == stock2][radioval][-1000:],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} of {dropdown[stock]} vs {dropdown[stock2]} Over Time (1000 iterations)",
                xaxis={"title": stock,}, yaxis={"title": stock2}, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(
    dash.dependencies.Output('my-graph2', 'figure'),
    [dash.dependencies.Input('feature-selected1', 'value')])
def update_graph(selected_feature1):
    if selected_feature1 == None:
        selected_feature1 = 'Type'
        trace = go.Histogram(x= df2.Type,
                             marker=dict(color='rgb(0, 0, 100)'))
    else:
        trace = go.Histogram(x=df2[selected_feature1],
                         marker=dict(color='rgb(0, 0, 100)'))

    return {
        'data': [trace],
        'layout': go.Layout(title=f'Metric: {selected_feature1.title()}',
                            colorway=["#EF963B", "#EF533B"], hovermode="closest",
                            xaxis={'title': "Distribution", 'titlefont': {'color': 'black', 'size': 14},
                                   'tickfont': {'size': 14, 'color': 'black'}},
                            yaxis={'title': "Frequency", 'titlefont': {'color': 'black', 'size': 14, },
                                   'tickfont': {'color': 'black'}},     paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}


@app.callback(
    dash.dependencies.Output('hist-graph', 'figure'),
    [dash.dependencies.Input('select-survival', 'value'),])
def update_graph(selected):
    dff = df2[df2['Paid'] == int(selected)]
    trace = go.Histogram(x=dff['Type'], marker=dict(color='rgb(0, 0, 100)'))
    layout = go.Layout(xaxis={'title': 'Post distribution types', 'showgrid': False},
                       yaxis={'title': 'Frequency', 'showgrid': False}, paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)' )
    figure2 = {'data': [trace], 'layout': layout}

    return figure2


@app.callback(Output('traintest', 'figure'),
              [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
def update_graph(stock , radioval):
    dropdown = {'AAPL':'Apple','ADBE':'Adobe','ADP':'Automatic Data Processing','AMAT':'Applied Materials','AMD':'AMD','AMZN':'Amazon',
            'AVGO':'Broadcom','CRM':'Salesforce','CSCO':'Cisco','DIS':'Disney','EBAY':'eBay','FB':'Facebook','GOOG':'Google',
           'IBM':'IBM','INTC':'Intel','INTU':'Intuit','LRCX':'Lam Research','MSFT':'Microsoft','NFLX':'Netflix','NVDA':'NVIDIA',
           'ORCL':'Oracle','PYPL':'PayPal','QCOM':'QUALCOMM','TSLA':'Tesla','TWTR':'Twitter','TXN':'Texas Instruments',}
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    trace1 = []
    trace2 = []
    train_data = df[df['Stock'] == stock][-1000:][0:int(1000 * 0.8)]
    test_data = df[df['Stock'] == stock][-1000:][int(1000 * 0.8):]
    if (stock == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        trace1.append(go.Scatter(x=train_data['Date'],y=train_data[radioval], mode='lines',
            opacity=0.7,name=f'Training Set',textposition='bottom center'))
        trace2.append(go.Scatter(x=test_data['Date'],y=test_data[radioval],mode='lines',
            opacity=0.6,name=f'Test Set',textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} Train-Test Sets for {dropdown[stock]}",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('preds', 'figure'),
              [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
def update_graph(stock, radioval):
    dropdown = {'AAPL':'Apple','ADBE':'Adobe','ADP':'Automatic Data Processing','AMAT':'Applied Materials','AMD':'AMD','AMZN':'Amazon',
            'AVGO':'Broadcom','CRM':'Salesforce','CSCO':'Cisco','DIS':'Disney','EBAY':'eBay','FB':'Facebook','GOOG':'Google',
           'IBM':'IBM','INTC':'Intel','INTU':'Intuit','LRCX':'Lam Research','MSFT':'Microsoft','NFLX':'Netflix','NVDA':'NVIDIA',
           'ORCL':'Oracle','PYPL':'PayPal','QCOM':'QUALCOMM','TSLA':'Tesla','TWTR':'Twitter','TXN':'Texas Instruments',}
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    dropdown = {'AAPL':'Apple','ADBE':'Adobe','ADP':'Automatic Data Processing','AMAT':'Applied Materials','AMD':'AMD','AMZN':'Amazon',
            'AVGO':'Broadcom','CRM':'Salesforce','CSCO':'Cisco','DIS':'Disney','EBAY':'eBay','FB':'Facebook','GOOG':'Google',
           'IBM':'IBM','INTC':'Intel','INTU':'Intuit','LRCX':'Lam Research','MSFT':'Microsoft','NFLX':'Netflix','NVDA':'NVIDIA',
           'ORCL':'Oracle','PYPL':'PayPal','QCOM':'QUALCOMM','TSLA':'Tesla','TWTR':'Twitter','TXN':'Texas Instruments',}
    trace1 = []
    trace2 = []
    if (stock == None):
        trace1.append(
        go.Scatter(x=[0],y=[0],mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        test_data = df[df['Stock'] == stock][-1000:][int(1000 * 0.8):]
        train_data = df[df['Stock'] == stock][-1000:][0:int(1000 * 0.8)]
        train_ar = train_data[radioval].values
        test_ar = test_data[radioval].values
        history = [x for x in train_ar]
        predictions = list()
        for t in range(len(test_ar)):
            model = ARIMA(history, order=(3, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test_ar[t]
            history.append(obs)
        error = r2_score(test_ar, predictions)
        trace1.append(go.Scatter(x=test_data['Date'],y=test_data['High'],mode='lines',
            opacity=0.6,name=f'Actual Series',textposition='bottom center'))
        trace2.append(go.Scatter(x=test_data['Date'],y= np.concatenate(predictions).ravel(), mode='lines',
            opacity=0.7,name=f'Predicted Series (R2: {error})',textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} ARIMA Predictions vs Actual for {dropdown[stock]}",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
