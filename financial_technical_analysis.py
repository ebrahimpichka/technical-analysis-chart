import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import offline
import pandas_datareader.data as web
from datetime import datetime
import MetaTrader5 as mt5
from plotly.offline import plot
import pytz
import requests

# -------------------------------------------------------
# getting data yahoo

def get_data_yahoo(symbol,start_date,end_date,resolution=None):
    df = web.DataReader(symbol,'yahoo',start_date,end_date)
    return(df)

 # -------------------------------------------------------
 # technical analysis Indicators

def calc_sma(df,window,on):
    sma = df[on].rolling(window).mean()
    return(sma)

def calc_ema(df,window,on):
    ema = df[on].ewm(span=window, adjust=False).mean()
    return(ema)

def calc_rsi(df,period=14):
    change = df['Close'].diff(1)
    gain = change.mask(change>0,other=0)
    loss = change.mask(change<0,other=0)
    avg_gain = gain.ewm(span = period).mean()
    avg_loss = loss.ewm(span = period).mean()
    rs = abs(avg_gain/avg_loss)
    rsi = (100/(1+rs))
    return(rsi)

def calc_bollinger_bands(df,window=22):
    bb = pd.DataFrame()
    bb['center_line'] = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    bb['upper_band'] = bb['center_line'] + 2*std
    bb['lower_band'] = bb['center_line'] - 2*std
    return(bb)

# -------------------------------------------------------
# plotting the chart

def plot_chart(df, symbol='', plot_sma=False, plot_ema=True, plot_rsi=False, plot_bollingerband=False, sma=[22,45] ,ema=[22,45], bb_window=22, rsi_period=14,
                sma_on = 'Close',ema_on = 'Close'):
    """plots an interactive candle-stick chart for a specific financial asset and its specified technichal indicators

    Args:
        df (pd.DataFrame): pandas dataframe time series containing 'Open','High','Low','Close','Volume' columns
        symbol (str, optional): symbol of the financial asset to be plotted . Defaults to ''.
        plot_sma (bool, optional): wether to plot Simple Moving Average. Defaults to False.
        plot_ema (bool, optional): wether to plot Exponential Moving Average. Defaults to True.
        plot_rsi (bool, optional): wether to plot Reletive Strength Index. Defaults to False.
        plot_bollingerband (bool, optional): wether to plot Bollinger Bands. Defaults to False.
        sma (list, optional): iterable containing integer `window` values for SMA. Defaults to [22,45].
        ema (list, optional): iterable containing integer `window` values for EMA. Defaults to [22,45].
        bb_window (int, optional): integer `window` value for BB. Defaults to 22.
        rsi_period (int, optional): integer `period` value for RSI. Defaults to 14.
        sma_on (str, optional): the column among 'Open','High','Low','Close' on which SMA should be plotted. Defaults to 'Close'.
        ema_on (str, optional): the column among 'Open','High','Low','Close' on which EMA should be plotted. Defaults to 'Close'.
    """
    # seperting required data for each type of plot
    price_data = df[['Open','High','Low','Close']]
    volume_data = df['Volume']

    if plot_sma:
        sma_data = pd.DataFrame()
        for sma_window in sma:
            sma_data['SMA'+str(sma_window)] = calc_sma(df, sma_window ,on=sma_on)
    
    if plot_ema:
        ema_data = pd.DataFrame()
        for ema_window in ema:
            ema_data['EMA'+str(ema_window)] = calc_ema(df, ema_window ,on=ema_on)
    
    if plot_bollingerband:
        bollinger_band_data = calc_bollinger_bands(df,bb_window)
    
    if plot_rsi:
        rsi_data = calc_rsi(df,rsi_period)

    # color mapping for Volumes
    INCREASING_COLOR = 'rgb(0,140,240)'
    DECREASING_COLOR = 'rgb(200,0,90)'
    colors = []
    for i in range(len(df.Close)):
        if i != 0:
            if df.Close[i] > df.Close[i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)

    # ------------------------------------
    # creating graph objects for ech type of plot
    # candlestick graph object
    candle_stick = go.Candlestick(x=df.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price OHLC',
                whiskerwidth=0.2,
                increasing_line_color= INCREASING_COLOR,
                decreasing_line_color= DECREASING_COLOR)
    # volume graph object            
    volume = go.Bar(x=df.index,y=volume_data,name="Volume",marker_color = colors)

    # SMA graph object
    if plot_sma:
        SMA = []
        for col in sma_data.columns:
            SMA.append(go.Scatter(x = df.index, y= sma_data[col].values,mode='lines',name=col,line=dict(width=2)))
    # EMA graph object
    if plot_ema:
        EMA = []
        for col in ema_data.columns:
            EMA.append(go.Scatter(x = df.index, y= ema_data[col].values,mode='lines',name=col,line=dict(width=2)))
    # Bollinger Bands graph object
    if plot_bollingerband:
        bb_upper = go.Scatter(x = df.index, y= bollinger_band_data['upper_band'] ,mode='lines',name='bb_upper_band',line=dict(width=3))
        bb_mid = go.Scatter(x = df.index, y= bollinger_band_data['center_line'] ,mode='lines',name='bb_center_line',line=dict(width=3),fill='tonexty')
        bb_lower = go.Scatter(x = df.index, y= bollinger_band_data['lower_band'] ,mode='lines',name='bb_lower_band',line=dict(width=3),fill='tonexty')

    # RSI graph object
    if plot_rsi:
        RSI = go.Scatter(x = df.index, y= rsi_data ,mode='lines',name='RSI',line=dict(width=3))
    
    if not plot_rsi:
        fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.01,
                        specs=[[{"rowspan": 2}],
                            [{}],
                            [{}]])
    else:
        fig = make_subplots(rows=5, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.02,
                        specs=[[{"rowspan": 3}],
                            [{}],
                            [{}],
                            [{}],
                            [{}]])

    # adding graph objects to the figure
    fig.add_trace(candle_stick,row=1, col=1)

    if plot_rsi:
        fig.add_trace(volume,row=4, col=1)
        fig.update_yaxes(title_text="Volume", row=4, col=1)
    else:
        fig.add_trace(volume,row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)

    if plot_sma:
        for element in SMA:
            fig.add_trace(element,row=1, col=1)
    
    if plot_ema:
        for element in EMA:
            fig.add_trace(element,row=1, col=1)
    
    if plot_bollingerband:
        fig.add_trace(bb_upper,row=1, col=1)
        fig.add_trace(bb_mid,row=1, col=1)
        fig.add_trace(bb_lower,row=1, col=1)
    
    if plot_rsi:
        fig.add_trace(RSI,row=5, col=1)
        fig.add_shape(type="line",x0=df.index[0], y0=30, x1=df.index[-1], y1=30,line=dict(color="LightSeaGreen",width=2,dash="dashdot",),row=5, col=1)
        fig.add_shape(type="line",x0=df.index[0], y0=70, x1=df.index[-1], y1=70,line=dict(color="LightSeaGreen",width=2,dash="dashdot",),row=5, col=1)
        fig.update_yaxes(title_text="RSI "+str(rsi_period), range=[0, 100], row=5, col=1)

    
    fig.update_layout(title=symbol+' Price Chart',
                        yaxis=dict(title='Price',side='right'))
    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_layout(plot_bgcolor='rgb(230,240,240)')
    fig.update_layout(xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,label="1m",step="month",stepmode="backward"),
                dict(count=6,label="6m",step="month",stepmode="backward"),
                dict(count=1,label="YTD",step="year",stepmode="todate"),
                dict(count=1,label="1y",step="year",stepmode="backward"),
                dict(step="all")])),
                rangeslider=dict(visible=True),type="date"))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(side='right')

    # fig.show()
    # fig.to_html()
    # plot(fig)
    offline.plot(fig, filename = f'D:\Python Projects\personal_projects\CandleStick\{symbol}-chart.html', auto_open=False)
# -------------------------------------------------------


if __name__ == '__main__':

    # symbols_for_yahoo = ['GOOG','AAPL','FB','BTC-USD','ETH-USD','LTC-USD']
    # symbol = input('Enter symbol:')

    # parameter setting:

    symbol = 'GOOG'
    start_date = datetime(2020,1,15) 
    end_date = datetime.today()

    # SMA parameters
    plot_sma = False
    sma = [45,22]
    sma_on = 'Close'

    # EMA parameters
    plot_ema = True
    ema = [45,22]
    ema_on = 'Close'

    # RSI parameters
    plot_rsi = True
    rsi_period=14

    # Bollinger Bands parameters
    plot_bollingerband = True
    bb_window=22

    # geting bars data from yahoo
    df = get_data_yahoo(symbol,start_date=start_date,end_date=end_date)

    # plottig
    plot_chart(df,symbol,plot_sma=plot_sma, plot_ema=plot_ema, plot_rsi=plot_rsi, plot_bollingerband=plot_bollingerband)


