# technical-analysis-chart
An interactive candle-stick chart for technical analysis of a financial asset using plotly

## parameters to be set in code:
  
```
    symbol = 'GOOG'     # financial asset symbol
    start_date = datetime(2020,1,15) 
    end_date = datetime.today()

    # SMA parameters
    plot_sma = False
    sma = [45,22]       # SMA window values
    sma_on = 'Close'

    # EMA parameters
    plot_ema = True
    ema = [45,22]       # EMA window values
    ema_on = 'Close'

    # RSI parameters
    plot_rsi = True
    rsi_period=14       # RSI period value

    # Bollinger Bands parameters
    plot_bollingerband = True
    bb_window=22        # BB window value
    
```
## example

![chart](https://github.com/ebrahimpichka/technical-analysis-chart/blob/main/img.png)


## TODO:
 <ul>
  <li>move to OOP</li>
  <li>add indicators</li>
  <li>create cli</li>
</ul>
