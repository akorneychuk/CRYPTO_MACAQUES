import plotly.graph_objects as go
from SRC.CORE._CONSTANTS import _KIEV_TIMESTAMP
import numpy as np

def add_bars(col, name, color, row, fig, df):
    fig.add_trace(
        go.Bar(
            x=df[_KIEV_TIMESTAMP],
            y=df[col],
            name=name,
            marker=dict(color=color),
            width=(df.index[1] - df.index[0]).total_seconds() * 1000,
        ),
        row=row, col=1
)

def add_scatter(col, name, color, row, fig, df, fill=None, fillcolor=None, width=2, dash=None):
    fig.add_trace(
        go.Scatter(
            x=df[_KIEV_TIMESTAMP],
            y=df[col],
            name=name,
            line=dict(color=color, width=width, dash=dash),
            mode='lines',
            fill=fill,
            fillcolor=fillcolor
        ),
        row=row, col=1
    )

def add_over_zone(y0, y1, fillcolor, row, fig):
    fig.add_hrect(
        y0=y0, y1=y1,
        line_width=0,
        fillcolor=fillcolor,
        opacity=0.2,
        row=row, col=1
    )

def add_central_line(row, fig, y=50, line_dash="dot"):
    fig.add_hline(
        y=y,
        line_dash=line_dash,
        line_color="white",
        line_width=1,
        opacity=0.3,
        row=row, col=1
    )

def add_over_zones_and_a_central_line(row, fig):
    add_over_zone(80, 100, "red", row, fig)
    add_over_zone(0, 20, "green", row, fig)
    add_central_line(row, fig)

def add_mrc(candlestick_row, fig, df):
    add_scatter('meanline', "MRC Mean", '#FFCD00', candlestick_row, fig, df)
    add_scatter('upband1', "MRC R1", 'green', candlestick_row, fig, df, width=1, dash='dot')
    add_scatter('loband1', "MRC S1", 'green', candlestick_row, fig, df, width=1, dash='dot')
    add_scatter('upband2', "MRC R2", 'red', candlestick_row, fig, df, width=1)
    add_scatter('loband2', "MRC S2", 'red', candlestick_row, fig, df, width=1)

def add_volume(df, volume_row, fig):
    add_bars(
        "volume",
        "Volume",
        ["green" if c > o else "red" for o, c in zip(df["open"], df["close"])],
        volume_row,
        fig,
        df
    )

def add_rsi(rsi_row, fig, df):
    add_scatter('rsi', "RSI", 'purple', rsi_row, fig, df)
    add_over_zones_and_a_central_line(rsi_row, fig)

def add_stoch_scatter(speed, color, stoch_row, fig, df):
    add_scatter('stoch_' + speed, "Stoch %" + speed.capitalize(), color, stoch_row, fig, df)

def add_stoch(stoch_row, fig, df):
    add_stoch_scatter('k', "lightblue", stoch_row, fig, df)
    add_stoch_scatter('d', "orange", stoch_row, fig, df)
    add_over_zones_and_a_central_line(stoch_row, fig)

def add_macd(macd, df, macd_row, fig):
    full_histogram = macd.macd_diff()
    prev_histogram = full_histogram.shift(1).loc[df.index]
    conditions = [
        (df['macd_histogram'] >= 0) & (df['macd_histogram'] >= prev_histogram),
        (df['macd_histogram'] >= 0) & (df['macd_histogram'] < prev_histogram),
        (df['macd_histogram'] < 0) & (df['macd_histogram'] <= prev_histogram),
        (df['macd_histogram'] < 0) & (df['macd_histogram'] > prev_histogram)
    ]
    choices = ['green', 'lightgreen', 'red', 'lightcoral']
    macd_colors = np.select(conditions, choices, default='rgba(128, 128, 128, 0.3)')
    add_scatter("macd_line", "MACD Line", 'lightblue', macd_row, fig, df)
    add_scatter("macd_signal", "Signal Line", 'orange', macd_row, fig, df)
    add_bars("macd_histogram", "MACD Histogram", macd_colors, macd_row, fig, df)
    add_central_line(macd_row, fig, line_dash="solid")

def add_atr(atr_row, fig, df):
    add_scatter('atr', "ATR (14)", 'orange', atr_row, fig, df, fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.1)')
    add_central_line(atr_row, fig, y=df['atr'].mean())