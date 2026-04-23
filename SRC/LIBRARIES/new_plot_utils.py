import math
import os
import random
from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.core.display_functions import display
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from SRC.CORE._CONSTANTS import _DASHBOARD_SEGMENT_AUTOTRADING, __DIFF, __DIST, _INVERTED_TRADES_INFO, _PLOT_ENABLED, COLORS, _CANDLE_FILE_PATH, __TPR, _TRADES_DF_FIG_HTML_FILE_PATH
from SRC.CORE._CONSTANTS import _DISCRETIZATION, _SYMBOL, _DASHBOARD_SEGMENT_NET_FULL_PATH, _TRADES_DF_FIG_IMG_FILE_PATH, UTC_TZ, _TRADES_FILE_PATH, IS_INVERTED_TRADES_INFO, KIEV_TZ, _UTC_TIMESTAMP, _CLOSE, __SIGNAL, __INCLUDED
from SRC.CORE._CONSTANTS import _KIEV_TIMESTAMP, _NET_FOLDER, _DASHBOARD_SEGMENT, _BALANCE_FILE_PATH
from SRC.CORE._CONSTANTS import _SIGNAL, SIGNAL_IGNORE, SIGNAL_LONG_OUT, SIGNAL_SHORT_OUT, SIGNAL_SHORT_IN, SIGNAL_LONG_IN, RSI_TOP, RSI_BOTTOM
from SRC.CORE._FUNCTIONS import AUC_ROC_DOWN_SAMPLING_RATIO_PRODUCER
from SRC.CORE.debug_utils import printmd, is_cloud, printmd_high, NOTICE, DEBUG
from SRC.CORE.plot_utils import get_grad_color, mcad_bar_color_selector, display_plot, produce_bins_presentation, lighten_color
from SRC.CORE.utils import do_build_presentation_coordinates, datetime_h_m__d_m, produce_timedelta_ticks, _float_n, datetime_h_m_s, datetime_h_m__d_m_y, read_json_safe, timedelta_h_m_s, datetime_m_d__h_m
from SRC.LIBRARIES.new_data_utils import retrieve_all, produce_balance_df, fetch_all, produce_signal_encoder, get_ignore_diff_dist_cl, get_diff_center_cl, get_clazzes_count
from SRC.LIBRARIES.new_utils import remove_list_duplicates, populate_char_n_times, string_bool, format_num, normalize, is_close_to_zero, color_with_opacity

from SRC.WEBAPP.libs.dashboard_app_plot_utils import produce_trade_info_str
from SRC.LIBRARIES.time_utils import TIME_DELTA, round_down_to_nearest_step, kiev_now

NA_PRESENT = '-'

try:
    from backports.zoneinfo import ZoneInfo
except:
    from zoneinfo import ZoneInfo

try:
    from shapely.geometry import Polygon, Point
except:
    from shapely import Polygon, Point


RECOMMENDATION_COLOR_D = {
    'STRONG_SELL': lighten_color('red', 1),
    'SELL': lighten_color('red', 0.5),
    'NEUTRAL': lighten_color('gray', 0.5),
    'BUY': lighten_color('green', 0.5),
    'STRONG_BUY': lighten_color('green', 1)
}

RECOMMENDATION_MAP_D = {
    'STRONG_SELL': {
        'color': lighten_color('red', 1),
        'value': 1
    },
    'SELL': {
        'color': lighten_color('red', 0.5),
        'value': 0.5
    },
    'NEUTRAL': {
        'color': lighten_color('gray', 0.5),
        'value': 0
    },
    'BUY': {
        'color': lighten_color('green', 0.5),
        'value': -0.5
    },
    'STRONG_BUY': {
        'color': lighten_color('green', 1),
        'value': -1
    }
}

MARGIN_LOAN_AVAILIBILITY_D = {
    'AVAILABLE': lighten_color('blue', 0.5),
    'NOT_AVAILABLE': lighten_color('yellow', 0.5)
}


def plot_timeseries_discreate_feature_s(df, fig, fig_row, time_feature, feature_color_d_s, tick_font_size=12):
    min_y = df['price'].min()
    max_y = df['price'].max()

    if len(feature_color_d_s) > 0:
        d_y = (max_y - min_y) / len(feature_color_d_s)

        def fill_feature_distribution(x0, y0, x1, y1, feature_value, color_d):
            color = color_d[feature_value]
            fig.add_shape(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                fillcolor=f"rgb{color}",
                line=dict(color=f"rgb(255,255,255, 0)", width=0),
                xref=f"x{1}",
                yref=f"y{fig_row}",
                opacity=0.4
            )

        for i, feature_color_d in enumerate(feature_color_d_s[::-1]):
            feature = feature_color_d['feature']
            color_d = feature_color_d['color_d']

            y0 = min_y + d_y * i
            y1 = y0 + d_y

            start_row = df.iloc[0]
            end_row = df.iloc[-1]
            start_idx = start_row[time_feature]
            end_idx = end_row[time_feature]
            start_recommend = start_row[feature]
            for _, row in df.iterrows():
                idx = row[time_feature]
                recommend = row[feature]
                if recommend == start_recommend:
                    continue

                fill_feature_distribution(x0=start_idx, y0=y0, x1=idx, y1=y1, feature_value=start_recommend, color_d=color_d)

                start_idx = idx
                start_recommend = recommend

            if start_idx < idx:
                fill_feature_distribution(x0=start_idx, y0=y0, x1=end_idx, y1=y1, feature_value=recommend, color_d=color_d)

        y_range = [min_y, max_y]
        x_range = [df[time_feature].min(), df[time_feature].max()]
        fig.update_yaxes(tickfont=dict(size=1, color='red'), range=y_range, row=fig_row, col=1)
        fig.update_xaxes(range=x_range, row=fig_row, col=1)


def plot_bar_timeseries_discreate_feature_s(df, fig, fig_row, time_feature, feature_map_d_s, tick_font_size=12):
    max_y = 1.2
    min_y = -1.2

    def get_bar_color_val_d(df, feature, feature_map_d):
        idx_s = []
        color_s = []
        value_s = []
        for _, row in df.iterrows():
            idx = row[time_feature]
            recommendation = row[feature]
            color = feature_map_d[recommendation]['color']
            value = feature_map_d[recommendation]['value']

            idx_s.append(idx)
            color_s.append(color)
            value_s.append(value)

        return idx_s, value_s, color_s

    if len(feature_map_d_s) > 0:
        for i, feature_color_d in enumerate(feature_map_d_s[::-1]):
            feature = feature_color_d['feature']
            map_d = feature_color_d['map_d']

            idx_s, value_s, color_s = get_bar_color_val_d(df, feature, map_d)

            # fig.add_trace(go.Bar(
            #     x=idx_s,
            #     y=value_s,
            #     marker=dict(
            #         color=color_s,
            #         showscale=False
            #     ),
            #     showlegend=False
            # ), row=fig_row, col=1)

            x = np.array(idx_s)
            y = np.array(value_s)

            y_positive = np.where(y > 0, y, 0)
            y_negative = np.where(y < 0, y, 0)

            fig.add_trace(go.Scatter(
                x=x,
                y=y_positive,
                fill='tozeroy',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.5)',
                name='Above 0'
            ), row=fig_row, col=1)

            fig.add_trace(go.Scatter(
                x=x,
                y=y_negative,
                fill='tozeroy',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(0, 200, 0, 0.5)',
                name='Below 0'
            ), row=fig_row, col=1)

            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color='darkgray', width=0.5),
                name='Value'
            ), row=fig_row, col=1)

            fig.add_hline(y=0, line=dict(color='gray', dash='dash'), row=fig_row, col=1)

        y_range = [min_y, max_y]
        x_range = [df[time_feature].min(), df[time_feature].max()]
        fig.update_yaxes(tickfont=dict(size=1, color='red'), range=y_range, row=fig_row, col=1)
        fig.update_xaxes(range=x_range, row=fig_row, col=1)
        fig.update_layout(bargap=0.0, showlegend=True)


def calc_now_y_position(y_s):
    price_max = max(y_s)
    price_min = min(y_s)
    price_diff = price_max - price_min
    price_mean = (price_max + price_min) / 2
    price_close_last = y_s[-1]

    if price_close_last > price_mean:
        now_y_position = price_min + price_diff * 0.1
    else:
        now_y_position = price_max - price_diff * 0.1

    return now_y_position


def plot_candles(df, fig, fig_row, time_feature=None, tick_font_size=12, y_range=None, x_range=None):
    discretization = df.iloc[0][_DISCRETIZATION]
    name = f'Candles {discretization}'

    if time_feature is not None:
        time_series = df[time_feature]
    else:
        time_series = df.index

    fig.add_trace(go.Candlestick(
        x=time_series, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing=dict(fillcolor='#a9e8d4', line=dict(color='#53c9a4')),
        decreasing=dict(fillcolor='#fcc1bd', line=dict(color='#fa847d')), showlegend=False), row=fig_row, col=1)

    y_range = [df['low'].min() / 1.005, df['high'].max() * 1.005] if y_range is None else y_range
    x_range = [time_series.min(), time_series.max()] if x_range is None else x_range
    fig.update_yaxes(tickfont=dict(size=tick_font_size), range=y_range, row=fig_row, col=1)
    fig.update_xaxes(range=x_range, row=fig_row, col=1)


def plot_signals(df, fig, fig_row, time_feature):
    close_col = 'close'
    BUY_MARKER_SIZE = 5
    SELL_MARKER_SIZE = 5

    df_action_signal = df[df[_SIGNAL] != SIGNAL_IGNORE]
    last_index = None
    scatter_s = []
    for indx, row in df_action_signal.iterrows():
        if last_index is None:
            last_index = indx
            continue

        last_row = df_action_signal.loc[last_index]
        last_signal = last_row[_SIGNAL]
        last_time = last_row[time_feature]
        curr_time = row[time_feature]
        curr_signal = row[_SIGNAL]
        last_close = last_row[close_col]
        curr_close = row[close_col]

        if (last_signal == SIGNAL_LONG_IN and curr_signal == SIGNAL_LONG_OUT) or (last_signal == SIGNAL_SHORT_OUT and curr_signal == SIGNAL_LONG_OUT) or (
                last_signal == SIGNAL_LONG_IN and curr_signal == SIGNAL_SHORT_IN):
            scatter_s.append(go.Scatter(x=[last_time, curr_time], y=[last_close, curr_close], mode='lines', line=dict(color='green', width=2), name='LONG`s'))
        if (last_signal == SIGNAL_SHORT_IN and curr_signal == SIGNAL_SHORT_OUT) or (last_signal == SIGNAL_LONG_OUT and curr_signal == SIGNAL_SHORT_OUT) or (
                last_signal == SIGNAL_SHORT_IN and curr_signal == SIGNAL_LONG_IN):
            scatter_s.append(go.Scatter(x=[last_time, curr_time], y=[last_close, curr_close], mode='lines', line=dict(color='red', width=2), name='SHORT`s'))

        last_index = indx

    for scatter in scatter_s:
        fig.add_trace(scatter, row=fig_row, col=1)

    df_signal_long = df[(df[_SIGNAL] == SIGNAL_LONG_IN) | (df[_SIGNAL] == SIGNAL_SHORT_OUT)]
    df_signal_short = df[(df[_SIGNAL] == SIGNAL_LONG_OUT) | (df[_SIGNAL] == SIGNAL_SHORT_IN)]

    buy_actions_df = df.dropna(subset=['long'])
    sell_actions_df = df.dropna(subset=['short'])

    symbol_s = "circle-cross"
    symbol_color_s = "green"
    # fig.add_trace(go.Scatter(x=buy_actions_df[time_feature], y=buy_actions_df['long'], mode="markers", marker=dict(size=BUY_MARKER_SIZE, symbol=symbol_s, color=symbol_color_s), name='Local Min`s'), row=fig_row, col=1)
    fig.add_trace(go.Scatter(x=df_signal_long[time_feature], y=df_signal_long[close_col], mode="markers", marker=dict(size=BUY_MARKER_SIZE * 1.5, symbol=symbol_s, color=symbol_color_s), name='BUY`s'),
                  row=fig_row, col=1)

    symbol_s = "circle-x"
    symbol_color_s = "red"
    # fig.add_trace(go.Scatter(x=sell_actions_df[time_feature], y=sell_actions_df['short'], mode="markers", marker=dict(size=SELL_MARKER_SIZE, symbol=symbol_s, color=symbol_color_s), name='Local Max`s'), row=fig_row, col=1)
    fig.add_trace(
        go.Scatter(x=df_signal_short[time_feature], y=df_signal_short[close_col], mode="markers", marker=dict(size=SELL_MARKER_SIZE * 1.5, symbol=symbol_s, color=symbol_color_s), name='SELL`s'),
        row=fig_row, col=1)


def plot_grad_diffs(df, fig, fig_row, time_feature, grad_diff_col, focused=True, tick_font_size=12):
    time_s = df[time_feature].to_list()
    close_grad_diff_s = df[grad_diff_col].to_list()

    line_type = 'solid' if focused else 'dash'
    width = 3 if focused else 1.5

    abs_max = df[grad_diff_col].abs().max()
    fig.add_trace(go.Scatter(
        x=time_s,
        y=close_grad_diff_s,
        text=[f"{y:.8f}" for y in close_grad_diff_s],
        hoverinfo='x+y',
        hovertemplate=(
            f"<br>"
            f"%{{x}}<br>"
            f"%{{y:.8f}}<br>"
        ),
        mode='lines',
        line=dict(dash=line_type, width=width),
        showlegend=False
    ), row=fig_row, col=1)
    fig.update_yaxes(tickfont=dict(size=tick_font_size), range=[-abs_max * 1.1, abs_max * 1.1], row=fig_row, col=1)


def plot_grads(df, fig, fig_row, time_feature, feature, grad_window, grad_each=1, grads_threshold=None, focused=True, tick_font_size=12):
    discretization = df.iloc[0][_DISCRETIZATION]
    df = do_build_presentation_coordinates(df, grad_window, feature)

    grad_col = f'{feature}_grad_{grad_window}'
    grad_ys_col = f'{feature}_grad_ys_{grad_window}'
    grad_ye_col = f'{feature}_grad_ye_{grad_window}'

    df_shifted = df.copy()
    df_shifted[grad_ys_col] = df_shifted[grad_ys_col].shift(grad_window)

    if grad_each > 1:
        df_shifted = df_shifted.iloc[::-1].iloc[::grad_each].iloc[::-1]

    name = f'{feature} GRAD {grad_window}'
    first_loc = df_shifted.iloc[0]

    for dtx, row in df_shifted.iterrows():
        dtx_kiev = row[time_feature]
        if np.isnan(row[grad_ye_col]):
            continue

        if np.isnan(row[grad_col]):
            continue

        dtx_start = dtx_kiev - TIME_DELTA(discretization) * grad_window
        dtx_end = dtx_kiev

        if dtx_start < first_loc.name:
            continue

        y_start = row[grad_ys_col]
        y_end = row[grad_ye_col]

        grad = row[grad_col]
        grad_color = f'rgb{get_grad_color(grad)}'

        line_type = 'solid' if focused else 'dash'
        width = 3 if focused else 1.5
        fig.add_trace(go.Scatter(x=[dtx_start, dtx_end], y=[y_start, y_end], hoverinfo=None, showlegend=False, line=dict(color=grad_color, dash=line_type, width=width)), row=fig_row, col=1)

    if focused:
        grad_max = df[grad_col].abs().iloc[-1]
        y_max = df[grad_ye_col].abs().iloc[-1]
    else:
        grad_max = df[grad_col].abs().max()
        y_max = df[grad_ye_col].abs().max()

    grad_min = -grad_max
    y_min = -y_max

    grad_present_multiplier = y_max / grad_max

    grad_max_adjusted = grad_max if grads_threshold is None else grads_threshold
    grad_min_adjusted = grad_min if grads_threshold is None else -grads_threshold

    y_range = [grad_min_adjusted * grad_present_multiplier, grad_max_adjusted * grad_present_multiplier]

    y_ticks = np.linspace(y_range[0], y_range[1], 5)
    y_texts = [f"{_float_n(y_tick / grad_present_multiplier, 3)}" for y_tick in y_ticks]

    fig.update_yaxes(tickfont=dict(size=tick_font_size), tickvals=y_ticks, ticktext=y_texts, range=y_range, row=fig_row, col=1)

    grad_y_max = df_shifted[grad_ys_col].abs().max()

    return grad_y_max


def plot_rsi(df, fig, fig_row, time_feature):
    rsi_window = 14
    rsi_top = RSI_TOP()
    rsi_bottom = RSI_BOTTOM()
    name = f'RSI {rsi_window}'
    rsi_feature = f'RSI_{rsi_window}'

    fig.add_trace(go.Scatter(x=df[time_feature], y=df[rsi_feature], mode='lines', line=dict(color='violet', width=1.5), name=name), row=fig_row, col=1)
    fig.add_trace(go.Scatter(x=[df[time_feature].iloc[0], df[time_feature].iloc[-1]], y=[rsi_top, rsi_top], mode='lines', line=dict(color='black', width=1, dash='dash'), showlegend=False),
                  row=fig_row, col=1)
    fig.add_trace(go.Scatter(x=[df[time_feature].iloc[0], df[time_feature].iloc[-1]], y=[rsi_bottom, rsi_bottom], mode='lines', line=dict(color='black', width=1, dash='dash'), showlegend=False),
                  row=fig_row, col=1)
    fig.add_shape(x0=min(df[time_feature]), x1=max(df[time_feature]), y0=rsi_bottom, y1=rsi_top, type="rect", fillcolor="rgba(128, 128, 128, 0.5)", line=dict(width=0), row=fig_row, col=1)


def plot_macd(df, fig, fig_row, time_feature):
    macd_low = 12
    macd_high = 26
    macd_window = 9
    name = f'MACD {macd_low}/{macd_high}/{macd_window}'
    macd_feature = f'MACD_{macd_low}_{macd_high}_{macd_window}'
    macd_signal_feature = f'MACDs_{macd_low}_{macd_high}_{macd_window}'
    macd_histogram_feature = f'MACDh_{macd_low}_{macd_high}_{macd_window}'
    values = df[macd_histogram_feature].to_list()
    histogram_colors = list(map(lambda indx: mcad_bar_color_selector(None if indx == 0 else values[indx - 1], values[indx]), range(len(values))))

    fig.add_trace(go.Scatter(x=df[time_feature], y=df[macd_feature], mode='lines', line=dict(color='blue', width=1), name='MACD Line'), row=fig_row, col=1)
    fig.add_trace(go.Scatter(x=df[time_feature], y=df[macd_signal_feature], mode='lines', line=dict(color='red', width=1), name='Signal Line'), row=fig_row, col=1)
    fig.add_trace(go.Bar(x=df[time_feature], y=df[macd_histogram_feature], name=name, marker=dict(color=histogram_colors, colorbar=dict(title='Value'), showscale=False)), row=fig_row, col=1)


def plot_imacd(df, fig, fig_row, time_feature):
    name = f'iMACD'
    macd_feature = 'iMACD'
    macd_signal_feature = 'iMACDs'
    macd_histogram_feature = 'iMACDh'
    values = df[macd_histogram_feature].to_list()
    histogram_colors = list(map(lambda indx: mcad_bar_color_selector(None if indx == 0 else values[indx - 1], values[indx]), range(len(values))))

    fig.add_trace(go.Scatter(x=df[time_feature], y=df[macd_feature], mode='lines', line=dict(color='blue', width=1), name='MACD Line'), row=fig_row, col=1)
    fig.add_trace(go.Scatter(x=df[time_feature], y=df[macd_signal_feature], mode='lines', line=dict(color='red', width=1), name='Signal Line'), row=fig_row, col=1)
    fig.add_trace(go.Bar(x=df[time_feature], y=df[macd_histogram_feature], name=name, marker=dict(color=histogram_colors, colorbar=dict(title='Value'), showscale=False)), row=fig_row, col=1)


def plot_atr(df, fig, fig_row, time_feature):
    atr = 'ATRr_14'
    name = 'ATRr 14'
    fig.add_trace(go.Scatter(x=df[time_feature], y=df[atr], mode='lines', line=dict(color='green', width=1), name=name), row=fig_row, col=1)


def plot_feature_line(df, fig, fig_row, time_feature, feature, color):
    fig.add_trace(go.Scatter(x=df[time_feature], y=df[feature], name=feature, line=dict(color=color, dash='solid')), row=fig_row, col=1)


def plot_candles_signals_grads(df, start_dt, end_dt, time_feature):
    df = df[df[time_feature] >= start_dt][df[time_feature] <= end_dt]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    plot_candles(df=df, fig=fig, fig_row=1, time_feature=time_feature)

    plot_signals(df=df, fig=fig, fig_row=1, time_feature=time_feature)

    plot_grads(df=df, fig=fig, fig_row=2, time_feature=time_feature, feature='close', grad_window=1)
    plot_grads(df=df, fig=fig, fig_row=3, time_feature=time_feature, feature='close', grad_window=2)
    plot_grads(df=df, fig=fig, fig_row=4, time_feature=time_feature, feature='close', grad_window=5)

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))
    fig.update_layout(yaxis_title='Price', xaxis_rangeslider_visible=False, xaxis_rangeslider_thickness=0.03)
    fig.update_layout(height=1200, paper_bgcolor="LightSteelBlue")
    fig.update_layout(margin=dict(l=30, r=20, t=40, b=20))
    fig.update_layout(hovermode='x unified')
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_traces(xaxis="x")
    names = set()
    fig.for_each_trace(lambda trace: trace.update(showlegend=False) if (trace.name in names) else names.add(trace.name))
    fig.show()


def plot_zigzags(df, fig, fig_row, time_feature, threshold):
    _pivot_col = f'{_SIGNAL}_{threshold}'
    _close_col = 'close'
    BUY_MARKER_SIZE = 5
    SELL_MARKER_SIZE = 5

    # zig_zags = df.loc[df[_pivot_col] != SIGNAL_IGNORE]
    # for (idx_curr, row_curr), (idx_next, row_next) in zip(zig_zags.iloc[:-1].iterrows(), zig_zags.iloc[1:].iterrows()):
    # 	if row_curr[_pivot_col] == row_next[_pivot_col]:
    # 		continue
    # 	fig.add_trace(go.Scatter(x=[idx_curr, idx_next], y=[row_curr[_close_col], row_next[_close_col]], mode='lines', line=dict(color='gray', width=2), name=f'ZigZag_{threshold}'), row=fig_row, col=1)

    long_signal = df.loc[df[_pivot_col] == SIGNAL_LONG_IN]
    fig.add_trace(go.Scatter(x=long_signal[time_feature], y=long_signal[_close_col], mode="markers", marker=dict(size=BUY_MARKER_SIZE, symbol="circle-cross", color="green"), name='Local Min`s'), row=fig_row, col=1)

    short_signal = df.loc[df[_pivot_col] == SIGNAL_SHORT_IN]
    fig.add_trace(go.Scatter(x=short_signal[time_feature], y=short_signal[_close_col], mode="markers", marker=dict(size=SELL_MARKER_SIZE, symbol="circle-x", color="red"), name='Local Max`s'), row=fig_row, col=1)


def plot_candles_zigzags(df, fig, fig_row, time_feature, threshold):
    plot_candles(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature)
    plot_zigzags(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, threshold=threshold)


def plot_candles_signals(df, fig, fig_row, time_feature):
    plot_candles(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature)
    plot_signals(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature)


def plot_supertrend_signals(df, fig, fig_row, time_feature):
    y_min = df['low'].min()
    y_max = df['high'].max()

    fig.add_trace(go.Candlestick(
        x=df[time_feature],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ), row=fig_row, col=1)

    fig.add_trace(go.Scatter(
        x=df[time_feature],
        y=df['supertrend'],
        mode='lines',
        line=dict(color='green', width=2),
        name='Supertrend'
    ), row=fig_row, col=1)

    buy_signals = df[df['supertrend_buy']]
    fig.add_trace(go.Scatter(
        x=buy_signals[time_feature],
        y=buy_signals['close'],
        mode='markers+text',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        text=["Buy"] * len(buy_signals),
        textposition="bottom center",
        name='Buy Signal'
    ), row=fig_row, col=1)

    sell_signals = df[df['supertrend_sell']]
    fig.add_trace(go.Scatter(
        x=sell_signals[time_feature],
        y=sell_signals['close'],
        mode='markers+text',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        text=["Sell"] * len(sell_signals),
        textposition="top center",
        name='Sell Signal'
    ), row=fig_row, col=1)

    fig.update_layout(
        title='TradingView-Style Supertrend (10, HL2, 3)',
        height=800,
        yaxis=dict(title='Price', range=[y_min * 0.98, y_max * 1.02]),
        xaxis=dict(title='Date', rangeslider=dict(visible=True)),
        template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )


def plot_volumes(df, fig, fig_row, time_feature):
    # Compute candle width in milliseconds for volume bars
    # Using difference between first two datetimes
    dt_width = (df.iloc[1][time_feature] - df.iloc[0][time_feature]).total_seconds() * 1000

    volume_bars_color_s = [
        "green" if c > o else "red"
        for o, c in zip(df["open"], df["close"])
    ]

    fig.add_trace(
        go.Bar(
            x=df[time_feature],
            y=df['volume'],
            marker=dict(color=volume_bars_color_s),
            width=dt_width,  # makes bar width match one candle
            name='Volume'
        ),
        row=fig_row, col=1
    )


def plot_continous_tpr_pred_true_bars(df, fig, fig_row, time_feature, threshold):
    _actual_trp = f'tpr_{threshold}'
    _predicted_trp = f'pred_tpr_{threshold}'

    fig.add_trace(
        go.Bar(
            x=df[time_feature],
            y=df[_actual_trp],
            name="Actual TPR",
            marker_color="blue"
        ),
        row=fig_row, col=1
    )

    fig.add_trace(
        go.Bar(
            x=df[time_feature],
            y=df[_predicted_trp],
            name="Predicted TPR",
            marker_color="red"
        ),
        row=fig_row, col=1
    )

    fig.add_shape(
        type="line",
        x0=df.iloc[0][time_feature],
        x1=df.iloc[-1][time_feature],
        y0=0,
        y1=0,
        line=dict(color="black", width=1),
        row=fig_row, col=1
    )


def plot_features(df, start_dt, end_dt, time_feature, plot_s):
    fig = fig_features(df, start_dt, end_dt, time_feature, plot_s)
    display_plot(fig)


def fig_features(df, start_dt, end_dt, time_feature, plot_s):
    discretization = df.iloc[0]['discretization']
    printmd(f"**############################################################ {discretization} ###########################################################**")

    close_col = 'close'

    # plot_s = []
    # plot_s.append(lambda fig, fig_row: plot_atr(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature))
    # plot_s.append(lambda fig, fig_row: plot_rsi(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='RSI_14', grad_window=2))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='RSI_14', grad_window=5))

    # plot_s.append(lambda fig, fig_row: plot_imacd(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature))

    # plot_s.append(lambda fig, fig_row: plot_macd(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_D', grad_window=2))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_H', grad_window=2))

    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature=close_col, grad_window=1))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature=close_col, grad_window=2))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature=close_col, grad_window=5))
    # plot_s.append(lambda fig, fig_row: plot_feature_line(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_DIFF_grad_1', color='cyan'))
    # plot_s.append(lambda fig, fig_row: plot_feature_line(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_DIFF_grad_2', color='cyan'))
    # plot_s.append(lambda fig, fig_row: plot_feature_line(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_DIFF_grad_5', color='cyan'))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_DIFF', grad_window=1))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_SLOPE', grad_window=1))

    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_S', grad_window=1))

    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_D', grad_window=1))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_D', grad_window=5))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_H', grad_window=1))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_H', grad_window=5))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_DIFF', grad_window=2))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_DIFF', grad_window=5))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9', grad_window=1))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_S_N', grad_window=1))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_D_N', grad_window=1))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_H_N', grad_window=1))
    # plot_s.append(lambda fig, fig_row: plot_feature_line(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_SLOPE_grad_1', color='violet'))
    # plot_s.append(lambda fig, fig_row: plot_feature_line(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_SLOPE_grad_2', color='cyan'))
    # plot_s.append(lambda fig, fig_row: plot_feature_line(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_SLOPE_grad_5', color='violet'))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_SLOPE', grad_window=2))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_SLOPE', grad_window=5))
    # plot_s.append(lambda fig, fig_row: plot_feature_line(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_rel_diff', color='orange'))
    # plot_s.append(lambda fig, fig_row: plot_feature_line(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9_abs_diff', color='orange'))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9', grad_window=1))
    # plot_s.append(lambda fig, fig_row: plot_grads(df=df, fig=fig, fig_row=fig_row, time_feature=time_feature, feature='MACDh_12_26_9', grad_window=5))

    fig = make_subplots(rows=len(plot_s), cols=1, vertical_spacing=0.03, shared_xaxes=True)

    for fig_row in range(1, len(plot_s) + 1):
        plot = plot_s[fig_row - 1]
        plot(fig, fig_row)
        tickvals = df[time_feature]
        # ticktext = list(map(lambda t: datetime_h_m(t), tickvals))
        ticktext = list(map(lambda t: datetime_h_m__d_m(t), tickvals))
        fig.update_xaxes(tickangle=90, tickvals=tickvals, ticktext=ticktext, rangeslider_visible=False, range=[start_dt, end_dt], row=fig_row, col=1)
        fig.update_traces(xaxis=f"x{fig_row}")

    fig.update_layout(hovermode='x unified')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis_rangeslider_thickness=0.03)
    fig.update_layout(height=len(plot_s) * 400, paper_bgcolor="LightSteelBlue")
    fig.update_layout(margin=dict(l=30, r=20, t=40, b=20))
    names = set()
    fig.for_each_trace(lambda trace: trace.update(showlegend=False) if (trace.name in names) else names.add(trace.name))

    return fig


def fill_confusion_matrix(conf_mtx, net_cpu_empty, fig: Figure, row, col, group, take_each_n=1):
    ignore_clazz = net_cpu_empty.ignore_clazz()
    label_set = net_cpu_empty.label_s()

    exclude_conf_mtx_cl_s = []
    result_classes_count = len(label_set) - len(exclude_conf_mtx_cl_s)

    class_s = list(range(len(label_set)))
    included_conf_mtx = np.zeros((result_classes_count, result_classes_count))
    # included_label_set_formatted = [label_set[clazz] if clazz != ignore_clazz else f"<b>{label_set[clazz]}</b>" for clazz in class_s if clazz not in exclude_conf_mtx_cl_s]
    included_label_set_formatted = [f"[{clazz}]" if clazz != ignore_clazz else f"<b>[{clazz}]</b>" for clazz in class_s if clazz not in exclude_conf_mtx_cl_s]

    included_class_s = [clazz for clazz in class_s if clazz not in exclude_conf_mtx_cl_s]
    conf_mtx_np = np.array(conf_mtx)
    for clazz_x in included_class_s:
        for clazz_y in included_class_s:
            included_conf_mtx[included_class_s.index(clazz_x), included_class_s.index(clazz_y)] = conf_mtx_np[clazz_x, clazz_y]

    included_conf_mtx_annot = np.vectorize(lambda float_val: format_num(int(float_val)))(included_conf_mtx)
    heatmap = go.Heatmap(
        z=included_conf_mtx[::take_each_n, ::take_each_n],
        x=included_label_set_formatted[::take_each_n],
        y=included_label_set_formatted[::take_each_n],
        name='Confusion Matrix',
        text=included_conf_mtx_annot[::take_each_n, ::take_each_n],
        texttemplate="%{text}",
        textfont={"size": 9},
        colorscale='Reds',
        legendgroup=group,
        showscale=False
    )

    if is_cloud():
        fig.update_xaxes(tickfont=dict(size=10), ticklabelposition="inside", row=row, col=col)
        fig.update_yaxes(tickfont=dict(size=10), ticklabelposition="inside", row=row, col=col)
    else:
        fig.update_xaxes(tickfont=dict(size=10), row=row, col=col)
        fig.update_yaxes(tickfont=dict(size=10), row=row, col=col)

    fig.add_trace(heatmap, row=row, col=col)


def fill_auc_roc_map_stage3(auc_roc, signal_encoder, fig, row, col):
    auc_roc_s = auc_roc['roc_auc']
    xcross_diff_dist_cl_s = signal_encoder.xcross_diff_dist_cl_s()
    full_diff_dist_cl_s = signal_encoder.full_diff_dist_cl_s()

    x_s = []
    y_s = []
    z_s = []
    text_s = []

    for diff_dist_cl in full_diff_dist_cl_s:
        dist_cl = diff_dist_cl[1]
        diff_cl = diff_dist_cl[0]
        dist_label = signal_encoder.dist_label_s()[dist_cl]
        diff_label = signal_encoder.diff_label_s()[diff_cl]
        if diff_dist_cl in xcross_diff_dist_cl_s:
            auc_roc_val = np.nan
            text = '-'
        else:
            clazz = signal_encoder.diff_dist_cl__dd_cl__s(diff_dist_cl)[0]
            auc_roc = auc_roc_s[clazz]
            auc_roc_val = auc_roc
            text = round(auc_roc, 4) if not np.isnan(auc_roc) else 'NaN'

        x_s.append(dist_label)
        y_s.append(diff_label)
        z_s.append(auc_roc_val)
        text_s.append(text)

    heatmap = go.Heatmap(
        x=x_s,
        y=y_s,
        z=z_s,
        text=text_s,
        name='AUC ROC Map',
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale='Blues',
        showscale=False,
        zmin=0.5,
        zmax=1,
    )

    if is_cloud():
        fig.update_xaxes(tickfont=dict(size=12), tickvals=x_s, ticktext=x_s, ticklabelposition="inside", showgrid=False, zeroline=False, row=row, col=col)
        fig.update_yaxes(tickfont=dict(size=12), tickvals=y_s, ticktext=y_s, ticklabelposition="inside", showgrid=False, zeroline=False, row=row, col=col)

    fig.add_trace(heatmap, row=row, col=col)


def fill_auc_roc_map_stage4(auc_roc_map, net_cpu_empty, fig, row, col):
    zmin = 0.2
    zmax = 0.8

    x_s = []
    y_s = []
    z_s = []
    text_s = []

    for clazz_str, roc_auc in auc_roc_map['roc_auc'].items():
        clazz = int(clazz_str)
        text = _float_n(roc_auc, 5) if not np.isnan(roc_auc) else 'NaN'

        x_s.append(1)
        y_s.append(clazz)
        z_s.append(roc_auc)

        if clazz == net_cpu_empty.ignore_clazz():
            text_s.append(f"<b>[{clazz}]</b> | {text}")
        else:
            text_s.append(f"[{clazz}] | {text}")

    heatmap = go.Heatmap(
        x=x_s,
        y=y_s,
        z=z_s,
        name='AUC ROC',
        colorscale=[
            [0, 'rgb(255, 0, 0)'],
            [0.5, 'rgb(255, 255, 255)'],
            [1, 'rgb(0, 255, 0)']
        ],
        zmin=zmin,
        zmax=zmax,
        showscale=False
    )

    for clazz, text in enumerate(text_s):
        annotation = dict(
            x=1,
            y=clazz,
            text=text,
            font=dict(size=11, color='black'),
            showarrow=False,
            row=row, col=col)
        fig.add_annotation(**annotation)

    fig.update_xaxes(showticklabels=False, row=row, col=col)
    fig.update_yaxes(showticklabels=False, row=row, col=col)

    fig.add_trace(heatmap, row=row, col=col)


def fill_auc_roc_curve(auc_roc_map, net_cpu_empty, auc_roc_clazz_color_s, fig, row, col):
    annotation_font_size = 10

    tpr_s = auc_roc_map['tpr']
    fpr_s = auc_roc_map['fpr']
    auc_roc_s = auc_roc_map['roc_auc']

    auc_roc_min = np.abs(np.array(list(auc_roc_s.values()))).min()
    auc_roc_max = np.abs(np.array(list(auc_roc_s.values()))).max()

    ignore_annotation = None
    for clazz_str, auc_roc in dict(sorted(auc_roc_s.items(), key=lambda item: item[1])).items():
        clazz = int(clazz_str)
        if clazz == net_cpu_empty.ignore_clazz():
            color = 'black'
        else:
            color = auc_roc_clazz_color_s[clazz]

        fpr = fpr_s[clazz_str]
        take_each_n_item = AUC_ROC_DOWN_SAMPLING_RATIO_PRODUCER(len(fpr))

        fpr = fpr_s[clazz_str]
        tpr = tpr_s[clazz_str]
        fpr = [*fpr[::take_each_n_item], *[fpr[-1]]]
        tpr = [*tpr[::take_each_n_item], *[tpr[-1]]]

        if clazz == int(len(auc_roc_s) / 2):
            printmd_high(f"**CLASS = {clazz} | FPR size = {len(fpr)} | Take each {take_each_n_item}'s item**")
            pass

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                line=dict(color=color, width=1, dash='dash')
            ),
            row=row, col=col
        )

        auc_roc_annot_x = normalize(abs(auc_roc), auc_roc_min, auc_roc_max, 0.1, 0.9)
        auc_roc_annot_y = normalize(abs(auc_roc), auc_roc_min, auc_roc_max, 0.1, 0.9)

        # direction = signal_encoder.direction_label(clazz)
        # label = signal_encoder.label(clazz)

        auc_roc_formatted = "%0.4f" % auc_roc
        if clazz == net_cpu_empty.ignore_clazz():
            auc_roc_annot_text = f"<b>[{clazz}] IGNORE:</b> {auc_roc_formatted}"
        else:
            auc_roc_annot_text = f"[{clazz}]: {auc_roc_formatted}"

        annotation = dict(
                x=auc_roc_annot_x,
                y=auc_roc_annot_y,
                text=auc_roc_annot_text,
                font=dict(size=annotation_font_size, color='white'),
                bgcolor=color,
                showarrow=False,
                bordercolor="black",
                borderwidth=1,
                row=row, col=col
            )

        if clazz == net_cpu_empty.ignore_clazz():
            ignore_annotation = annotation
            continue

        fig.add_annotation(**annotation)

    fig.add_annotation(**ignore_annotation)

    if is_cloud():
        fig.update_xaxes(ticklabelposition="inside", showgrid=False, zeroline=True, row=row, col=col)
        fig.update_yaxes(ticklabelposition="inside", showgrid=False, zeroline=True, row=row, col=col)


def produce_model_precision_fig_stage4(net_cpu_empty, plot_data, resources_usage_present_plotly=None, existing_epochs=0, relayout_data=None, height=1800):
    annotation_font_size = 10

    clazzes_count = net_cpu_empty.clazzes_count()
    train_config = plot_data['train_config']

    roc_auc_clazz_color_s = [*list(COLORS.values()), *list(COLORS.values()), *list(COLORS.values())]
    assert len(roc_auc_clazz_color_s) > clazzes_count, f"len(colors): {len(roc_auc_clazz_color_s)} <= clazzes_count: {clazzes_count}"

    fig = make_subplots(
        rows=6, cols=6,
        horizontal_spacing=0.02,
        vertical_spacing=0.01,
        subplot_titles=(
            "CM INIT", "AR MAP INIT",
            "AR CURVE INIT", "AR CURVE TRAIN", "AR CURVE TEST",
            "CM TRAIN", "AR MAP TRAIN",
            "Duration / Cross entropy loss", "AUC ROC TRAIN", "AUC ROC TEST",
            "CM TEST", "AR MAP TEST",
            "Resources"
        ),
        specs=[
            [{"colspan": 5}, None, None, None, None, {"colspan": 1}],
            [{"colspan": 2}, None, {"colspan": 2}, None, {"colspan": 2}, None],
            [{"colspan": 5}, None, None, None, None, {"colspan": 1}],
            [{"colspan": 2, "secondary_y": True}, None, {"colspan": 2}, None, {"colspan": 2}, None],
            [{"colspan": 5}, None, None, None, None, {"colspan": 1}],
            [{"colspan": 6}, None, None, None, None, None],
        ],
        row_heights=[0.3, 0.15, 0.3, 0.115, 0.3, 0.03]
    )

    if plot_data is None:
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="LightSteelBlue",
            title_text=f"Evaluation..")

        if relayout_data is not None:
            fig.update_layout(height=height)

        return fig

    auc_roc_test = plot_data['test_validation_data']['auc_roc_map'] if plot_data['test_validation_data'] is not None else None
    auc_roc_train = plot_data['train_validation_data']['auc_roc_map'] if plot_data['train_validation_data'] is not None else None
    auc_roc_final = plot_data['final_validation_data']['auc_roc_map'] if plot_data['final_validation_data'] is not None else None

    conf_mtx_test = plot_data['test_validation_data']['conf_mtx'] if plot_data['test_validation_data'] is not None else None
    conf_mtx_train = plot_data['train_validation_data']['conf_mtx'] if plot_data['train_validation_data'] is not None else None
    conf_mtx_final = plot_data['final_validation_data']['conf_mtx'] if plot_data['final_validation_data'] is not None else None

    train_roc_auc_val_s = [validation['auc_roc_map']['roc_auc'] for validation in plot_data['train_validation_s']] if 'train_validation_s' in plot_data else None
    test_roc_auc_val_s = [validation['auc_roc_map']['roc_auc'] for validation in plot_data['test_validation_s']] if 'test_validation_s' in plot_data else None

    train_loss_s = plot_data['train_loss_s']
    test_loss_s = plot_data['test_loss_s']

    epoch_exec_time_s = plot_data['epoch_exec_time_s']
    current_exec_time_seconds = plot_data['current_exec_time_seconds']
    started_at = plot_data['started_at']
    last_updated_at = plot_data['last_updated_at']
    last_rendered_at = datetime.now(tz=ZoneInfo("Europe/Istanbul"))
    no_update_gap = last_rendered_at - last_updated_at

    epochs = train_config['EPOCHS']

    epoch_s = np.array(list(range(len(train_loss_s)))) + 1
    epoch = epoch_s[-1]

    epoch_exec_time_sec_s, dutationTickvals, durationTicktext = produce_timedelta_ticks(epoch_exec_time_s, 10)

    epoch_x_s = list(map(lambda e: e - 1, epoch_s))
    epoch_x_range = [0, epochs + 1]
    epoch_tick_s = list(range(epochs + 2))
    epoch_text_s = [*['E'], *list(np.array(epoch_tick_s[1:-1]) + existing_epochs), *['V']]

    duration_y_s = epoch_exec_time_sec_s
    duration_y_tick_text_s = [timedelta_h_m_s(timedelta(seconds=epoch_exec_time_sec)) for epoch_exec_time_sec in epoch_exec_time_sec_s]

    train_loss_annotation_s = ["%0.7f" % loss for loss in train_loss_s]
    test_loss_annotation_s = ["%0.7f" % loss for loss in test_loss_s]

    loss_title_map = {}
    conf_mtx_info_map = {}

    if auc_roc_test is not None and conf_mtx_test is not None:
        fill_confusion_matrix(conf_mtx_test, net_cpu_empty, fig, row=1, col=1, group=4)
        fill_auc_roc_map_stage4(auc_roc_test, net_cpu_empty, fig, row=1, col=6)
        fill_auc_roc_curve(auc_roc_test, net_cpu_empty, roc_auc_clazz_color_s, fig, row=2, col=1)
        # conf_mtx_test_info = build_conf_mtx_info_str(conf_mtx_test, signal_encoder)
        # if conf_mtx_test_info is not None:
        #     conf_mtx_info_map['CM INIT'] = conf_mtx_test_info

        loss_ = _float_n(test_loss_s[0], 5)
        loss_title_map['AR CURVE INIT'] = loss_

    if auc_roc_train is not None and conf_mtx_train is not None:
        fill_confusion_matrix(conf_mtx_train, net_cpu_empty, fig, row=3, col=1, group=5)
        fill_auc_roc_map_stage4(auc_roc_train, net_cpu_empty, fig, row=3, col=6)
        fill_auc_roc_curve(auc_roc_train, net_cpu_empty, roc_auc_clazz_color_s, fig, row=2, col=3)
        # conf_mtx_train_info = build_conf_mtx_info_str(conf_mtx_train, signal_encoder)
        # if conf_mtx_train_info is not None:
        #     conf_mtx_info_map['CM TRAIN'] = conf_mtx_train_info

        if auc_roc_final is None:
            loss_ = _float_n(train_loss_s[-1], 5)
        else:
            loss_ = _float_n(train_loss_s[-2], 5)
        loss_title_map['AR CURVE TRAIN'] = loss_

    if auc_roc_final is not None:
        fill_confusion_matrix(conf_mtx_final, net_cpu_empty, fig, row=5, col=1, group=6)
        fill_auc_roc_map_stage4(auc_roc_final, net_cpu_empty, fig, row=5, col=6)
        fill_auc_roc_curve(auc_roc_final, net_cpu_empty, roc_auc_clazz_color_s, fig, row=2, col=5)
        # conf_mtx_final_info = build_conf_mtx_info_str(conf_mtx_final, signal_encoder)
        # if conf_mtx_final_info is not None:
        #     conf_mtx_info_map['CM TEST'] = conf_mtx_final_info

        loss_ = _float_n(test_loss_s[-1], 5)
        loss_title_map['AR CURVE TEST'] = loss_

    fig.for_each_annotation(lambda a: a.update(text=a.text + ' | LOSS: ' + loss_title_map[a.text] if a.text in loss_title_map else a.text))
    fig.for_each_annotation(lambda a: a.update(text=conf_mtx_info_map[a.text] if a.text in conf_mtx_info_map else a.text))

    fig.add_trace(
        go.Scatter(
            x=epoch_x_s,
            y=duration_y_s,
            mode='lines+markers',
            marker=dict(size=5, color='yellow'),
            line=dict(color='yellow', width=2),
        ),
        row=4, col=1
    )

    for i in range(len(epoch_x_s)):
        fig.add_annotation(
            x=epoch_x_s[i],
            y=duration_y_s[i],
            text=duration_y_tick_text_s[i],
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=0,
            font=dict(size=annotation_font_size, color="black"),
            bgcolor="yellow",
            bordercolor="black",
            borderwidth=1,
            row=4, col=1
        )

    # fig.update_xaxes(tickvals=epoch_tick_s, ticktext=epoch_text_s, range=epoch_x_range, row=4, col=1)
    # fig.update_yaxes(showticklabels=False, row=4, col=1)

    fig.add_trace(
        go.Scatter(
            x=epoch_x_s,
            y=train_loss_s,
            mode='lines+markers',
            marker=dict(size=5, color='lightskyblue'),
            line=dict(color='lightskyblue', width=2),
        ),
        row=4, col=1,
        secondary_y=True
    )

    for i in range(len(epoch_x_s)):
        fig.add_annotation(
            x=epoch_x_s[i],
            y=train_loss_s[i],
            text=f"<b>{train_loss_annotation_s[i]}</b>",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-20,
            font=dict(size=annotation_font_size, color='white'),
            arrowcolor=color_with_opacity("lightskyblue", 0.5),
            bgcolor="lightskyblue",
            bordercolor="blue",
            borderwidth=1,
            row=4, col=1,
            secondary_y=True
        )

    fig.add_trace(
        go.Scatter(
            x=epoch_x_s,
            y=test_loss_s,
            mode='lines+markers',
            marker=dict(size=5, color='limegreen'),
            line=dict(color='limegreen', width=2),
        ),
        row=4, col=1,
        secondary_y=True
    )

    for i in range(len(epoch_x_s)):
        fig.add_annotation(
            x=epoch_x_s[i],
            y=test_loss_s[i],
            text=f"<b>{test_loss_annotation_s[i]}</b>",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=+20,
            font=dict(size=annotation_font_size, color='white'),
            arrowcolor=color_with_opacity("limegreen", 0.5),
            bgcolor="limegreen",
            bordercolor="green",
            borderwidth=1,
            row=4, col=1,
            secondary_y=True
        )

    fig.update_xaxes(tickvals=epoch_tick_s, ticktext=epoch_text_s, range=epoch_x_range, row=4, col=1)
    fig.update_yaxes(showticklabels=False, row=4, col=1, secondary_y=True)

    train_aucroc_max = max([max(train_roc_auc_val.values()) for train_roc_auc_val in train_roc_auc_val_s])
    test_aucroc_max = max([max(test_roc_auc_val.values()) for test_roc_auc_val in test_roc_auc_val_s])
    aucroc_abs_max = max(train_aucroc_max, test_aucroc_max)

    train_aucroc_min = min([min(train_roc_auc_val.values()) for train_roc_auc_val in train_roc_auc_val_s])
    test_aucroc_min = min([min(test_roc_auc_val.values()) for test_roc_auc_val in test_roc_auc_val_s])
    aucroc_abs_min = min(train_aucroc_min, test_aucroc_min)

    if train_roc_auc_val_s is not None:
        fill_roc_auc_series(net_cpu_empty, epoch_x_s, epoch_tick_s, epoch_text_s, epoch_x_range, train_roc_auc_val_s, roc_auc_clazz_color_s, aucroc_abs_min, aucroc_abs_max, fig, row=4, col=3)

    if test_roc_auc_val_s is not None:
        fill_roc_auc_series(net_cpu_empty, epoch_x_s, epoch_tick_s, epoch_text_s, epoch_x_range, test_roc_auc_val_s, roc_auc_clazz_color_s, aucroc_abs_min, aucroc_abs_max, fig, row=4, col=5)

    fig.update_xaxes(tickvals=[0, 1], ticktext=[0, 1], range=[0, 1], row=2, col=1)
    fig.update_yaxes(tickvals=[0, 1], ticktext=[0, 1], range=[0, 1], row=2, col=1)

    fig.update_xaxes(tickvals=[0, 1], ticktext=[0, 1], range=[0, 1], row=2, col=3)
    fig.update_yaxes(tickvals=[0, 1], ticktext=[0, 1], range=[0, 1], row=2, col=3)

    fig.update_xaxes(tickvals=[0, 1], ticktext=[0, 1], range=[0, 1], row=2, col=5)
    fig.update_yaxes(tickvals=[0, 1], ticktext=[0, 1], range=[0, 1], row=2, col=5)

    started_at = datetime_h_m_s(started_at)
    updated_at = datetime_h_m_s(last_updated_at)
    duration = str(current_exec_time_seconds).split(".")[0]
    rendered_at = datetime_h_m_s(last_rendered_at)
    no_update_gap = str(no_update_gap).split(".")[0]

    if current_exec_time_seconds < timedelta(seconds=1):
        title_text = f"Evaluated > Epoch: {existing_epochs + 1}.. || Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"
    else:
        title_text = f"Epoch: {existing_epochs + epoch - 1} > {existing_epochs + epoch}.. || Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"

    if epoch > epochs:
        title_text = f"Epoch {existing_epochs + epoch - 1} > Validation.. || Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"

    if epoch > epochs + 1:
        title_text = f"Validated > Finished || Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"

    if auc_roc_final is not None and auc_roc_test is None and auc_roc_train is None:
        title_text = f"Evaluated trained model"

    if resources_usage_present_plotly is not None:
        fig.add_trace(go.Scatter(x=[1], y=[1], opacity=0.0), row=6, col=1)

        fig.add_annotation(
            text=resources_usage_present_plotly,
            x=0, y=5,
            ax=0, ay=5,
            yanchor="middle",
            showarrow=False,
            align="left",
            font=dict(size=11, color="black"),
            opacity=0.8,
            row=6, col=1
        )

        fig.update_xaxes(visible=False, row=6, col=1)
        fig.update_yaxes(visible=False, row=6, col=1)

    fig.update_layout(
        title={
            'text': title_text,
            'font': {
                'size': 12
            }
        },
        paper_bgcolor="LightSteelBlue",
        showlegend=False,
        margin=dict(l=5, r=5, t=100, b=5)
    )

    if relayout_data is not None:
        fig.plotly_relayout(relayout_data)
        fig.update_layout(height=height)

    return fig


def produce_model_precision_fig_continous(net_cpu_empty, plot_data, resources_usage_present_plotly=None, existing_epochs=0, relayout_data=None, height=700):
    annotation_font_size = 10

    train_config = plot_data['train_config']

    fig = make_subplots(
        rows=2, cols=6,
        horizontal_spacing=0.03,
        vertical_spacing=0.04,
        subplot_titles=(
            "Duration", "TRAIN (Hubber loss)", "TEST (Hubber loss)"
        ),
        specs=[
            [{"colspan": 2}, None, {"colspan": 2}, None, {"colspan": 2}, None],
            [{"colspan": 6}, None, None, None, None, None],
        ],
        row_heights=[0.85, 0.15]
    )

    fig.update_yaxes(matches='y2', row=1, col=5)

    if plot_data is None:
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="LightSteelBlue",
            title_text=f"Evaluation..")

        if relayout_data is not None:
            fig.update_layout(height=height)

        return fig

    train_loss_s = plot_data['train_loss_s']
    test_loss_s = plot_data['test_loss_s']

    epoch_exec_time_s = plot_data['epoch_exec_time_s']
    current_exec_time_seconds = plot_data['current_exec_time_seconds']
    started_at = plot_data['started_at']
    last_updated_at = plot_data['last_updated_at']
    last_rendered_at = datetime.now(tz=ZoneInfo("Europe/Istanbul"))
    no_update_gap = last_rendered_at - last_updated_at

    epochs = train_config['EPOCHS']

    epoch_s = np.array(list(range(len(train_loss_s)))) + 1
    epoch = epoch_s[-1]

    epoch_exec_time_sec_s, dutationTickvals, durationTicktext = produce_timedelta_ticks(epoch_exec_time_s, 10)

    loss_title_map = {}

    epoch_x_s = list(map(lambda e: e - 1, epoch_s))
    epoch_x_range = [0, epochs + 1]
    epoch_tick_s = list(range(epochs + 2))
    epoch_text_s = [*['E'], *list(np.array(epoch_tick_s[1:-1]) + existing_epochs), *['V']]

    duration_y_s = epoch_exec_time_sec_s
    duration_y_tick_text_s = [timedelta_h_m_s(timedelta(seconds=epoch_exec_time_sec)) for epoch_exec_time_sec in epoch_exec_time_sec_s]

    train_loss_annotation_s = ["%0.7f" % loss for loss in train_loss_s]
    test_loss_annotation_s = ["%0.7f" % loss for loss in test_loss_s]

    fig.for_each_annotation(lambda a: a.update(text=a.text + ' | LOSS: ' + loss_title_map[a.text] if a.text in loss_title_map else a.text))

    fig.add_trace(
        go.Scatter(
            x=epoch_x_s,
            y=duration_y_s,
            mode='lines+markers',
            marker=dict(size=5, color='yellow'),
            line=dict(color='yellow', width=2),
        ),
        row=1, col=1
    )

    for i in range(len(epoch_x_s)):
        fig.add_annotation(
            x=epoch_x_s[i],
            y=duration_y_s[i],
            text=duration_y_tick_text_s[i],
            showarrow=True,
            arrowhead=2,
            ax=50,
            ay=0,
            font=dict(size=annotation_font_size, color="black"),
            bgcolor="yellow",
            bordercolor="black",
            borderwidth=1,
            row=1, col=1
        )

    fig.update_xaxes(tickvals=epoch_tick_s, ticktext=epoch_text_s, range=epoch_x_range, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=epoch_x_s,
            y=train_loss_s,
            mode='lines+markers',
            marker=dict(size=5, color='lightskyblue'),
            line=dict(color='lightskyblue', width=2),
        ),
        row=1, col=3,
    )

    for i in range(len(epoch_x_s)):
        fig.add_annotation(
            x=epoch_x_s[i],
            y=train_loss_s[i],
            text=f"<b>{train_loss_annotation_s[i]}</b>",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-20,
            font=dict(size=annotation_font_size, color='white'),
            arrowcolor=color_with_opacity("lightskyblue", 0.5),
            bgcolor="lightskyblue",
            bordercolor="blue",
            borderwidth=1,
            row=1, col=3,
        )

    fig.update_xaxes(tickvals=epoch_tick_s, ticktext=epoch_text_s, range=epoch_x_range, row=1, col=3)

    fig.add_trace(
        go.Scatter(
            x=epoch_x_s,
            y=test_loss_s,
            mode='lines+markers',
            marker=dict(size=5, color='limegreen'),
            line=dict(color='limegreen', width=2),
        ),
        row=1, col=5,
    )

    for i in range(len(epoch_x_s)):
        fig.add_annotation(
            x=epoch_x_s[i],
            y=test_loss_s[i],
            text=f"<b>{test_loss_annotation_s[i]}</b>",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=+20,
            font=dict(size=annotation_font_size, color='white'),
            arrowcolor=color_with_opacity("limegreen", 0.5),
            bgcolor="limegreen",
            bordercolor="green",
            borderwidth=1,
            row=1, col=5,
        )

    fig.update_xaxes(tickvals=epoch_tick_s, ticktext=epoch_text_s, range=epoch_x_range, row=1, col=5)

    if resources_usage_present_plotly is not None:
        fig.add_trace(go.Scatter(x=[1], y=[1], opacity=0.0), row=2, col=1)

        fig.add_annotation(
            text=resources_usage_present_plotly,
            x=0, y=5,
            ax=0, ay=5,
            yanchor="middle",
            showarrow=False,
            align="left",
            font=dict(size=11, color="black"),
            opacity=0.8,
            row=2, col=1
        )

        fig.update_xaxes(visible=False, row=2, col=1)
        fig.update_yaxes(visible=False, row=2, col=1)

    started_at = datetime_h_m_s(started_at)
    updated_at = datetime_h_m_s(last_updated_at)
    duration = str(current_exec_time_seconds).split(".")[0]
    rendered_at = datetime_h_m_s(last_rendered_at)
    no_update_gap = str(no_update_gap).split(".")[0]

    if current_exec_time_seconds < timedelta(seconds=1):
        title_text = f"Evaluated > Epoch: {existing_epochs + 1}.. || Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"
    else:
        title_text = f"Epoch: {existing_epochs + epoch - 1} > {existing_epochs + epoch}.. || Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"

    if epoch > epochs:
        title_text = f"Epoch {existing_epochs + epoch - 1} > Validation.. || Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"

    if epoch > epochs + 1:
        title_text = f"Validated > Finished || Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"

    # if auc_roc_final is not None and auc_roc_test is None and auc_roc_train is None:
    #     title_text = f"Evaluated trained model"

    fig.update_layout(
        title={
            'text': title_text,
            'font': {
                'size': 12
            }
        },
        paper_bgcolor="LightSteelBlue",
        showlegend=False,
        margin=dict(l=5, r=5, t=100, b=5)
    )

    if relayout_data is not None:
        fig.plotly_relayout(relayout_data)
        fig.update_layout(height=height)

    return fig


def fill_roc_auc_series(net_cpu_empty, epoch_x_s, epoch_tick_s, epoch_text_s, epoch_x_range, roc_auc_val_s, roc_auc_clazz_color_s, aucroc_abs_min, aucroc_abs_max, fig, row, col):
    annotation_font_size = 10

    roc_auc_clazz_s = {}
    all_roc_auc_s = []
    for clazz in range(net_cpu_empty.clazzes_count()):
        clazz_str = str(clazz)
        auc_roc_s = []
        for roc_auc in roc_auc_val_s:
            auc_roc_clazz = roc_auc[clazz_str]
            auc_roc_s.append(auc_roc_clazz)
            all_roc_auc_s.append(auc_roc_clazz)

        roc_auc_clazz_s[clazz_str] = auc_roc_s

    def plot_auc_roc_serie(clazz, roc_auc_s):
        color = roc_auc_clazz_color_s[clazz] if clazz != net_cpu_empty.ignore_clazz() else 'black'

        fig.add_trace(
            go.Scatter(
                x=epoch_x_s,
                y=roc_auc_s,
                mode='lines+markers',
                marker=dict(size=5, color=color),
                line=dict(color=color, width=2),
                opacity=0.75
            ),
            row=row, col=col
        )

        auc_roc_annot_x = epoch_x_s[-1]
        auc_roc_annot_y = roc_auc_s[-1]

        direction = net_cpu_empty.direction_label(clazz)
        label = net_cpu_empty.label(clazz)

        auc_roc_formatted = "%0.4f" % auc_roc_annot_y
        # auc_roc_formatted = f"{auc_roc_annot_y}"
        if clazz == net_cpu_empty.ignore_clazz():
            auc_roc_annot_text = f"<b>[{clazz}] IGNORE:</b> {auc_roc_formatted}"
        else:
            auc_roc_annot_text = f"[{clazz}]: {auc_roc_formatted}"

        fig.add_annotation(
            x=auc_roc_annot_x,
            y=auc_roc_annot_y,
            text=auc_roc_annot_text,
            showarrow=True,
            arrowhead=2,
            ax=50,
            font=dict(size=annotation_font_size, color='white'),
            bgcolor=color,
            bordercolor="black",
            borderwidth=1,
            row=row, col=col
        )

    ignore_auc_roc_s = None
    for clazz_str, auc_roc_s in roc_auc_clazz_s.items():
        clazz = int(clazz_str)
        if clazz == net_cpu_empty.ignore_clazz():
            ignore_auc_roc_s = auc_roc_s
            continue

        plot_auc_roc_serie(clazz, auc_roc_s)

    plot_auc_roc_serie(net_cpu_empty.ignore_clazz(), ignore_auc_roc_s)

    fig.update_xaxes(tickvals=epoch_tick_s, ticktext=epoch_text_s, range=epoch_x_range, row=row, col=col)
    fig.update_yaxes(showticklabels=False, range=[aucroc_abs_min / 1.05, aucroc_abs_max * 1.05], row=row, col=col)


def plot__clazzes_weights__distribution(label_s, original_weights, adjusted_weights=None, log_y=False, show_as_img=False):
    import plotly.graph_objects as go

    fig = go.Figure()
    clazz_s = list(range(len(original_weights)))

    norm_sum_original = round(sum(original_weights), 5)
    fig.add_trace(go.Scatter(x=clazz_s, y=list(original_weights), mode='lines+markers', fillcolor='red', name=f'Original weights [{norm_sum_original}]', hovertemplate='%{x:d}, %{y:.6f}'))

    if adjusted_weights is not None:
        norm_sum_adjusted = round(sum(adjusted_weights), 5)
        fig.add_trace(go.Scatter(x=clazz_s, y=list(adjusted_weights), mode='lines+markers', fillcolor='green', name=f'Adjusted weights [{norm_sum_adjusted}]', hovertemplate='%{x:d}, %{y:.6f}'))

    fig.update_layout(
        title='Clazz - Weights distribution',
        xaxis_title='Clazzes',
        yaxis_title='Weights',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        legend_orientation="h",
        paper_bgcolor="LightSteelBlue")

    x_ticks = clazz_s
    x_texts = [label_s[clazz] for clazz in clazz_s]
    fig.update_layout(xaxis=dict(tickangle=-90, tickvals=x_ticks, ticktext=x_texts, range=[0, x_ticks[-1]]))
    if log_y:
        fig.update_yaxes(type="log")

    display_plot(fig, show_as_img=show_as_img)


btn_stop_style = {'width': '100%', 'color': 'white', 'backgroundColor': 'red', 'border': 'none', 'padding': '5px 5px', 'cursor': 'pointer'}
btn_finished_style = {'width': '100%', 'color': 'white', 'backgroundColor': 'green', 'border': 'none', 'padding': '5px 5px', 'cursor': 'pointer'}
btn_pending_style = {'width': '100%', 'color': 'white', 'backgroundColor': 'yellow', 'border': 'none', 'padding': '5px 5px', 'cursor': 'pointer'}

net_discretization_map = {}
symbol_discretization_df_map = {}


def plot_input_feature_abs_max_distribution(configs_suffix, discretization, num_workers=5):
    import pandas as pd
    from SRC.LIBRARIES.new_data_utils import initialize_dataframe_s
    from SRC.CORE.utils import read_json
    import plotly.express as px
    from SRC.CORE._CONSTANTS import CONFIGS_FILE_PATH

    configs = read_json(CONFIGS_FILE_PATH(suffix=configs_suffix))
    all_symbol_s = list(set([conf['symbol'] for conf in configs['train']]))

    input_features = configs['input_features']
    exclude_symbol_s = configs['exclude_symbol_s']

    filtered_symbol_s = [symbol for symbol in all_symbol_s if symbol not in exclude_symbol_s]

    df_s = initialize_dataframe_s(filtered_symbol_s, discretization, feature_s=['symbol', *input_features], num_workers=num_workers)
    concated_df = pd.concat(df_s)
    symbol_group_df = concated_df.groupby('symbol')[input_features].max().reset_index()

    for input_feature in input_features:
        line_df = symbol_group_df[['symbol', input_feature]]
        fig_bins = px.bar(line_df, x="symbol", y=f"{input_feature}", title=f"Distribution through symbols of {input_feature} feature")
        fig_bins.update_xaxes(tickangle=-75)
        fig_bins.update_layout(xaxis=dict(tickangle=-75, tickvals=line_df['symbol'].to_list(), ticktext=line_df['symbol'].to_list()))
        fig_bins.update_layout(paper_bgcolor="LightSteelBlue")
        fig_bins.show()


def plot_diff_clustering(df, threshold, include_poligon:Polygon, size=700):
    _included = __INCLUDED(threshold)
    _curr_diff = f'curr_abs_{__DIFF(threshold)}'
    _prev_diff = f'prev_abs_{__DIFF(threshold)}'

    included_data = df[df[_included] == True]
    excluded_data = df[df[_included] == False]

    x_s, y_s = include_poligon.exterior.xy
    figure_x_s = list(x_s)
    figure_y_s = list(y_s)

    total_points_count = 3_000
    included_points_count = len(included_data)
    excluded_points_cunt = len(excluded_data)
    included_part = included_points_count / (included_points_count + excluded_points_cunt)
    excluded_part = excluded_points_cunt / (included_points_count + excluded_points_cunt)

    included_show_each_n = int(included_points_count / (total_points_count * included_part)) if included_part > 0 else 0
    excluded_show_each_n = int(excluded_points_cunt / (total_points_count * excluded_part)) if excluded_part > 0 else 0

    included_data = included_data[::included_show_each_n] if included_show_each_n >= 1 else included_data
    excluded_data = excluded_data[::excluded_show_each_n] if excluded_show_each_n >= 1 else excluded_data

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=included_data[_curr_diff].to_list(), y=included_data[_prev_diff].to_list(), mode='markers', marker=dict(color='green')))
    fig.add_trace(go.Scatter(x=excluded_data[_curr_diff].to_list(), y=excluded_data[_prev_diff].to_list(), mode='markers', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=figure_x_s, y=figure_y_s, name="Closed figure", line=dict(color='yellow')))
    fig.update_xaxes(tickangle=-75)
    fig.update_layout(paper_bgcolor="LightSteelBlue")
    fig.update_layout(width=size, height=size)
    fig.update_layout(showlegend=False)

    display_plot(fig)


def plot_diff_dist_data_distribution(diff_dist_data):
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('matplotlib', 'inline')

    # plot__diff_dist__heatmap_1(diff_dist_data)
    plot__diff_dist__heatmap_clazzmap(diff_dist_data)
    # plot__diff_dist__heatmap_3(diff_dist_data)

    # if ipython is not None:
    # 	ipython.run_line_magic('matplotlib', 'notebook')
    # plot__diff_dist__bar_3d(diff_dist_data)


def plot__diff_dist__heatmap_1(data):
    if _PLOT_ENABLED in os.environ and not string_bool(os.environ[_PLOT_ENABLED]):
        return

    heat_map = data['heat_map']
    diff_counts = data['diff']['counts']
    diff_ticks = data['diff']['classes']
    diff_labels = data['diff']['labels']
    dist_counts = data['dist']['counts']
    dist_ticks = data['dist']['classes']
    dist_labels = data['dist']['labels']

    plt.rcParams["figure.figsize"] = (10, 10)
    fig, ax = plt.subplots()

    heat_map_annot = np.where(np.isnan(heat_map), NA_PRESENT, heat_map.astype(int))
    heat_map_no_nan = np.nan_to_num(heat_map, nan=0)

    vmax = sorted(list(heat_map_no_nan.flatten()))[-2] * 1.3
    im = ax.imshow(heat_map_no_nan, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
    for i in range(len(diff_counts)):
        for j in range(len(dist_counts)):
            text = ax.text(j, i, heat_map_annot[i][j], ha="center", va="center", color="w")

    plt.ylim([0 - 0.5, len(diff_ticks) - 0.5])
    plt.xlim([0 - 0.5, len(dist_ticks) - 0.5])
    plt.yticks(ticks=diff_ticks, labels=diff_labels, rotation='horizontal')
    plt.xticks(ticks=dist_ticks, labels=dist_labels, rotation='vertical')
    plt.grid(None)


def plot__diff_dist__heatmap_clazzmap(diff_dist_data):
    import numpy as np
    import plotly.graph_objects as go
    from SRC.CORE.plot_utils import display_plot
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.15,
        subplot_titles=['Clazzes distribution', 'Diff Dist - Clazz Map']
    )

    diff_cl_s, dist_cl_s = diff_dist_data['diff']['classes'], diff_dist_data['dist']['classes']
    ignore_diff_cl = get_diff_center_cl(diff_cl_s)
    ignore_diff_dist_cl = get_ignore_diff_dist_cl(diff_cl_s)

    heat_map = np.array(diff_dist_data['heat_map'])
    clazz_map = np.array(diff_dist_data['clazz_map'])
    diff_label_s = diff_dist_data['diff']['labels']
    diff_cl_s = diff_dist_data['diff']['classes']
    dist_label_s = diff_dist_data['dist']['labels']
    dist_cl_s = diff_dist_data['dist']['classes']
    clazzes_count = get_clazzes_count(diff_cl_s, dist_cl_s)

    heat_map_no_nan = np.nan_to_num(heat_map, nan=0)

    heat_map_annot = np.where(np.isnan(heat_map), NA_PRESENT, heat_map.astype(int))
    heat_map_annot = np.vectorize(lambda num_str: format_num(int(num_str)) if num_str != NA_PRESENT else NA_PRESENT)(heat_map_annot)

    clazz_map_annot = np.where(np.isnan(clazz_map), NA_PRESENT, clazz_map.astype(int))
    clazz_map_no_nan = np.vectorize(lambda clazz: np.nan if np.isnan(clazz) else normalize(clazz + 1, 1, clazzes_count, 0, 1))(clazz_map)

    # fig.update_xaxes(ticklabelposition="inside", showgrid=False, tickfont=dict(size=14), tickvals=dist_label_s, ticktext=dist_label_s, row=1, col=2)
    # fig.update_yaxes(ticklabelposition="inside", showgrid=False, tickfont=dict(size=14), tickvals=diff_label_s, ticktext=diff_label_s, row=1, col=2)

    fig.add_trace(go.Heatmap(
        x=dist_label_s,
        y=diff_label_s,
        z=clazz_map_no_nan,
        name='Diff Diss - Clazz Map',
        text=clazz_map_annot,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale=[
            [0, 'rgb(229, 245, 224)'],
            [1, 'rgb(0, 109, 44)']
        ],
        showscale=False
    ), row=1, col=2)

    dist_cl_s_entries_count = 0
    for i in range(1, len(dist_label_s)):
        dist_cl_i_entries_count = int(heat_map_no_nan[:, i].sum())
        dist_label_s[i] = f"{dist_label_s[i]} ({format_num(dist_cl_i_entries_count)})"
        dist_cl_s_entries_count += dist_cl_i_entries_count

    dist_label_s[0] = f"{dist_label_s[0]} ({format_num(dist_cl_s_entries_count)})"

    diff_cl_s_entries_count = 0
    for i in range(0, len(diff_label_s)):
        if i == ignore_diff_cl:
            continue

        diff_cl_i_entries_count = int(heat_map_no_nan[i].sum())
        diff_label_s[i] = f"{diff_label_s[i]} ({format_num(diff_cl_i_entries_count)})"
        diff_cl_s_entries_count += diff_cl_i_entries_count

    diff_label_s[ignore_diff_cl] = f"{diff_label_s[ignore_diff_cl]} ({format_num(diff_cl_s_entries_count)})"

    # fig.update_xaxes(ticklabelposition="inside", showgrid=False, tickfont=dict(size=14), tickvals=dist_label_s, ticktext=dist_label_s, row=1, col=1)
    # fig.update_yaxes(ticklabelposition="inside", showgrid=False, tickfont=dict(size=14), tickvals=diff_label_s, ticktext=diff_label_s, row=1, col=1)

    heat_map_no_nan_optim = np.vectorize(lambda float_val: np.nan if is_close_to_zero(float_val) or np.isnan(float_val) else math.log(float_val))(heat_map_no_nan)

    fig.add_trace(go.Heatmap(
        x=dist_label_s,
        y=diff_label_s,
        z=heat_map_no_nan_optim,
        name='Clazzes distribution',
        text=heat_map_annot,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale='Blues',
        zmin=0,
        zmax=heat_map_no_nan_optim[ignore_diff_dist_cl[0], ignore_diff_dist_cl[1]],
        showscale=False
    ), row=1, col=1)


    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        paper_bgcolor="LightSteelBlue")

    display_plot(fig, show_as_img=True)


def plot__diff_dist__heatmap_3(data):
    import seaborn

    heat_map = data['heat_map']
    diff_labels = data['diff']['labels']
    dist_labels = data['dist']['labels']

    heat_map_annot = np.where(np.isnan(heat_map), NA_PRESENT, heat_map.astype(int))
    heat_map_no_nan = np.nan_to_num(heat_map, nan=0)

    vmax = sorted(list(heat_map_no_nan.flatten()))[-2] * 1.3
    ax = seaborn.heatmap(heat_map_no_nan, annot=heat_map_annot, fmt='', xticklabels=dist_labels, yticklabels=diff_labels, vmin=0, vmax=vmax)
    ax.invert_yaxis()


def plot__pivot_diff__histogram(data):
    import seaborn

    heat_map = data['heat_map']
    diff_counts = data['diff']['counts']
    diff_labels = data['diff']['labels']

    vmax = sorted(list(heat_map.flatten()))[-2] * 1.3

    ax = seaborn.barplot(x=diff_labels, y=diff_counts)
    ax.set_xticklabels(rotation=90, labels=diff_labels)


def plot__pivot_dist__histogram(data):
    import seaborn

    dist_counts = data['dist']['counts']
    dist_labels = data['dist']['labels']

    ax = seaborn.barplot(x=dist_labels, y=dist_counts)
    ax.set_xticklabels(rotation=90, labels=dist_labels)


def plot__pivot_diff_dist__histograms(data):
    import seaborn
    from matplotlib import pyplot as plt

    seaborn.set()

    diff_counts = data['diff']['counts']
    diff_labels = data['diff']['labels']

    dist_counts = data['dist']['counts']
    dist_labels = data['dist']['labels']

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('DIFF + DIST pivot distribution')

    axes[0].set_title('DIFF pivot distribution')
    seaborn.barplot(ax=axes[0], x=diff_labels, y=diff_counts)
    axes[0].set_xticklabels(rotation=90, labels=diff_labels)

    axes[1].set_title('DIST pivot distribution')
    seaborn.barplot(ax=axes[1], x=dist_labels, y=dist_counts)
    axes[1].set_xticklabels(rotation=90, labels=dist_labels)

    display_plot(fig)


def plot__diff_dist__bar_3d(data):
    diff_bins = data['diff']['bins']
    diff_classes = data['diff']['classes']
    diff_labels = data['diff']['labels']
    dist_bins = data['dist']['bins']
    dist_classes = data['dist']['classes']
    dist_labels = data['dist']['labels']
    x_s = data['bar_3D']['x_s']
    y_s = data['bar_3D']['y_s']
    z_s = data['bar_3D']['z_s']

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    xpos = x_s
    ypos = y_s
    dz = z_s

    num_elements = len(xpos)
    zpos = np.zeros(num_elements)
    dx = np.ones(num_elements)
    dy = np.ones(num_elements)

    ax.bar3d(xpos, ypos, zpos, dx, dy, np.log10(dz) + 1, color='#00ceaa', shade=True)
    plt.axis('equal')
    plt.gca().set_xlim(0, len(diff_bins))
    plt.gca().set_ylim(0, len(diff_bins))
    plt.xticks(dist_classes, dist_labels)
    ax.set_xticklabels(dist_labels, rotation=+45, horizontalalignment='right')
    plt.yticks(diff_classes, diff_labels)
    ax.set_yticklabels(diff_labels, rotation=-20, horizontalalignment='left')
    plt.show()


def plot_centered_signal_df(df:pd.DataFrame, curr_idx, threshold, window=150):
    symbol = df.iloc[0][_SYMBOL]
    discretization = df.iloc[0][_DISCRETIZATION]

    _signal = __SIGNAL(threshold)
    _included = __INCLUDED(threshold)
    _prev_abs_diff = f'prev_abs_{__DIFF(threshold)}'
    _curr_abs_diff = f'curr_abs_{__DIFF(threshold)}'

    time_delta = TIME_DELTA(discretization=discretization) * window

    df = df[(df[_UTC_TIMESTAMP] >= curr_idx - time_delta / 2) & (df[_UTC_TIMESTAMP] <= curr_idx + time_delta / 2)]

    fig = make_subplots(cols=1, rows=1)
    plot_candles(df=df, fig=fig, fig_row=1, time_feature=_UTC_TIMESTAMP)

    if _included not in df.columns:
        time_ticks = df[(df[_signal] != SIGNAL_IGNORE)][_UTC_TIMESTAMP]

        plot_candles_zigzags(df=df, fig=fig, fig_row=1, time_feature=_UTC_TIMESTAMP, threshold=threshold)
    else:
        time_ticks = df[(df[_signal] != SIGNAL_IGNORE) & (df[_included] == True)][_UTC_TIMESTAMP]

        excluded_longin_df = df[(df[_included] == False) & (df[_signal] == SIGNAL_LONG_IN)]
        excluded_shortin_df = df[(df[_included] == False) & (df[_signal] == SIGNAL_SHORT_IN)]

        included_longin_df = df[(df[_included] == True) & (df[_signal] == SIGNAL_LONG_IN)]
        included_shortin_df = df[(df[_included] == True) & (df[_signal] == SIGNAL_SHORT_IN)]

        fig.add_trace(go.Scatter(x=excluded_longin_df[_UTC_TIMESTAMP],  y=excluded_longin_df[_CLOSE], mode="markers", marker=dict(size=7, symbol="circle-x", color="green"), name='Excluded LONG'), row=1, col=1)
        fig.add_trace(go.Scatter(x=excluded_shortin_df[_UTC_TIMESTAMP], y=excluded_shortin_df[_CLOSE], mode="markers", marker=dict(size=7, symbol="circle-x", color="red"), name='Excluded SHORT'), row=1, col=1)

        fig.add_trace(go.Scatter(x=included_longin_df[_UTC_TIMESTAMP],  y=included_longin_df[_CLOSE], mode="markers", marker=dict(color='rgba(0, 255, 0, 0.2)', size=8, line=dict(color='green', width=3)), name='Included LONG'), row=1, col=1)
        fig.add_trace(go.Scatter(x=included_shortin_df[_UTC_TIMESTAMP], y=included_shortin_df[_CLOSE], mode="markers", marker=dict(color='rgba(255, 0, 0, 0.2)', size=8, line=dict(color='red', width=3)), name='Included SHORT'), row=1, col=1)

        df['curr_prev'] = df[[_prev_abs_diff, _curr_abs_diff]].apply(lambda row: f"{round(row[_prev_abs_diff], 4)}|{round(row[_curr_abs_diff], 4)}", axis=1)
        fig.add_trace(go.Scatter(x=df[_UTC_TIMESTAMP], y=df[_CLOSE], hovertext=df['curr_prev'], mode='markers', marker=dict(size=5, color='rgba(255, 255, 255, 0.2)'), showlegend=False), row=1, col=1)

    time_labels = [datetime_m_d__h_m(time_tick, tz=UTC_TZ) for time_tick in time_ticks]
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis_rangeslider_thickness=0.03)
    fig.update_xaxes(tickvals=time_ticks, ticktext=time_labels)
    fig.update_yaxes(tickformat=",.5f")
    fig.update_layout(title_text=f"<b>{symbol} | {discretization} | {threshold}</b>")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis_rangeslider_thickness=0.03)
    fig.update_layout(height=400, paper_bgcolor="LightSteelBlue")
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.update_layout(hovermode='x unified')
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_traces(xaxis="x")
    names = set()
    fig.for_each_trace(lambda trace: trace.update(showlegend=False) if (trace.name in names) else names.add(trace.name))
    # display_plot(fig)
    fig.show()

    # display(df[[_CLOSE, __SIGNAL(threshold), __INCLUDED(threshold)]])


def plot_centered_tpr_extremums(df_concat, df_s, threshold, take=25, window=300, per_symbol=1):
    printmd(populate_char_n_times("#", 70, "**TPR EXTREMUMS**"))

    feature = __TPR(threshold)
    _predicate = lambda df_concat_nona, feature_max: np.isclose(df_concat_nona[feature].abs(), feature_max, rtol=1e-5)

    return plot_centered_feature_extremums(df_concat, df_s, feature, _predicate, take, window, per_symbol)


def plot_centered_diff_extremums(df_concat, df_s, threshold, take=25, window=300, per_symbol=1):
    printmd(populate_char_n_times("#", 70, "**DIFF EXTREMUMS**"))

    feature = __DIFF(threshold)
    _predicate = lambda df_concat_nona, feature_max: np.isclose(df_concat_nona[feature].abs(), feature_max, rtol=1e-5)

    return plot_centered_feature_extremums(df_concat, df_s, feature, _predicate, take, window, per_symbol)


def plot_centered_dist_extremums(df_concat, df_s, threshold, take=25, window=300, per_symbol=1):
    printmd(populate_char_n_times("#", 70, "**DIST EXTREMUMS**"))

    feature = __DIST(threshold)
    _predicate = lambda df_concat_nona, feature_max: df_concat_nona[feature].astype('int16') == int(feature_max)

    return plot_centered_feature_extremums(df_concat, df_s, feature, _predicate, take, window, per_symbol)


def plot_centered_feature_extremums(df_concat, df_s, feature, _predicate, take=25, window=300, per_symbol=1):
    threshold = float(feature.split("_")[1])
    df_concat_nona = df_concat.dropna(subset=[feature])
    feature_series_concat_nona = df_concat_nona[feature]
    feature_max_s_desc_ordered = [dist for dist in sorted(feature_series_concat_nona.abs().to_list())[::-1]]

    # printmd(f"len(df_concat): ***{len(df_concat)}*** | len(df_concat_nona): ***{len(df_concat_nona)}*** | len(feature_max_s_desc_ordered): ***{len(feature_max_s_desc_ordered)}***")

    already_shown_symbol_s = []
    for feature_max in feature_max_s_desc_ordered:
        if len(already_shown_symbol_s) >= take:
            break

        df_concat_nona_filtered = df_concat_nona[_predicate(df_concat_nona, feature_max)]
        if len(df_concat_nona_filtered) == 0:
            break

        row = df_concat_nona_filtered.iloc[0]
        symbol = row[_SYMBOL]

        if already_shown_symbol_s.count(symbol) >= per_symbol:
            continue

        idx = row[_UTC_TIMESTAMP]
        printmd(f"***{symbol}*** MAX {feature}: **{feature_max}** at: **{idx}**")
        df_ = [result_df for result_df in df_s if result_df.iloc[0][_SYMBOL] == symbol]
        if len(df_) == 0:
            already_shown_symbol_s.append(symbol)
            continue

        plot_centered_signal_df(df_[0], idx, threshold, window=window)

        already_shown_symbol_s.append(symbol)

    symbol_s = remove_list_duplicates(already_shown_symbol_s)

    return symbol_s


def plot_centered_feature_ohlc(df, idx, window=150, img_path=None, img_width=None, img_height=None):
    fig = make_subplots(cols=1, rows=1)
    if isinstance(idx, int):
        idx = df.index[idx]

    rand_i = random.randint(5, len(df) - 5)
    discretization_td = df.iloc[rand_i].name - df.iloc[rand_i-1].name
    side_width = int(window / 2)
    filtered_df = df.loc[idx - discretization_td * side_width:idx + discretization_td * side_width]
    plot_candles(df=filtered_df, fig=fig, fig_row=1, time_feature=None)
    fig.add_vline(x=idx, line_width=3, line_dash="dash", line_color="cyan", row=1, col=1)

    if img_path is None:
        fig.show()
    else:
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        fig.write_image(img_path, width=img_width, height=img_height)


def plot__diff_tpr__distribution(bins, counts, log_y=True, show_as_img=False):
    import plotly.express as px

    title = f"Distribution DIFF/TPR [LOG Y: {log_y}]"
    bin_presentation_s = produce_bins_presentation(bins)

    line_df = pd.DataFrame({"counts": counts, "bins": list(range(len(bin_presentation_s))), "bin_presentation": bin_presentation_s})
    if log_y:
        fig_bins = px.bar(line_df, x="bins", y="counts", custom_data="bin_presentation", title=title, log_y=log_y)
        fig_bins.update_yaxes(type="log")
    else:
        fig_bins = px.bar(line_df, x="bins", y="counts", custom_data="bin_presentation", title=title)

    fig_bins.update_xaxes(tickangle=90)
    fig_bins.update_traces(hovertemplate='Bin: %{customdata}<br>Occurency: %{y}')
    fig_bins.update_layout(paper_bgcolor="LightSteelBlue")
    fig_bins.update_layout(xaxis=dict(tickvals=line_df['bins'].to_list(), ticktext=line_df['bin_presentation'].to_list()))
    fig_bins.update_layout(title_text=f"{title}")
    fig_bins.update_layout(paper_bgcolor="LightSteelBlue")

    display_plot(fig_bins, show_as_img=show_as_img)


def show__plot_data(model_name_suffix):
    import os
    from SRC.CORE.utils import read_json
    from SRC.CORE._CONSTANTS import project_root_dir, _RESOURCES_FORMAT_PLOTLY
    from SRC.LIBRARIES.new_utils import produce_resource_usage_format, produce_empty_net, produce_inference_model
    from IPython.display import Image

    from SRC.NN.BaseContinousNN import BaseContinousNN
    from SRC.NN.BaseDiscreateNN import BaseDiscreateNN

    data_path = f"{project_root_dir()}/plot_data.json"
    img_path = f"{project_root_dir()}/plot_data.png"

    os.environ['STAGE'] = '4'

    net_cpu_empty = produce_empty_net(model_name_suffix)

    plot_data = read_json(data_path)
    resources_usage_format_plotly = produce_resource_usage_format(_RESOURCES_FORMAT_PLOTLY)

    if isinstance(net_cpu_empty, BaseDiscreateNN):
        fig = produce_model_precision_fig_stage4(net_cpu_empty, plot_data, resources_usage_format_plotly())
        fig.write_image(img_path, width=1500, height=3800, engine="kaleido")
    elif isinstance(net_cpu_empty, BaseContinousNN):
        height_fig = 600
        height_img = height_fig
        width_img = height_img * 2
        fig = produce_model_precision_fig_continous(net_cpu_empty, plot_data, resources_usage_format_plotly())
        fig.write_image(img_path, width=width_img, height=height_img, engine="kaleido")
    else:
        raise RuntimeError(f"WRONG NET TYPE PROVIDED TO run_nn_training_result_console: {type(net_cpu_empty)}")

    display(Image(filename=img_path))


def display_df_full(df):
    with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.max_colwidth', None,
            'display.width', None
    ):
        display(df)


if __name__ == "__main__":
    show__plot_data(model_name_suffix='CC2_x14_71__IMP1')