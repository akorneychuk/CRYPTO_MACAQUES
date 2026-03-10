import colorsys
import copy as cp
import math
import uuid
from functools import reduce
from statistics import mean, stdev

import matplotlib.colors as mc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from SRC.CORE._CONSTANTS import COLORS, LOG_BASE, CLASSES, POWER_DEGREE, UNBALANCED_CENTER_RATIO, \
    NON_LINEARITY_TOP, EPOCHS, TARGET_FEATURE, MEAN_GRAD, ACTION_BUY, ACTION_SELL, STATE_OUT, DISCRETIZATION, ACTION_NO, _REGIME, \
    NETWORK_KEY, FEE_KEY, FORCE_DISCRETIZATION, FORCE_INTERVAL, _PLOT_ENABLED, project_root_dir
from SRC.CORE._FUNCTIONS import AUC_ROC_DOWN_SAMPLING_RATIO_PRODUCER, get_feature_abs_max
from SRC.CORE.debug_utils import *
from SRC.CORE.utils import calc_linear_regression_coefs, build_gradient_presentation_coordinates, produce_timedelta_ticks, datetime_h_m__d_m, datetime_Y_m_d__h_m_s, datetime_Y_m_d, \
    datetime_h_m_s, datetime_h_m, _float_6, _float_n, make_dir
from SRC.CORE.utils import calc_symmetric_log_space, calc_symmetric_pow_space, ___calc_space_bins

try:
    from backports.zoneinfo import ZoneInfo
except:
    from zoneinfo import ZoneInfo

np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:.5f}'.format
pd.options.plotting.backend = "plotly"

color_vals = COLORS.values()
_colors = list([*color_vals, *color_vals, *color_vals, *color_vals, *color_vals, *color_vals])
__colors = cp.copy(_colors)

btn_stop_style = {'width': '100%', 'color': 'white', 'backgroundColor': 'red', 'border': 'none', 'padding': '5px 5px', 'cursor': 'pointer'}
btn_finished_style = {'width': '100%', 'color': 'white', 'backgroundColor': 'green', 'border': 'none', 'padding': '5px 5px', 'cursor': 'pointer'}


def draw_graph(layout_fig_secs_candles, layout_fig_balance, relayout_data_2, data_state, dataframes):
    data_state = data_state[0]

    regime_network_title, pair_state_title, = get_dashboard_header_info(data_state)

    is_stopped_or_finished = data_state['is_stopped_or_finished']

    trade_symbol = data_state['trade_symbol']
    discretization = data_state['discretization']

    ticks_df = dataframes['ticks_df']
    peaks_df = dataframes['peaks_df']
    candles_secs_df = dataframes['candles_secs_df']

    candles_trade_df = dataframes['candles_trade_df']

    result_df_dict = data_state['result_df_dict']
    target_candles_df = data_state['target_candles_df']

    from _DASHBOARD.TradingBot import TradingBot
    if TradingBot.SECS_DATA is not None and not TradingBot.SECS_DATA['df'].empty:
        secs_df = TradingBot.SECS_DATA['df']

        xaxes_tickvals = list(filter(lambda ts: ts.second % 60 == 0, secs_df['close_time'].tolist()))
        xaxes_ticktext = [tickval.strftime('%H:%M') if tickval.minute % FORCE_INTERVAL() != 0 else f'<b>{tickval.strftime("%H:%M")}</b>' for tickval in xaxes_tickvals]

        secs_ema_low_col_name = TradingBot.SECS_DATA['ema_low_col_name']
        secs_ema_medium_col_name = TradingBot.SECS_DATA['ema_medium_col_name']
        secs_ema_high_col_name = TradingBot.SECS_DATA['ema_high_col_name']

        sec_candles_figure = go.Figure(data=[
            go.Candlestick(
                x=secs_df['close_time'],
                open=secs_df['open'], high=secs_df['high'], low=secs_df['low'], close=secs_df['close'],
                name='Secs candles'
            )
        ])

        sec_candles_figure.add_trace(go.Scatter(
            x=secs_df['close_time'].to_list(),
            y=secs_df[f'{secs_ema_low_col_name}'].to_list(),
            mode='lines', line=dict(color='orange', width=0.8),
            name=f'{secs_ema_low_col_name}'
        ))

        sec_candles_figure.add_trace(go.Scatter(
            x=secs_df['close_time'].to_list(),
            y=secs_df[f'{secs_ema_medium_col_name}'].to_list(),
            mode='lines', line=dict(color='cyan', width=1),
            name=f'{secs_ema_medium_col_name}'
        ))

        sec_candles_figure.add_trace(go.Scatter(
            x=secs_df['close_time'].to_list(),
            y=secs_df[f'{secs_ema_high_col_name}'].to_list(),
            mode='lines', line=dict(color='magenta', width=1.2),
            name=f'{secs_ema_high_col_name}'
        ))

        sec_candles_figure.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))
        sec_candles_figure.update_layout(yaxis_title='Price', xaxis_rangeslider_visible=False)
        sec_candles_figure.update_layout(height=350, paper_bgcolor="LightSteelBlue")
        sec_candles_figure.update_layout(margin=dict(l=30, r=20, t=40, b=20))
        sec_candles_figure.update_layout(xaxis=dict(tickmode='array', tickvals=xaxes_tickvals, ticktext=xaxes_ticktext))
    else:
        sec_candles_figure = px.line()
        sec_candles_figure.update_layout(height=13, paper_bgcolor="LightSteelBlue")

    if layout_fig_secs_candles is not None:
        sec_candles_figure.plotly_relayout(layout_fig_secs_candles)

    fig_balance_stable = produce__trade_sim__balance_price__combined_result_figure(trade_symbol, target_candles_df, result_df_dict, layout_fig_balance=layout_fig_balance)

    # if ticks_df.empty or peaks_df.empty or candles_secs_df.empty:
    #     main_fig = px.line()
    #     main_fig.update_layout(height=13, paper_bgcolor="LightSteelBlue")
    # else:
    #     # main_fig = produce_main_fig(ticks_df, peaks_df, candles_secs_df, relayout_data)
    #     main_fig = px.line()
    #     main_fig.update_layout(height=13, paper_bgcolor="LightSteelBlue")

    if candles_trade_df.empty:
        fig_candle_minutes = px.line()
        fig_candle_minutes.update_layout(height=13, paper_bgcolor="LightSteelBlue")
    else:
        # fig_candle_minutes = produce_candles_mins_fig(candles_trade_df, ROLLING_GRAD_WINDOW_FREQs, relayout_data_2)
        fig_candle_minutes = px.line()
        fig_candle_minutes.update_layout(height=13, paper_bgcolor="LightSteelBlue")

    btn_title = "FINISHED" if is_stopped_or_finished else "STOP"
    btn_style = btn_finished_style if is_stopped_or_finished else btn_stop_style

    price_style = {"font-size": "120%"}
    ticks_alive_style = {'color': 'cyan'} if 'ticks_alive' in data_state and data_state['ticks_alive'] else {'color': '#FFCCCB'}
    try:
        candles_target_alive_style = {'color': 'cyan'} if f'candles_{discretization.lower()}_alive' in data_state and data_state[f'candles_{discretization.lower()}_alive'] else {'color': '#FFCCCB'}
        candles_secs_alive_style = {'color': 'cyan'} if 'candles_1s_alive' in data_state and data_state['candles_1s_alive'] else {'color': '#FFCCCB'}
    except:
        candles_target_alive_style = {'color': '#FFCCCB'}
        candles_secs_alive_style = {'color': '#FFCCCB'}

    if len(candles_secs_df) > 1 and len(ticks_df) > 1:
        # price = str(candles_secs_df.iloc[-1]['close'])[0:8]
        time = candles_secs_df.iloc[-1]['close_time']
        window_prices = ticks_df['price'].to_list()#list(map(lambda tick: tick["price"], ticks_df))
        avg = mean(window_prices)
        std = stdev(window_prices)
        std_mean_ratio = std / avg * 100
        # price = "{0: <8}".format(price)[0:8]
        avg = "{0: <8}".format(avg)[0:8]
        std = "{0: <4}".format(std)[0:4]
        std_mean = "{0: <4}".format("{:f}".format(std_mean_ratio))[0:4]

        tick_curr_price = float(ticks_df.iloc[-1]['price'])
        tick_prev_price = float(ticks_df.iloc[-2]['price'])
        if tick_curr_price > tick_prev_price:
            price_style['color'] = 'green'
            indicator = '<'
        elif tick_curr_price < tick_prev_price:
            price_style['color'] = 'red'
            indicator = '>'
        else:
            price_style['color'] = 'black'
            indicator = '='

        price = f"{_float_6(tick_prev_price)} {indicator} {_float_6(tick_curr_price)}"
    else:
        return sec_candles_figure, fig_balance_stable, fig_candle_minutes, regime_network_title, pair_state_title, '???', '???', '???', '???', '???', price_style, ticks_alive_style, candles_secs_alive_style, candles_target_alive_style, btn_title, btn_style

    return sec_candles_figure, fig_balance_stable, fig_candle_minutes, regime_network_title, pair_state_title, time, price, avg, std, std_mean, price_style, ticks_alive_style, candles_secs_alive_style, candles_target_alive_style, btn_title, btn_style


def produce_candles_mins_fig(segment, grad_window_s, relayout_data=None, title=None):
    from SRC.CORE.utils import flatten

    try:
        unique_day_dates = segment['utc_timestamp'].dt.date.unique()
        unique_day_dates_present = list(map(lambda dt: datetime_Y_m_d(dt), unique_day_dates))
    except:
        unique_day_dates_present = ""

    height = 1400
    bottom_rows_slider_thickness = 0.03
    subplots_vertical_spacing = 0.05

    discretization = DISCRETIZATION()
    fake_candles_row_heigh_ratio = 0.001
    candles_row_heigh_ratio = 0.22
    mean_row_heigh_ratio = 0.22
    top_rows_heighs = [fake_candles_row_heigh_ratio, candles_row_heigh_ratio, mean_row_heigh_ratio]
    bottom_row_heigths = [(1 - sum(top_rows_heighs)) / (len(grad_window_s) * 2) for i in range(len(grad_window_s) * 2)]
    row_heights=[*top_rows_heighs, *bottom_row_heigths]
    plot_grad_titles = flatten([[f'Mean grad {grad_window} value {discretization}: {unique_day_dates_present}', f'Mean grad {grad_window} present {discretization}: {unique_day_dates_present}'] for grad_window in grad_window_s])
    subplot_titles = [*[f'Candles {discretization}: {unique_day_dates_present}', '', f'Mean relative diff {discretization}: {unique_day_dates_present}'], *plot_grad_titles]

    fig = make_subplots(
        subplot_titles=subplot_titles,
        rows=len(row_heights),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=subplots_vertical_spacing,
        row_heights=row_heights
    )

    subplot_layout_setup = dict()

    if len(segment) == 0:
        fig.update_layout(paper_bgcolor="LightSteelBlue")

        return fig

    custom_tickvals = segment['utc_timestamp'].to_list()
    custom_ticktext = list(map(lambda ts: datetime_h_m(ts), custom_tickvals))

    segment['color'] = segment.apply(lambda row: 'green' if row['open'] <= row['close'] else 'red', axis=1)

    fig.add_trace(go.Candlestick(
        x=custom_tickvals,
        open=segment['open'],
        high=segment['high'],
        low=segment['low'],
        close=segment['close'],
        showlegend=False
    ), row=1, col=1)

    xaxis_range_slider = dict(
        rangeslider=dict(visible=True, thickness=bottom_rows_slider_thickness),
        type="date",
        showticklabels=False,
    )

    xaxis_no_range_slider = dict(
        rangeslider=dict(visible=False, thickness=0),
        type="date",
        showticklabels=True,
    )

    yaxis_no_tick_labels=dict(showticklabels=False)
    yaxis_tick_labels=dict(showticklabels=True)

    subplot_layout_setup['1'] = dict(
        xaxis=xaxis_range_slider,
        yaxis=yaxis_no_tick_labels
    )

    fig.add_trace(go.Candlestick(
        x=custom_tickvals,
        open=segment['open'].to_list(),
        high=segment['high'].to_list(),
        low=segment['low'].to_list(),
        close=segment['close'].to_list(),
        showlegend=True,
        name=f'Candles {discretization}'
    ), row=2, col=1)

    subplot_layout_setup['2'] = dict(
        xaxis=xaxis_no_range_slider,
        yaxis=yaxis_tick_labels
    )

    fig.add_trace(go.Scatter(
        x=custom_tickvals,
        y=segment['mean_rel_diff'].to_list(),
        showlegend=True,
        name=f'Mean {discretization}',
        line=dict(color=_colors[0], dash='solid')
    ), row=3, col=1)

    subplot_layout_setup['3'] = dict(
        xaxis=xaxis_no_range_slider,
        yaxis=yaxis_tick_labels
    )

    for indx, grad_window in enumerate(grad_window_s):
        feature = f'{MEAN_GRAD}_{grad_window}'

        series = segment[['utc_timestamp', f'{MEAN_GRAD}_ys_{grad_window}', f'{MEAN_GRAD}_ye_{grad_window}', feature]]

        row = len(top_rows_heighs) + 1 + indx * 2

        fig.add_trace(go.Scatter(
            x=series['utc_timestamp'],
            y=series[feature],
            showlegend=True,
            name=feature,
            line=dict(color=_colors[indx + 1], dash='solid')
        ), row=row, col=1)

        subplot_layout_setup[f'{row}'] = dict(
            xaxis=xaxis_no_range_slider,
            yaxis=yaxis_tick_labels
        )

        np_series = series.to_numpy()

        for i in range(len(np_series)):
            if i < grad_window:
                continue

            start = np_series[i - grad_window]
            end = np_series[i]
            x_start = start[0]
            x_end = end[0]
            y_start = start[1]
            y_end = end[2]
            grad = end[3]
            grad_color = f'rgb{get_grad_color(grad)}'

            if (math.isnan(grad)):
                continue

            fig.add_trace(go.Scatter(
                x=[x_start, x_end], y=[y_start, y_end],
                showlegend=False,
                line=dict(color=grad_color, dash='solid'),
                name=grad
            ), row=row + 1, col=1)

            subplot_layout_setup[f'{row + 1}'] = dict(
                xaxis=xaxis_no_range_slider,
                yaxis=yaxis_no_tick_labels
            )

    if title is not None:
        title_text = f"{title} | {subplot_titles}"
    else:
        title_text = f"{subplot_titles}"

    fig.update_layout(
        height=height,
        xaxis=dict(autorange=True),
        yaxis=dict(autorange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="LightSteelBlue",
        # title_text=title_text
    )

    fig.update_xaxes(
        tickvals=custom_tickvals,
        ticktext=custom_ticktext,
        tickmode='array',
        tickangle=-45
    )

    for key,val in subplot_layout_setup.items():
        fig.layout[f'xaxis{key}'] = fig.layout[f'xaxis{key}'].update(val['xaxis'])
        fig.layout[f'yaxis{key}'] = fig.layout[f'yaxis{key}'].update(val['yaxis'])

    if relayout_data is not None:
        fig.plotly_relayout(relayout_data)

    return fig


def produce_main_fig(ticks_df, peaks_df, candles_secs_df, relayout_data):
    height = 1050

    if not candles_secs_df.empty:
        ticks_each_n = 15
        tickvals = candles_secs_df['close_time'].to_list()
        ticktext = list(map(lambda ts: datetime_h_m_s(ts), tickvals))
    elif not ticks_df.empty:
        ticks_each_n = 15
        tickvals = ticks_df['close_time'].to_list()
        ticktext = list(map(lambda ts: datetime_h_m_s(ts), tickvals))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.1, 0.45, 0.45]
    )

    if not ticks_df.empty:
        price = go.Scatter(
            x=ticks_df['close_time'].to_list(),
            y=ticks_df['price'],
            marker_color='black',
            mode='lines',
            name='Price'
        )

        fig.add_trace(price, row=2, col=1)

    if not peaks_df.empty:
        # Peak step
        peak = go.Scattergl(
            x=peaks_df['close_time'].to_list(),
            y=peaks_df['signal'],
            name='Peak',
            mode='lines',
            marker_color='red',
            line=dict(shape='hv')
        )

        fig.add_trace(peak, row=1, col=1)

        avg = go.Scatter(
            x=peaks_df['close_time'].to_list(),
            y=peaks_df['avg'],
            marker_color='cyan',
            mode='lines',
            name='AVG'
        )

        fig.add_trace(avg, row=2, col=1)

        std_upper = go.Scatter(
            x=peaks_df['close_time'].to_list(),
            y=peaks_df["upper_std"],
            marker_color='green',
            mode='lines',
            name='Window upper',
            legendgroup='AvgStdWindow'
        )

        fig.add_trace(std_upper, row=2, col=1)

        std_lower = go.Scatter(
            x=peaks_df['close_time'].to_list(),
            y=peaks_df["lower_std"],
            marker_color='green',
            mode='lines',
            name='Window lower'
        )

        fig.add_trace(std_lower, row=2, col=1)

    if not candles_secs_df.empty:
        trace = go.Candlestick(
            x=candles_secs_df['close_time'].to_list(),
            open=candles_secs_df['open'],
            high=candles_secs_df['high'],
            low=candles_secs_df['low'],
            close=candles_secs_df['close'],
            name='Second candle'
        )

        fig.add_trace(
            trace,
            row=3, col=1
        )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        paper_bgcolor="LightSteelBlue",
        legend_orientation="h",
        xaxis1_rangeslider_visible=False,
        xaxis1_rangeslider_thickness=0,
        xaxis2_rangeslider_visible=False,
        xaxis2_rangeslider_thickness=0,
        xaxis3_rangeslider_visible=True,
        xaxis3_rangeslider_thickness=0.04,
        yaxis1=dict(tickvals=[-1, 0, 1]),
        xaxis1=dict(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=10, label="10m", step="minute", stepmode="backward"),
                    dict(count=30, label="30m", step="minute", stepmode="backward"),
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=3, label="3h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            type="date",
            showticklabels=True
        ),
        xaxis2=dict(
            rangeslider_visible=False,
            rangeslider=dict(visible=False),
            type="date",
            showticklabels=True
        ),
        xaxis3=dict(
            rangeslider_visible=True,
            rangeslider=dict(visible=True),
            type="date",
            showticklabels=True
        )
    )

    if not candles_secs_df.empty or not ticks_df.empty:
        fig.update_xaxes(
            tickvals=tickvals[::ticks_each_n][1:],
            ticktext=ticktext[::ticks_each_n][1:],
            tickmode='array',
            tickangle=-45
        )

    if relayout_data is not None:
        fig.plotly_relayout(relayout_data)

    return fig


def lighten_color(color, amount=0):
    try:
        c = mc.cnames[color]
    except:
        c = color
    hls = colorsys.rgb_to_hls(*mc.to_rgb(c))
    rgb = colorsys.hls_to_rgb(hls[0], 1 - amount * (1 - hls[1]), hls[2])
    rgb_color = tuple(int(x * 255) for x in rgb)

    return rgb_color


def lighten_color_rgba(color, amount=0):
    color = lighten_color(color, amount)

    return f"rgb{color}"


def color_interpolator(grad):
    upper = 0.005
    normalized = grad if grad <= upper else upper
    grad_normalized = normalize(
        [normalized],
        {'actual': {'lower': 0, 'upper': upper}, 'desired': {'lower': 0, 'upper': 1}}
    )[0]

    # return grad_normalized ** (1/1.1)
    # return grad_normalized ** (1/1.3)
    # return grad_normalized ** (1/2)
    return grad_normalized ** (1/5)
    return grad_normalized
    # return (grad_normalized ** (1/1.5)) ** (1/1.5)
    # return (grad_normalized ** (1/2))


def get_grad_color(grad):
    grad_abs = abs(grad)
    color = 'green' if grad > 0 else 'red' if grad < 0 else 'gray'
    normalized_grad = color_interpolator(grad_abs)
    color_shaded = lighten_color(color, normalized_grad)

    return color_shaded


def display_plot(fig, show_as_img=False):
    if _PLOT_ENABLED in os.environ and not string_bool(os.environ[_PLOT_ENABLED]):
        return

    from plotly.offline import iplot

    if show_as_img:
        from IPython.core.display_functions import DisplayHandle
        from IPython.display import Image, Markdown

        folder_path = f"{project_root_dir()}/killme"
        make_dir(folder_path)
        img_path = f"{folder_path}/{str(uuid.uuid4())}.png"
        fig.write_image(img_path, width=1500, height=600, engine="kaleido")
        display_handle = DisplayHandle()
        display_handle.display(Image(filename=img_path))
        os.remove(img_path)
        return

    iplot(go.FigureWidget(fig))


def normalize(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]


def show_feature_distribution(df, feature, bins_count, ticks=None, offset_condition=lambda val: val != 0):
    df_range = df[df[feature].apply(offset_condition)]
    range_presentation = get_range_presentation(df_range)
    counts, bins = np.histogram(df_range[feature], bins=bins_count, range=(df_range[feature].min(), df_range[feature].max()))
    bins = 0.5 * (bins[:-1] + bins[1:])
    fig = px.bar(x=bins, y=counts, labels={'x':feature, 'y':'occurrency'}, title=f'Feature: {feature}, Bins: {bins_count}, Total: {sum(counts)} in range: {range_presentation}', log_y=True)
    if ticks is None:
        fig.update_layout(paper_bgcolor="LightSteelBlue", xaxis=dict(tickformat='%.format.%3f'))
    else:
        fig.update_layout(paper_bgcolor="LightSteelBlue", xaxis=dict(tickformat='%.format.%3f', tickvals=ticks))

    # fig.update_xaxes(tickangle=75)
    fig.show()


def show_series_gradient(series, reducer_func):
    entry = 1
    display(Markdown("**INPUT:**"))
    print(series)
    result = reduce(reducer_func, series, [entry])
    display(Markdown("**OUTPUT:**"))

    print(result)
    plt.plot(range(len(result)), result)

    coef = calc_linear_regression_coefs(np.asarray(range(len(result))), result)
    approx_x = np.asarray(range(6))
    y_reg = approx_x * coef[1] + coef[0]
    plt.plot(approx_x, y_reg)


def show_mean_grad_distribution(segment, grad_window_s, title=""):
    first_row_height_ratio = 0.35
    row_heigths = [(1 - first_row_height_ratio) / (len(grad_window_s) * 2) for i in range(len(grad_window_s) * 2)]
    fig = make_subplots(rows=1+len(grad_window_s) * 2, cols=1, shared_xaxes=True, row_heights=[*[first_row_height_ratio], *row_heigths])
    fig.add_trace(go.Scatter(x=segment['utc_timestamp'], y=segment['mean'], showlegend=True, name='mean'), row=1, col=1)
    for indx, grad_window in enumerate(grad_window_s):
        row = 2 + indx * 2

        feature = f'{MEAN_GRAD}_{grad_window}'

        fig.add_trace(go.Scatter(x=segment['utc_timestamp'], y=segment[feature], showlegend=True, name=feature), row=row, col=1)
        series = segment[['utc_timestamp', f'{MEAN_GRAD}_ys_{grad_window}', f'{MEAN_GRAD}_ye_{grad_window}']].to_numpy()
        for i in range(len(series)):
            if i < grad_window:
                continue
            start = series[i - grad_window]
            end = series[i]
            fig.add_trace(go.Scatter(x=[start[0], end[0]], y=[start[1], end[2]], showlegend=False), row=row + 1, col=1)

    fig.update_layout(
        height=1200,
        width=1000,
        xaxis=dict(autorange=True),
        yaxis=dict(autorange=True),
        title_text=f"{title} + Mean price & Gradient sequence")
    fig.show()


def show_feature_distribution_by_condition_in_range(loc, df_range, feature, title, feature_space_producer=None):
    printmd_low(f"**####### {title} #######**")

    range_presentation = get_range_presentation(df_range)

    title_loc = f"{title} {feature} item"
    title_range = f"{title} {feature} in range: {range_presentation}"

    printmd_medium(f"**{title_loc}**")
    display_medium(loc)

    printmd_medium(f"**{title_range}**")
    display_medium(df_range)

    POWER_SPACE_MEAN_REF_DIFF = calc_symmetric_pow_space(CLASSES(), POWER_DEGREE(), 0.1, 1, UNBALANCED_CENTER_RATIO)
    space = POWER_SPACE_MEAN_REF_DIFF if feature_space_producer is None else feature_space_producer(feature)
    plot_feature_distribution(df_range, feature, space)


def show_feature_in_range(df_range, feature_1, feature_2, title=''):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(
        go.Candlestick(
            x=df_range['utc_timestamp'],
            open=df_range['open'],
            high=df_range['high'],
            low=df_range['low'],
            close=df_range['close'],
            showlegend=True,
            name='Second candle'
        ), row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df_range['utc_timestamp'], y=df_range[feature_1], showlegend=True, name=feature_1), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_range['utc_timestamp'], y=df_range[feature_2], showlegend=True, name=feature_2), row=3, col=1)
    fig.update_layout(height=800, width=1000, title_text=f"{title} + Candles price sequence", xaxis=dict(rangeslider=dict(visible=False), type="date"))
    fig.show()


def show_range_distribution(df, PAIR, ROLLING_GRAD_WINDOW_FREQs, symmetric_space, feature=TARGET_FEATURE()):
    range_presentation = get_range_presentation(df)
    printmd_low(f"**####### {range_presentation}: #######**")

    df = build_gradient_presentation_coordinates(PAIR, df, ROLLING_GRAD_WINDOW_FREQs)
    plot_feature_distribution(df, feature, symmetric_space)
    produce_candles_mins_fig(df, ROLLING_GRAD_WINDOW_FREQs).show()


def get_range_presentation(df):
    from_timestamp_str = datetime_Y_m_d__h_m_s(df.iloc[0]['utc_timestamp'])
    to_timestamp_str= datetime_Y_m_d__h_m_s(df.iloc[-1]['utc_timestamp'])

    return f"{from_timestamp_str} ~ {to_timestamp_str}"


def plot_feature_distribution(df, feature, symmetric_space):
    range_presentation = get_range_presentation(df)
    printmd_low(f"**Feature: {feature} distribution in range: {range_presentation}**")
    title = f"Feature: {feature} in range: {range_presentation}"

    __plot_feature_distribution(df[feature].dropna().to_list(), symmetric_space, title)


def produce_bins_presentation(bins):
    from SRC.CORE.utils import pairwise

    bin_presentation_s = []
    for start_bin, end_bin in pairwise(list(bins)):
        bin_presentation = f"{'{:.5f}'.format(start_bin)} | {'{:.5f}'.format(end_bin)} [{len(bin_presentation_s)}]"
        bin_presentation_s.append(bin_presentation)

    return bin_presentation_s


def __plot_feature_distribution(series, symmetric_space, title_distribution='Distribution', title_nonlinearity='Nonlinearity'):
    calc_bins = lambda data: ___calc_space_bins(data, symmetric_space)
    bins, counts, weights, weights_count_product, space = calc_bins(series)
    bin_presentation_s = produce_bins_presentation(bins)

    title_bins = f"{title_distribution} BINS"
    line_df = pd.DataFrame({"counts": counts, "bins": list(range(len(bin_presentation_s))), "bin_presentation": bin_presentation_s})
    fig_bins = px.bar(line_df, x="bins", y="counts", custom_data="bin_presentation", title=title_bins, log_y=True)
    fig_bins.update_xaxes(tickangle=-75)
    fig_bins.update_traces(hovertemplate='Bin: %{customdata}<br>Occurency: %{y}')
    fig_bins.update_layout(xaxis=dict(tickangle=-75, tickvals=line_df['bins'].to_list(), ticktext=line_df['bin_presentation'].to_list()))
    fig_bins.update_layout(paper_bgcolor="LightSteelBlue")

    title_nonlinearity = f"{title_nonlinearity} space distribution"
    fig_log_space = get_symmetric_space_distribution_fig(space, title_nonlinearity)

    fig_combined = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35])
    fig_combined.add_trace(fig_bins.data[0], row=1, col=1)
    fig_combined.add_trace(fig_log_space.data[0], row=1, col=2)
    fig_combined.update_yaxes(type="log", row=1, col=1)
    fig_combined.update_layout(xaxis=dict(tickvals=line_df['bins'].to_list(), ticktext=line_df['bin_presentation'].to_list()))
    fig_combined.update_layout(title_text=f"{title_bins} | {title_nonlinearity}")
    fig_combined.update_layout(paper_bgcolor="LightSteelBlue")

    display_plot(fig_combined)

    if is_high_log_level():
        title_classes = f"{title_distribution} WEIGHTS x COUNT"
        line_df = pd.DataFrame({"classes": list(range(len(weights_count_product))), "counts": weights_count_product})
        fig = px.bar(line_df, x="classes", y="counts", title=title_classes, log_y=True)
        fig.update_xaxes(tickangle=-75)
        fig.update_traces(hovertemplate='Class: %{x}<br>Count: %{y}')
        fig.update_layout(xaxis=dict(tickvals=line_df['classes'].to_list(), ticktext=line_df['classes'].to_list()))
        fig.update_layout(paper_bgcolor="LightSteelBlue")

        display_plot(fig)


def plot_series_correlation(df, feature1, feature2):
    range_presentation = get_range_presentation(df)
    title = f"Features correlation"
    description = f"{feature1} vs {feature2} | range: {range_presentation}"
    print_action_title_description__low(title, description)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df[feature1], name=f"{feature1} values"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df[feature2] , name=f"{feature2} values"),secondary_y=True)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_traces(hovertemplate='Time: %{x}<br>%{y:.f}')
    fig.update_layout(paper_bgcolor="LightSteelBlue")
    fig.update_layout(title_text=f"{title} || {description}")

    display_plot(fig)


def plot_series_dependency(df, feature1, feature2, xaxis_ticks=None, yaxis_ticks=None, is_permuted=False, size=700):
    if is_permuted:
        title = f'Dependency: {feature1} <> {feature2} count: {len(df)}'
    else:
        range_presentation = get_range_presentation(df)
        title = f'Dependency: {feature1} <> {feature2} range: {range_presentation}'

    printmd_low(f"**{title}**")

    feature_x_max = df[feature1].max()
    feature_x_min = df[feature1].min()
    printmd(f"`{feature1}_max: {feature_x_max}, {feature1}_min: {feature_x_min}`")

    feature_y_max = df[feature2].max()
    feature_y_min = df[feature2].min()
    printmd(f"`{feature2}_max: {feature_y_max}, {feature2}_min: {feature_y_min}`")

    fig = px.scatter(df, x=f"{feature1}", y=f"{feature2}", title=title)

    fig.update_xaxes(tickangle=-75)
    fig.update_traces(hovertemplate=f'{feature1}: {"%{x:.f}"}<br>{feature2}: {"%{y:.f}"}')
    fig.update_layout(paper_bgcolor="LightSteelBlue")

    if xaxis_ticks is None:
        xaxis_ticks = np.linspace(feature_x_min, -feature_x_min)

    if yaxis_ticks is None:
        yaxis_ticks = np.linspace(feature_y_min, -feature_y_min)

    for xaxis_tick in xaxis_ticks:
        fig.add_vline(x=xaxis_tick, line_width=1, line_dash="solid", line_color="fuchsia")

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=xaxis_ticks,
            ticktext=['{:.5f}'.format(x) for x in xaxis_ticks],
        ))
    fig.update_xaxes(tickangle=-75)

    for yaxis_tick in yaxis_ticks:
        fig.add_hline(y=yaxis_tick, line_width=1, line_dash="solid", line_color="cyan")

    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=yaxis_ticks,
            ticktext=['{:.5f}'.format(y) for y in yaxis_ticks],
        ))
    # fig.update_yaxes(tickangle=-75)

    if xaxis_ticks is not None and yaxis_ticks is not None:
        xmax = max(xaxis_ticks)
        xmin = min(xaxis_ticks)
        ymax = max(yaxis_ticks)
        ymin = min(yaxis_ticks)

        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=[xmin - 0.05 * abs(xmax - xmin), xmax + 0.05 * abs(xmax - xmin)], ),
                yaxis=dict(nticks=4, range=[ymin - 0.05 * abs(ymax - ymin), ymax + 0.05 * abs(ymax - ymin)], ),))

        fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=(ymax - ymin) / (xmax - xmin)))

    fig.update_layout(width=size, height=size)

    display_plot(fig)


def fill_auc_roc_curve(auc_roc, fig, colors, row, col, group):
    from SRC.CORE._CONSTANTS import SYMMETRIC_CLASSES

    fpr_s = auc_roc[0]
    tpr_s = auc_roc[1]
    auc_roc_s = auc_roc[2]

    symmetric_classes = SYMMETRIC_CLASSES()
    for class_indx in range(len(auc_roc_s)):
        symmetric_class = symmetric_classes[class_indx]
        symmetric_class_present = f" {symmetric_class}" if symmetric_class >= 0 else f"{symmetric_class}"

        take_each_n_item = AUC_ROC_DOWN_SAMPLING_RATIO_PRODUCER(len(fpr_s[class_indx]))

        fpr = fpr_s[class_indx]
        tpr = tpr_s[class_indx]
        fpr = [*fpr[::take_each_n_item], *[fpr[-1]]]
        tpr = [*tpr[::take_each_n_item], *[tpr[-1]]]

        auc_roc = auc_roc_s[class_indx]

        if class_indx == int(len(auc_roc_s) / 2):
            printmd_high(f"**CLASS = {class_indx} | FPR size = {len(fpr)} | Take each {take_each_n_item}'s item**")
            pass

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                name=f'{symmetric_class_present} | {class_indx} {"  " if class_indx < 10 else ""}= %0.2f' % auc_roc,
                mode='lines',
                line=dict(color=colors[class_indx], width=1, dash='dash'),
                legendgroup=f'{group}'
            ),
            row=row, col=col
        )


def produce_model_precision_fig(plot_data, relayout_data=None):
    height = 1000
    colors = list(COLORS.values())

    fig = make_subplots(
        rows=2, cols=6,
        specs=[
            [{"colspan": 2}, None, {"colspan": 2}, None, {"colspan": 2}, None],
            [{"colspan": 3}, None, None, {"colspan": 3}, None, None],
        ],
        horizontal_spacing=0.075,
        vertical_spacing=0.075,
        subplot_titles=(
            "AUC ROC INIT",
            "AUC ROC TRAIN",
            "AUC ROC TEST",
            "Epoch duration",
            "Cross entropy loss"),
    )

    if plot_data is None:
        fig.update_layout(
            height = height,
            showlegend=False,
            paper_bgcolor="LightSteelBlue",
            title_text=f"Evaluation..")

        return fig

    auc_roc_test = plot_data['auc_roc_test'] if 'auc_roc_test' in plot_data else None
    auc_roc_train = plot_data['auc_roc_train'] if 'auc_roc_train' in plot_data else None
    auc_roc_final = plot_data['auc_roc_final'] if 'auc_roc_final' in plot_data else None

    loss_s = plot_data['loss_s']
    epoch_exec_time_s = plot_data['epoch_exec_time_s']
    current_exec_time_seconds = plot_data['current_exec_time_seconds']
    started_at = plot_data['started_at']
    last_updated_at = plot_data['last_updated_at']
    last_rendered_at = datetime.now(tz=ZoneInfo("Europe/Istanbul"))
    no_update_gap = last_rendered_at - last_updated_at

    epoch_s = np.array(list(range(len(loss_s)))) + 1
    epoch = epoch_s[-1]

    if auc_roc_test is not None:
        fill_auc_roc_curve(auc_roc_test, fig, colors, 1, 1, 1)

    if auc_roc_train is not None:
        fill_auc_roc_curve(auc_roc_train, fig, colors, 1, 3, 2)

    if auc_roc_final is not None:
        fill_auc_roc_curve(auc_roc_final, fig, colors, 1, 5, 3)

    epoch_exec_time_sec_s, tickvals, ticktext = produce_timedelta_ticks(epoch_exec_time_s, 10)

    loss_x = list(map(lambda e: e - 1, epoch_s))
    loss_y = loss_s
    initial_loss = _float_n(loss_y[0], 4) if len(loss_y) > 0 else -1
    if len(loss_x) > 1 and len(loss_y) > 1:
        loss_x = loss_x[1:]
        loss_y = loss_y[1:]

    fig.add_trace(
        go.Scatter(
            x=list(map(lambda e: e - 1, epoch_s)),
            y=epoch_exec_time_sec_s,
            name="Epoch takes",
            mode='lines+markers',
            legendgroup='4'
        ),
        row=2, col=1
    )

    fig.update_yaxes(
        tickvals=tickvals, ticktext=ticktext,
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=loss_x,
            y=loss_y,
            legendgroup='5',
            mode='lines+markers',
            name="CEL"
        ),
        row=2, col=4
    )

    started_at = datetime_h_m_s(started_at)
    # duration = str(current_exec_time_seconds).split(".")[0]
    duration = datetime_h_m_s(datetime.now() - started_at)
    updated_at = datetime_h_m_s(last_updated_at)
    rendered_at = datetime_h_m_s(last_rendered_at)
    no_update_gap = str(no_update_gap).split(".")[0]

    if current_exec_time_seconds < timedelta(seconds=1):
        title_text = f"Evaluated > Epoch: 1.. | Initial loss: {initial_loss} | Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"
    else:
        title_text = f"Epoch: {epoch - 1} > {epoch}.. | Initial loss: {initial_loss} | Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"

    if epoch > EPOCHS():
        title_text = f"Epoch {epoch - 1} > Validation.. | Initial loss: {initial_loss} | Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"

    if epoch > EPOCHS() + 1:
        title_text = f"Validated > Finished | Initial loss: {initial_loss} | Started: {started_at} | Rendered: {rendered_at} | Duration: {duration} | Updated: {updated_at} | No update: {no_update_gap}"

    if auc_roc_final is not None and auc_roc_test is None and auc_roc_train is None:
        title_text = f"Evaluated trained model"

    fig.update_layout(
        height=height,
        title_text=title_text,
        showlegend=True,
        legend_tracegroupgap=30,
        paper_bgcolor="LightSteelBlue"
    )

    if relayout_data is not None:
        fig.plotly_relayout(relayout_data)

    return fig


def plot_evaluated_model(plot_data_queue, auc_roc_test, auc_roc_train, auc_roc_final, loss_s, epoch_exec_time_s, current_exec_time_seconds, started_at, force_plot=False):
    last_updated_at = datetime.now(tz=ZoneInfo("Europe/Istanbul"))

    plot_data = {
        'auc_roc_test': auc_roc_test,
        'auc_roc_train': auc_roc_train,
        'auc_roc_final': auc_roc_final,

        'loss_s': loss_s,
        'epoch_exec_time_s': epoch_exec_time_s,
        'current_exec_time_seconds': current_exec_time_seconds,
        'started_at': started_at,
        'last_updated_at': last_updated_at,
    }
    plot_data_queue.append(plot_data)


def get_symmetric_space_distribution_fig(log_space, title):
    import plotly.express as px

    x_s = np.linspace(log_space[0], log_space[-1], len(log_space))
    y_s = log_space
    fig = px.line(x=x_s, y=y_s, title=title)
    fig.update_layout(width=500, height=500)
    fig.update_xaxes(title_text="X space")
    fig.update_yaxes(title_text="Custom LOG(X)")
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(paper_bgcolor="LightSteelBlue")

    return fig


def plot__pow_vs_log__space_distribution():
    from SRC.CORE._CONSTANTS import CLASSES, LOG_START, NON_LINEARITY_TOP_DEFAULT, POWER_DEGREE

    classes = 15

    power_degree = 4
    power_start = 0.0035

    log_base = 4
    log_start = 0.0035

    non_linearity_top = 0.3
    space_top = 0.3

    log_space_no_completion = calc_symmetric_log_space(classes, log_base=log_base, non_linearity_top=non_linearity_top, log_start=log_start)
    log_space_space_top_1 = calc_symmetric_log_space(classes, log_base=log_base, non_linearity_top=non_linearity_top, log_start=log_start, space_top=non_linearity_top)
    log_space_space_top_07 = calc_symmetric_log_space(classes, log_base=log_base, non_linearity_top=non_linearity_top, log_start=log_start, space_top=space_top)

    pow_space_no_completion = calc_symmetric_pow_space(classes, power_degree=power_degree, non_linearity_top=non_linearity_top, power_start=power_start)
    pow_space_space_top_1 = calc_symmetric_pow_space(classes, power_degree=power_degree, non_linearity_top=non_linearity_top, power_start=power_start, space_top=non_linearity_top)
    pow_space_space_top_07 = calc_symmetric_pow_space(classes, power_degree=power_degree, non_linearity_top=non_linearity_top, power_start=power_start, space_top=space_top)

    # plt.plot(np.linspace(log_space_no_completion[0], log_space_no_completion[-1], len(log_space_no_completion)), log_space_no_completion, label=f"LOG(X) non_linearity_top={non_linearity_top} space_top={1}")
    plt.plot(np.linspace(log_space_space_top_1[0], log_space_space_top_1[-1], len(log_space_space_top_1)), log_space_space_top_1, label=f"LOG(X) non_linearity_top={non_linearity_top} space_top={non_linearity_top}")
    plt.plot(np.linspace(log_space_space_top_07[0], log_space_space_top_07[-1], len(log_space_space_top_07)), log_space_space_top_07, label=f"LOG(X) non_linearity_top={non_linearity_top} space_top={space_top}")

    # plt.plot(np.linspace(pow_space_no_completion[0], pow_space_no_completion[-1], len(pow_space_no_completion)), pow_space_no_completion, label=f"POW(X) non_linearity_top={non_linearity_top} space_top={1}")
    plt.plot(np.linspace(pow_space_space_top_1[0], pow_space_space_top_1[-1], len(pow_space_space_top_1)), pow_space_space_top_1, label=f"POW(X) non_linearity_top={non_linearity_top} space_top={non_linearity_top}")
    plt.plot(np.linspace(pow_space_space_top_07[0], pow_space_space_top_07[-1], len(pow_space_space_top_07)), pow_space_space_top_07, label=f"POW(X) non_linearity_top={non_linearity_top} space_top={space_top}")
    plt.xlabel('X')
    plt.ylabel('LOG(X) / POW(X)')
    plt.legend()
    plt.figure(figsize=(16, 9))
    plt.show()


def plot_feature_space(feature, extremums):
    classes = CLASSES()
    power_degree = POWER_DEGREE()
    non_linearity_top = get_feature_abs_max(feature, extremums) * NON_LINEARITY_TOP()
    sapce_top = get_feature_abs_max(feature, extremums)

    feature_space = calc_symmetric_pow_space(classes=classes, power_degree=power_degree, non_linearity_top=non_linearity_top, space_top=sapce_top, unbalanced_center_ratio=UNBALANCED_CENTER_RATIO)

    printmd(f"**classes = {classes} | power_degree = {power_degree} | non_linearity_top = {non_linearity_top} | sapce_top = {sapce_top}**")
    plt.plot(np.linspace(feature_space[0], feature_space[-1], len(feature_space)), feature_space)
    plt.xlabel('X')
    plt.ylabel('POW(X)')
    plt.legend()
    plt.figure(figsize=(16, 9))
    plt.show()


def produce_trading_actions_distribution_figure(action_s):
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(action_s, columns=['actions']).groupby('actions').size().reset_index(name='occurencies')
    fig = px.bar(df, x="actions", y="occurencies")
    fig.update_layout(height=250, paper_bgcolor="LightSteelBlue")
    fig.update_layout(margin=dict(l=30, r=20, t=40, b=20))

    return fig


def produce_trading_states_distribution_figure(state_s):
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(state_s, columns=['states']).groupby('states').size().reset_index(name='occurencies')
    fig = px.bar(df, x="states", y="occurencies")
    fig.update_layout(height=250, paper_bgcolor="LightSteelBlue")
    fig.update_layout(margin=dict(l=30, r=20, t=40, b=20))

    return fig


def produce_class_sequence_series_figure(class_s, title):
    import pandas as pd
    import plotly.express as px

    ticks_limit = 10_000
    ticks_each_n = int(len(class_s) / ticks_limit) if len(class_s) / ticks_limit > 1 else 1
    ticks_deal = list(range(len(class_s)))[::ticks_each_n]
    ticks_class = class_s[::ticks_each_n]
    tickYvals = list(range(CLASSES()))
    tickYtext = list(map(lambda c: f"{c}", tickYvals))
    df = pd.DataFrame({'deal': ticks_deal, 'class': ticks_class})
    fig = px.line(df, x='deal', y='class', title='Classes')
    fig.update_layout(height=270, paper_bgcolor="LightSteelBlue")
    fig.update_layout(margin=dict(l=30, r=20, t=40, b=20))
    fig.update_yaxes(tickangle=0, tickvals=tickYvals, ticktext=tickYtext)

    return fig


def produce__act_vs_pred__class_sequence_series_figure(df, title):
    import plotly.express as px

    df = df.iloc[2:]
    tickYvals = list(range(CLASSES()))
    tickYtext = list(map(lambda c: f"{c}", tickYvals))
    label_s = {'act_prev_class_s': "ACTUAL's", 'pred_class': "PREDICTED's"}

    fig = px.line(title=title, labels=label_s, markers=True)
    fig.add_trace(go.Scatter(x=df['close_dt_s'], y=df['pred_class_s'], name=f"PREDICTED"))
    fig.add_trace(go.Scatter(x=df['prev_close_dt_s'], y=df['act_prev_class_s'], name=f"ACTUAL PREV"))

    fig.update_layout(height=270, paper_bgcolor="LightSteelBlue")
    fig.update_layout(margin=dict(l=30, r=20, t=40, b=20))
    fig.update_yaxes(tickangle=0, tickvals=tickYvals, ticktext=tickYtext)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    return fig


def plot_class_distribution_diagram_figure(class__s, title):
    import pandas as pd
    import plotly.express as px

    class_distribution_df = pd.DataFrame(class__s, columns=['classes']).groupby('classes').size().reset_index(name='occurencies')
    display_high(class_distribution_df)

    fig = px.bar(class_distribution_df, x="classes", y="occurencies", title=title, log_y=True)
    fig.update_layout(xaxis=dict(tickvals=class_distribution_df['classes'].to_list(), ticktext=class_distribution_df['classes'].to_list()))
    fig.update_layout(paper_bgcolor="LightSteelBlue")
    fig.update_layout(height=250, paper_bgcolor="LightSteelBlue")
    fig.update_layout(margin=dict(l=30, r=20, t=40, b=20))
    display_plot(fig)


def plot__act_vs_pred__class_distribution_diagram_figure(df, title):
    fig = px.bar(title=title, log_y=True)
    for col_class_s in ['act_prev_class_s', 'pred_class_s']:
        class_distribution_df = pd.DataFrame(df[col_class_s].to_list(), columns=['classes']).groupby('classes').size().reset_index(name='occurencies')
        fig.add_trace(
            go.Bar(
                x=class_distribution_df['classes'],
                y=class_distribution_df['occurencies'],
                name=col_class_s,
                offsetgroup=col_class_s,
                legendgroup=col_class_s,
            )
        )
    fig.update_layout(barmode='group', xaxis_tickangle=-45)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_layout(xaxis=dict(tickvals=class_distribution_df['classes'].to_list(), ticktext=class_distribution_df['classes'].to_list()))
    fig.update_layout(paper_bgcolor="LightSteelBlue")
    fig.update_layout(height=250, paper_bgcolor="LightSteelBlue")
    fig.update_layout(margin=dict(l=30, r=20, t=40, b=20))
    display_plot(fig)


def produce_balance_figure(date_s, balance_s_s, title_s, log_y=False):
    if len(balance_s_s) > 1:
        label_s = {'balance_s_1': title_s[0], 'balance_s_2': title_s[1]}
        df = pd.DataFrame({'deal': date_s, 'balance_s_1': balance_s_s[0], 'balance_s_2': balance_s_s[1]})
        # fig = go.Figure()
        # fig.add_traces(go.Scatter(x=df['id'], y=df['a'], mode='lines', line=dict(color="#0")))
        # fig.add_traces(go.Scatter(x=df['id'], y=df['c'], mode='lines', line=dict(color=colors[2])))
        # fig.show()
        fig = px.line(df, x='deal', y=['balance_s_1', 'balance_s_2'], log_y=log_y, labels=label_s, markers=False)
        fig.for_each_trace(lambda t: t.update(name=label_s[t.name], legendgroup=label_s[t.name], hovertemplate=t.hovertemplate.replace(t.name, label_s[t.name])))
    else:
        fig = go.Figure(data=go.Scatter(x=date_s, y=balance_s_s[0], mode='lines', line=dict(width=1.5), name=title_s[0]))

    fig.update_layout(height=250, paper_bgcolor="LightSteelBlue")

    if len(date_s) >= 2:
        init_balance = balance_s_s[0][0] if len(balance_s_s[0]) > 0 else 100
        fig.add_trace(go.Scatter(x=[date_s[0], date_s[-1]], y=[init_balance, init_balance], mode='lines', line=dict(color='violet', width=1, dash='dash'), showlegend=False))

    fig.update_layout(yaxis=dict(title="Balance", range=[50, 150]))

    return fig


def produce_price_figure(date_s, price_s, log_y=False):
    title = "Price"
    if len(date_s) != len(price_s):
        return go.Figure()

    fig = go.Figure(data=go.Scatter(x=date_s, y=price_s, mode='lines', line=dict(color='cyan', width=1.5), name=title))
    fig.update_layout(height=250, paper_bgcolor="LightSteelBlue")
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.update_layout(yaxis=dict(title=title))

    return fig


def plot_trade_sim_combined_result(trade_symbol, target_candles_df, result_df_s):
    fig = produce__trade_sim__balance_price__combined_result_figure(trade_symbol, target_candles_df, result_df_s)
    display_plot(fig)


def mcad_bar_color_selector(prev, val):
    COLOR_LIGHT_GRAY = '#e2dee3'

    COLOR_RED = '#fa0000'
    COLOR_LIGHT_RED = '#ffbdbd'

    COLOR_GREEN = '#00917b'
    COLOR_LIGHT_GREEN = '#02e3c1'

    if prev is None:
        return COLOR_LIGHT_GRAY

    if val > 0:
        if prev < val:
            return COLOR_GREEN
        else:
            return COLOR_LIGHT_GREEN
    elif val < 0:
        if prev < val:
            return COLOR_LIGHT_RED
        else:
            return COLOR_RED
    else:
        return COLOR_LIGHT_GRAY

def produce__trade_sim__balance_price__combined_result_figure(trade_symbol, target_candles_df, result_df_s, layout_fig_balance=None):
    BUY_MARKER_SIZE = 4
    SELL_MARKER_SIZE = 4

    from SRC.CORE.utils import process_format_precision_order

    stable_symbol = trade_symbol.split("/")[1]

    stable_balance_s_s = []
    title_s = []
    for key, df in result_df_s.items():
        regime = 'SIM' if 'sim' in key else "BIN"

        filtered_df = df[df['state_s'] == STATE_OUT]

        stable_balance_s = filtered_df['stable_balance_s'].to_list()
        balance_date_s = filtered_df['close_dt_s'].to_list()
        stable_balance_s_s.append(stable_balance_s)

        stable_init_balance = process_format_precision_order(stable_balance_s[0]) if len(stable_balance_s) > 0 else '??'
        stable_final_balance = process_format_precision_order(stable_balance_s[-1]) if len(stable_balance_s) > 0 else '??'
        title = f"{regime} {stable_symbol}: {stable_init_balance} > {stable_final_balance}"
        title_s.append(title)

    key, df = next(iter(result_df_s.items()))
    price_s = df['price_s'].to_list()
    price_date_s = df['close_dt_s'].to_list()

    balance_price_combined_fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    range_title = f'{datetime_h_m__d_m(price_date_s[0]) if len(price_date_s) > 0 else "??"} - {datetime_h_m__d_m(price_date_s[-1]) if len(price_date_s) > 0 else "??"}'

    balance_fig = produce_balance_figure(balance_date_s, stable_balance_s_s, title_s, log_y=False)
    if len(stable_balance_s_s) > 0 and len(stable_balance_s_s[0]) > 0:
        last_balance = stable_balance_s_s[0][-1]
        if len(stable_balance_s_s) > 1:
            balance_range = [last_balance - last_balance * 0.1, last_balance + last_balance * 0.1]
        else:
            minimum = min(stable_balance_s_s[0])
            maximum = max(stable_balance_s_s[0])
            balance_range = [minimum - minimum * 0.1, maximum + maximum * 0.1]
    else:
        balance_range = [0, 1000]

    if balance_fig is not None and 'data' in balance_fig:
        for trace in balance_fig['data']:
            balance_price_combined_fig.add_trace(trace, row=1, col=1)

    price_figure = produce_price_figure(price_date_s, price_s, log_y=False)
    if price_figure is not None and 'data' in price_figure:
        for trace in price_figure['data']:
            balance_price_combined_fig.add_trace(trace, row=2, col=1)

    buy_actions_df = df[df['action_s'] == 'BUY']
    symbol_s = "circle-cross"
    symbol_color_s = "green"

    balance_price_combined_fig.add_trace(go.Scatter(x=buy_actions_df['close_dt_s'], y=buy_actions_df['price_s'], mode="markers", marker=dict(size=BUY_MARKER_SIZE, symbol=symbol_s, color=symbol_color_s), name='BUY`s'), row=2, col=1)

    buy_actions_df = df[df['action_s'] == 'SELL']
    symbol_s = "circle-x"
    symbol_color_s = "red"

    balance_price_combined_fig.add_trace(go.Scatter(x=buy_actions_df['close_dt_s'], y=buy_actions_df['price_s'], mode="markers", marker=dict(size=SELL_MARKER_SIZE, symbol=symbol_s, color=symbol_color_s), name='SELL`s'), row=2, col=1)

    from SRC.CORE._CONSTANTS import RSI_TOP, RSI_BOTTOM
    rsi_top = RSI_TOP()
    rsi_bottom = RSI_BOTTOM()

    if len(target_candles_df) > 0:
        balance_price_combined_fig.add_trace(go.Scatter(x=target_candles_df['close_time'], y=target_candles_df['RSI_14'], mode='lines', line=dict(color='violet', width=1.5), name='RSI 14'), row=3, col=1)
        balance_price_combined_fig.add_trace(go.Scatter(x=[target_candles_df['close_time'].iloc[0], target_candles_df['close_time'].iloc[-1]], y=[rsi_top, rsi_top], mode='lines', line=dict(color='black', width=1, dash='dash'), showlegend=False), row=3, col=1)
        balance_price_combined_fig.add_trace(go.Scatter(x=[target_candles_df['close_time'].iloc[0], target_candles_df['close_time'].iloc[-1]], y=[rsi_bottom, rsi_bottom], mode='lines', line=dict(color='black', width=1, dash='dash'), showlegend=False), row=3, col=1)
        balance_price_combined_fig.add_shape(x0=target_candles_df['close_time'].iloc[0], x1=target_candles_df['close_time'].iloc[-1], y0=rsi_bottom, y1=rsi_top, type="rect", fillcolor="rgba(128, 128, 128, 0.5)", line=dict(width=0), row=3, col=1)

        values = target_candles_df['MACDh_12_26_9'].to_list()
        histogram_colors = list(map(lambda indx: mcad_bar_color_selector(None if indx == 0 else values[indx - 1], values[indx]), range(len(values))))
        balance_price_combined_fig.add_trace(go.Scatter(x=target_candles_df['close_time'], y=target_candles_df['MACD_12_26_9'], line=dict(color='blue', width=1), name='MACD Line'), row=4, col=1)
        balance_price_combined_fig.add_trace(go.Scatter(x=target_candles_df['close_time'], y=target_candles_df['MACDs_12_26_9'], line=dict(color='red', width=1), name='Signal Line'), row=4, col=1)
        balance_price_combined_fig.add_trace(go.Bar(x=target_candles_df['close_time'], y=target_candles_df['MACDh_12_26_9'], name='MACD Histogram', marker=dict(color=histogram_colors, colorbar=dict(title='Value'), showscale=False)), row=4, col=1)

    balance_price_combined_fig.add_trace(
        go.Candlestick(
            x=target_candles_df['close_time'],
            open=target_candles_df['open'],
            high=target_candles_df['high'],
            low=target_candles_df['low'],
            close=target_candles_df['close'],
            showlegend=True,
            name=f'{FORCE_DISCRETIZATION()} candles'
        ), row=5, col=1
    )

    ticks_limit = 100
    xaxes_tickvals = list(filter(lambda ts: ts.minute % FORCE_INTERVAL() == 0, target_candles_df['close_time'].tolist()))
    ticks_each_n = int(len(xaxes_tickvals) / ticks_limit) if len(xaxes_tickvals) / ticks_limit > 1 else 1
    xaxes_tickvals = xaxes_tickvals[::ticks_each_n]
    xaxes_ticktext = [datetime_h_m__d_m(tickval) for tickval in xaxes_tickvals]

    balance_price_combined_fig.update_layout(title_text=f"{range_title}")
    balance_price_combined_fig.update_layout(height=1500)
    balance_price_combined_fig.update_layout(paper_bgcolor="LightSteelBlue")
    balance_price_combined_fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1))
    balance_price_combined_fig.update_layout(margin=dict(l=30, r=20, t=40, b=20))
    balance_price_combined_fig.update_layout(
        yaxis=dict(title="Balance", range=balance_range),
        yaxis2=dict(title=f"Price {trade_symbol}"),
        yaxis3=dict(title=f"RSI {rsi_top}/{rsi_bottom}", range=[0, 100]),
        yaxis4=dict(title=f'MCAD {target_candles_df.columns.values[6]}'),
        yaxis5=dict(title=f'{FORCE_DISCRETIZATION()} candles'))
    balance_price_combined_fig.update_layout(xaxis=dict(tickmode='array', tickvals=xaxes_tickvals, ticktext=xaxes_ticktext))
    balance_price_combined_fig.update_layout(dragmode='pan')

    bottom_rows_slider_thickness = 0.05
    xaxis_range_slider = dict(
        rangeslider=dict(visible=True, thickness=bottom_rows_slider_thickness),
        type="date",
        showticklabels=False,
        tickmode='array', tickvals=xaxes_tickvals, ticktext=xaxes_ticktext, tickangle=-45
    )

    xaxis_no_range_slider = dict(
        rangeslider=dict(visible=False, thickness=0),
        type="date",
        showticklabels=False,
        tickmode='array', tickvals=xaxes_tickvals, ticktext=xaxes_ticktext, tickangle=-45
    )

    subplot_layout_setup = dict()

    subplot_layout_setup['1'] = dict(
        xaxis=xaxis_no_range_slider,
        yaxis=dict(showticklabels=True)
    )

    subplot_layout_setup['2'] = dict(
        xaxis=xaxis_no_range_slider,
        yaxis=dict(showticklabels=True)
    )

    subplot_layout_setup['3'] = dict(
        xaxis=xaxis_no_range_slider,
        yaxis=dict(showticklabels=True, tickmode='array', tickvals=[0, rsi_bottom, 50, rsi_top, 100], ticktext=[0, rsi_bottom, 50, rsi_top, 100])
    )

    subplot_layout_setup['4'] = dict(
        xaxis=xaxis_no_range_slider,
        yaxis=dict(showticklabels=True)
    )

    subplot_layout_setup['5'] = dict(
        xaxis=xaxis_range_slider,
        yaxis=dict(showticklabels=True)
    )

    balance_price_combined_fig.update_xaxes(showspikes=True, spikemode='across+marker', spikesnap='cursor', showline=False, spikedash='solid', spikecolor="grey", spikethickness=1)
    balance_price_combined_fig.update_yaxes(showspikes=True, spikemode='across+marker', spikesnap='cursor', showline=False, spikedash='solid', spikecolor="grey", spikethickness=1)

    balance_price_combined_fig.update_layout(dragmode='pan', hovermode='x unified')
    balance_price_combined_fig.update_traces(xaxis="x5")

    for key, val in subplot_layout_setup.items():
        balance_price_combined_fig.layout[f'xaxis{key}'] = balance_price_combined_fig.layout[f'xaxis{key}'].update(val['xaxis'])
        balance_price_combined_fig.layout[f'yaxis{key}'] = balance_price_combined_fig.layout[f'yaxis{key}'].update(val['yaxis'])

    if layout_fig_balance is not None:
        balance_price_combined_fig.plotly_relayout(layout_fig_balance)

    return balance_price_combined_fig


def plot_trade_sim_actions_distribution(trade_symbol, result_df_s):
    from SRC.CORE.plot_utils import display_plot
    from SRC.CORE.plot_utils import produce_trading_actions_distribution_figure

    key, df = next(iter(result_df_s.items()))

    action_s = df['action_s'].to_list()

    fig = produce_trading_actions_distribution_figure(action_s)
    display_plot(fig)


def plot_trade_sim_states_distribution(trade_symbol, result_df_s):
    from SRC.CORE.plot_utils import display_plot
    from SRC.CORE.plot_utils import produce_trading_states_distribution_figure

    key, df = next(iter(result_df_s.items()))

    state_s = df['state_s'].to_list()

    fig = produce_trading_states_distribution_figure(state_s)
    display_plot(fig)


def plot__trade_sim__act_pred__class_series_correlation(trade_symbol, result_df_s):
    from SRC.CORE.plot_utils import display_plot

    key, df = next(iter(result_df_s.items()))

    title = f'{trade_symbol} >> {"ACTUAL"} vs. {"PREDICTED"} class correlation'
    # printmd(f'**{title}**')
    act_fig = produce__act_vs_pred__class_sequence_series_figure(df, title)
    display_plot(act_fig)


def plot__trade_sim__act_vs_pred__class_distribution_separate(trade_symbol, result_df_s):
    from SRC.CORE.plot_utils import plot_class_distribution_diagram_figure

    key, df = next(iter(result_df_s.items()))

    act_class_s = df['act_prev_class_s'].to_list()
    pred_class_s = df['pred_class_s'].to_list()

    title = f"{trade_symbol} >> {'ACTUAL'} class distribution"
    # printmd(f'**{title}**')
    plot_class_distribution_diagram_figure(act_class_s, title)

    title = f"{trade_symbol} >> {'PREDICTED'} class distribution"
    # printmd(f'**{title}**')
    plot_class_distribution_diagram_figure(pred_class_s, title)


def plot__trade_sim__act_vs_pred__class_distribution_combined(trade_symbol, result_df_s):
    key, df = next(iter(result_df_s.items()))

    title = f'{trade_symbol} >> {"ACTUAL"} vs. {"PREDICTED"} class distribution'
    # printmd(f'**{title}**')
    plot__act_vs_pred__class_distribution_diagram_figure(df, title)


def get_dashboard_header_info(data_state):
    regime = data_state[_REGIME]
    network = data_state[NETWORK_KEY].replace("-model.pt", "")
    fee = data_state[FEE_KEY]
    trade_symbol = data_state['trade_symbol']
    # discretization = data_state['discretization']
    discretization = '5M'
    result_df_dict = data_state['result_df_dict']
    target_df = result_df_dict['bin_df'] if 'bin_df' in result_df_dict else result_df_dict['sim_df']

    if target_df.empty:
        state_s = []
        action_s = []
        transaction_fee_s = []
        state = '??'
    else:
        state_s = target_df['state_s'].to_list()
        action_s = target_df['action_s'].to_list()
        transaction_fee_s = target_df['transaction_fee_s'].to_list()
        state = state_s[-1]

    buys_count = len(list(filter(lambda a: a == ACTION_BUY, action_s)))
    sells_count = len(list(filter(lambda a: a == ACTION_SELL, action_s)))
    ignore_count = len(list(filter(lambda a: a == ACTION_NO, action_s)))
    total = buys_count + sells_count + ignore_count
    transaction_fees_spent = sum(transaction_fee_s)

    regime_network_title = f"{regime} | {fee} | {network}"
    trades_title = f"buy's: {buys_count} | sell's: {sells_count} | ignore's: {ignore_count} | total: {total} | fee's: {_float_6(transaction_fees_spent)}"
    pair_state_title = f"{trade_symbol} || {discretization} || {state} || {trades_title}" if len(state_s) > 0 else f"{trade_symbol} || {discretization} || {trades_title}"

    return regime_network_title, pair_state_title


def optimized_ticks(target_variable, _lambda, ticks_limit=70):
    ticks_each_n = int(len(target_variable) / ticks_limit) if len(target_variable) / ticks_limit > 1 else 1
    tickval_s = target_variable[::ticks_each_n]
    ticktext_s = [_lambda(tickval) for tickval in tickval_s]

    return tickval_s, ticktext_s
