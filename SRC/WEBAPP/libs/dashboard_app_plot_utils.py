import math
import os
import plotly.io as pio

from SRC.CORE._CONFIGS import get_config
from SRC.CORE._CONSTANTS import _KIEV_TIMESTAMP, _SHORT, _LONG
from SRC.NN.IModelBase import produce_model

pio.renderers.default = "notebook_connected"

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from SRC.CORE.plot_utils import lighten_color_rgba
from SRC.WEBAPP.libs.exceptions import NoTradesInfoError, NoActivityError

from SRC.CORE._CONSTANTS import _INF_DISCR, _DASHBOARD_CORRELATION_STATUS__ALIVE, _DASHBOARD_CORRELATION_STATUS__DIED, _OUT_DISCRETIZATION_FILE_PATH, _UTC_TIMESTAMP, _DASHBOARD_SEGMENT_NET_FULL_PATH, _TRADES_FILE_PATH, \
    _TRADES_DF_FIG_HTML_FILE_PATH, _WEBAPP_DASHBOARD_NETFOLDER_TRADES_HTML_FILE_PATH, _TRANSACTIONS_INFO_JSON_FILE_PATH, _STATS_DF_FILE_PATH, _INVERTED_TRADES_INFO, IS_INVERTED_TRADES_INFO, _FUTURES, _MARGIN, IGNORE_SEPARATOR, \
    POSITION_SEPARATOR, ALERT_SEPARATOR, EXCEPTION_SEPARATOR, _DASHBOARD_CORRELATION_STATUS__ERROR, _TRADES_ERRORS_FILE_PATH
from SRC.LIBRARIES.time_utils import TIME_DELTA, kiev_now, as_kiev_tz, get_datetime_splitters, as_utc_tz

from SRC.CORE.debug_utils import SET_SYMBOL, ERROR_SPLITTED, is_autotrading, is_backtesting
from SRC.LIBRARIES.new_data_utils import produce_signal_encoder, candelify, produce_balance_df, fetch_cached, read_df_cached
from SRC.LIBRARIES.new_utils import produce_net_folder, parse_net_folder, parse_string_variables, produce_empty_net, nexter, floor, get_datetime_price, get_datetime_price_s, is_running, parse_net_folder_hashed, \
    is_close_to_zero, read_text_to_list, filter_dict, get_balance_s
from SRC.CORE.utils import read_json_safe, datetime_h_m__d_m_y, datetime_h_m_s, _float_n, datetime_h_m_s__d_m, pairwise, write_json, _float_5, _float_2, _float_3
from SRC.CORE._CONSTANTS import _DASHBOARD_SEGMENT, _AUTOTRADING, _BALANCE_FILE_PATH, _MOCK, _DASHBOARD_SEGMENT_AUTOTRADING, _AUTOTRADING_REGIME, _AUTOMATION_TYPE, _MARKET_TYPE, _SYMBOL_JOIN, _NET_FOLDER, _TRADES_DF_FIG_IMG_FILE_PATH, _SYMBOL, _DISCRETIZATION, \
    _KIEV_TIMESTAMP, _WEBAPP_DASHBOARD_NETFOLDER_IMAGE_FILE_PATH


def get_correlation_status_rgba_color(status):
    status_color = 'rgb(255, 255, 255)'

    if status == _DASHBOARD_CORRELATION_STATUS__ALIVE:
        status_color = 'rgb(176, 196, 222)'

    if status == _DASHBOARD_CORRELATION_STATUS__DIED:
        status_color = lighten_color_rgba('gray', 0.15)

    if status == _DASHBOARD_CORRELATION_STATUS__ERROR:
        status_color = lighten_color_rgba('violet', 0.5)

    return status_color


def draw_dashboard_automation_trades_fig(running_net_folder, intercept_errors=False, empty_net_d=None, force_draw=False):
    os.environ[_NET_FOLDER] = produce_net_folder(parse_net_folder_hashed(running_net_folder))

    try:
        net_folder_full_path = _DASHBOARD_SEGMENT_NET_FULL_PATH()
        if not os.path.exists(net_folder_full_path):
            return {
                "L/P": 0,
                "boost_fail": 1,
                "profit_loss": 0.1,
                "epd": 0,
                "DETAILS": f"NOT EXISTS: {net_folder_full_path}"
            }

        transactions_info_json_file_path = _TRANSACTIONS_INFO_JSON_FILE_PATH(net_folder=running_net_folder)
        if is_backtesting() and not force_draw and os.path.exists(transactions_info_json_file_path):
            transactions_info = read_json_safe(transactions_info_json_file_path, {})
            if transactions_info:
                return transactions_info

        fig, ohlc_df, transactions_info = produce_dashboard_automation_trades_fig(running_net_folder, empty_net_d=empty_net_d, force_draw=force_draw)

        write_json(transactions_info, transactions_info_json_file_path)

        if force_draw:
            trades_df_fig_file_path = _TRADES_DF_FIG_IMG_FILE_PATH(net_folder=running_net_folder)
            img_path = _WEBAPP_DASHBOARD_NETFOLDER_IMAGE_FILE_PATH(net_folder=running_net_folder)
            os.makedirs(os.path.dirname(img_path), exist_ok=True)

            img_width = 1300 + math.pow(len(ohlc_df), 1 / 2.2) * 50

            fig.update_layout(width=img_width, height=1200)
            fig.write_image(img_path, width=img_width, height=1200)
            fig.write_image(trades_df_fig_file_path, width=img_width, height=1200)

            if is_autotrading() or is_backtesting():
                trade_info_s = read_text_to_list(_TRADES_FILE_PATH(net_folder=running_net_folder))
                trades_df_fig_file_path = _TRADES_DF_FIG_HTML_FILE_PATH(net_folder=running_net_folder)
                trades_df_fig_out_file_path = _WEBAPP_DASHBOARD_NETFOLDER_TRADES_HTML_FILE_PATH(net_folder=running_net_folder)
                os.makedirs(os.path.dirname(trades_df_fig_out_file_path), exist_ok=True)

                trades_info_fig = produce_trades_info_fig(trade_info_s, transactions_info)
                trades_info_fig.write_html(trades_df_fig_file_path)
                trades_info_fig.write_html(trades_df_fig_out_file_path)

        return transactions_info
    except NoTradesInfoError as err:
        if intercept_errors:
            ERROR_SPLITTED(f"{str(err)}")

            return None
        else:
            raise
    except NoActivityError as err:
        if intercept_errors:
            ERROR_SPLITTED(f"{str(err)}")

            return None
        else:
            raise


def produce_pseudo_ignore_balance_record(dt_utc, balance):
    pseudo_balance = {
        'date_time': dt_utc,
        'transaction_id': 'UNKNOWN',
        'correlation_id': 'UNKNOWN',
        'transaction_type': 'ALIGN',
        'transaction_result': 'ALIGN',
        'balance_movement': 0,
        'balance': balance
    }

    return pseudo_balance


def produce_dashboard_automation_trades_fig(net_folder, empty_net_d=None, force_draw=False):
    net_data = parse_net_folder(net_folder)
    net_folder = produce_net_folder(net_data)

    market = net_data['market']
    market_type = market.split('__')[0]
    symbol = net_data['symbol']

    _regime = 'REGIME_'
    _inf_discr = 'INF_DISCR_'
    var_s = parse_string_variables(market, [_regime, _inf_discr])
    autotrading_regime = var_s[_regime] if _regime in var_s else _MOCK

    SET_SYMBOL(symbol)

    os.environ[_MARKET_TYPE] = market_type
    os.environ[_AUTOTRADING_REGIME] = autotrading_regime

    _symbol_join = os.environ[_SYMBOL_JOIN]

    try:
        balance_s = get_balance_s()
    except FileNotFoundError as err:
        raise NoTradesInfoError(net_folder, str(err)) from err

    price = get_datetime_price()

    inf_discr = var_s[_inf_discr] if _inf_discr in var_s else 'MODEL'
    os.environ[_INF_DISCR] = inf_discr

    model_name = net_data['model_name']
    if empty_net_d is not None:
        if model_name in empty_net_d:
            inference_discretization_s = empty_net_d[model_name]
        else:
            net = produce_model(model_name)
            inference_discretization_s = net.inference_discretization_s()
            empty_net_d[model_name] = inference_discretization_s
    else:
        net = produce_model(model_name)
        inference_discretization_s = net.inference_discretization_s()

    ohlc_discretization = inference_discretization_s[0]
    signal_discretization = inference_discretization_s[0]

    if price['date_time'] > balance_s[-1]['date_time']:
        last_pseudo_balance = produce_pseudo_ignore_balance_record(price['date_time'], balance_s[-1]['balance'])
        balance_s.append(last_pseudo_balance)

    end_dt = price['date_time']
    start_dt = balance_s[0]['date_time'] - TIME_DELTA(ohlc_discretization) * 2

    balance_df = produce_balance_df(balance_s)
    out_discretization_ohlc_file_path = _OUT_DISCRETIZATION_FILE_PATH()
    if os.path.exists(out_discretization_ohlc_file_path):
        ohlc_df = read_df_cached(out_discretization_ohlc_file_path)
    else:
        ohlc_df = fetch_cached(market_type, _symbol_join, ohlc_discretization, start_dt, end_dt, validate=False)

    ohlc_df = ohlc_df[ohlc_df[_UTC_TIMESTAMP] >= balance_df.iloc[0][_UTC_TIMESTAMP]].iloc[:-1]

    fig, transactions_info = fig_balances_candles(balance_df, ohlc_df, net_folder, end_dt, market_type)

    if is_autotrading():
        filtered = balance_df[balance_df["correlation_id"] != "UNKNOWN"]
        if not filtered.empty:
            last_correlation_id = filtered.iloc[-1]["correlation_id"]
        else:
            last_correlation_id = None

        transactions_info['last_correlation_id'] = last_correlation_id

    return fig, ohlc_df, transactions_info


def fig_balances_candles(balance_df, ohlc_df, net_folder, end_dt, market_type):
    time_feature = _KIEV_TIMESTAMP
    end_dt_kiev = as_kiev_tz(end_dt)

    net_folder_present = " | ".join([f"<b>{seg}</b>" for seg in net_folder.split("|")])
    initial_balance = balance_df.iloc[0]['balance']
    symbol = ohlc_df.iloc[0][_SYMBOL]
    discretization = ohlc_df.iloc[0][_DISCRETIZATION]
    start_dt_st = datetime_h_m__d_m_y(ohlc_df.iloc[0][time_feature])
    end_dt_st = datetime_h_m__d_m_y(ohlc_df.iloc[-1][time_feature])
    ticks_limit = int(35 + (math.sqrt(len(ohlc_df)) * 1.3))
    vertical_spacing = 0.015

    min_price = ohlc_df[['open', 'high', 'low', 'close']].min().min() * 0.997
    max_price = ohlc_df[['open', 'high', 'low', 'close']].max().max() * 1.003
    start_ohlc_dt = min(ohlc_df[time_feature])
    end_ohlc_dt = max(ohlc_df[time_feature])
    last_price = ohlc_df.iloc[-1]['close']

    min_balance = balance_df[['balance']].min().min()
    max_balance = balance_df[['balance']].max().max()

    start_balance_dt = min(balance_df[time_feature])
    end_balance_dt = max(balance_df[time_feature])

    ohlc_xs = ohlc_df[time_feature].to_list()

    balance_xs = balance_df[time_feature].to_list()
    balance_ys = balance_df['balance'].to_list()
    current_balance = balance_ys[-1]

    tickvals = balance_xs
    last_ohlc_dt = max(ohlc_xs)
    last_balance_dt = as_kiev_tz(balance_df.iloc[-1][_KIEV_TIMESTAMP])
    # tickvals = ohlc_xs

    aligns_df = balance_df[balance_df['transaction_type'] == 'ALIGN']
    ignores_df = aligns_df[aligns_df['transaction_type'] == 'IGNORE']
    errors_df = aligns_df[aligns_df['transaction_type'] == 'ERROR']
    warns_df = aligns_df[aligns_df['transaction_type'] == 'WARNING']
    transactions_df = balance_df[(balance_df['transaction_type'] != 'ALIGN')]

    if is_close_to_zero(max_balance - min_balance):
        shift_arrow_delta = initial_balance * 1.145
        shift_arrow_padding = initial_balance * 1.055
    else:
        shift_arrow_delta = (max_balance - min_balance) * 0.145
        shift_arrow_padding = (max_balance - min_balance) * 0.055

    max_y = max_balance + shift_arrow_delta + shift_arrow_padding
    min_y = min_balance - shift_arrow_delta - shift_arrow_padding

    ignore_df = transactions_df[(transactions_df['transaction_type'] == 'IGNORE')]
    long_profit_df = transactions_df[(transactions_df['transaction_type'] == 'LONG') & (transactions_df['transaction_result'] == 'PROFIT')]
    long_loss_df = transactions_df[(transactions_df['transaction_type'] == 'LONG') & (transactions_df['transaction_result'] == 'LOSS')]
    short_profit_df = transactions_df[(transactions_df['transaction_type'] == 'SHORT') & (transactions_df['transaction_result'] == 'PROFIT')]
    short_loss_df = transactions_df[(transactions_df['transaction_type'] == 'SHORT') & (transactions_df['transaction_result'] == 'LOSS')]

    stats_df_file_path = _STATS_DF_FILE_PATH(net_folder=net_folder)
    stats_df = produce_transactions_stats_df(transactions_df)
    stats_df.to_csv(stats_df_file_path, index=False)

    fig = make_subplots(rows=2, cols=1, vertical_spacing=vertical_spacing, shared_xaxes=True)
    fig.add_trace(go.Candlestick(
        x=ohlc_df[time_feature],
        open=ohlc_df['open'],
        high=ohlc_df['high'],
        low=ohlc_df['low'],
        close=ohlc_df['close'],
        name=f'Candles {discretization}',
        increasing=dict(fillcolor='#a9e8d4', line=dict(color='#53c9a4')),
        decreasing=dict(fillcolor='#fcc1bd', line=dict(color='#fa847d'))), row=1, col=1)

    fig.add_trace(go.Scatter(x=[balance_df.iloc[0][time_feature], balance_df.iloc[-1][time_feature]], y=[initial_balance, initial_balance], name='Initial, $', line=dict(color='red', dash='dash', width=0.5), mode='lines'), row=2, col=1)

    interpolation_color = 'darkgray'
    interpolation_width = 1.5
    balance_market_size = 4
    transaction_marker_size = 5
    error_warning_marker_size = 8
    font_size = 18

    fig.add_trace(go.Scatter(x=balance_df[time_feature], y=balance_df['balance'], name='Interpol, $', line=dict(color=interpolation_color, dash='solid', width=interpolation_width), mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=transactions_df[time_feature], y=transactions_df['balance'], name='Balance, $', marker=dict(color='black', size=balance_market_size), mode='markers'), row=2, col=1)

    fig.add_trace(go.Scatter(x=ignores_df[time_feature], y=ignores_df['balance'], name='IGNORE', marker=dict(color='gray', size=5), mode='markers'), row=2, col=1)

    fig.add_trace(go.Scatter(x=long_profit_df[time_feature], y=long_profit_df['balance'] - shift_arrow_delta, name='LONG PROFIT', mode="markers", marker=dict(size=transaction_marker_size, symbol="triangle-up", color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=long_loss_df[time_feature], y=long_loss_df['balance'] - shift_arrow_delta, name='LONG LOSS', mode="markers", marker=dict(size=transaction_marker_size, symbol="triangle-up", color="red")), row=2, col=1)

    fig.add_trace(go.Scatter(x=short_profit_df[time_feature], y=short_profit_df['balance'] + shift_arrow_delta, name='SHORT PROFIT', mode="markers", marker=dict(size=transaction_marker_size, symbol="triangle-down", color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=short_loss_df[time_feature], y=short_loss_df['balance'] + shift_arrow_delta, name='SHORT LOSS', mode="markers", marker=dict(size=transaction_marker_size, symbol="triangle-down", color="red")), row=2, col=1)

    fig.add_trace(go.Scatter(x=warns_df[time_feature], y=warns_df['balance'], name='WARNING', marker=dict(color='yellow', size=error_warning_marker_size, symbol='x'), mode='markers'), row=2, col=1)
    fig.add_trace(go.Scatter(x=errors_df[time_feature], y=errors_df['balance'], name='ERROR', marker=dict(color='red', size=error_warning_marker_size, symbol='x'), mode='markers'), row=2, col=1)

    days_splitter_s = get_datetime_splitters(ohlc_xs, discretization='1D', as_tz=as_kiev_tz)
    for days_splitter in days_splitter_s:
        fig.add_vline(x=days_splitter, line_width=4, line_dash="dash", line_color="lightgray", row=1, col=1)

    week_splitter_s = get_datetime_splitters(ohlc_xs, discretization='1W', as_tz=as_kiev_tz)
    for days_splitter in week_splitter_s:
        fig.add_vline(x=days_splitter, line_width=4, line_dash="dash", line_color="gray", row=1, col=1)

    if current_balance > initial_balance:
        state_shape_y_end = max_y
        state_color = 'green'
    elif current_balance < initial_balance:
        state_shape_y_end = min_y
        state_color = 'red'
    else:
        state_shape_y_end = initial_balance
        state_color = 'LightSteelBlue'

    fig.add_shape(
        type="rect",
        x0=tickvals[0], y0=initial_balance, x1=tickvals[-1], y1=state_shape_y_end,
        fillcolor=state_color,
        opacity=0.1,
        row=2, col=1
    )

    arrow_position_bottom = False
    top_idx = 0
    bottom_idx = 0

    next_positive = nexter([20, 45, 70])
    next_negative = nexter([-20, -45, -70])
    for idx, row in pd.concat([errors_df, warns_df]).sort_index().iterrows():
        ay = next_positive() if arrow_position_bottom else next_negative()
        transactions_info_annotation = dict(
            x=row[time_feature],
            y=row['balance'],
            text=f"{row['transaction_result']}",
            showarrow=True,
            yshift=-5 if arrow_position_bottom else 5,
            arrowhead=2,
            font=dict(color='#6300ff', size=font_size * 0.75),
            ay=ay,
            xref="x2",
            yref="y2"
        )
        fig.add_annotation(**transactions_info_annotation)
        if arrow_position_bottom:
            bottom_idx += 1
        else:
            top_idx += 1
        arrow_position_bottom = not arrow_position_bottom

    is_real_time = tickvals[-1] >= kiev_now() - TIME_DELTA('1M')
   
    ticktext = list(map(lambda t: datetime_h_m__d_m_y(t), tickvals))
    ticks_count_multiplier = 1
    time_feature_max = max([end_ohlc_dt, end_balance_dt])
    x_range = [start_ohlc_dt, time_feature_max]

    fig.add_vline(x=end_dt_kiev, line_width=5, line_dash="dash", line_color="blue", row=2, col=1)

    if len(balance_df) <= 3:
        fig.add_shape(
            type="rect",
            x0=tickvals[0], y0=min_y, x1=tickvals[-1], y1=max_y,
            fillcolor='yellow',
            opacity=0.1,
            row=2, col=1
        )

    if len(ohlc_df['close']) > 0:
        current_price_annotation = dict(
            x=time_feature_max,
            y=last_price,
            text=f'PRICE: <b>{last_price}</b>',
            showarrow=False,
            xshift=-100,
            font=dict(color='#6300ff', size=font_size),
            xref="x1",
            yref="y1"
        )
        fig.add_annotation(**current_price_annotation)

    balance_star_suffix = ""
    start_balance = initial_balance
    final_balance = current_balance
    transactions_info = None
    if len(balance_xs) > 0:
        start_balance_annotation = dict(
            x=x_range[0],
            y=balance_ys[0],
            text=f'INITIAL: <b>${_float_n(start_balance, 5)}{balance_star_suffix}</b>',
            showarrow=False,
            yshift=-20 if current_balance > initial_balance else 20,
            xshift=100,
            font=dict(color='#6300ff', size=font_size),
            xref="x2",
            yref="y2"
        )
        fig.add_annotation(**start_balance_annotation)

        last_balance_annotation = dict(
            x=balance_xs[-1],
            y=current_balance,
            text=f'CURRENT: <b>${_float_n(final_balance, 5)}{balance_star_suffix}</b>',
            showarrow=False,
            yshift=-20 if current_balance > initial_balance else 20,
            xshift=-100,
            font=dict(color='#6300ff', size=font_size),
            xref="x2",
            yref="y2"
        )
        fig.add_annotation(**last_balance_annotation)

        if len(balance_xs) > 1:
            no_nan_transactions_df = transactions_df.dropna(subset=['balance'])
            candles = len(ohlc_df)
            long = len(long_profit_df) + len(long_loss_df)
            short = len(short_profit_df) + len(short_loss_df)
            profit = len(long_profit_df) + len(short_profit_df)
            loss = len(long_loss_df) + len(short_loss_df)
            loss_profit_ratio = round(loss / profit if profit > 0 else math.inf, 2)
            profit_loss = round(profit / loss if profit > 0 and loss > 0 else -math.inf, 2)
            positions = long + short
            ignores = candles - positions
            start_balance = no_nan_transactions_df.iloc[0]['balance'] if len(no_nan_transactions_df) > 0 else 1
            end_balance = no_nan_transactions_df.iloc[-1]['balance'] if len(no_nan_transactions_df) > 0 else 1
            boost_fail = floor(end_balance / start_balance, 3) if start_balance > 0 else "X"
            earn_per_deal_percent = floor((boost_fail - 1) * 100 / (positions if positions > 0 else 1), 3) if start_balance > 0 else "X"

            transactions_info = {
                'boost_fail': boost_fail ,
                'profit_loss': profit_loss,
                'epd': earn_per_deal_percent,
                'L/P': loss_profit_ratio,
                'PROF': profit,
                'LOSS': loss,
                'CAND': candles,
                'POS': positions,
                'IG': ignores,
                'LO': long,
                'SH': short,
                'L PROF': len(long_profit_df),
                'S PROF': len(short_profit_df),
                'L LOSS': len(long_loss_df),
                'S LOSS': len(short_loss_df),

                'last_drawn_dt': end_dt_kiev,
                'last_ohlc_dt': last_ohlc_dt,
                'last_balance_dt': last_balance_dt
            }

    ticks_each_n = int(((len(tickvals) / ticks_limit) if len(tickvals) / ticks_limit > 1 else 1) * ticks_count_multiplier)
    ticks_vals_n = tickvals[-1::-ticks_each_n][::-1]
    ticks_text_n = ticktext[-1::-ticks_each_n][::-1]

    ticks_vals_n = [tickvals[0], *ticks_vals_n, tickvals[-1]]
    ticks_text_n = [ticktext[0], *ticks_text_n, ticktext[-1]]

    alive_time_threshold = get_config("automation_dashboard_app.backtest.alive_time_threshold", default="30S")

    is_net_running = is_running(net_folder, alive_time_threshold)
    is_fresh_dfs = (kiev_now() - end_dt_kiev) < TIME_DELTA('1M')
  
    if is_net_running:
        bg_color = get_correlation_status_rgba_color(_DASHBOARD_CORRELATION_STATUS__ALIVE)
        fig.update_layout(paper_bgcolor=bg_color)
    else:
        bg_color = get_correlation_status_rgba_color(_DASHBOARD_CORRELATION_STATUS__DIED)
        fig.update_layout(paper_bgcolor=bg_color)

    error_log_file_path = _TRADES_ERRORS_FILE_PATH(dashboard_segment=_DASHBOARD_SEGMENT_AUTOTRADING, net_folder=net_folder)
    if os.path.exists(error_log_file_path):
        bg_color = get_correlation_status_rgba_color(_DASHBOARD_CORRELATION_STATUS__ERROR)
        fig.update_layout(paper_bgcolor=bg_color)

    fig.update_xaxes(range=x_range, row=1, col=1)
    fig.update_xaxes(tickangle=-45, tickvals=ticks_vals_n, ticktext=ticks_text_n, rangeslider_visible=False, range=x_range, row=2, col=1)
    fig.update_yaxes(range=[min_price, max_price], row=1, col=1)
    fig.update_yaxes(range=[min_y, max_y], row=2, col=1)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis_rangeslider_thickness=0.03)
    fig.update_layout(height=900)
    fig.update_layout(margin=dict(l=30, r=20, t=40, b=20))
    names = set()
    fig.for_each_trace(lambda trace: trace.update(showlegend=False) if (trace.name in names) or 'Interpol' in trace.name else names.add(trace.name))

    return fig, transactions_info


def produce_transactions_stats_df(transactions_df):
    bins = [0, 6, 12, 18, 24]
    labels = ["night", "morning", "day", "evening"]

    transactions_df = transactions_df[(transactions_df['transaction_type'] == _SHORT) | (transactions_df['transaction_type'] == _LONG)]
    transactions_df["hour"] = transactions_df[_KIEV_TIMESTAMP].dt.hour
    transactions_df["day_part"] = pd.cut(transactions_df["hour"], bins=bins, labels=labels, right=False)

    transactions_df['transaction_result'] = transactions_df['transaction_result'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)

    stats_df = transactions_df.groupby("day_part")["transaction_result"].value_counts().unstack(fill_value=0)
    stats_df["total"] = stats_df.sum(axis=1)
    if 'PROFIT' in stats_df.columns:
        stats_df["profit_rate"] = stats_df["PROFIT"] / stats_df["total"]
    else:
        stats_df["profit_rate"] = 0

    return stats_df


def group_trades_lines(lines):
    starts_with_s = [f"{IGNORE_SEPARATOR} ", f"{POSITION_SEPARATOR} ", f"{ALERT_SEPARATOR} ", f"{EXCEPTION_SEPARATOR} "]

    groups = []
    current_group = []
    open_marker = None

    def is_start(line):
        return any(line.startswith(prefix) for prefix in starts_with_s)

    for line in lines:
        if is_start(line):
            if open_marker is None:
                open_marker = line
                current_group = [line]
            else:
                if line == open_marker:
                    current_group.append(line)
                    groups.append(current_group)
                    current_group = []
                    open_marker = None
                else:
                    groups.append(current_group)
                    open_marker = line
                    current_group = [line]
        else:
            if open_marker is not None:
                current_group.append(line)

    return groups


def produce_trades_info_fig(trade_info_s, transactions_info):
    ignore_field_s = ['last_correlation_id', 'last_drawn_dt', 'last_ohlc_dt']
    trade_info_s = trade_info_s
    trades_info_title = produce_trade_info_str(transactions_info, ignore_field_s=ignore_field_s)
    trades_info_title = trades_info_title.replace(" | ", "|")

    if IS_INVERTED_TRADES_INFO():
        trade_info_s = trade_info_s[::-1]

    separator = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    if separator not in trade_info_s:
        trades_info_title_second = "<br>".join(trade_info_s)
        trades_info_title_final = "<br>".join([trades_info_title, trades_info_title_second])
        trades_info_format_groupings_present_s = ["NO TRADES DETAILS YET.."]

        trades_info_fig = go.Figure(data=[go.Table(header=dict(values=[trades_info_title_final], align='left'), cells=dict(values=[trades_info_format_groupings_present_s], align='left', format=[""]))])
        trades_info_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        trades_info_fig.update_layout(autosize=True, height=None)

        return trades_info_fig

    index = trade_info_s.index(separator)
    header_info_s = trade_info_s[:index]  # elements before the split string
    body_info_s = trade_info_s[index + 1:]

    trades_info_title_second = "<br>".join(header_info_s)
    trades_info_title_final = "<br>".join([trades_info_title, trades_info_title_second])

    if any([_FUTURES in header_info_item for header_info_item in header_info_s]):
        trades_info_grouping_s = group_trades_lines(body_info_s)
        trades_info_format_grouping_s = [format_trades_info(trades_info_format_grouping) for trades_info_format_grouping in trades_info_grouping_s]
        trades_info_format_groupings_present_s = ["<br>".join(lines[:-1]) for lines in trades_info_format_grouping_s]
    elif any([_MARGIN in header_info_item for header_info_item in header_info_s]):
        trades_info_grouping_s = group_trades_lines(body_info_s)
        trades_info_format_grouping_s = [format_trades_info(trades_info_format_grouping) for trades_info_format_grouping in trades_info_grouping_s]
        trades_info_format_groupings_present_s = ["<br>".join(lines[:-1]) for lines in trades_info_format_grouping_s]
    else:
        trades_info_format_groupings_present_s = ["BACKTESTING NOT IMPLEMENTED"]

    trades_info_fig = go.Figure(data=[go.Table(header=dict(values=[trades_info_title_final], align='left'), cells=dict(values=[trades_info_format_groupings_present_s], align='left', format=[""]))])
    trades_info_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    trades_info_fig.update_layout(autosize=True, height=None)

    return trades_info_fig


def format_trades_info(trade_info_s):
    _long_in_text = lambda text: text.replace("LONG IN", f'<span style="color:black"><b>{"LONG IN"}</b></span>')
    _short_in_text = lambda text: text.replace("SHORT IN", f'<span style="color:black"><b>{"SHORT IN"}</b></span>')
    _profit_out_text = lambda text: text.replace("PROFIT OUT", f'<span style="color:green"><b>{"PROFIT OUT"}</b></span>')
    _loss_out_text = lambda text: text.replace("LOSS OUT", f'<span style="color:red"><b>{"LOSS OUT"}</b></span>')

    _loss_short_text = lambda text: text.replace("LOSS|SHORT", f'<span style="color:red"><b>{"LOSS|SHORT"}</b></span>')
    _profit_short_text = lambda text: text.replace("PROFIT|SHORT", f'<span style="color:green"><b>{"PROFIT|SHORT"}</b></span>')
    _loss_long_text = lambda text: text.replace("LOSS|LONG", f'<span style="color:red"><b>{"LOSS|LONG"}</b></span>')
    _profit_long_text = lambda text: text.replace("PROFIT|LONG", f'<span style="color:green"><b>{"PROFIT|LONG"}</b></span>')

    _profit_exception_text = lambda text: text.replace("!!! PROFIT [EXCEPTION]", f'<span style="color:red"><b><i>{"!!! PROFIT [EXCEPTION]"}</i></b></span>')
    _loss_exception_text = lambda text: text.replace("!!! LOSS [EXCEPTION]", f'<span style="color:red"><b><i>{"!!! LOSS [EXCEPTION]"}</i></b></span>')

    _exception_title_text = lambda text: f'<span style="color:violet"><b>{text}</b></span>'
    _error_text = lambda text: f'<span style="color:orange">{text}</span>'
    _exception_text = lambda text: f'<span style="color:violet">{text}</span>'
    _exit_success_text = lambda text: f'<span style="color:lime"><b>{text}</b></span>'
    _exit_failed_text = lambda text: f'<span style="color:violet"><b>{text}</b></span>'

    exception_regime = False
    trade_info_format_s = []

    api_error_s = [
        'APIError(code=-3045)',
        'APIError(code=-2010)',
        'APIError(code=-1102)',
        'APIError(code=-3044)',
        'APIError(code=-2013)',
    ]

    for trade_info in trade_info_s:
        trade_info = trade_info.replace('***', '').replace('**', '')

        if 'NO ACTION' in trade_info:
            trade_info_format = _exception_title_text(trade_info)
            trade_info_format_s.append(trade_info_format)
            continue
        if any([api_error in trade_info for api_error in api_error_s]):
            trade_info_format = _error_text(trade_info)
            trade_info_format_s.append(trade_info_format)
            continue
        if 'Traceback' in trade_info or 'APIError' in trade_info or exception_regime:
            exception_regime = not 'APIError' in trade_info
            trade_info_format = _exception_text(trade_info)
            trade_info_format_s.append(trade_info_format)
            continue
        if ('BINANCE API ERROR' in trade_info or 'EXCEPTION' in trade_info) and 'BALANCE' in trade_info:
            trade_info_format = _exit_failed_text(trade_info)
            trade_info_format_s.append(trade_info_format)
            continue

        trade_info_format = _long_in_text(trade_info)
        trade_info_format = _short_in_text(trade_info_format)
        trade_info_format = _profit_out_text(trade_info_format)
        trade_info_format = _loss_out_text(trade_info_format)
        trade_info_format = _loss_short_text(trade_info_format)
        trade_info_format = _profit_short_text(trade_info_format)
        trade_info_format = _loss_long_text(trade_info_format)
        trade_info_format = _profit_long_text(trade_info_format)
        trade_info_format = _profit_exception_text(trade_info_format)
        trade_info_format = _loss_exception_text(trade_info_format)

        trade_info_format_s.append(trade_info_format)

    return trade_info_format_s


def produce_trade_info_str(transactions_info, ignore_field_s=['last_correlation_id']):
    original_transactions_info = transactions_info
    if transactions_info is None:
        return "NO TRADE INFO YET"

    if "DETAILS" in transactions_info:
        return transactions_info["DETAILS"]

    ignore_field_s.append('L/P')
    ignore_field_s.append('profit_loss')
    ignore_field_s.append('boost_fail')
    ignore_field_s.append('epd')

    for ignore_field in ignore_field_s:
        transactions_info = filter_dict(transactions_info, ignore_field)

    boost_fail = original_transactions_info['boost_fail']
    profit_loss = original_transactions_info['profit_loss'] if 'profit_loss' in original_transactions_info else 1 / original_transactions_info['L/P']
    earn_per_deal_percent = original_transactions_info['epd']

    boost_fail_title = (f'BOOST' if boost_fail > 1 else 'FAIL' if boost_fail < 1 else 'NEUTRAL') if isinstance(boost_fail, float) or isinstance(boost_fail, int) else 'X'
    boost_fail_suffix = f"{boost_fail_title} x<b>{boost_fail}</b>" if isinstance(boost_fail, int) or isinstance(boost_fail, float) else "x<b>XXX</b>"
    profit_loss_title = f'P/L: <b>{_float_3(profit_loss)}</b>'
    earn_per_deal_percent_title = f"{earn_per_deal_percent}%"

    transactions_info_str = ' | '.join([f"{key}: <b>{val}</b>" for key, val in transactions_info.items()])
    transactions_info_str = f"{boost_fail_suffix} || {profit_loss_title} | {earn_per_deal_percent_title} | {transactions_info_str}"

    return transactions_info_str


def get_scoring_bg_color(metric_value):
    if metric_value == 0:
        return 'neutral'

    if metric_value == 1:
        return 'ignore'

    if metric_value > 1:
        return 'profit'
    else:
        return 'loss'


if __name__ == "__main__":
    pass