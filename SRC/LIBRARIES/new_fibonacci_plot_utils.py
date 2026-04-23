import plotly.graph_objects as go
from SRC.CORE._CONSTANTS import _KIEV_TIMESTAMP
import pandas as pd
from LIBRARIES.time_utils import as_kiev_tz
from SRC.CORE.utils import featurize_lambda
from SRC.CORE._CONSTANTS import _UTC_TIMESTAMP
import LIBRARIES.new_fibonacci_statistics_utils as nfsu
from plotly.subplots import make_subplots

def get_display_timeframe(trading_timeframe):
    """
    Возвращает таймфрейм для отображения графика в зависимости от торгового таймфрейма
    """
    mapping = {
        '1D': '1H',
        '8H': '30M',
        '4H': '15M',
        '1H': '5M',
        '30M': '1M',
        '15M': '1M',
        '5M': '1M'
    }
    return mapping.get(trading_timeframe, '1M')

def resample_to_timeframe(df_1m, timeframe):
    """
    Преобразует 1-минутные данные в указанный таймфрейм
    """
    interval_minutes = {
        '1M': 1,
        '5M': 5,
        '15M': 15,
        '30M': 30,
        '1H': 60,
        '4H': 240,
        '8H': 480,
        '1D': 1440
    }.get(timeframe, 60)

    if timeframe == '1M':
        return df_1m

    # Убираем часовой пояс
    if df_1m.index.tz is not None:
        df_no_tz = df_1m.tz_localize(None)
        original_tz = df_1m.index.tz
    else:
        df_no_tz = df_1m
        original_tz = None

    # Используем стандартный resample вместо groupby
    rule = f'{interval_minutes}T'
    df_resampled = df_no_tz.resample(rule, closed='left', label='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Возвращаем часовой пояс
    if original_tz is not None:
        df_resampled.index = df_resampled.index.tz_localize(original_tz)

    df_resampled[_UTC_TIMESTAMP] = df_resampled.index
    df_resampled = featurize_lambda(df_resampled, _UTC_TIMESTAMP, _KIEV_TIMESTAMP, lambda utc_ts: as_kiev_tz(utc_ts))

    return df_resampled

def add_target_candle_scatter(position, position_multiplier, name, color, symbol_direction, df, fig, candlestick_row):
    signals = df[df['is_target_candle']]
    tcs_count = len(signals)

    if tcs_count > 0:
        fig.add_trace(
            go.Scatter(
                x=signals[_KIEV_TIMESTAMP],
                y=signals[position] * position_multiplier,
                name=name + " TC",
                mode='markers',
                marker=dict(color=color, size=10, symbol='triangle-' + symbol_direction)
            ),
            row=candlestick_row, col=1
        )

    return tcs_count

def add_fibonacci_levels(fig, target_data, target_idx, df_subset, levels, sl_level, row=1, col=1, fill_start_level=None, fill_end_level=None, fill_status=None):
    import math
    high = target_data['high'].iloc[0]
    low = target_data['low'].iloc[0]

    log_start = math.log(low)
    log_end = math.log(high)
    log_range = log_end - log_start

    # Базовые уровни: 0, 1 и все торговые уровни + стоп-лосс
    all_display_levels = [0, 1] + levels + [sl_level]
    # Цвета (можно настроить)
    colors_map = {0: 'yellow', 1: 'yellow'}
    color_palette = ['violet', 'lavender', 'orange', 'pink', 'aqua']
    for i, lvl in enumerate(levels):
        colors_map[lvl] = color_palette[i % len(color_palette)]
    colors_map[sl_level] = 'aqua'

    prices = {lvl: math.exp(log_start + log_range * lvl) for lvl in all_display_levels}

    if len(df_subset.index) >= 2:
        time_offset = df_subset.index[0] - (df_subset.index[1] - df_subset.index[0])
    else:
        time_offset = df_subset.index[0] - pd.Timedelta(hours=1)

    end_date = df_subset.index[-1]

    # Заливка (если есть)
    if fill_start_level is not None and fill_end_level is not None:
        if fill_start_level in prices and fill_end_level in prices:
            y_lower = min(prices[fill_start_level], prices[fill_end_level])
            y_upper = max(prices[fill_start_level], prices[fill_end_level])
            fill_color = 'red' if fill_status == 'failure' else colors_map.get(fill_start_level, 'gray')
            fig.add_trace(
                go.Scatter(
                    x=[as_kiev_tz(target_idx), as_kiev_tz(end_date), as_kiev_tz(end_date), as_kiev_tz(target_idx)],
                    y=[y_lower, y_lower, y_upper, y_upper],
                    fill='toself',
                    mode='lines',
                    name=f"Fill {fill_start_level}→{fill_end_level}",
                    line=dict(width=0),
                    fillcolor=fill_color,
                    opacity=0.3,
                    showlegend=False
                ),
                row=row, col=col
            )

    # Линии и текст
    for lvl, p in prices.items():
        fig.add_trace(
            go.Scatter(
                x=[as_kiev_tz(target_idx), as_kiev_tz(end_date)],
                y=[p, p],
                mode='lines',
                name=f"Fib {lvl:.3f}: {p:.2f}",
                line=dict(color=colors_map[lvl], width=1),
                showlegend=True
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=[as_kiev_tz(time_offset)],
                y=[p],
                mode='text',
                text=[f"{lvl:.3f} ({p:.2f})"],
                textposition="middle left",
                textfont=dict(size=12, color=colors_map[lvl], family="Arial"),
                showlegend=False,
                hoverinfo='none'
            ),
            row=row, col=col
        )

    return fig

def plot_target_candle_with_fib_multiframe(df_trading, df_1m, target_idx, future_bars, trading_timeframe, levels, sl_level, commission_rate, position_sizes, capital_per_trade, use_stop_loss):
    """
    Строит график для целевой свечи
    """
    import math

    display_timeframe = get_display_timeframe(trading_timeframe)
    df_display = resample_to_timeframe(df_1m, display_timeframe)

    # Определяем начало отображения в зависимости от торгового ТФ
    if trading_timeframe == '1D':
        next_day = target_idx + pd.Timedelta(days=1)
        mask = df_display.index >= next_day
        if mask.any():
            target_idx_display = df_display.index[mask][0]
        else:
            target_idx_display = target_idx
    else:
        interval_minutes = {
            '1M': 1, '5M': 5, '15M': 15, '30M': 30, '1H': 60, '4H': 240, '8H': 480
        }.get(trading_timeframe, 1)
        candle_close = target_idx + pd.Timedelta(minutes=interval_minutes)
        mask = df_display.index >= candle_close
        if mask.any():
            target_idx_display = df_display.index[mask][0]
        else:
            target_idx_display = target_idx

    if target_idx_display not in df_display.index:
        target_pos = df_display.index.get_indexer([target_idx_display], method='nearest')[0]
        target_idx_display = df_display.index[target_pos]
        print(f"Adjusted target_idx_display: {target_idx_display}")

    target_data = df_trading.loc[[target_idx]]
    outcome = nfsu.analyze_target_candle_outcome(df_trading, df_1m, target_idx, trading_timeframe, levels, sl_level, commission_rate, position_sizes, capital_per_trade, use_stop_loss)

    # Определяем диапазон для отображения
    if outcome['status'] in ['success', 'failure'] and outcome['timestamp'] is not None:
        result_ts = outcome['timestamp']

        if df_display.index.tz is not None and result_ts.tz is None:
            result_ts = result_ts.tz_localize(df_display.index.tz)
        elif df_display.index.tz is None and result_ts.tz is not None:
            result_ts = result_ts.tz_localize(None)

        df_after_target = df_display.loc[target_idx_display:]
        mask = df_after_target.index <= result_ts
        if mask.any():
            end_dt = df_after_target.index[mask][-1]
        else:
            end_dt = target_idx_display
    else:
        target_pos = df_display.index.get_loc(target_idx_display) if target_idx_display in df_display.index else 0
        end_pos = min(len(df_display) - 1, target_pos + future_bars)
        end_dt = df_display.index[end_pos]

    if end_dt <= target_idx_display:
        next_pos = df_display.index.get_loc(target_idx_display) + 1
        if next_pos < len(df_display):
            end_dt = df_display.index[next_pos]
        else:
            end_dt = target_idx_display

    df_subset = df_display.loc[target_idx_display:end_dt].copy()

    if len(df_subset) < 2:
        current_pos = df_display.index.get_loc(end_dt)
        next_pos = current_pos + 1
        if next_pos < len(df_display):
            end_dt = df_display.index[next_pos]
            df_subset = df_display.loc[target_idx_display:end_dt].copy()
            print(f"Extended df_subset to {len(df_subset)} candles (was 1)")

    fig_target_candle = make_subplots(rows=1, cols=1)

    fig_target_candle.add_trace(
        go.Candlestick(
            x=df_subset[_KIEV_TIMESTAMP],
            open=df_subset["open"],
            high=df_subset["high"],
            low=df_subset["low"],
            close=df_subset["close"],
            name=f"OHLC ({display_timeframe})",
            showlegend=True
        ),
        row=1, col=1
    )

    # Информация о комиссии
    commission_usd = outcome.get('commission_usd', 0)
    commission_info = f" | 💸 Commission: ${commission_usd:.2f} (Parameter: {commission_rate*100:.3f}%)"

    date_str = as_kiev_tz(target_idx).strftime('%Y-%m-%d %H:%M') if hasattr(as_kiev_tz(target_idx), 'strftime') else str(as_kiev_tz(target_idx))

    if outcome['status'] == 'success':
        result_str = f"✅ SUCCESS (level: {outcome['level']:.3f})"
        result_time = as_kiev_tz(outcome['timestamp']).strftime('%Y-%m-%d %H:%M') if hasattr(as_kiev_tz(outcome['timestamp']), 'strftime') else str(as_kiev_tz(outcome['timestamp']))
        level_return = outcome.get('level_return_pct', 0)
        profit_info = f" | 💰 Profit: ${outcome['profit_usd']:.2f} ({outcome['profit_pct']:.2f}%)"
        title = f"Trading TF: {trading_timeframe} | Display TF: {display_timeframe} | 🎯 TC: {date_str} | {result_str} | 📈 Level return: {level_return:.2f}% | 📍 Result: {result_time}{profit_info}{commission_info}"

    elif outcome['status'] == 'failure':
        result_str = f"❌ FAILURE (SL: {outcome['level']:.3f})"
        result_time = as_kiev_tz(outcome['timestamp']).strftime('%Y-%m-%d %H:%M') if hasattr(as_kiev_tz(outcome['timestamp']), 'strftime') else str(as_kiev_tz(outcome['timestamp']))
        level_return = outcome.get('level_return_pct', 0)
        loss_info = f" | 💰 Loss: ${outcome['profit_usd']:.2f} ({outcome['profit_pct']:.2f}%)"
        title = f"Trading TF: {trading_timeframe} | Display TF: {display_timeframe} | 🎯 TC: {date_str} | {result_str} | 📈 Level return: {level_return:.2f}% | 📍 Result: {result_time}{loss_info}{commission_info}"

    elif outcome['status'] == 'in_progress':
        title = f"Trading TF: {trading_timeframe} | Display TF: {display_timeframe} | 🎯 TC: {date_str} | 🔄 IN PROGRESS"
    else:
        title = f"Trading TF: {trading_timeframe} | Display TF: {display_timeframe} | 🎯 TC: {date_str} | ⚪ UNCERTAIN"

    fig_target_candle = add_fibonacci_levels(
        fig_target_candle, target_data, target_idx_display, df_subset, levels, sl_level, row=1, col=1,
        fill_start_level=outcome.get('fill_start_level'),
        fill_end_level=outcome.get('fill_end_level'),
        fill_status=outcome['status'],
    )

    if sl_level is not None:
        high = target_data['high'].iloc[0]
        low = target_data['low'].iloc[0]
        log_start = math.log(low)
        log_end = math.log(high)

        log_range = log_end - log_start
        sl_price = math.exp(log_start + log_range * sl_level)

        fig_target_candle.add_trace(
            go.Scatter(
                x=[as_kiev_tz(target_idx_display), as_kiev_tz(df_subset.index[-1])],
                y=[sl_price, sl_price],
                mode='lines',
                name=f"🛑 Stop Loss ({sl_level})",
                line=dict(color='crimson', width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )

        fig_target_candle.add_trace(
            go.Scatter(
                x=[as_kiev_tz(target_idx_display)],
                y=[sl_price],
                mode='text',
                text=[f"🛑"],
                textposition="middle right",
                textfont=dict(size=11, color='red', family="Arial"),
                showlegend=False,
                hoverinfo='none'
            ),
            row=1, col=1
        )

    if outcome['status'] in ['success', 'failure'] and outcome['timestamp'] is not None:
        result_dt = outcome['timestamp']

        if df_subset.index.tz is not None and result_dt.tz is None:
            result_dt = result_dt.tz_localize(df_subset.index.tz)
        elif df_subset.index.tz is None and result_dt.tz is not None:
            result_dt = result_dt.tz_localize(None)

        indices = df_subset.index.tolist()
        target_candle_idx = None
        for idx in indices:
            if idx <= result_dt:
                target_candle_idx = idx
            else:
                break

        if target_candle_idx is not None:
            fig_target_candle.add_vline(
                x=as_kiev_tz(target_candle_idx),
                line_dash="dash",
                line_color='lime' if outcome['status'] == 'success' else 'red',
                line_width=0.4,
                opacity=1,
                row=1, col=1
            )

    fig_target_candle.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        yaxis_type="log",
        yaxis_title="Price (log scale)",
        template="plotly_dark",
        height=1700,
        showlegend=True,
        hovermode='x unified'
    )

    return fig_target_candle

def plot_all_target_candles_multiframe_and_save_htmls_to_disc(df_trading, df_1m, trading_timeframe, levels, sl_level, commission_rate, position_sizes, capital_per_trade, use_stop_loss, future_bars=1000, output_dir="target_candles", clean_dir=True):
    """
    Сохраняет графики для всех целевых свечей с учетом мультитаймфрейма
    """
    import shutil
    import os

    target_candle_indices = df_trading[df_trading['is_target_candle']].index.tolist()

    if not target_candle_indices:
        print("No target candles to display")
        return

    if clean_dir and os.path.exists(output_dir):
        print(f"Cleaning folder '{output_dir}'...")
        shutil.rmtree(output_dir)
        print(f"Folder '{output_dir}' deleted")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Folder '{output_dir}' created")
    print(f"Trading timeframe: {trading_timeframe}")
    print(f"Display timeframe: {get_display_timeframe(trading_timeframe)}")
    print(f"Total target candles: {len(target_candle_indices)}")

    for i, target_idx in enumerate(target_candle_indices):
        date_str = as_kiev_tz(target_idx).strftime('%Y-%m-%d_%H-%M') if hasattr(as_kiev_tz(target_idx), 'strftime') else str(as_kiev_tz(target_idx))
        filename = f"{output_dir}/target_{i+1:03d}_{trading_timeframe}_{date_str}.html"

        fig = plot_target_candle_with_fib_multiframe(
            df_trading, df_1m, target_idx, future_bars, trading_timeframe, levels, sl_level, commission_rate, position_sizes, capital_per_trade, use_stop_loss
        )
        fig.write_html(filename)

    print(f"\n✅ All {len(target_candle_indices)} charts saved to folder '{output_dir}'")
    print(f"   Open the files in your browser to view")