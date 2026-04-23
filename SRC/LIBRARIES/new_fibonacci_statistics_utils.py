import pandas as pd
import time

def calculate_current_level_profit_usd(size, price_end, price_start, commission_rate):
    """
    Точный расчет прибыли для одной позиции с учетом комиссий.
    Использует последовательное вычисление: покупка, затем продажа.
    """
    # Покупка: списываем комиссию с депозита
    amount_invested = size
    commission_buy = amount_invested * commission_rate
    net_invested = amount_invested - commission_buy
    quantity = net_invested / price_start

    # Продажа: выручка и комиссия
    gross_revenue = quantity * price_end
    commission_sell = gross_revenue * commission_rate
    net_revenue = gross_revenue - commission_sell

    profit = net_revenue - amount_invested
    commission_total = commission_buy + commission_sell
    return profit, commission_total

def calculate_stop_loss_level_loss_usd(level_prices, sl_level, col, commission_rate, position_sizes):
    """
    Убыток по позиции, открытой на уровне col и закрытой по стоп-лоссу sl_level.
    """
    price_start = level_prices[col]
    price_end = level_prices[sl_level]
    size = position_sizes[col]
    # Покупка
    commission_buy = size * commission_rate
    net_invested = size - commission_buy
    quantity = net_invested / price_start
    # Продажа
    gross_revenue = quantity * price_end
    commission_sell = gross_revenue * commission_rate
    net_revenue = gross_revenue - commission_sell
    loss = net_revenue - size  # отрицательное число
    return loss

def calculate_stop_loss_level_commission_usd(level_prices, sl_level, col, commission_rate, position_sizes):
    """
    Комиссия по позиции, закрытой по стоп-лоссу.
    """
    price_start = level_prices[col]
    size = position_sizes[col]
    commission_buy = size * commission_rate
    quantity = (size - commission_buy) / price_start
    price_end = level_prices[sl_level]
    commission_sell = (quantity * price_end) * commission_rate
    return commission_buy + commission_sell

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours} h {minutes} min {secs:.0f} sec"
    elif minutes > 0:
        return f"{minutes} min {secs:.0f} sec"
    else:
        return f"{secs:.0f} sec"

def analyze_target_candle_outcome(df_trading, df_1m, target_idx, trading_timeframe, levels, sl_level, commission_rate, position_sizes, capital_per_trade, use_stop_loss):
    import math

    target_data = df_trading.loc[[target_idx]]
    high = target_data['high'].iloc[0]
    low = target_data['low'].iloc[0]

    # логарифмическая шкала от минимума к максимуму
    log_start = math.log(low)
    log_end = math.log(high)
    log_range = log_end - log_start

    all_levels = [0] + levels + [sl_level]
    level_prices = {level: math.exp(log_start + log_range * level) for level in all_levels}

    # Определяем начало анализа после закрытия ЦС
    if trading_timeframe == '1D':
        start_analysis = target_idx + pd.Timedelta(days=1)
    else:
        interval_minutes = {'1M':1,'5M':5,'15M':15,'30M':30,'1H':60,'4H':240,'8H':480}.get(trading_timeframe,1)
        start_analysis = target_idx + pd.Timedelta(minutes=interval_minutes)

    df_1m_after = df_1m.loc[start_analysis:].copy()
    if len(df_1m_after) == 0:
        return {'status':'in_progress','level':None,'timestamp':None,
                'fill_start_level':None,'fill_end_level':None,
                'profit_usd':0,'profit_pct':0,'level_return_pct':0,
                'commission_usd':0,'details':'No data after TC'}

    current_deep_level = None
    level_index = -1

    for idx, row in df_1m_after.iterrows():
        current_high = row['high']
        current_low = row['low']

        # Проверяем достижение нового глубокого уровня
        for i, lvl in enumerate(levels):
            if current_low <= level_prices[lvl] <= current_high:
                if current_deep_level is None or lvl < current_deep_level:
                    current_deep_level = lvl
                    level_index = i

        # Успех: разворот от глубокого уровня к предыдущему
        if current_deep_level is not None:
            if level_index == 0:
                upper_level = 0
            else:
                upper_level = levels[level_index-1]

            upper_price = level_prices[upper_level]
            if current_low <= upper_price <= current_high:
                price_start = level_prices[current_deep_level]
                price_end = level_prices[upper_level]
                price_change_abs = price_end - price_start
                price_change_pct = (price_change_abs / price_start) * 100

                profit_usd = 0.0
                commission_usd = 0.0

                # Прибыль от самого глубокого уровня
                size_deep = position_sizes[current_deep_level]
                profit_deep, comm_deep = calculate_current_level_profit_usd(
                    size_deep, price_end, price_start, commission_rate
                )
                profit_usd += profit_deep
                commission_usd += comm_deep

                # Убытки по всем более ранним уровням (закрываются на upper_level)
                for j in range(level_index):
                    lvl_j = levels[j]
                    size_j = position_sizes[lvl_j]
                    price_close_j = level_prices[upper_level]
                    price_open_j = level_prices[lvl_j]
                    loss_j, comm_j = calculate_current_level_profit_usd(
                        size_j, price_close_j, price_open_j, commission_rate
                    )
                    profit_usd += loss_j
                    commission_usd += comm_j

                return {
                    'status': 'success',
                    'level': current_deep_level,
                    'fill_start_level': current_deep_level,
                    'fill_end_level': upper_level,
                    'timestamp': idx,
                    'price_start': price_start,
                    'price_end': price_end,
                    'price_change_abs': price_change_abs,
                    'price_change_pct': price_change_pct,
                    'level_return_pct': price_change_pct,
                    'profit_usd': profit_usd,
                    'profit_pct': (profit_usd / capital_per_trade) * 100,
                    'commission_usd': commission_usd,
                    'details': f'Price reached {current_deep_level}, returned to {upper_level}'
                }

        # Стоп-лосс
        if use_stop_loss and current_low <= level_prices[sl_level] <= current_high:
            last_level = current_deep_level if current_deep_level is not None else 0
            price_start = level_prices[last_level]
            price_end = level_prices[sl_level]
            price_change_abs = price_end - price_start
            price_change_pct = (price_change_abs / price_start) * 100

            loss_usd = 0.0
            commission_usd = 0.0
            for lvl in levels:
                loss = calculate_stop_loss_level_loss_usd(level_prices, sl_level, lvl, commission_rate, position_sizes)
                comm = calculate_stop_loss_level_commission_usd(level_prices, sl_level, lvl, commission_rate, position_sizes)
                loss_usd += loss
                commission_usd += comm

            return {
                'status': 'failure',
                'level': sl_level,
                'fill_start_level': last_level,
                'fill_end_level': sl_level,
                'timestamp': idx,
                'price_start': price_start,
                'price_end': price_end,
                'price_change_abs': price_change_abs,
                'price_change_pct': price_change_pct,
                'level_return_pct': price_change_pct,
                'profit_usd': loss_usd,
                'profit_pct': (loss_usd / capital_per_trade) * 100,
                'commission_usd': commission_usd,
                'details': f'Stop Loss at {sl_level}'
            }

    return {'status':'in_progress','level':None,'timestamp':None,
            'fill_start_level':None,'fill_end_level':None,
            'profit_usd':0,'profit_pct':0,'level_return_pct':0,
            'commission_usd':0,'details':'Analysis not completed'}

def collect_statistics(df, df_1m, target_indices, trading_timeframe, start_time, levels, sl_level, commission_rate, position_sizes, capital_per_trade, use_stop_loss):
    """
    Собирает статистику по всем целевым свечам, включая комиссии.
    Учитывает только завершённые сделки (success и failure) для расчёта средней комиссии.
    """
    stats = {
        'total': len(target_indices),
        'success': 0,
        'failure': 0,
        'in_progress': 0,
        'by_level': {},
        'by_level_avg_pct': {},
        'by_level_avg_abs': {},
        'by_level_avg_profit_usd': {},
        'by_level_avg_profit_pct': {},
        'stop_loss': 0,
        'total_avg_pct': 0,
        'total_avg_abs': 0,
        'total_avg_profit_usd': 0,
        'total_avg_profit_pct': 0,
        'total_avg_loss_usd': 0,
        'total_avg_loss_pct': 0,
        'total_profit_sum': 0,
        'total_loss_sum': 0,
        'net_profit': 0,
        'profit_factor': None,
        'total_commission_usd': 0,
        'avg_commission_per_trade_usd': 0
    }

    # Временные хранилища
    level_pct_sum = {}
    level_abs_sum = {}
    level_count = {}
    level_profit_usd_sum = {}
    level_profit_pct_sum = {}

    total_pct_sum = 0
    total_abs_sum = 0
    total_profit_usd_sum = 0
    total_success_count = 0
    total_loss_usd_sum = 0
    total_loss_count = 0
    total_commission = 0.0
    total_completed_trades = 0

    target_indices_len = len(target_indices)
    i = 0

    for idx in target_indices:
        i += 1
        print(f"\r{i} of {target_indices_len} target candles. Running time: {format_duration(time.perf_counter() - start_time)}", end='', flush=True)
        outcome = analyze_target_candle_outcome(df, df_1m, idx, trading_timeframe, levels, sl_level, commission_rate, position_sizes, capital_per_trade, use_stop_loss)

        if outcome['status'] == 'success':
            stats['success'] += 1
            level = outcome['level']
            pct_change = outcome['price_change_pct']
            abs_change = outcome['price_change_abs']
            profit_usd = outcome.get('profit_usd', 0)
            profit_pct = outcome.get('profit_pct', 0)
            commission = outcome.get('commission_usd', 0)

            total_commission += commission
            total_completed_trades += 1

            if level not in stats['by_level']:
                stats['by_level'][level] = 0
                level_pct_sum[level] = 0
                level_abs_sum[level] = 0
                level_profit_usd_sum[level] = 0
                level_profit_pct_sum[level] = 0
                level_count[level] = 0
            stats['by_level'][level] += 1
            level_pct_sum[level] += pct_change
            level_abs_sum[level] += abs_change
            level_profit_usd_sum[level] += profit_usd
            level_profit_pct_sum[level] += profit_pct
            level_count[level] += 1

            total_pct_sum += pct_change
            total_abs_sum += abs_change
            total_profit_usd_sum += profit_usd
            total_success_count += 1

        elif outcome['status'] == 'failure':
            stats['failure'] += 1
            stats['stop_loss'] += 1
            loss_usd = outcome.get('profit_usd', 0)
            commission = outcome.get('commission_usd', 0)

            total_commission += commission
            total_completed_trades += 1

            total_loss_usd_sum += loss_usd
            total_loss_count += 1

        elif outcome['status'] == 'in_progress':
            stats['in_progress'] += 1

    # Средние по уровням (только для успешных)
    for level in level_count:
        stats['by_level_avg_pct'][level] = level_pct_sum[level] / level_count[level]
        stats['by_level_avg_abs'][level] = level_abs_sum[level] / level_count[level]
        stats['by_level_avg_profit_usd'][level] = level_profit_usd_sum[level] / level_count[level]
        stats['by_level_avg_profit_pct'][level] = level_profit_pct_sum[level] / level_count[level]

    if total_success_count > 0:
        stats['total_avg_pct'] = total_pct_sum / total_success_count
        stats['total_avg_abs'] = total_abs_sum / total_success_count
        stats['total_avg_profit_usd'] = total_profit_usd_sum / total_success_count
        stats['total_avg_profit_pct'] = (total_profit_usd_sum / (total_success_count * capital_per_trade)) * 100

    if total_loss_count > 0:
        stats['total_avg_loss_usd'] = total_loss_usd_sum / total_loss_count
        stats['total_avg_loss_pct'] = (total_loss_usd_sum / (total_loss_count * capital_per_trade)) * 100

    stats['total_profit_sum'] = total_profit_usd_sum
    stats['total_loss_sum'] = total_loss_usd_sum
    stats['net_profit'] = total_profit_usd_sum + total_loss_usd_sum

    if total_loss_usd_sum != 0:
        stats['profit_factor'] = abs(total_profit_usd_sum / total_loss_usd_sum)

    stats['total_commission_usd'] = total_commission
    if total_completed_trades > 0:
        stats['avg_commission_per_trade_usd'] = total_commission / total_completed_trades

    stats['duration'] = time.perf_counter() - start_time

    return stats

def print_statistics(stats, symbol, discretization, display_start_date_str, load_end_date, measure_percentile, use_candle_size_instead_of_shadow, filter_by_measure_range, measure_lower_mult, measure_upper_mult, use_mrc, use_mrc_r2, use_mrc_s2, use_stop_loss, sl_level, position_sizes, capital_per_trade, commission_rate):
    print("\n" + "="*60)
    print("📊 TARGET CANDLE STATISTICS")
    print("="*60)
    print(f"\n📌 GENERAL STATISTICS:")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {discretization}")
    print(f"   Start date: {display_start_date_str}")
    print(f"   End date: {load_end_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Duration: {format_duration(stats['duration'])}")
    print(f"   Total target candles: {stats['total']}")
    print(f"   Measure percentile: {measure_percentile}")
    print(f"   Use candle size instead of shadow: {use_candle_size_instead_of_shadow}")
    print(f"   Filter by measure range: {filter_by_measure_range}" + (f", lower mult: {measure_lower_mult}, upper mult: {measure_upper_mult}" if filter_by_measure_range else ''))
    print(f"   Use MRC: {use_mrc}")
    if use_mrc:
        print(f"   Use MRC R2: {use_mrc_r2}")
        print(f"   Use MRC S2: {use_mrc_s2}")
    print(f"   Use SL: {use_stop_loss}")
    if use_stop_loss:
        print(f"   SL level: {sl_level}")
    print(f"\n📊 POSITION SIZES:")
    for level, size in position_sizes.items():
        print(f"   Level {level:.3f}: ${size:.2f} ({size/capital_per_trade*100:.1f}%)")
    print(f"\n💰 CAPITAL & FEES:")
    print(f"   Capital per trade: ${capital_per_trade}")
    print(f"   Commission rate: {commission_rate*100:.3f}% per transaction")
    print(f"\n📈 RESULTS:")
    completed = stats['success'] + stats['failure']
    print(f"   ✅ SUCCESSFUL: {stats['success']} ({stats['success']/stats['total']*100:.1f}% of total, {stats['success']/completed*100:.1f}% of completed)" if completed else "   ✅ SUCCESSFUL: 0")
    print(f"   ❌ UNSUCCESSFUL: {stats['failure']} ({stats['failure']/stats['total']*100:.1f}% of total, {stats['failure']/completed*100:.1f}% of completed)" if completed else "   ❌ UNSUCCESSFUL: 0")
    print(f"   🔄 IN PROGRESS: {stats['in_progress']} ({stats['in_progress']/stats['total']*100:.1f}%)")
    # остальные строки без изменений...
    print(f"\n🎯 SUCCESS BY LEVEL:")
    for level in reversed(sorted(stats['by_level'].keys())):
        count = stats['by_level'][level]
        avg_usd = stats['by_level_avg_profit_usd'][level]
        avg_pct = stats['by_level_avg_profit_pct'][level]
        print(f"   Level {level:.3f}: {count} ({count/stats['success']*100:.1f}% of successful) | 📊 Avg profit: ${avg_usd:.2f} ({avg_pct:.2f}%)")
    print(f"\n📊 AVERAGE SUCCESS METRICS:")
    print(f"   📈 Avg profit per success: ${stats['total_avg_profit_usd']:.2f} ({stats['total_avg_profit_pct']:.2f}%)")
    print(f"\n🛑 FAILURES:")
    print(f"   Stop Loss: {stats['stop_loss']}")
    if stats['total_avg_loss_usd'] != 0:
        print(f"   📉 Avg loss per failure: ${stats['total_avg_loss_usd']:.2f} ({stats['total_avg_loss_pct']:.2f}%)")
    print(f"\n💰 FINANCIAL SUMMARY:")
    print(f"   Total profit: ${stats['total_profit_sum']:.2f}")
    print(f"   Total loss: ${stats['total_loss_sum']:.2f}")
    print(f"   Net profit: ${stats['net_profit']:.2f}")
    if stats['profit_factor'] is not None:
        print(f"   Profit Factor: {stats['profit_factor']:.2f}")
    else:
        print(f"   Profit Factor: N/A (no losses)")
    print(f"\n💸 COMMISSIONS:")
    print(f"   Total commissions paid: ${stats['total_commission_usd']:.2f}")
    if stats['total'] > 0:
        print(f"   Avg commission per trade: ${stats['avg_commission_per_trade_usd']:.2f}")
    print("\n" + "="*60)

def analyze_and_print_statistics(df, df_1m, start_time, levels, sl_level, commission_rate, position_sizes, capital_per_trade, use_stop_loss, symbol, discretization, display_start_date_str, load_end_date, measure_percentile, use_candle_size_instead_of_shadow, filter_by_measure_range, measure_lower_mult, measure_upper_mult, use_mrc, use_mrc_r2, use_mrc_s2):
    target_candle_indices = df[df['is_target_candle']].index.tolist()
    if not target_candle_indices:
        print("There are no target candles for analysis")
        return
    stats = collect_statistics(df, df_1m, target_candle_indices, discretization, start_time, levels, sl_level, commission_rate, position_sizes, capital_per_trade, use_stop_loss)
    print_statistics(stats, symbol, discretization, display_start_date_str, load_end_date, measure_percentile, use_candle_size_instead_of_shadow, filter_by_measure_range, measure_lower_mult, measure_upper_mult, use_mrc, use_mrc_r2, use_mrc_s2, use_stop_loss, sl_level, position_sizes, capital_per_trade, commission_rate)
    return stats