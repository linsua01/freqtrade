# --- Do not remove these libs ---
from datetime import datetime
from functools import reduce
from typing import Optional, Union
from freqtrade.persistence import Trade
from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy, stoploss_from_open
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

DEBUG = False

class ERMD_Strategy_total_risk(IStrategy):
    INTERFACE_VERSION: int = 3

    can_short = False  # enable short direction
    timeframe = "1d"
    startup_candle_count = 300  # how many candles we need to skip to start receiving robust signals
    risk = 0.01
    min_pos_pct = 0.01      # minimum position size, percent of current portfolio value
    max_pos_pct = DecimalParameter(0.01, 0.5, default=0.5, decimals=2, optimize=True, space="buy")         # maximum position size, percent of current portfolio value

    # stop loss
    stoploss = -0.333                # maximum stop loss distance
    tsl_break_even = DecimalParameter(0.0, 0.1, default=0.06, decimals=2, optimize=True, space="buy")       # place break-even stop loss when desired profit is reached
    tsl_start_offset = DecimalParameter(0.0, 0.1, default=0.036, decimals=3, optimize=True, space="buy")    # start trailing stop loss when this profit level is reached
    tsl_trailing_distance = DecimalParameter(0.01, 0.2, default=0.17, decimals=2, optimize=True, space="buy")      # distance to price

    exit_profit_only = False
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # take profit
    tp1_dist = 0.2
    tp1_size = 0.3
    tp2_dist = 0.4
    tp2_size = 0.3
    tp3_dist = 0.6
    tp3_size = 0.4
    minimal_roi = {
        "0": 0.159,
        "4377": 0.09,
        "21282": 0.041,
        "28618": 0}       # optimized tp

    # Parameters: signal
    buy_ema_enabled = BooleanParameter(default=False, optimize=True)
    buy_ema1_length = IntParameter(3, 100, default=6, optimize=True)
    buy_ema2_length = IntParameter(3, 100, default=64, optimize=True)
    buy_ema3_length = IntParameter(3, 100, default=76, optimize=True)
    buy_rsi_enabled = BooleanParameter(default=False, optimize=True)
    buy_rsi_length = IntParameter(3, 100, default=39, optimize=True)
    buy_rsi_threshold = IntParameter(50, 100, default=52, optimize=True)
    buy_macd_enabled = BooleanParameter(default=True, optimize=True)
    buy_macd_fast = IntParameter(3, 50, default=28, optimize=True)
    buy_macd_slow = IntParameter(20, 200, default=100, optimize=True)
    buy_macd_smooth = IntParameter(1, 30, default=8, optimize=True)
    buy_candlestick_enabled = BooleanParameter(default=False, optimize=True)

    sell_ema_enabled = BooleanParameter(default=False, optimize=True)
    sell_ema1_length = IntParameter(3, 100, default=70, optimize=True)
    sell_ema2_length = IntParameter(3, 100, default=84, optimize=True)
    sell_ema3_length = IntParameter(3, 100, default=33, optimize=True)
    sell_rsi_enabled = BooleanParameter(default=False, optimize=True)
    sell_rsi_length = IntParameter(3, 100, default=46, optimize=True)
    sell_rsi_threshold = IntParameter(0, 50, default=7, optimize=True)
    sell_macd_enabled = BooleanParameter(default=True, optimize=True)
    sell_macd_fast = IntParameter(3, 50, default=8, optimize=True)
    sell_macd_slow = IntParameter(20, 200, default=113, optimize=True)
    sell_macd_smooth = IntParameter(1, 30, default=3, optimize=True)
    sell_candlestick_enabled = BooleanParameter(default=False, optimize=True)

    sell_willy_exit = BooleanParameter(default=True, space='sell', optimize=True)
    sell_high_line = IntParameter(-25, 0, default=0, space="sell", optimize=True)
    sell_low_line = IntParameter(-100, -75, default=-81, space="sell", optimize=True)
    sell_ema = IntParameter(7, 22, default=11, space='sell', optimize=True)
    sell_willy = IntParameter(15, 30, default=22, space='sell', optimize=True)

    # signals
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """ adds indicators to dataframe """

        # BUY
        # EMAs
        for val in self.buy_ema1_length.range:
            dataframe[f"buy_ema1_{val}"] = ta.EMA(dataframe, val)
        for val in self.buy_ema2_length.range:
            dataframe[f"buy_ema2_{val}"] = ta.EMA(dataframe, val)
        for val in self.buy_ema3_length.range:
            dataframe[f"buy_ema3_{val}"] = ta.EMA(dataframe, val)

        # RSI
        for val in self.buy_rsi_length.range:
            dataframe[f"buy_rsi_{val}"] = ta.RSI(dataframe, val)

        # MACD
        for fast_val in self.buy_macd_fast.range:
            for slow_val in self.buy_macd_slow.range:
                for smooth_val in self.buy_macd_smooth.range:
                    if smooth_val < fast_val < slow_val:
                        macd = ta.MACD(dataframe, fast_val, slow_val, smooth_val)
                        dataframe[f'buy_macd_{fast_val}_{slow_val}_{smooth_val}'] = macd['macd']
                        dataframe[f'buy_macds_{fast_val}_{slow_val}_{smooth_val}'] = macd['macdsignal']
                    else:
                        macd = ta.MACD(dataframe, 12, 26, 9)
                        dataframe[f'buy_macd_{fast_val}_{slow_val}_{smooth_val}'] = macd['macd']
                        dataframe[f'buy_macds_{fast_val}_{slow_val}_{smooth_val}'] = macd['macdsignal']


        # SELL
        # EMAs
        for val in self.sell_ema1_length.range:
            dataframe[f"sell_ema1_{val}"] = ta.EMA(dataframe, val)
        for val in self.sell_ema2_length.range:
            dataframe[f"sell_ema2_{val}"] = ta.EMA(dataframe, val)
        for val in self.sell_ema3_length.range:
            dataframe[f"sell_ema3_{val}"] = ta.EMA(dataframe, val)

        # RSI
        for val in self.sell_rsi_length.range:
            dataframe[f"sell_rsi_{val}"] = ta.RSI(dataframe, val)

        # MACD
        for fast_val in self.sell_macd_fast.range:
            for slow_val in self.sell_macd_slow.range:
                for smooth_val in self.sell_macd_smooth.range:
                    if smooth_val < fast_val < slow_val:
                        macd = ta.MACD(dataframe, fast_val, slow_val, smooth_val)
                        dataframe[f'sell_macd_{fast_val}_{slow_val}_{smooth_val}'] = macd['macd']
                        dataframe[f'sell_macds_{fast_val}_{slow_val}_{smooth_val}'] = macd['macdsignal']
                    else:
                        macd = ta.MACD(dataframe, 12, 26, 9)
                        dataframe[f'sell_macd_{fast_val}_{slow_val}_{smooth_val}'] = macd['macd']
                        dataframe[f'sell_macds_{fast_val}_{slow_val}_{smooth_val}'] = macd['macdsignal']


        for val in self.sell_willy.range:
            dataframe[f'sell_willy{val}'] = self.willy_ema(dataframe, low_line=-80,
                                                           up_line=-20, willyLen=val, emaLen=13)[
                'Willy']
        for val in self.sell_ema.range:
            dataframe[f'sell_ema_will{val}'] = \
                self.willy_ema(dataframe, low_line=-80, up_line=-20, willyLen=21, emaLen=val)[
                    'Ema_will']


        # DOJI candlestick pattern
        doji = ta.CDLDOJI(dataframe)
        doji_dir = (dataframe["high"] - dataframe["close"]) / (dataframe["close"] - dataframe["low"])
        dataframe.loc[((doji == 100) & (doji_dir > 1), "doji")] = dataframe["high"]
        dataframe.loc[((doji == 100) & (doji_dir < 1), "doji")] = dataframe["low"]

        # MARUBOZU candlestick pattern
        body = (dataframe["close"] - dataframe["open"]).abs()
        wick_top = dataframe["high"] - dataframe[["open", "close"]].max(axis=1)
        wick_bottom = dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
        marubozu = (wick_top/body < 0.25) & (wick_bottom/body < 0.25)
        dataframe.loc[(marubozu & (body > 0), 'marubozu')] = dataframe["high"]
        dataframe.loc[(marubozu & (body < 0), 'marubozu')] = dataframe["low"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """ generate signals """

        # BUY signal
        # only trade if there is volume
        conditions = [(dataframe['volume'] > 0)]

        # Close > EMA1 > EMA2 > EMA3
        if self.buy_ema_enabled.value:
            conditions.append((dataframe["close"] > dataframe[f"buy_ema1_{self.buy_ema1_length.value}"]) &
                              (dataframe[f"buy_ema1_{self.buy_ema1_length.value}"] > dataframe[f"buy_ema2_{self.buy_ema2_length.value}"]) &
                              (dataframe[f"buy_ema2_{self.buy_ema2_length.value}"] > dataframe[f"buy_ema3_{self.buy_ema3_length.value}"]))

        # RSI > RSI_Threshold_Long
        if self.buy_rsi_enabled:
            conditions.append(dataframe[f"buy_rsi_{self.buy_rsi_length.value}"] > self.buy_rsi_threshold.value)

        # MACD cross MACDs above or MACD > 0
        if self.buy_macd_enabled.value:
            macd = dataframe[f'buy_macd_{self.buy_macd_fast.value}_{self.buy_macd_slow.value}_{self.buy_macd_smooth.value}']
            macds = dataframe[f'buy_macds_{self.buy_macd_fast.value}_{self.buy_macd_slow.value}_{self.buy_macd_smooth.value}']
            conditions.append(qtpylib.crossed_above(macd, macds) | (macd > 0))

        # Candlestick patterns: any DOJI and green MARUBOZU
        if self.buy_candlestick_enabled.value:
            doji = dataframe['doji'] == dataframe["high"]
            marubozu = dataframe['marubozu'] == dataframe["high"]
            conditions.append(doji | marubozu)

        # combine conditions and return
        buy_conditions = reduce(lambda x, y: x & y, conditions)
        dataframe.loc[buy_conditions, 'enter_long'] = 1

        # SELL signal
        # only trade if there is volume
        conditions = [(dataframe['volume'] > 0)]

        # Close < EMA1 < EMA2 < EMA3
        if self.sell_ema_enabled.value:
            conditions.append((dataframe["close"] < dataframe[f"sell_ema1_{self.sell_ema1_length.value}"]) &
                              (dataframe[f"sell_ema1_{self.sell_ema1_length.value}"] < dataframe[f"sell_ema2_{self.sell_ema2_length.value}"]) &
                              (dataframe[f"sell_ema2_{self.sell_ema2_length.value}"] < dataframe[f"sell_ema3_{self.sell_ema3_length.value}"]))

        # RSI < RSI_Threshold_Long
        if self.sell_rsi_enabled:
            conditions.append(dataframe[f"sell_rsi_{self.sell_rsi_length.value}"] < self.sell_rsi_threshold.value)

        # MACD cross MACDs below or MACD < 0
        if self.sell_macd_enabled.value:
            macd = dataframe[f'sell_macd_{self.sell_macd_fast.value}_{self.sell_macd_slow.value}_{self.sell_macd_smooth.value}']
            macds = dataframe[f'sell_macds_{self.sell_macd_fast.value}_{self.sell_macd_slow.value}_{self.sell_macd_smooth.value}']
            conditions.append(qtpylib.crossed_below(macd, macds) | (macd < 0))

        # Candlestick patterns, any DOJI and red MARUBOZU
        if self.sell_candlestick_enabled.value:
            doji = dataframe['doji'] == dataframe['low']
            marubozu = dataframe['marubozu'] == dataframe['low']
            conditions.append(doji | marubozu)

        # combine conditions and return
        sell_conditions = reduce(lambda x, y: x & y, conditions)
        dataframe.loc[sell_conditions, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.sell_willy_exit.value:

            dataframe.loc[(dataframe[f'sell_ema_will{self.sell_ema.value}'] > self.sell_high_line.value),
            'exit_long'] = 1

        if self.sell_willy_exit.value:

            dataframe.loc[(dataframe[f'sell_ema_will{self.sell_ema.value}'] < self.sell_low_line.value),
            'exit_short'] = 1


        return dataframe

    stop_info = {}

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        """ exit on EMA3 cross, or exit at entry candle low (long) or high (short) """

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        close = current_rate

        # exit on EMA20 cross
        if trade.is_short:
            ema3 = last_candle[f"sell_ema3_{self.sell_ema3_length.value}"]
            if close > ema3:
                if DEBUG:
                    print(f"{pair} EMA20 exit short at {current_time}, price={current_rate}, level={ema3}, profit={current_profit}")
                return "ema20_cross_exit"
        else:
            ema3 = last_candle[f"buy_ema3_{self.buy_ema3_length.value}"]
            if close < ema3:
                if DEBUG:
                    print(f"{pair} EMA20 exit long at {current_time}, price={current_rate}, level={ema3}, profit={current_profit}")
                return "ema20_cross_exit"

        # exit on entry candle low/high
        if trade.pair in self.stop_info:
            if trade.is_short:
                if close > self.stop_info[trade.pair]["level"]:
                    if DEBUG:
                        print(f"{pair} entry candle high break exit short at {current_time}, price={current_rate}, "
                              f"level={self.stop_info[trade.pair]['level']}, profit={current_profit}")
                    return "candle-based stop_loss"
            else:
                if close < self.stop_info[trade.pair]['level']:
                    if DEBUG:
                        print(f"{pair} entry candle low break exit long at {current_time}, price={current_rate}, "
                              f"level={self.stop_info[trade.pair]['level']}, profit={current_profit}")
                    return "candle-based stop_loss"

        # exit on 3rd take profit, if it sums to 100%
        if self.tp1_size + self.tp2_size + self.tp3_size == 1:
            if trade.is_short:
                ema1 = last_candle[f"sell_ema1_{self.sell_ema1_length.value}"]
                if close < ema1 * (1-self.tp3_dist):
                    if DEBUG:
                        print(f"{pair} 3rd take profit filled => exit short at {current_time}, price={current_rate}, "
                              f"level={ema1 * (1-self.tp3_dist)}, profit={current_profit}")
                    return "take profit 3"
            else:
                ema1 = last_candle[f"buy_ema1_{self.buy_ema1_length.value}"]
                if close > ema1 * (1+self.tp3_dist):
                    if DEBUG:
                        print(f"{pair} 3rd take profit filled => exit long at {current_time}, price={current_rate}, "
                              f"level={ema1 * (1+self.tp3_dist)}, profit={current_profit}")
                    return "take profit 3"

        return False

    # Stop Loss
    # use_custom_stoploss = True
    #
    # def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
    #                     current_profit: float, **kwargs) -> float:
    #     """ custom trailing stop loss """
    #
    #     if self.tsl_break_even.value <= current_profit < self.tsl_start_offset.value:
    #         return stoploss_from_open(0, current_profit, is_short=trade.is_short)
    #     elif current_profit >= self.tsl_start_offset.value:
    #         return self.tsl_trailing_distance.value
    #     else:
    #         return -1

    # Position Size
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """ custom position size """

        if self.wallets:
            total = self.wallets.get_total("USDT")
        else:
            if DEBUG:
                print("Wallets are not available, using max_stake instead of portfolio value to calculate position size!")
            total = max_stake

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if side == "long":
            level = last_candle["low"]
            stop_distance = current_rate / level - 1
            self.stop_info[pair] = {'level': level}
        else:
            level = last_candle["high"]
            stop_distance = level / current_rate - 1
            self.stop_info[pair] = {'level': level}

        if stop_distance > abs(self.stoploss):
            if DEBUG:
                print(f"{pair} at {current_time}: stop distance {round(stop_distance*100)}% is too large, "
                      f"reducing stop loss to {abs(round(self.stoploss*100))}!")
            stop_distance = abs(self.stoploss)

        result = total * self.risk / stop_distance

        if DEBUG:
            print(f"{pair} at {current_time} position calculation: level={level}; current price={current_rate}; "
                  f"distance={stop_distance}; result={result}; min_stake={min_stake}; max_stake={max_stake}")

        # apply maximum/minimum position size
        min_pos = self.min_pos_pct * total
        max_pos = self.max_pos_pct.value * total
        if result < min_pos:
            if DEBUG:
                print(f"Calculated position ({round(result, 2)}) is less than minimum position size ({round(min_pos)})! Skipping...")
            return 0
        if result > max_pos:
            if DEBUG:
                print(f"Calculated position ({round(result, 2)}) is greater than maximum position size ({round(min_pos)})! Reduced!")
            result = max_pos

        # check min/max stake
        if result < min_stake:
            if DEBUG:
                print(f"Calculated stake ({round(result,2)}) is less than minimum stake ({min_stake})! Skipping trade...")
            return 0
        elif result > max_stake:
            if DEBUG:
                print(f"Calculated stake ({round(result,2)}) is greater than maximum stake ({max_stake})! Reducing to maximum!")
            result = max_stake

        return result

    # Take Profit
    position_adjustment_enable = True
    max_entry_position_adjustment = -1

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """ custom take profits """

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        if trade.is_short:
            if trade.nr_of_successful_exits == 0:
                if current_rate < (1-self.tp1_dist) * dataframe[f"sell_ema1_{self.sell_ema1_length.value}"].iloc[-1].squeeze():
                    return -trade.amount_requested * self.tp1_size * current_rate
            if trade.nr_of_successful_exits == 1:
                if current_rate < (1-self.tp2_dist) * dataframe[f"sell_ema1_{self.sell_ema1_length.value}"].iloc[-1].squeeze():
                    return -trade.amount_requested * self.tp2_size * current_rate
            if trade.nr_of_successful_exits == 2:
                if current_rate < (1-self.tp3_dist) * dataframe[f"sell_ema1_{self.sell_ema1_length.value}"].iloc[-1].squeeze():
                    if self.tp1_size + self.tp2_size + self.tp3_size != 1:
                        return -trade.amount_requested * self.tp3_size * current_rate
        else:
            if trade.nr_of_successful_exits == 0:
                if current_rate > (1 + self.tp1_dist) * dataframe[f"buy_ema1_{self.buy_ema1_length.value}"].iloc[-1].squeeze():
                    return -trade.amount_requested * self.tp1_size * current_rate
            if trade.nr_of_successful_exits == 1:
                if current_rate > (1 + self.tp2_dist) * dataframe[f"buy_ema1_{self.buy_ema1_length.value}"].iloc[-1].squeeze():
                    return -trade.amount_requested * self.tp2_size * current_rate
            if trade.nr_of_successful_exits == 2:
                if current_rate > (1 + self.tp3_dist) * dataframe[f"buy_ema1_{self.buy_ema1_length.value}"].iloc[-1].squeeze():
                    if self.tp1_size + self.tp2_size + self.tp3_size != 1:
                        return -trade.amount_requested * self.tp3_size * current_rate

        return None

    # set up plotting
    @property
    def plot_config(self):
        plot_config = {
            'main_plot': {
                f"buy_ema1_{self.buy_ema1_length.value}": {},
                f"buy_ema2_{self.buy_ema2_length.value}": {},
                f"buy_ema3_{self.buy_ema3_length.value}": {},
                f"doji": {"type": "scatter"},
                f"marubozu": {"type": "scatter"}
            },
            'subplots': {
                "MACD": {
                    f'buy_macd_{self.buy_macd_fast.value}_{self.buy_macd_slow.value}_{self.buy_macd_smooth.value}': {},
                    f'buy_macds_{self.buy_macd_fast.value}_{self.buy_macd_slow.value}_{self.buy_macd_smooth.value}': {},
                },
                "RSI": {
                    f'buy_rsi_{self.buy_rsi_length.value}': {}
                }
            }}

        return plot_config

    def willy_ema(self, dataframe: DataFrame, low_line=-80.0, up_line=-20.0, willyLen=21, emaLen=13):

        def max_help(open, high, low, close):
            return max(open, high, low, close)

        def min_help(open, high, low, close):
            return min(open, high, low, close)

        df = dataframe.copy()
        df['highest'] = np.vectorize(max_help)(df['open'], df['high'], df['low'], df['close'])
        df['lowest'] = np.vectorize(min_help)(df['open'], df['high'], df['low'], df['close'])
        df['upper'] = df['highest'].rolling(willyLen).max()
        df['lower'] = df['lowest'].rolling(willyLen).min()
        df['willy'] = 100 * (df['close'] - df['upper']) / (df['upper'] - df['lower'])
        df['ema_will'] = ta.EMA(df['willy'], emaLen)
        # df['ema_willy'] =
        df.drop(['highest', 'lowest', 'upper', 'lower'], inplace=True, axis=1)
        df['low_line'] = low_line
        df['up_line'] = up_line
        return DataFrame(index=df.index, data={
            'Willy': df['willy'],
            'Ema_will': df['ema_will'],
            'Low_line': df['low_line'],
            'Up_line': df['up_line']
        })


    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        hal_per = self.wallets.get_total_stake_amount() / 200
        usdt_amount = amount * rate
        if usdt_amount > hal_per:
            if Debug_Amount:
                print(f"Amount of trade USDT{usdt_amount} -- (minimal{hal_per})")
            return True
        if Debug_Amount:
            print(f"Trade was rejected ============================>>>>> actual === {usdt_amount} - minimal === {hal_per}")
        return False
