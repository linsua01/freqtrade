# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from functools import reduce

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union, Dict

from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair, stoploss_from_absolute)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib
from technical.candles import heikinashi

Debug = False


class Heikin_Ashi1D174_spot_SI(IStrategy):
    INTERFACE_VERSION = 3

    # class HyperOpt:
    #     def stoploss_space():
    #         return [SKDecimal(-0.9, -0.5, decimals=2, name='stoploss')]
    #
    #     # Define custom ROI space
    #     def roi_space():
    #         return [
    #             Integer(0, 1, name='roi_t'),
    #             Integer(100, 101, name='roi_p'),
    #         ]
    #
    #     def generate_roi_table(params: Dict) -> Dict[int, float]:
    #         roi_table = {}
    #         roi_table[0] = params['roi_p']
    #
    #         return roi_table

    timeframe = '1d'

    # Can this strategy go short?
    can_short: bool = False

    minimal_roi = {
        "0": 0.249,
        "9457": 0.133,
        "23285": 0.068,
        "40133": 0
    }

    stoploss = -0.342

    trailing_stop = True

    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 300

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
                f'sl_l{self.buy_sl_mult.value}': {'color': 'red'},
                f'sl_h{self.buy_sl_mult.value}': {'color': 'green'}
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
                },
                "Willy": {
                    f'willy{self.buy_willy.value}': {'color': 'red'},
                    f'ema_will{self.buy_ema.value}': {'color': 'blue'},
                },
                "sell_willy": {
                    f'sell_willy{self.sell_willy.value}': {'color': 'blue'},
                    f'sell_ema_will{self.sell_ema.value}': {'color': 'blue'}
                },
                "Doji_in_row": {
                    f"doji_in_row{self.sell_doji_in_row.value}": {'color': 'blue', 'type': 'bar'}
                },
            }
        }

    buy_RSI_long = IntParameter(30, 60, default=58, space="buy", optimize=True)
    buy_RSI_short = IntParameter(30, 60, default=55, space="buy", optimize=True)
    buy_bear_row = IntParameter(1, 3, default=1, space='buy', optimize=True)
    buy_bull_row = IntParameter(1, 3, default=1, space='buy', optimize=True)
    buy_ema = IntParameter(7, 22, default=18, space='buy', optimize=True)
    buy_high_line = IntParameter(-30, 0, default=-2, space="buy", optimize=True)
    buy_low_line = IntParameter(-100, -70, default=-89, space="buy", optimize=True)
    buy_on_willy = BooleanParameter(default=False, space='buy', optimize=True)
    buy_sl_mult = DecimalParameter(0.5, 5, default=3.4, decimals=1, space='buy', optimize=True)
    buy_willy = IntParameter(15, 30, default=23, space='buy', optimize=True)
    sell_doji_in_row = IntParameter(1, 6, default=2, space="sell", optimize=True)
    sell_ema = IntParameter(7, 22, default=12, space='sell', optimize=True)
    sell_high_line = IntParameter(-25, 0, default=-10, space="sell", optimize=True)
    sell_low_line = IntParameter(-100, -75, default=-92, space="sell", optimize=True)
    sell_willy = IntParameter(15, 30, default=22, space='sell', optimize=True)
    sell_willy_exit = BooleanParameter(default=True, space='sell', optimize=True)



    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1w') for pair in pairs]
        informative_pairs += [("ETH/USDT", "1w"), ("BTC/USDT", "1w"), ("DOGE/USDT", "1w"), ("BNB/USDT", "1w"),
                              ("XRP/USDT", "1w"),
                              ("ADA/USDT", "1w"), ("MATIC/USDT", "1w"), ("DOT/USDT", "1w"),
                              ("TRX/USDT", "1w"), ("LTC/USDT", "1w"),
                              ("SOL/USDT", "1w"), ("UNI/USDT", "1w"), ("AVAX/USDT", "1w"), ("LPT/USDT", "1w"),
                              ("LINK/USDT", "1w"), ("XMR/USDT", "1w"), ("ATOM/USDT", "1w"),
                              ("ETC/USDT", "1w"), ("BCH/USDT", "1w"), ("XLM/USDT", "1w"), ("APE/USDT", "1w"),
                              ("QNT/USDT", "1w"),
                              ("ALGO/USDT", "1w"), ("VET/USDT", "1w"), ("NEAR/USDT", "1w"),
                              ("HBAR/USDT", "1w"), ("ICP/USDT", "1w"), ("FIL/USDT", "1w"), ("EOS/USDT", "1w"),
                              ("EGLD/USDT", "1w"),
                              ("FLOW/USDT", "1w"), ("THETA/USDT", "1w"), ("AAVE/USDT", "1w"), ("XTZ/USDT", "1w"),
                              ("AXS/USDT", "1w"), ("CHZ/USDT", "1w"), ("SAND/USDT", "1w"), ("ZEC/USDT", "1w"),
                              ("MANA/USDT", "1w"),
                              ("FTM/USDT", "1w"),
                              ("MKR/USDT", "1w"),
                              ("GRT/USDT", "1w"),
                              ("KLAY/USDT", "1w"), ("DASH/USDT", "1w"), ("APT/USDT", "1w"),
                              ("IOTA/USDT", "1w"),
                              ("RUNE/USDT", "1w"), ("NEO/USDT", "1w"), ("SNX/USDT", "1w"),
                              ("IMX/USDT", "1w"),
                              ("FTT/USDT", "1w"), ("1INCH/USDT", "1w"),
                              ("LDO/USDT", "1w"), ("ZIL/USDT", "1w"), ("BAT/USDT", "1w"),
                              ("CRV/USDT", "1w"), ("LRC/USDT", "1w"),
                              ("XEM/USDT", "1w"), ("ENJ/USDT", "1w"),
                              ("HNT/USDT", "1w"), ("CVX/USDT", "1w"), ("HOT/USDT", "1w"), ("BAL/USDT", "1w"),
                              ("KAVA/USDT", "1w"), ("RVN/USDT", "1w"),
                              ("COMP/USDT", "1w"), ("CELO/USDT", "1w"), ("ENS/USDT", "1w"),
                              ("OP/USDT", "1w"),
                              ("KSM/USDT", "1w"), ("AR/USDT", "1w"), ("SUSHI/USDT", "1w"),
                              ("QTUM/USDT", "1w"),
                              ("YFI/USDT", "1w"), ("ROSE/USDT", "1w"), ("ONE/USDT", "1w"),
                              ("BNX/USDT", "1w"),
                              ("IOTX/USDT", "1w"), ("WAVES/USDT", "1w"),
                              ("GMT/USDT", "1w"), ("ANKR/USDT", "1w"),
                              ("MASK/USDT", "1w")
                              ]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        inf_tf = '1w'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        informative['bear_cand'] = self.heikin_asi_trigger(dataframe, metadata, 1, 1, 1)['Bear_cand']
        informative['bull_cand'] = self.heikin_asi_trigger(dataframe, metadata, 1, 1, 1)['Bull_cand']
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['plot_rsi'] = self.plot_RSI(dataframe)['Plot_rsi']
        dataframe['ha_open'] = heikinashi(dataframe)['open']
        dataframe['ha_high'] = heikinashi(dataframe)['high']
        dataframe['ha_low'] = heikinashi(dataframe)['low']
        dataframe['ha_close'] = heikinashi(dataframe)['close']
        dataframe['doji_cand'] = self.heikin_asi_trigger(dataframe, metadata, 1, 1, 1)['Doji_cand']
        dataframe['bear_cand'] = self.heikin_asi_trigger(dataframe, metadata, 1, 1, 1)['Bear_cand']
        dataframe['bull_cand'] = self.heikin_asi_trigger(dataframe, metadata, 1, 1, 1)['Bull_cand']
        dataframe['red_or_green'] = self.heikin_asi_trigger(dataframe, metadata, 1, 1, 1)['Red_or_green']
        dataframe['plot_bull'] = self.heikin_asi_trigger(dataframe, metadata, 1, 1, 1)['Plot_bull']
        dataframe['plot_bear'] = self.heikin_asi_trigger(dataframe, metadata, 1, 1, 1)['Plot_bear']
        dataframe['plot_doji'] = self.heikin_asi_trigger(dataframe, metadata, 1, 1, 1)['Plot_doji']
        for val in self.buy_willy.range:
            dataframe[f'willy{val}'] = self.willy_ema(dataframe, low_line=self.buy_low_line.value,
                                                      up_line=self.buy_high_line.value, willyLen=val, emaLen=13)[
                'Willy']
        for val in self.buy_ema.range:
            dataframe[f'ema_will{val}'] = self.willy_ema(dataframe, low_line=-80, up_line=-20, willyLen=21, emaLen=val)[
                'Ema_will']
        for val in self.buy_sl_mult.range:
            dataframe[f'sl_l{val}'] = self.cust_stoploss(dataframe, length=20, mult=val)['Sl_low']
        for val in self.buy_sl_mult.range:
            dataframe[f'sl_h{val}'] = self.cust_stoploss(dataframe, length=20, mult=val)['Sl_high']

        for val in self.sell_willy.range:
            dataframe[f'sell_willy{val}'] = self.willy_ema(dataframe, low_line=self.buy_low_line.value,
                                                           up_line=self.buy_high_line.value, willyLen=val, emaLen=13)[
                'Willy']
        for val in self.sell_ema.range:
            dataframe[f'sell_ema_will{val}'] = \
                self.willy_ema(dataframe, low_line=-80, up_line=-20, willyLen=21, emaLen=val)[
                    'Ema_will']
        for val in self.sell_doji_in_row.range:
            dataframe[f'doji_in_row{val}'] = self.heikin_asi_trigger(dataframe, metadata, val, 1, 1)['Doji_in_row']
        for val in self.buy_bull_row.range:
            dataframe[f'buy_bull_row{val}'] = self.heikin_asi_trigger(dataframe, metadata, 1, val, 1)['Bull_in_row']
        for val in self.buy_bear_row.range:
            dataframe[f'buy_bear_row{val}'] = self.heikin_asi_trigger(dataframe, metadata, 1, 1, val)['Bear_in_row']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = [(dataframe['volume'] > 0)]

        conditions.append((dataframe['rsi'] > self.buy_RSI_long.value) &  # Signal: RSI
                          (dataframe[f'buy_bull_row{self.buy_bull_row.value}'] == self.buy_bull_row.value))

        if self.buy_on_willy.value:
            conditions.append((dataframe[f'willy{self.buy_willy.value}'] < self.buy_low_line.value) &
                              (dataframe[f'ema_will{self.buy_ema.value}'] < self.buy_low_line.value))

        buy_conditions = reduce(lambda x, y: x & y, conditions)
        dataframe.loc[buy_conditions, 'enter_long'] = 1

        conditions = [(dataframe['volume'] > 0)]
        conditions.append((dataframe['rsi'] < self.buy_RSI_short.value) &  # Signal: RSI
                          (dataframe[f'buy_bear_row{self.buy_bear_row.value}'] == self.buy_bear_row.value))

        if self.buy_on_willy.value:
            conditions.append((dataframe[f'willy{self.buy_willy.value}'] > self.buy_high_line.value) &
                              (dataframe[f'ema_will{self.buy_ema.value}'] > self.buy_high_line.value))

        if Debug:
            print(conditions)
        sell_conditions = reduce(lambda x, y: x & y, conditions)
        dataframe.loc[sell_conditions, 'enter_short'] = 1

        if Debug:
            print(dataframe.tail(20))

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.sell_willy_exit.value:
            conditions = [(dataframe['bear_cand'] > 0) |
                          (dataframe[f'doji_in_row{self.sell_doji_in_row.value}'] == self.sell_doji_in_row.value)
                          ]
        else:
            conditions = [(dataframe['bear_cand'] > 0) |
                          (dataframe[f'doji_in_row{self.sell_doji_in_row.value}'] == self.sell_doji_in_row.value) |
                          ((dataframe[f'sell_willy{self.sell_willy.value}'] > self.sell_high_line.value) &
                           (dataframe[f'sell_ema_will{self.sell_ema.value}'] > self.sell_high_line.value))
                          ]

        sell_conditions = reduce(lambda x, y: x & y, conditions)
        dataframe.loc[sell_conditions, 'exit_long'] = 1

        if not self.sell_willy_exit.value:
            conditions = [(dataframe['bull_cand'] > 0) |
                          (dataframe[f'doji_in_row{self.sell_doji_in_row.value}'] == self.sell_doji_in_row.value)]
        else:
            conditions = [(dataframe['bull_cand'] > 0) |
                          (dataframe[f'doji_in_row{self.sell_doji_in_row.value}'] == self.sell_doji_in_row.value) |
                          ((dataframe[f'sell_willy{self.sell_willy.value}'] < self.sell_low_line.value) &
                          (dataframe[f'sell_ema_will{self.sell_ema.value}'] < self.sell_low_line.value))
                          ]
        sell_conditions = reduce(lambda x, y: x & y, conditions)
        dataframe.loc[sell_conditions, 'exit_short'] = 1

        if Debug:
            print(dataframe.tail(30))
        return dataframe

    def plot_RSI(self, dataframe: DataFrame):
        df = dataframe.copy()
        df['drop_rsi'] = ta.RSI(df)
        df['plot_rsi'] = np.where((df["drop_rsi"] > self.buy_RSI_long.value), df['high'] + (df['high'] * 0.01),
                                  np.where((df["drop_rsi"] < self.buy_RSI_short.value),
                                           df['low'] - (df['low'] * 0.01), np.nan))

        df.drop(['drop_rsi'], inplace=True, axis=1)
        return DataFrame(index=df.index, data={
            'Plot_rsi': df['plot_rsi']
        })

    def heikin_asi_trigger(self, dataframe: DataFrame, metadata: dict, doji_sum, bull_row, bear_row):
        def _flat_top(close, low, open, high):

            if high == open and low < close:
                return 1
            else:
                return 0

        def _wick_length(close, low, open, high):

            if close > open:
                top_wick = high - close
                bottom_wick = open - low

            else:
                top_wick = high - open
                bottom_wick = close - low

            if top_wick > 0 and bottom_wick > 0:
                return 1
            else:
                return 0

        def _is_red_green(open, close):
            if open < close:
                return 1
            else:
                return 0

        def _flat_bottom(close, low, open, high):

            if open == low and high > close:
                return 1
            else:
                return 0

        df = dataframe.copy()
        df['ha_open'] = heikinashi(df)['open']
        df['ha_high'] = heikinashi(df)['high']
        df['ha_low'] = heikinashi(df)['low']
        df['ha_close'] = heikinashi(df)['close']
        df['ha_red_or_green'] = np.vectorize(_is_red_green)(df['ha_open'], df['ha_close'])
        df['bull_cand'] = np.vectorize(_flat_bottom)(df['ha_close'], df['ha_low'], df['ha_open'], df['ha_high'])
        df['plot_bull'] = np.where(df['bull_cand'] > 0, df['high'], np.NaN)
        df['bear_cand'] = np.vectorize(_flat_top)(df['ha_close'], df['ha_low'], df['ha_open'], df['ha_high'])
        df['plot_bear'] = np.where(df['bear_cand'] > 0, df['low'], np.NaN)
        df['doji_cand'] = np.vectorize(_wick_length)(df['ha_close'], df['ha_low'], df['ha_open'], df['ha_high'])
        df['plot_doji'] = np.where(df['doji_cand'] > 0, df['close'], np.NaN)
        df['doji_in_row'] = df['doji_cand'].rolling(window=doji_sum).sum()
        df['bull_in_row'] = df['bull_cand'].rolling(window=bull_row).sum()
        df['bear_in_row'] = df['bear_cand'].rolling(window=bear_row).sum()

        df.drop(['ha_open', 'ha_high', 'ha_low', 'ha_close'], inplace=True, axis=1)

        return DataFrame(index=df.index, data={
            'Doji_cand': df['doji_cand'],
            'Bear_cand': df['bear_cand'],
            'Bull_cand': df['bull_cand'],
            'Red_or_green': df['ha_red_or_green'],
            'Plot_bull': df['plot_bull'],
            'Plot_bear': df['plot_bear'],
            'Plot_doji': df['plot_doji'],
            'Doji_in_row': df['doji_in_row'],
            'Bull_in_row': df['bull_in_row'],
            'Bear_in_row': df['bear_in_row']

        })

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

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = dataframe.iloc[-1].squeeze()
        stoploss_pricel = candle[f'sl_l{self.buy_sl_mult.value}']
        stoploss_prices = candle[f'sl_h{self.buy_sl_mult.value}']
        middle_of_cand = (candle['high'] + candle['low'] + candle['close'] + candle['open']) / 4
        if candle['ha_open'] < candle['ha_close']:
            stoploss_ = (middle_of_cand - stoploss_pricel) / candle['close'] * 100

            amount_ = self.wallets.get_total_stake_amount() / 100
            if Debug:
                print('stoploss from amount = >', stoploss_)
                print('полній кошилек =>>>', self.wallets.get_total_stake_amount())
                print('amount', amount_)
                print('Amount final =>>', amount_ / stoploss_ * 100)
            return (amount_ / stoploss_) * 100

        if candle['ha_open'] > candle['ha_close']:
            stoploss_ = (middle_of_cand - stoploss_prices) / candle['close'] * -1 * 100
            amount_ = self.wallets.get_total_stake_amount() / 100
            if Debug:
                print('stoploos from amount =>', stoploss_)
                print('полній кошилек =>>>', self.wallets.get_total_stake_amount())
                print('amount', amount_)
                print('Amount =>>', (amount_ / stoploss_) * 100)
            return (amount_ / stoploss_) * 100

        return 2000

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = dataframe.iloc[-1].squeeze()

        stoploss_pricel = candle[f'sl_l{self.buy_sl_mult.value}']
        stoploss_prices = candle[f'sl_h{self.buy_sl_mult.value}']
        middle_of_cand = (candle['high'] + candle['low'] + candle['close'] + candle['open']) / 4
        if Debug:
            print('trade ID =>', trade.id, 'Trade is short? =>', trade.is_short)
            print('sl hihg =>', candle[f'sl_h{self.buy_sl_mult.value}'], 'sl low =>', candle[f'sl_l{self.buy_sl_mult.value}'])
            print('candle middle =>', middle_of_cand)
        if not trade.is_short:
            if Debug:
                print('   ')
                print('Trade is long', 'Stoploss ================================>',
                      (middle_of_cand - stoploss_pricel) / candle['close'] * -1)
                print('   ')
            return (middle_of_cand - stoploss_pricel) / candle['close'] * -1

        if trade.is_short:
            if Debug:
                print('   ')
                print('Trade is short', 'Stoploss ==============================>',
                      (middle_of_cand - stoploss_prices) / candle['close'])
                print('   ')
            return (middle_of_cand - stoploss_prices) / candle['close']

        return 100

    def cust_stoploss(self, dataframe: DataFrame, mult, length):
        df = dataframe.copy()
        df['sma'] = qtpylib.sma(df['close'], length)
        df['tr'] = qtpylib.atr(df, length)

        df['sl_high'] = qtpylib.sma(df['tr'], length)

        def help1(x, mul, high):
            return x * mul + high

        df['sl_high'] = np.vectorize(help1)(df['sl_high'], mult, df['high'])

        df['sl_low'] = qtpylib.sma(df['tr'], length)

        def help1(x, mul, low):
            return low - x * mul

        df['sl_low'] = np.vectorize(help1)(df['sl_low'], mult, df['low'])
        df.drop(['sma', 'tr'], axis=1, inplace=True)
        return DataFrame(index=df.index, data={
            'Sl_high': df['sl_high'],
            'Sl_low': df['sl_low'],
        })
