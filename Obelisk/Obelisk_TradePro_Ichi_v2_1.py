# --- Do not remove these libs ---
from functools import reduce

from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy, \
    stoploss_from_open
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real
from typing import Optional, Any, Callable, Dict, List
# --------------------------------
import pandas as pd
import numpy as np
import technical.indicators as ftt

pd.options.mode.chained_assignment = None  # default='warn'


# Obelisk_TradePro_Ichi v2.1 - 2021-04-02
#
# by Obelisk
# https://twitter.com/brookmiles
#
# Originally based on "Crazy Results Best Ichimoku Cloud Trading Strategy Proven 100 Trades" by Trade Pro
# https://www.youtube.com/watch?v=8gWIykJgMNY
#
# Contributions:
#
# JimmyNixx
#  - SSL Channel confirmation
#  - ROCR & RMI confirmations
#
#
# Backtested with pairlist generated from:
# "pairlists": [
#     {
#         "method": "VolumePairList",
#         "number_assets": 50,
#         "sort_key": "quoteVolume",
#         "refresh_period": 1800
#     },
#     {"method": "AgeFilter", "min_days_listed": 10},
#     {"method": "PrecisionFilter"},
#     {"method": "PriceFilter",
#         "low_price_ratio": 0.001,
#         "max_price": 20,
#     },
#     {"method": "SpreadFilter", "max_spread_ratio": 0.002},
#     {
#         "method": "RangeStabilityFilter",
#         "lookback_days": 3,
#         "min_rate_of_change": 0.1,
#         "refresh_period": 1440
#     },
# ],


class Obelisk_TradePro_Ichi_v2_1(IStrategy):
    # Optimal timeframe for the strategy
    timeframe = '1h'
    can_short = False
    class HyperOpt:

        # def roi_space():
        #     return [
        #         Integer(0, 1, name='roi_t'),
        #         SKDecimal(0.010, 0.050, decimals=3, name='roi_p'),
        #     ]
        #
        # def generate_roi_table(params: Dict) -> Dict[int, float]:
        #     roi_table = {}
        #     roi_table[0] = params['roi_p']
        #     return roi_table

        def trailing_space() -> List[Dimension]:
            # All parameters here are mandatory, you can only modify their type or the range.
            return [
                # Fixed to true, if optimizing trailing_stop we assume to use trailing stop at all times.
                Categorical([True, False], name='trailing_stop'),

                SKDecimal(0.001, 0.01, decimals=3, name='trailing_stop_positive'),
                # 'trailing_stop_positive_offset' should be greater than 'trailing_stop_positive',
                # so this intermediate parameter is used as the value of the difference between
                # them. The value of the 'trailing_stop_positive_offset' is constructed in the
                # generate_trailing_params() method.
                # This is similar to the hyperspace dimensions used for constructing the ROI tables.
                SKDecimal(0.001, 0.05, decimals=3, name='trailing_stop_positive_offset_p1'),

                Categorical([True, False], name='trailing_only_offset_is_reached'),
            ]

        def stoploss_space():

            return [SKDecimal(-0.1, -0.05, decimals=3, name='stoploss')]

    # WARNING: ichimoku is a long indicator, if you remove or use a
    # shorter startup_candle_count your results will be unstable/invalid
    # for up to a week from the start of your backtest or dry/live run
    # (180 candles = 7.5 days)
    startup_candle_count = 180

    # NOTE: this strat only uses candle information, so processing between
    # new candles is a waste of resources as nothing will change
    process_only_new_candles = True

    minimal_roi = {
        "0": 0.231,
        "222": 0.165,
        "805": 0.075,
        "2074": 0
    }

    # Stoploss:
    stoploss = -0.075

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # plot_config = {
    #     # Main plot indicators (Moving averages, ...)
    #     'main_plot': {
    #         'senkou_a': {
    #             'color': 'green',
    #             'fill_to': 'senkou_b',
    #             'fill_label': 'Ichimoku Cloud',
    #             'fill_color': 'rgba(0,0,0,0.2)',
    #         },
    #         # plot senkou_b, too. Not only the area to it.
    #         'senkou_b': {
    #             'color': 'red',
    #         },
    #         'tenkan_sen': {'color': 'orange'},
    #         'kijun_sen': {'color': 'blue'},
    #
    #         'chikou_span': {'color': 'lightgreen'},
    #
    #         # 'ssl_up': { 'color': 'green' },
    #         # 'ssl_down': { 'color': 'red' },
    #     },
    #     'subplots': {
    #         "Signals": {
    #             'go_long': {'color': 'blue'},
    #             'future_green': {'color': 'green'},
    #             'chikou_high': {'color': 'lightgreen'},
    #             'ssl_high': {'color': 'orange'},
    #         },
    #     }
    # }
    # buy_conversion_line_period = IntParameter(15, 25, default=20, space="buy")
    # buy_base_line_periods = IntParameter(45, 75, default=60, space="buy")
    # buy_laggin_span = IntParameter(100, 140, default=120, space="buy")
    # buy_displacement = IntParameter(20, 40, default=30, space="buy")


    buy_1 = IntParameter(15, 25, default=16, space="buy")
    buy_2 = IntParameter(50, 70, default=67, space="buy")
    buy_3 = IntParameter(90, 150, default=142, space="buy")
    buy_4 = IntParameter(20, 40, default=30, space="buy", optimize=False)
    buy_rocr = IntParameter(15, 35, default=19, space="buy")
    buy_rmi_fast = IntParameter(5, 20, default=5, space="buy")
    buy_ssl = IntParameter(5, 15, default=6, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for va1 in self.buy_1.range:
            for va2 in self.buy_2.range:
                for va3 in self.buy_3.range:
                    ichimoku = ftt.ichimoku(dataframe, conversion_line_period=va1, base_line_periods=va2,
                                                                laggin_span=va3,
                                                                displacement=30)

                    dataframe[f'chikou_span{va1}{va2}{va3}{30}'] = ichimoku['chikou_span']

                    # cross indicators
                    dataframe[f'tenkan_sen{va1}{va2}{va3}{30}'] = ichimoku['tenkan_sen']
                    dataframe[f'kijun_sen{va1}{va2}{va3}{30}'] = ichimoku['kijun_sen']

                        # cloud, green a > b, red a < b
                    dataframe[f'senkou_a{va1}{va2}{va3}{30}'] = ichimoku['senkou_span_a']
                    dataframe[f'senkou_b{va1}{va2}{va3}{30}'] = ichimoku['senkou_span_b']
                    dataframe[f'leading_senkou_span_a{va1}{va2}{va3}{30}'] = ichimoku['leading_senkou_span_a']
                    dataframe[f'leading_senkou_span_b{va1}{va2}{va3}{30}'] = ichimoku['leading_senkou_span_b']
                    dataframe[f'cloud_green{va1}{va2}{va3}{30}'] = ichimoku['cloud_green'] * 1
                    dataframe[f'cloud_red{va1}{va2}{va3}{30}'] = ichimoku['cloud_red'] * -1
                    dataframe[f'future_green{va1}{va2}{va3}{30}'] = (dataframe[
                                                         f'leading_senkou_span_a{va1}{va2}{va3}{30}'] >
                                                     dataframe[
                                                         f'leading_senkou_span_b{va1}{va2}{va3}{30}']).astype(
                            'int') * 2
                    dataframe[f'chikou_high{va1}{va2}{va3}{30}'] = (
                                (dataframe[
                                     f'chikou_span{va1}{va2}{va3}{30}'] >
                                 dataframe[
                                     f'senkou_a{va1}{va2}{va3}{30}']) &
                                (dataframe[
                                     f'chikou_span{va1}{va2}{va3}{30}'] >
                                 dataframe[
                                     f'senkou_b{va1}{va2}{va3}{30}'])
                        ).shift(30).fillna(0).astype('int')
        # DANGER ZONE START

        # NOTE: Not actually the future, present data that is normally shifted forward for display as the cloud


        # The chikou_span is shifted into the past, so we need to be careful not to read the
        # current value.  But if we shift it forward again by displacement it should be safe to use.
        # We're effectively "looking back" at where it normally appears on the chart.


        # DANGER ZONE END
        for val in self.buy_ssl.range:
            dataframe[f'ssl_down{val}'] = self.SSLChannels(dataframe, val)['SslDown']
            dataframe[f'ssl_up{val}'] = self.SSLChannels(dataframe, val)['SslUp']
            dataframe[f'ssl_high{val}'] = (dataframe[f'ssl_up{val}'] > dataframe[f'ssl_down{val}']).astype('int') * 3
        for val in self.buy_rocr.range:
            dataframe[f'rocr{val}'] = ta.ROCR(dataframe, timeperiod=val)
        for val in self.buy_rmi_fast.range:
            dataframe[f'rmi-fast{val}'] = ftt.RMI(dataframe, length=val, mom=3)


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [(dataframe['volume'] > 0)]
        dataframe['go_long'] = (
                                       (dataframe[f'tenkan_sen{self.buy_1.value}{self.buy_2.value}{self.buy_3.value}{30}'] >
                                        dataframe[f'kijun_sen{self.buy_1.value}{self.buy_2.value}{self.buy_3.value}{30}']) &
                                       (dataframe['close'] > dataframe[f'senkou_a{self.buy_1.value}{self.buy_2.value}{self.buy_3.value}{30}']) &
                                       (dataframe['close'] > dataframe[f'senkou_b{self.buy_1.value}{self.buy_2.value}{self.buy_3.value}{30}']) &
                                       (dataframe[f'future_green{self.buy_1.value}{self.buy_2.value}{self.buy_3.value}{30}'] > 0) &
                                       (dataframe[f'chikou_high{self.buy_1.value}{self.buy_2.value}{self.buy_3.value}{30}'] > 0) &
                                       (dataframe[f'ssl_high{self.buy_ssl.value}'] > 0) &
                                       (dataframe[f'rocr{self.buy_rocr.value}'] > dataframe[f'rocr{self.buy_rocr.value}'].shift()) &
                                       (dataframe[f'rmi-fast{self.buy_rmi_fast.value}'] > dataframe[f'rmi-fast{self.buy_rmi_fast.value}'].shift(2))
                               ).astype('int') * 4

        conditions.append(qtpylib.crossed_above(dataframe['go_long'], 0))
        conditions = reduce(lambda x, y: x & y, conditions)
        dataframe.loc[conditions, 'enter_long'] = 1
        # print(dataframe.tail(50))

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe[f'ssl_high{self.buy_ssl.value}'] == 0)
            &
            (
                    (dataframe[f'tenkan_sen{self.buy_1.value}{self.buy_2.value}{self.buy_3.value}{30}'] < dataframe[f'kijun_sen{self.buy_1.value}{self.buy_2.value}{self.buy_3.value}{30}'])
                    |
                    (dataframe['close'] < dataframe[f'kijun_sen{self.buy_1.value}{self.buy_2.value}{self.buy_3.value}{30}'])
            )
            ,
            'exit_long'] = 1

        return dataframe

    def SSLChannels(self, dataframe: DataFrame, length=7):
        df = dataframe.copy()
        df['ATR'] = ta.ATR(df, timeperiod=14)
        df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
        df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
        df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
        df['hlv'] = df['hlv'].ffill()
        df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
        df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
        return DataFrame(index=df.index, data={
            'SslDown': df['sslDown'],
            'SslUp': df['sslUp']
        })
