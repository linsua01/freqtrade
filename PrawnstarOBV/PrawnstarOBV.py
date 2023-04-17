# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real
from typing import Optional, Any, Callable, Dict, List
from freqtrade.strategy import IStrategy, stoploss_from_open
from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy
pd.options.mode.chained_assignment = None  # default='warn'

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class PrawnstarOBV(IStrategy):

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

                SKDecimal(0.001, 0.06, decimals=3, name='trailing_stop_positive'),
                # 'trailing_stop_positive_offset' should be greater than 'trailing_stop_positive',
                # so this intermediate parameter is used as the value of the difference between
                # them. The value of the 'trailing_stop_positive_offset' is constructed in the
                # generate_trailing_params() method.
                # This is similar to the hyperspace dimensions used for constructing the ROI tables.
                SKDecimal(0.01, 0.1, decimals=2, name='trailing_stop_positive_offset_p1'),

                Categorical([True, False], name='trailing_only_offset_is_reached'),
            ]

        def stoploss_space():

            return [SKDecimal(-0.2, -0.05, decimals=2, name='stoploss')]

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3
    can_short = False
    # Optimal timeframe for the strategy
    timeframe = '1h'

    # ROI table:
    #minimal_roi = {
    #    "0": 0.8
    #}

    minimal_roi = {
        "0": 0.322,
        "316": 0.101,
        "574": 0.036,
        "1863": 0
    }

    # Stoploss:
    stoploss = -0.15

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.011
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = False
    use_buy_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    buy_obvSma = IntParameter(3, 20, default=7, space="buy")
    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Momentum Indicators
        # ------------------------------------
        
        # Momentum
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['obv'] = ta.OBV(dataframe)
        for val in self.buy_obvSma.range:
            dataframe[f'obvSma{val}'] = ta.SMA(dataframe['obv'], timeperiod=val)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['obv'], dataframe[f'obvSma{self.buy_obvSma.value}'])) &
                (dataframe['rsi'] < 50) |
                ((dataframe[f'obvSma{self.buy_obvSma.value}'] - dataframe['close']) / dataframe[f'obvSma{self.buy_obvSma.value}'] > 0.1) |
                (dataframe['obv'] > dataframe['obv'].shift(1)) &
                (dataframe[f'obvSma{self.buy_obvSma.value}'] > dataframe[f'obvSma{self.buy_obvSma.value}'].shift(5)) &
                (dataframe['rsi'] < 50)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
            ),
            'exit_long'] = 1

        return dataframe
