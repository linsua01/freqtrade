# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union, Dict

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.optimize.space import SKDecimal, Integer
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair, stoploss_from_absolute)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from technical.candles import heikinashi
# 1 backtest with roi, no stoploss + custom entry amount
# 2 backtest no roi, stoploss + custom entry amount
# 3 roi , stoploss + custom entry amount
# no roi, no / stoploss, no custom entry emount 10 orders

class Heikin_Ashi1D165(IStrategy):
    INTERFACE_VERSION = 3

    # class HyperOpt:
    #     # def stoploss_space():
    #     #     return [SKDecimal(-0.9, -0.1, decimals=1, name='stoploss')]
    #
    #     # Define custom ROI space
    #     def roi_space():
    #         return [
    #             Integer(10, 120, name='roi_t'),
    #             Integer(1, 20,  name='roi_p'),
    #         ]
    #
    #     def generate_roi_table(params: Dict) -> Dict[int, float]:
    #         roi_table = {}
    #         roi_table[0] = params['roi_p']
    #
    #         return roi_table


    buy_RSI_long = IntParameter(30, 60, default=58, space="buy")
    buy_RSI_short = IntParameter(30, 60, default=39, space="buy")
    # buy_cust_one_trade_limit_max = IntParameter(5, 30, default=10, space="buy")
    # buy_cust_one_trade_limit_min = IntParameter(1, 8, default=1, space="buy")

    timeframe = '1d'

    # Can this strategy go short?
    can_short: bool = True

    minimal_roi = {
        "0": 3
    }

    stoploss = -0.5

    trailing_stop = False

    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    process_only_new_candles = True





    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 10

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
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
                }
            }
        }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)

        dataframe['ha_open'] = heikinashi(dataframe)['open']
        dataframe['ha_high'] = heikinashi(dataframe)['high']
        dataframe['ha_low'] = heikinashi(dataframe)['low']
        dataframe['ha_close'] = heikinashi(dataframe)['close']
        # Heikin Ashi Strategy
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']
        dataframe['plot_rsi'] = self.plot_RSI(dataframe)['Plot_rsi']
        dataframe['doji_cand'] = self.heikin_asi_trigger(dataframe)['Doji_cand']
        dataframe['bear_cand'] = self.heikin_asi_trigger(dataframe)['Bear_cand']
        dataframe['bull_cand'] = self.heikin_asi_trigger(dataframe)['Bull_cand']
        dataframe['red_or_green'] = self.heikin_asi_trigger(dataframe)['Red_or_green']
        dataframe['plot_bull'] = self.heikin_asi_trigger(dataframe)['Plot_bull']
        dataframe['plot_bear'] = self.heikin_asi_trigger(dataframe)['Plot_bear']
        dataframe['plot_doji'] = self.heikin_asi_trigger(dataframe)['Plot_doji']

        # 'Doshi_cand': df['doshi_cand'],
        # 'Bear_cand': df['bear_cand'],
        # 'Bull_cand': df['bull_cand']
        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > self.buy_RSI_long.value) &  # Signal: RSI
                    (dataframe['bull_cand'] == 1) &
                    (dataframe['bull_cand'].shift(1) == 1)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                    (dataframe['rsi'] < self.buy_RSI_short.value) &  # Signal: RSI
                    (dataframe['bear_cand'] == 1) &
                    (dataframe['bear_cand'].shift(1) == 1)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['bear_cand'] > 0) |
                    ((dataframe['doji_cand'] > 0) &
                     (dataframe['doji_cand'].shift(1) > 0))
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                    (dataframe['bull_cand'] > 0) |
                    ((dataframe['doji_cand'] > 0) &
                     (dataframe['doji_cand'].shift(1) > 0))
            ),
            'exit_short'] = 1

        # print(dataframe.tail(30))
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
    def heikin_asi_trigger(self, dataframe: DataFrame):
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

        # heikinashi = qtpylib.heikinashi(df)
        # df['ha_open'] = heikinashi['open']
        # df['ha_close'] = heikinashi['close']
        # df['ha_high'] = heikinashi['high']
        # df['ha_low'] = heikinashi['low']
        df['ha_open'] = heikinashi(df)['open']
        df['ha_high'] = heikinashi(df)['high']
        df['ha_low'] = heikinashi(df)['low']
        df['ha_close'] = heikinashi(df)['close']

        # df['doshi_cand'] = 0.0
        # df['bull_cand'] = 0.0
        # df['bear_cand'] = 0.0

        df['ha_red_or_green'] = np.vectorize(_is_red_green)(df['ha_open'], df['ha_close'])
        df['bull_cand'] = np.vectorize(_flat_bottom)(df['ha_close'], df['ha_low'], df['ha_open'], df['ha_high'])
        df['plot_bull'] = np.where(df['bull_cand'] > 0, df['high'], np.NaN)
        df['bear_cand'] = np.vectorize(_flat_top)(df['ha_close'], df['ha_low'], df['ha_open'], df['ha_high'])
        df['plot_bear'] = np.where(df['bear_cand'] > 0, df['low'], np.NaN)
        df['doji_cand'] = np.vectorize(_wick_length)(df['ha_close'], df['ha_low'], df['ha_open'], df['ha_high'])
        df['plot_doji'] = np.where(df['doji_cand'] > 0, df['close'], np.NaN)

        df.drop(['ha_open', 'ha_high', 'ha_low', 'ha_close'], inplace=True, axis=1)

        # for i in range(len(df)):
        #     if df['ha_open'].iat[i] <= df['ha_close'].iat[i]:
        #         df['ha_red'].iat = 1
        #     else:
        #         df['ha_green'] = 0.0
        # for i in range(len(df)):
        #     if df['ha_open'].iat[i] == df['ha_low'].iat[i] and df['ha_close'].iat[i] != df['ha_high'].iat[i]:
        #         df['bull_cand'].iat[i] = 1
        #     elif df['ha_open'].iat[i] == df['ha_high'].iat[i] and df['ha_close'].iat[i] != df['ha_low'].iat[i]:
        #         df['bear_cand'].iat[i] = 1
        #     elif df['ha_open'].iat[i] != df['ha_low'].iat[i] != df['ha_close'].iat[i] != df['ha_high'].iat[i]:
        #         df['doshi_cand'].iat[i] = 1
        # df['doshi_cand'] = np.where(df['ha_open'] != df['ha_low'] != df['ha_close'] != df['ha_high'], 1, np.NaN)
        # df['bear_cand'] = np.where(df['ha_open'] == df['ha_high'] & df['ha_close'] != df['ha_low'], 1, np.NaN)
        # df['bull_cand'] = np.where(df['ha_open'] == df['ha_low'] & df['ha_close'] != df['ha_high'], 1, np.NaN)
        # df.drop(['ha_open', 'ha_close', 'ha_high', 'ha_low'], inplace=True, axis=1)

        return DataFrame(index=df.index, data={
            'Doji_cand': df['doji_cand'],
            'Bear_cand': df['bear_cand'],
            'Bull_cand': df['bull_cand'],
            'Red_or_green': df['ha_red_or_green'],
            'Plot_bull': df['plot_bull'],
            'Plot_bear': df['plot_bear'],
            'Plot_doji': df['plot_doji']
        })

    # def bot_loop_start(self, **kwargs) -> None:
    #     """
    #     Called at the start of the bot iteration (one loop).
    #     Might be used to perform pair-independent tasks
    #     (e.g. gather some remote ressource for comparison)
    #
    #     For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/
    #
    #     When not implemented by a strategy, this simply does nothing.
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     """
    #     pass
    #
    # def custom_entry_price(self, pair: str, current_time: 'datetime', proposed_rate: float,
    #                        entry_tag: 'Optional[str]', side: str, **kwargs) -> float:
    #     """
    #     Custom entry price logic, returning the new entry price.
    #
    #     For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/
    #
    #     When not implemented by a strategy, returns None, orderbook is used to set entry price
    #
    #     :param pair: Pair that's currently analyzed
    #     :param current_time: datetime object, containing the current datetime
    #     :param proposed_rate: Rate, calculated based on pricing settings in exit_pricing.
    #     :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     :return float: New entry price value if provided
    #     """
    #     return proposed_rate
    #
    # def adjust_entry_price(self, trade: 'Trade', order: 'Optional[Order]', pair: str,
    #                         current_time: datetime, proposed_rate: float, current_order_rate: float,
    #                         entry_tag: Optional[str], side: str, **kwargs) -> float:
    #     """
    #     Entry price re-adjustment logic, returning the user desired limit price.
    #     This only executes when a order was already placed, still open (unfilled fully or partially)
    #     and not timed out on subsequent candles after entry trigger.
    #
    #     For full documentation please go to https://www.freqtrade.io/en/latest/strategy-callbacks/
    #
    #     When not implemented by a strategy, returns current_order_rate as default.
    #     If current_order_rate is returned then the existing order is maintained.
    #     If None is returned then order gets canceled but not replaced by a new one.
    #
    #     :param pair: Pair that's currently analyzed
    #     :param trade: Trade object.
    #     :param order: Order object
    #     :param current_time: datetime object, containing the current datetime
    #     :param proposed_rate: Rate, calculated based on pricing settings in entry_pricing.
    #     :param current_order_rate: Rate of the existing order in place.
    #     :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
    #     :param side: 'long' or 'short' - indicating the direction of the proposed trade
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     :return float: New entry price value if provided
    #
    #     """
    #     return current_order_rate
    #
    # def custom_exit_price(self, pair: str, trade: 'Trade',
    #                       current_time: 'datetime', proposed_rate: float,
    #                       current_profit: float, exit_tag: Optional[str], **kwargs) -> float:
    #     """
    #     Custom exit price logic, returning the new exit price.
    #
    #     For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/
    #
    #     When not implemented by a strategy, returns None, orderbook is used to set exit price
    #
    #     :param pair: Pair that's currently analyzed
    #     :param trade: trade object.
    #     :param current_time: datetime object, containing the current datetime
    #     :param proposed_rate: Rate, calculated based on pricing settings in exit_pricing.
    #     :param current_profit: Current profit (as ratio), calculated based on current_rate.
    #     :param exit_tag: Exit reason.
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     :return float: New exit price value if provided
    #     """
    #     return proposed_rate
    #
    # def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
    #                         proposed_stake: float, min_stake: Optional[float], max_stake: float,
    #                         leverage: float, entry_tag: Optional[str], side: str,
    #                         **kwargs) -> float:
    #
    #
    #     dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
    #                                                              timeframe=self.timeframe)
    #     stoploss_amount = max_stake / 200  # 0.5%
    #     current_candle = dataframe.iloc[-1].squeeze()
    #     if current_candle['ha_close'] >= current_candle['ha_open']:
    #         if current_candle['ha_low'] >= current_candle['low']:
    #             entry_by_norm_cand = (current_candle['close'] - current_candle['low']) / current_candle['close'] * 100
    #             entry = stoploss_amount / entry_by_norm_cand * 100
    #             if entry < max_stake / 100 * self.buy_cust_one_trade_limit_min.value:
    #                 # print(max_stake / 100 * self.buy_cust_one_trade_limit_min.value, 1)
    #                 return max_stake / 100 * self.buy_cust_one_trade_limit_min.value
    #             elif entry > max_stake / 100 * self.buy_cust_one_trade_limit_max.value:
    #                 # print(max_stake / 100 * self.buy_cust_one_trade_limit_max.value , 2)
    #                 return max_stake / 100 * self.buy_cust_one_trade_limit_max.value
    #             else:
    #                 # print(entry, 3)
    #                 return entry
    #         else:
    #             entry_by_ha_cand = (current_candle['close'] - (
    #                     (current_candle['low'] + current_candle['ha_low']) / 2)) / \
    #                                current_candle['close'] * 100
    #             entry = stoploss_amount / entry_by_ha_cand * 100
    #             if entry < max_stake / 100 * self.buy_cust_one_trade_limit_min.value:
    #                 # print(max_stake / 100 * self.buy_cust_one_trade_limit_min.value ,4)
    #                 return max_stake / 100 * self.buy_cust_one_trade_limit_min.value
    #             elif entry > max_stake / 100 * self.buy_cust_one_trade_limit_max.value:
    #                 # print(max_stake / 100 * self.buy_cust_one_trade_limit_max.value, 5)
    #                 return max_stake / 100 * self.buy_cust_one_trade_limit_max.value
    #             else:
    #                 # print(entry,6)
    #                 return entry
    #
    #     else:
    #         if current_candle['ha_high'] <= current_candle['high']:
    #             entry_by_norm_cand = (current_candle['close'] - current_candle['high']) / current_candle[
    #                 'close'] * 100 * -1
    #             entry = stoploss_amount / entry_by_norm_cand * 100
    #             if entry < max_stake / 100 * self.buy_cust_one_trade_limit_min.value:
    #                 # print(max_stake / 100 * self.buy_cust_one_trade_limit_min.value,7)
    #                 return max_stake / 100 * self.buy_cust_one_trade_limit_min.value
    #             elif entry > max_stake / 100 * self.cbuy_cust_one_trade_limit_max:
    #                 # print(max_stake / 100 * self.buy_cust_one_trade_limit_max.value,8)
    #                 return max_stake / 100 * self.buy_cust_one_trade_limit_max.value
    #             else:
    #                 # print(entry, 9)
    #                 return entry
    #         else:
    #             entry_by_ha_cand = (current_candle['close'] - ((current_candle['high'] + current_candle['ha_high']) /
    #                                                            2)) / current_candle['close'] * 100 * -1
    #             entry = stoploss_amount / entry_by_ha_cand * 100
    #
    #             if entry < max_stake / 100 * self.buy_cust_one_trade_limit_min.value:
    #                 # print(max_stake / 100 * self.buy_cust_one_trade_limit_min.value ,10)
    #                 return max_stake / 100 * self.buy_cust_one_trade_limit_min.value
    #             elif entry > max_stake / 100 * self.buy_cust_one_trade_limit_max.value:
    #                 # print(max_stake / 100 * self.buy_cust_one_trade_limit_max.value,11)
    #                 return max_stake / 100 * self.buy_cust_one_trade_limit_max.value
    #             else:
    #                 # print(entry, 12)
    #                 return entry

    # use_custom_stoploss = True
    #
    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #
    #     trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
    #     # Look up trade candle.
    #     trade_candle = dataframe.loc[dataframe['date'] == trade_date]
    #     # trade_candle may be empty for trades that just opened as it is still incomplete.
    #
    #     if not trade_candle.empty:
    #         trade_candle = trade_candle.squeeze()
    #
    #         if trade_candle['ha_close'] >= trade_candle['ha_open']:
    #             if trade_candle['ha_low'] >= trade_candle['low']:
    #                 print((trade_candle['close'] - trade_candle['low']) / trade_candle['close'] * -1)
    #                 return (trade_candle['close'] - trade_candle['low']) / trade_candle['close'] * -1
    #
    #             else:
    #                 print((trade_candle['close'] - ((trade_candle['low'] + trade_candle['ha_low']) / 2))
    #                       / trade_candle['close'] * -1)
    #                 return (trade_candle['close'] - ((trade_candle['low'] + trade_candle['ha_low']) / 2)) \
    #                        / trade_candle['close'] * -1
    #         else:
    #             if trade_candle['ha_high'] <= trade_candle['high']:
    #                 print((trade_candle['close'] - trade_candle['high']) / trade_candle['close'])
    #                 return (trade_candle['close'] - trade_candle['high']) / trade_candle['close']
    #             else:
    #                 print((trade_candle['close'] - ((trade_candle['high'] + trade_candle['ha_high']) / 2))
    #                        / trade_candle['close'])
    #                 return (trade_candle['close'] - ((trade_candle['high'] + trade_candle['ha_high']) / 2))\
    #                        / trade_candle['close']
    #
    #     # return some value that won't cause stoploss to update
    #
    #     return 100
    #
    # def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
    #                 current_profit: float, **kwargs) -> 'Optional[Union[str, bool]]':
    #     """
    #     Custom exit signal logic indicating that specified position should be sold. Returning a
    #     string or True from this method is equal to setting sell signal on a candle at specified
    #     time. This method is not called when sell signal is set.
    #
    #     This method should be overridden to create sell signals that depend on trade parameters. For
    #     example you could implement a sell relative to the candle when the trade was opened,
    #     or a custom 1:2 risk-reward ROI.
    #
    #     Custom exit reason max length is 64. Exceeding characters will be removed.
    #
    #     :param pair: Pair that's currently analyzed
    #     :param trade: trade object.
    #     :param current_time: datetime object, containing the current datetime
    #     :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
    #     :param current_profit: Current profit (as ratio), calculated based on current_rate.
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     :return: To execute sell, return a string with custom exit reason or True. Otherwise return
    #     None or False.
    #     """
    #     return None
    #
    # def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
    #                         time_in_force: str, current_time: datetime, entry_tag: Optional[str],
    #                         side: str, **kwargs) -> bool:
    #     """
    #     Called right before placing a entry order.
    #     Timing for this function is critical, so avoid doing heavy computations or
    #     network requests in this method.
    #
    #     For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/
    #
    #     When not implemented by a strategy, returns True (always confirming).
    #
    #     :param pair: Pair that's about to be bought/shorted.
    #     :param order_type: Order type (as configured in order_types). usually limit or market.
    #     :param amount: Amount in target (base) currency that's going to be traded.
    #     :param rate: Rate that's going to be used when using limit orders
    #                  or current rate for market orders.
    #     :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
    #     :param current_time: datetime object, containing the current datetime
    #     :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
    #     :param side: 'long' or 'short' - indicating the direction of the proposed trade
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     :return bool: When True is returned, then the buy-order is placed on the exchange.
    #         False aborts the process
    #     """
    #     return True
    #
    # def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
    #                        rate: float, time_in_force: str, exit_reason: str,
    #                        current_time: 'datetime', **kwargs) -> bool:
    #     """
    #     Called right before placing a regular exit order.
    #     Timing for this function is critical, so avoid doing heavy computations or
    #     network requests in this method.
    #
    #     For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/
    #
    #     When not implemented by a strategy, returns True (always confirming).
    #
    #     :param pair: Pair for trade that's about to be exited.
    #     :param trade: trade object.
    #     :param order_type: Order type (as configured in order_types). usually limit or market.
    #     :param amount: Amount in base currency.
    #     :param rate: Rate that's going to be used when using limit orders
    #                  or current rate for market orders.
    #     :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
    #     :param exit_reason: Exit reason.
    #         Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
    #                         'exit_signal', 'force_exit', 'emergency_exit']
    #     :param current_time: datetime object, containing the current datetime
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     :return bool: When True, then the exit-order is placed on the exchange.
    #         False aborts the process
    #     """
    #     return True
    #
    # def check_entry_timeout(self, pair: str, trade: 'Trade', order: 'Order',
    #                         current_time: datetime, **kwargs) -> bool:
    #     """
    #     Check entry timeout function callback.
    #     This method can be used to override the entry-timeout.
    #     It is called whenever a limit entry order has been created,
    #     and is not yet fully filled.
    #     Configuration options in `unfilledtimeout` will be verified before this,
    #     so ensure to set these timeouts high enough.
    #
    #     For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/
    #
    #     When not implemented by a strategy, this simply returns False.
    #     :param pair: Pair the trade is for
    #     :param trade: Trade object.
    #     :param order: Order object.
    #     :param current_time: datetime object, containing the current datetime
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     :return bool: When True is returned, then the entry order is cancelled.
    #     """
    #     return False
    #
    # def check_exit_timeout(self, pair: str, trade: 'Trade', order: 'Order',
    #                        current_time: datetime, **kwargs) -> bool:
    #     """
    #     Check exit timeout function callback.
    #     This method can be used to override the exit-timeout.
    #     It is called whenever a limit exit order has been created,
    #     and is not yet fully filled.
    #     Configuration options in `unfilledtimeout` will be verified before this,
    #     so ensure to set these timeouts high enough.
    #
    #     For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/
    #
    #     When not implemented by a strategy, this simply returns False.
    #     :param pair: Pair the trade is for
    #     :param trade: Trade object.
    #     :param order: Order object.
    #     :param current_time: datetime object, containing the current datetime
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     :return bool: When True is returned, then the exit-order is cancelled.
    #     """
    #     return False
    #
    # def adjust_trade_position(self, trade: 'Trade', current_time: datetime,
    #                           current_rate: float, current_profit: float,
    #                           min_stake: Optional[float], max_stake: float,
    #                           current_entry_rate: float, current_exit_rate: float,
    #                           current_entry_profit: float, current_exit_profit: float,
    #                           **kwargs) -> Optional[float]:
    #     """
    #     Custom trade adjustment logic, returning the stake amount that a trade should be
    #     increased or decreased.
    #     This means extra buy or sell orders with additional fees.
    #     Only called when `position_adjustment_enable` is set to True.
    #
    #     For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/
    #
    #     When not implemented by a strategy, returns None
    #
    #     :param trade: trade object.
    #     :param current_time: datetime object, containing the current datetime
    #     :param current_rate: Current buy rate.
    #     :param current_profit: Current profit (as ratio), calculated based on current_rate.
    #     :param min_stake: Minimal stake size allowed by exchange (for both entries and exits)
    #     :param max_stake: Maximum stake allowed (either through balance, or by exchange limits).
    #     :param current_entry_rate: Current rate using entry pricing.
    #     :param current_exit_rate: Current rate using exit pricing.
    #     :param current_entry_profit: Current profit using entry pricing.
    #     :param current_exit_profit: Current profit using exit pricing.
    #     :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
    #     :return float: Stake amount to adjust your trade,
    #                     Positive values to increase position, Negative values to decrease position.
    #                     Return None for no action.
    #     """
    #     return None
    #
    # def leverage(self, pair: str, current_time: datetime, current_rate: float,
    #              proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
    #              side: str, **kwargs) -> float:
    #     """
    #     Customize leverage for each new trade. This method is only called in futures mode.
    #
    #     :param pair: Pair that's currently analyzed
    #     :param current_time: datetime object, containing the current datetime
    #     :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
    #     :param proposed_leverage: A leverage proposed by the bot.
    #     :param max_leverage: Max leverage allowed on this pair
    #     :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
    #     :param side: 'long' or 'short' - indicating the direction of the proposed trade
    #     :return: A leverage amount, which is between 1.0 and max_leverage.
    #     """
    #     return 1.0
