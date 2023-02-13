
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import (DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter, stoploss_from_open)
from pandas import DataFrame, Series
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
import talib.abstract as ta
import math
import pandas_ta as pta
import logging
from logging import FATAL
import time

logger = logging.getLogger(__name__)
# ZEBCD
def ewo_buy(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif
    
def ewo_sell(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['high'] * 100
    return emadif
# END   
class Z0BCDstra (IStrategy):

    def version(self) -> str:
        return "template-v1"

    INTERFACE_VERSION = 3

    # ROI table:
    # ROI is used for backtest/hyperopt to not overestimating the effectiveness of trailing stoploss
    # Remember to change this to 100000 for dry/live and turn on trailing stoploss below
    minimal_roi = {"0": 1}
   # minimal_roi = {"0": 0.5, "30": 0.75,"60": 0.05, "120": 0.025}
   #cminimal_roi = {"0": 0.14414,"13": 0.10123,"20": 0.03256,"47": 0.0177, "132": 0.01016,"177": 0.00328, "277": 0}
    optimize_buy_ema = True
    buy_length_ema = IntParameter(1, 15, default=6, optimize=optimize_buy_ema)

    optimize_buy_ema2 = True
    buy_length_ema2 = IntParameter(1, 15, default=6, optimize=optimize_buy_ema2)

    optimize_sell_ema = True
    sell_length_ema = IntParameter(1, 15, default=6, optimize=optimize_sell_ema)

    optimize_sell_ema2 = True
    sell_length_ema2 = IntParameter(1, 15, default=6, optimize=optimize_sell_ema2)

    optimize_sell_ema3 = True
    sell_length_ema3 = IntParameter(1, 15, default=6, optimize=optimize_sell_ema3)

    sell_min_profit = DecimalParameter(0, 0.05, default=0.005, decimals=2, optimize=False)

    sell_clear_old_trade = IntParameter(3, 10, default=10, optimize=False)
    sell_clear_old_trade_profit = IntParameter(-2, 2, default=1.5, optimize=False)
 #BCD hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)
#END
    # Stoploss:
    stoploss = -0.99
    use_custom_stoploss = False
    can_short = True
    # Trailing stop:
    # Turned off for backtest/hyperopt to not gaming the backtest.
    # Turn this on for dry/live
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 27
# ZEBCD
    is_optimize_ewo = True
    buy_rsi_fast = IntParameter(35, 50, default=45, space='buy', optimize=is_optimize_ewo)
    buy_rsi = IntParameter(15, 35, default=35, space='buy', optimize=is_optimize_ewo)
    buy_ewo = DecimalParameter(-6.0, 5, default=-5.585, space='buy', optimize=is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.942, space='buy', optimize=is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.95, 1.2, default=1.084, space='buy', optimize=is_optimize_ewo)
    #BCD
    sell_rsi_fast = IntParameter(70, 90, default=80, space='sell', optimize=is_optimize_ewo)
    sell_ema_low = DecimalParameter(0.9, 0.99, default=0.972, space='sell', optimize=is_optimize_ewo)
    sell_ewo = DecimalParameter(-6.0, 5, default=4.585, space='sell', optimize=is_optimize_ewo)
    sell_ema_high = DecimalParameter(0.95, 1.2, default=1.072, space='sell', optimize=is_optimize_ewo)
    sell_rsi = IntParameter(70, 90, default=80, space='sell', optimize=is_optimize_ewo)
    #BCD
    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=41, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=14, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.927, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 0, default=-0.91, decimals=2, space='buy', optimize=is_optimize_32)

    is_optimize_deadfish = True
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0, space='sell', optimize=is_optimize_deadfish)

    sell_fastx = IntParameter(50, 100, default=75, space='sell', optimize=True)
#END    

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # In-strat age filter
        dataframe['age_filter_ok'] = (dataframe['volume'].rolling(window=30, min_periods=30).min() > 0)

        # Drop unused columns to save memory
        drop_columns = ['open', 'high', 'low', 'close', 'volume']
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    # Use BTC indicators as informative for other pairs
    @informative('30m', 'BTC/USDT:USDT', '{base}_{column}_{timeframe}')
    def populate_indicators_btc_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        drop_columns = ['open', 'high', 'low', 'close', 'volume']
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # DOn't trade coins that have 0 volume candle on the past 72 candles
        dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Calculate EMA30 of RSI
        dataframe['ema_rsi_30'] = ta.EMA(dataframe['rsi'], 30)

        if not self.optimize_buy_ema:
            # Have the period of EMA on increment of 5 without having to use CategoricalParameter
            dataframe['ema_offset_buy'] = ta.EMA(dataframe, int(5 * self.buy_length_ema.value)) * 0.9

        if not self.optimize_buy_ema2:
            dataframe['ema_offset_buy2'] = ta.EMA(dataframe, int(5 * self.buy_length_ema2.value)) * 0.9

        if not self.optimize_sell_ema:
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(5 * self.sell_length_ema.value))

        if not self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(5 * self.sell_length_ema2.value))

        if not self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(5 * self.sell_length_ema3.value))
# BCD TEMA - Triple Exponential Moving Average
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)
# EZBCD
        # buy_1 indicators
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # ewo indicators
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['EWO_buy'] = ewo_buy(dataframe, 50, 200)
        dataframe['EWO_sell'] = ewo_sell(dataframe, 50, 200)

        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # loss sell indicators
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        dataframe['bb_width'] = (
                (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
#END

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions = []
        conditions_sell = []

        if self.optimize_buy_ema:
            dataframe['ema_offset_buy'] = ta.EMA(dataframe, int(5 * self.buy_length_ema.value)) * 0.9

        if self.optimize_buy_ema2:
            dataframe['ema_offset_buy2'] = ta.EMA(dataframe, int(5 * self.buy_length_ema2.value)) * 0.9

        if self.optimize_sell_ema:
            dataframe['ema_offset_sell'] = ta.EMA(dataframe, int(5 * self.sell_length_ema.value))

        if self.optimize_sell_ema2:
            dataframe['ema_offset_sell2'] = ta.EMA(dataframe, int(5 * self.sell_length_ema2.value))

        if self.optimize_sell_ema3:
            dataframe['ema_offset_sell3'] = ta.EMA(dataframe, int(5 * self.sell_length_ema3.value))

        dataframe['enter_tag'] = ''

        add_check = (
            dataframe['live_data_ok']
            &
            dataframe['age_filter_ok_1d']
            &
            (dataframe['close'] < dataframe['open'])
        )

        # Imitate exit signal colliding, where entry shouldn't happen when the exit signal is triggered.
        # So this check make sure no exit logics are triggered
        # 
        '''
        ema_check = (
            (dataframe['close'] > dataframe['ema_offset_sell'])
            &
            ((dataframe['close'] < dataframe['ema_offset_sell2']).rolling(2).min() == 0)
        )

        buy_offset_ema = (
            (dataframe['close'] < dataframe['ema_offset_buy'])
            &
            (dataframe['btc_rsi_30m'] >= 50)
            &
            ema_check
        )
        dataframe.loc[buy_offset_ema, 'enter_tag'] += 'ema_strong '
        conditions.append(buy_offset_ema)

        buy_offset_ema_2 = (
            (dataframe['close'] < dataframe['ema_offset_buy'])
            &
            (dataframe['btc_rsi_30m'] < 50)
            &
            ema_check
        )
        dataframe.loc[buy_offset_ema_2, 'enter_tag'] += 'ema_weak '
        conditions.append(buy_offset_ema_2)

        ema2_check = (
            ((dataframe['close'] < dataframe['ema_offset_sell3']).rolling(2).min() == 0)
        )

        buy_offset_ema2 = (
            ((dataframe['close'] < dataframe['ema_offset_buy2']).rolling(2).min() > 0)
            &
            (dataframe['btc_rsi_30m'] >= 50)
            &
            ema2_check
        )
        dataframe.loc[buy_offset_ema2, 'enter_tag'] += 'ema_2_strong '
        conditions.append(buy_offset_ema2)

        buy_offset_ema2_2 = (
            ((dataframe['close'] < dataframe['ema_offset_buy2']).rolling(2).min() > 0)
            &
            (dataframe['btc_rsi_30m'] < 50)
            &
            ema2_check
        )
        dataframe.loc[buy_offset_ema2_2, 'enter_tag'] += 'ema_2_weak '
        conditions.append(buy_offset_ema2_2)
        '''
# ZEBCD
        is_ewo_buy = (
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low.value) &
                (dataframe['EWO_buy'] > self.buy_ewo.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high.value) &
                (dataframe['rsi'] < self.buy_rsi.value)
        )

        buy_1 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
                (dataframe['rsi'] > self.buy_rsi_32.value) &
                (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
                (dataframe['cti'] < self.buy_cti_32.value)
        )

        conditions.append(is_ewo_buy)
        dataframe.loc[is_ewo_buy, 'enter_tag'] += 'ewo_buy'

        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                #&
                #add_check,
                'enter_long',
            ]= 1
            
        is_ewo_sell = (
                (dataframe['rsi_fast'] > self.sell_rsi_fast.value) &
                (dataframe['close'] > dataframe['ema_8'] * self.sell_ema_low.value) &
                (dataframe['EWO_sell'] < self.sell_ewo.value) &
                (dataframe['close'] > dataframe['ema_16'] * self.sell_ema_high.value) &
                (dataframe['rsi'] > self.sell_rsi.value)
        )
        conditions_sell.append(is_ewo_sell)
        dataframe.loc[is_ewo_sell, 'enter_tag'] += 'ewo_sell'
#END
        
        if conditions_sell:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_sell),
                #&
                #add_check,
                'enter_short',
            ]= 1
        
        return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (   (dataframe["rsi"] > 50)
                & (dataframe["tema"] > dataframe["bb_middleband2"])
                & (dataframe["tema"] < dataframe["tema"].shift(1))
                & (dataframe["volume"] > 0)  
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (   (dataframe["rsi"] > 50)
                & (dataframe["tema"] <= dataframe["bb_middleband2"])
                & (dataframe["tema"] > dataframe["tema"].shift(1))
                & (dataframe["volume"] > 0) 
            ),
            "exit_short",
        ] = 1
        
        return dataframe
    '''    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        
        return dataframe
    '''  
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        if (len(dataframe) > 1):
            previous_candle_1 = dataframe.iloc[-2].squeeze()

        enter_tag = 'empty'
        if hasattr(trade, 'enter_tag') and trade.enter_tag is not None:
            enter_tag = trade.enter_tag
        enter_tags = enter_tag.split()
        # Change current profit value to be tied to latest candle's close value, so that backtest and dry/live behavior is the same
        current_profit = trade.calc_profit_ratio(current_candle['close'])
        sl_new = 1
        timeframe_minutes = timeframe_to_minutes(self.timeframe)
        
        if current_time - timedelta(minutes=int(timeframe_minutes * self.sell_clear_old_trade.value)) > trade.open_date_utc:
            if (current_profit >= (-0.01 * self.sell_clear_old_trade_profit.value)):
                return f"sell_old_trade ({enter_tag})"

        if ((current_time - timedelta(minutes=timeframe_minutes)) > trade.open_date_utc):
            if (current_profit > self.sell_min_profit.value):
                return f"take_profit ({enter_tag})"
        if current_time - timedelta(days=1) > trade.open_date_utc:
#            if (current_candle["fastk"] > self.sell_fastx.value) and (current_profit > -0.05):
            if (current_profit > -0.1):
                return f"Xoa lenh 1 day ({enter_tag})"
     #   if (dataframe["tema"] < dataframe["tema"].shift(1)):
        
      #      return f"take_profit2 ({enter_tag})"                
#1 ZEBCD
        '''
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

       
        if hasattr(trade, 'enter_tag') and trade.enter_tag is not None:
            enter_tag = trade.enter_tag
        enter_tags = enter_tag.split()

        if any(c in ["ewo_buy", "ewo_sell", "buy_1" ] for c in enter_tags):
            if current_profit >= 0.05:
                return -0.005

        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return -0.001

        return self.stoploss
        '''
#END
## Custom Trailing stoploss ( credit to Perkmeister for this custom stoploss to help the strategy ride a green candle )
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        '''
        # hard stoploss profit
        HSL = - 0.04 #self.pHSL.value
        PF_1 = 0.02 #self.pPF_1.value
        SL_1 = 0.008 #self.pSL_1.value
        PF_2 = 0.08 #self.pPF_2.value
        SL_2 = 0.04 #self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return min(-0.01, max(stoploss_from_open(sl_profit, current_profit), -0.99))
        '''

############################################################################
