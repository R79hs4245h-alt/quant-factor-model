"""
市场状态判别器 v6.1
基于 HMM / 多指标投票 动态识别市场状态(牛市/熊市/震荡)
并据此动态调整因子权重

核心逻辑:
  1. 收集市场特征(趋势/波动率/广度/情绪/资金)
  2. 多维度投票判别市场状态
  3. 不同状态使用不同因子权重配置
  4. 平滑过渡避免频繁切换
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from enum import Enum
from collections import deque

from config import (
    WEIGHTS, TIMING_MA_SHORT, TIMING_MA_LONG, TIMING_RSI_PERIOD,
    BEAR_MARKET_MAX_POSITION, DRAWDOWN_ALERT
)
from utils import get_logger

logger = get_logger("regime")


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL = "bull"         # 牛市: 趋势向上,波动适中
    BEAR = "bear"         # 熊市: 趋势向下,波动大
    RANGE = "range"       # 震荡: 无明显趋势
    TRANSITION = "transition"  # 转换期


# 不同市场状态下的因子权重调整系数
REGIME_FACTOR_ADJUSTMENTS = {
    MarketRegime.BULL: {
        # 牛市: 强调动量、技术、资金流
        "momentum": 1.5, "technical": 1.3, "fund_flow": 1.3,
        "trend_strength": 1.4, "volume_price": 1.2,
        "value": 0.7, "reversal": 0.6, "risk": 0.8,
    },
    MarketRegime.BEAR: {
        # 熊市: 强调价值、质量、风险控制
        "value": 1.5, "quality": 1.4, "risk": 1.6,
        "fundamental": 1.3, "liquidity": 1.2,
        "momentum": 0.5, "technical": 0.6, "fund_flow": 0.7,
        "trend_strength": 0.6, "sentiment": 0.7,
    },
    MarketRegime.RANGE: {
        # 震荡: 强调反转、价值、流动性
        "reversal": 1.4, "value": 1.2, "liquidity": 1.2,
        "fundamental": 1.1, "chip_distribution": 1.2,
        "momentum": 0.7, "trend_strength": 0.6,
    },
    MarketRegime.TRANSITION: {
        # 转换期: 均衡配置,偏向防御
        "quality": 1.2, "risk": 1.3, "liquidity": 1.1,
        "momentum": 0.8, "technical": 0.8,
    },
}


class MarketRegimeDetector:
    """市场状态判别器"""

    def __init__(self,
                 lookback: int = 60,
                 smoothing_window: int = 5,
                 transition_threshold: float = 0.6):
        """
        :param lookback: 回看窗口(交易日)
        :param smoothing_window: 状态平滑窗口
        :param transition_threshold: 状态确认阈值(0-1)
        """
        self.lookback = lookback
        self.smoothing_window = smoothing_window
        self.transition_threshold = transition_threshold

        # 历史状态记录(用于平滑)
        self._regime_history: deque = deque(maxlen=smoothing_window)
        self._current_regime: MarketRegime = MarketRegime.RANGE
        self._regime_confidence: float = 0.5
        self._regime_scores: Dict[str, float] = {}

    def detect(self, index_data: pd.DataFrame,
               breadth_data: Optional[Dict] = None,
               sentiment_data: Optional[Dict] = None) -> Dict:
        """
        判别当前市场状态

        :param index_data: 指数日线数据(date, close, volume, high, low)
        :param breadth_data: 市场广度数据(涨跌家数, 新高新低等)
        :param sentiment_data: 情绪指标(换手率, 融资余额变化等)
        :return: {
            "regime": MarketRegime,
            "confidence": float,
            "scores": {bull_score, bear_score, range_score},
            "indicators": {...},
            "adjustments": {factor: multiplier},
        }
        """
        if index_data is None or index_data.empty:
            return self._default_result()

        indicators = self._compute_regime_indicators(index_data, breadth_data, sentiment_data)

        # 多维度投票
        bull_votes = 0
        bear_votes = 0
        range_votes = 0
        total_votes = 0

        for indicator, value in indicators.items():
            if value is None or np.isnan(value):
                continue

            total_votes += 1

            # 趋势类指标
            if indicator in ["ma_trend", "macd_trend", "price_momentum"]:
                if value > 0.3:
                    bull_votes += 1
                elif value < -0.3:
                    bear_votes += 1
                else:
                    range_votes += 1

            # 波动率指标
            elif indicator == "volatility_regime":
                if value < 0.8:  # 低波动
                    bull_votes += 0.5
                elif value > 1.5:  # 高波动
                    bear_votes += 1
                else:
                    range_votes += 0.5

            # 广度指标
            elif indicator in ["breadth_ratio", "advance_decline"]:
                if value > 0.6:
                    bull_votes += 1
                elif value < 0.4:
                    bear_votes += 1
                else:
                    range_votes += 1

            # RSI
            elif indicator == "rsi_regime":
                if value > 55:
                    bull_votes += 0.5
                elif value < 45:
                    bear_votes += 0.5
                else:
                    range_votes += 1

            # 资金流
            elif indicator == "fund_flow_signal":
                if value > 0:
                    bull_votes += 1
                elif value < 0:
                    bear_votes += 1
                else:
                    range_votes += 0.5

        # 计算得分
        total = max(total_votes, 1)
        bull_score = bull_votes / total
        bear_score = bear_votes / total
        range_score = range_votes / total

        self._regime_scores = {
            "bull": bull_score,
            "bear": bear_score,
            "range": range_score,
        }

        # 判定状态
        if bull_score >= self.transition_threshold:
            raw_regime = MarketRegime.BULL
        elif bear_score >= self.transition_threshold:
            raw_regime = MarketRegime.BEAR
        elif range_score >= 0.5:
            raw_regime = MarketRegime.RANGE
        else:
            raw_regime = MarketRegime.TRANSITION

        # 平滑处理: 如果新状态与历史不一致但信心不足,保持原状态
        smoothed_regime = self._smooth_regime(raw_regime, max(bull_score, bear_score, range_score))

        # 获取调整因子
        adjustments = REGIME_FACTOR_ADJUSTMENTS.get(smoothed_regime, {})

        # 确定目标仓位
        target_position = self._get_target_position(smoothed_regime, bear_score)

        result = {
            "regime": smoothed_regime,
            "regime_name": smoothed_regime.value,
            "confidence": max(bull_score, bear_score, range_score),
            "scores": self._regime_scores,
            "indicators": indicators,
            "adjustments": adjustments,
            "target_position": target_position,
            "raw_regime": raw_regime.value,
        }

        logger.info(f"市场状态: {smoothed_regime.value} "
                    f"(信心={result['confidence']:.0%}, "
                    f"牛={bull_score:.0%}/熊={bear_score:.0%}/震荡={range_score:.0%}) "
                    f"目标仓位={target_position:.0%}")

        return result

    def _compute_regime_indicators(self, index_data: pd.DataFrame,
                                    breadth_data: Optional[Dict],
                                    sentiment_data: Optional[Dict]) -> Dict:
        """计算市场状态指标"""
        indicators = {}

        try:
            df = index_data.sort_values("date").tail(self.lookback).copy()
            if len(df) < 20:
                return indicators

            close = df["close"].astype(float)
            high = df.get("high", close).astype(float)
            low = df.get("low", close).astype(float)
            volume = df.get("volume", pd.Series([1]*len(df))).astype(float)

            # 1. 均线趋势: 价格 vs MA20/MA60
            ma_short = close.rolling(TIMING_MA_SHORT).mean().iloc[-1]
            ma_long = close.rolling(TIMING_MA_LONG).mean().iloc[-1] if len(close) >= TIMING_MA_LONG else ma_short
            current = close.iloc[-1]

            if current > ma_short > ma_long:
                indicators["ma_trend"] = 1.0
            elif current > ma_short:
                indicators["ma_trend"] = 0.5
            elif current < ma_short < ma_long:
                indicators["ma_trend"] = -1.0
            elif current < ma_short:
                indicators["ma_trend"] = -0.5
            else:
                indicators["ma_trend"] = 0.0

            # 2. MACD趋势
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            macd_hist = (dif - dea).iloc[-1]
            indicators["macd_trend"] = float(np.tanh(macd_hist / (close.iloc[-1] * 0.01)))

            # 3. 价格动量
            if len(close) >= 20:
                mom = close.iloc[-1] / close.iloc[-20] - 1
                indicators["price_momentum"] = float(np.tanh(mom * 10))

            # 4. 波动率体制
            returns = close.pct_change().dropna()
            vol_recent = returns.tail(20).std()
            vol_long = returns.tail(60).std() if len(returns) >= 60 else vol_recent
            indicators["volatility_regime"] = float(vol_recent / (vol_long + 1e-10))

            # 5. RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(TIMING_RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(TIMING_RSI_PERIOD).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            indicators["rsi_regime"] = float(rsi.iloc[-1])

            # 6. 成交量趋势
            vol_ma = volume.rolling(20).mean()
            vol_ratio = volume.iloc[-1] / (vol_ma.iloc[-1] + 1e-10)
            indicators["volume_trend"] = float(vol_ratio)

        except Exception as e:
            logger.debug(f"指标计算异常: {e}")

        # 市场广度
        if breadth_data:
            indicators["breadth_ratio"] = breadth_data.get("advance_ratio", 0.5)
            indicators["advance_decline"] = breadth_data.get("advance_decline_ratio", 1.0)

        # 情绪指标
        if sentiment_data:
            indicators["fund_flow_signal"] = sentiment_data.get("fund_flow_signal", 0)
            indicators["sentiment_score"] = sentiment_data.get("sentiment_score", 50)

        return indicators

    def _smooth_regime(self, raw_regime: MarketRegime, confidence: float) -> MarketRegime:
        """平滑状态切换,避免频繁跳变"""
        self._regime_history.append(raw_regime)

        if len(self._regime_history) < 2:
            self._current_regime = raw_regime
            return raw_regime

        # 如果信心很高(>0.8),直接切换
        if confidence > 0.8:
            self._current_regime = raw_regime
            return raw_regime

        # 否则需要历史多数确认
        history_list = list(self._regime_history)
        regime_counts = {}
        for r in history_list:
            regime_counts[r] = regime_counts.get(r, 0) + 1

        most_common = max(regime_counts, key=regime_counts.get)
        if regime_counts[most_common] >= len(history_list) * 0.6:
            self._current_regime = most_common
            return most_common
        else:
            # 不够确认,保持当前状态
            return self._current_regime

    def _get_target_position(self, regime: MarketRegime, bear_score: float) -> float:
        """根据市场状态确定目标仓位"""
        if regime == MarketRegime.BULL:
            return 1.0  # 满仓
        elif regime == MarketRegime.BEAR:
            # 熊市根据严重程度调整
            if bear_score > 0.8:
                return 0.2  # 极度熊市 20%
            else:
                return BEAR_MARKET_MAX_POSITION  # 一般熊市 30%
        elif regime == MarketRegime.RANGE:
            return 0.7  # 震荡 70%
        else:  # TRANSITION
            return 0.5  # 转换期 50%

    def _default_result(self) -> Dict:
        """默认结果(数据不足时)"""
        return {
            "regime": MarketRegime.RANGE,
            "regime_name": "range",
            "confidence": 0.5,
            "scores": {"bull": 0.33, "bear": 0.33, "range": 0.34},
            "indicators": {},
            "adjustments": {},
            "target_position": 0.7,
        }

    def get_dynamic_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """根据当前市场状态动态调整因子权重"""
        adjustments = REGIME_FACTOR_ADJUSTMENTS.get(self._current_regime, {})

        adjusted = {}
        for factor, weight in base_weights.items():
            multiplier = adjustments.get(factor, 1.0)
            adjusted[factor] = weight * multiplier

        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def get_current_regime(self) -> MarketRegime:
        """获取当前市场状态"""
        return self._current_regime
