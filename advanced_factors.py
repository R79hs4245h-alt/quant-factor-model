"""
高阶四维选股因子引擎 v7.3
==========================

基于专业短线交易逻辑的四维度选股体系:
  维度一: 估值相对合理 (PEG/历史分位/市值空间)
  维度二: 触底反弹拐点确认 (底背离/金针探底/倍量阳)
  维度三: 主力资金真流入 (DDX/大单/龙虎榜)
  维度四: 主升浪启动检测 (均线粘合/量能递增/筹码峰)

适用于: 个股选股 + ETF/LOF场内基金推荐增强

核心设计原则:
  - 每个维度独立评分 (0-100),可单独使用或组合
  - 四维共振时信号最强 (右侧确认买入框架)
  - 严格区分"数据可得"与"数据不可得"场景,后者使用代理因子
  - 所有技术指标基于K线数据计算,不依赖外部Level-2数据源

依赖: numpy, pandas, ta-lib(可选,不可用时用内置实现)
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

try:
    from utils import get_logger
    logger = get_logger("advanced_factors")
except Exception:
    import logging
    logger = logging.getLogger("advanced_factors")


# =====================================================================
# 维度一: 估值相对合理 (短线安全垫)
# =====================================================================

class ValuationDimension:
    """
    估值相对合理评分

    量化标准:
      1. PEG与行业对标: PE < 行业均值×80% 且 PEG < 1
      2. 历史分位点: PB/PE处于近3年30%分位以下
      3. 市值隐含空间: 距板块龙头有3-5倍市值差距
    """

    def __init__(self):
        self.name = "valuation"
        self.weight = 0.25

    @staticmethod
    def calc_peg(pe: float, growth_rate: float) -> float:
        """计算PEG: PE / 增长率(%). PEG<1 为低估"""
        if pe is None or pe <= 0 or growth_rate is None or growth_rate <= 0:
            return float('inf')
        return pe / growth_rate

    @staticmethod
    def calc_history_percentile(value: float, history_series: pd.Series) -> float:
        """计算当前值在历史序列中的分位点 (0-1)"""
        if history_series is None or len(history_series) < 10:
            return 0.5
        clean = history_series.dropna()
        if len(clean) < 10:
            return 0.5
        return float((clean < value).sum() / len(clean))

    @staticmethod
    def calc_market_cap_space(target_cap: float, leader_cap: float) -> float:
        """计算市值空间倍数: 龙头市值 / 目标市值"""
        if target_cap is None or target_cap <= 0 or leader_cap is None or leader_cap <= 0:
            return 1.0
        return leader_cap / target_cap

    def score(
        self,
        pe: float = None,
        industry_avg_pe: float = None,
        growth_rate: float = None,
        pb: float = None,
        pb_history: pd.Series = None,
        pe_history: pd.Series = None,
        market_cap: float = None,
        leader_market_cap: float = None,
    ) -> Dict[str, Any]:
        """
        计算估值维度得分

        返回:
          {
            'score': 0-100,
            'peg': float,
            'peg_ok': bool,         # PEG<1
            'pe_vs_industry': float, # PE/行业PE 比值
            'pe_vs_industry_ok': bool, # PE < 行业×80%
            'pb_percentile': float,  # PB历史分位
            'pe_percentile': float,  # PE历史分位
            'percentile_ok': bool,   # 分位<30%
            'cap_space': float,      # 市值空间倍数
            'cap_space_ok': bool,    # 空间>=3倍
            'details': str,
          }
        """
        result = {
            'score': 50.0,
            'peg': None,
            'peg_ok': False,
            'pe_vs_industry': None,
            'pe_vs_industry_ok': False,
            'pb_percentile': None,
            'pe_percentile': None,
            'percentile_ok': False,
            'cap_space': None,
            'cap_space_ok': False,
            'details': '',
        }

        sub_scores = []

        # 1. PEG评估
        if pe is not None and growth_rate is not None:
            peg = self.calc_peg(pe, growth_rate)
            result['peg'] = round(peg, 2)
            result['peg_ok'] = peg < 1.0
            if peg < 0.5:
                sub_scores.append(90)
            elif peg < 0.8:
                sub_scores.append(80)
            elif peg < 1.0:
                sub_scores.append(70)
            elif peg < 1.5:
                sub_scores.append(50)
            else:
                sub_scores.append(30)

            # PE vs 行业
            if industry_avg_pe is not None and industry_avg_pe > 0:
                ratio = pe / industry_avg_pe
                result['pe_vs_industry'] = round(ratio, 2)
                result['pe_vs_industry_ok'] = ratio < 0.8
                if ratio < 0.6:
                    sub_scores.append(85)
                elif ratio < 0.8:
                    sub_scores.append(70)
                elif ratio < 1.0:
                    sub_scores.append(55)
                else:
                    sub_scores.append(35)

        # 2. 历史分位
        if pb is not None and pb_history is not None:
            pct = self.calc_history_percentile(pb, pb_history)
            result['pb_percentile'] = round(pct, 3)
            if pct < 0.15:
                sub_scores.append(90)
            elif pct < 0.30:
                sub_scores.append(75)
            elif pct < 0.50:
                sub_scores.append(55)
            else:
                sub_scores.append(30)

        if pe is not None and pe_history is not None:
            pct = self.calc_history_percentile(pe, pe_history)
            result['pe_percentile'] = round(pct, 3)
            result['percentile_ok'] = pct < 0.30
            if pct < 0.15:
                sub_scores.append(90)
            elif pct < 0.30:
                sub_scores.append(75)
            elif pct < 0.50:
                sub_scores.append(55)
            else:
                sub_scores.append(30)

        # 3. 市值空间
        if market_cap is not None and leader_market_cap is not None:
            space = self.calc_market_cap_space(market_cap, leader_market_cap)
            result['cap_space'] = round(space, 1)
            result['cap_space_ok'] = space >= 3.0
            if space >= 5.0:
                sub_scores.append(90)
            elif space >= 3.0:
                sub_scores.append(75)
            elif space >= 2.0:
                sub_scores.append(55)
            else:
                sub_scores.append(35)

        # 汇总
        if sub_scores:
            result['score'] = round(np.mean(sub_scores), 1)

        result['details'] = (
            f"PEG={result['peg']}({'OK' if result['peg_ok'] else 'X'}), "
            f"PEvs行业={result['pe_vs_industry']}({'OK' if result['pe_vs_industry_ok'] else 'X'}), "
            f"PB分位={result['pb_percentile']}, "
            f"市值空间={result['cap_space']}x"
        )
        return result


# =====================================================================
# 维度二: 触底反弹拐点确认 (衰竭转强)
# =====================================================================

class BottomReversalDimension:
    """
    触底反弹拐点确认评分

    量化标准:
      1. 衰竭缺口+长下影: 金针探底/锤子线 (下影>实体2倍)
      2. 底背离共振: 价格新低但RSI/MACD未新低
      3. 标志性倍量阳: 涨幅>5%且量是前5日均量2倍以上
    """

    def __init__(self):
        self.name = "bottom_reversal"
        self.weight = 0.25

    @staticmethod
    def _calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """计算RSI"""
        if len(close) < period + 1:
            return np.array([50.0] * len(close))
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        # Wilder平滑
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50.0
        return rsi

    @staticmethod
    def _calc_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        """计算MACD: DIF, DEA, HIST"""
        if len(close) < slow + signal:
            n = len(close)
            return np.zeros(n), np.zeros(n), np.zeros(n)

        def ema(data, period):
            result = np.zeros(len(data))
            result[0] = data[0]
            k = 2 / (period + 1)
            for i in range(1, len(data)):
                result[i] = data[i] * k + result[i-1] * (1 - k)
            return result

        ema_fast = ema(close, fast)
        ema_slow = ema(close, slow)
        dif = ema_fast - ema_slow
        dea = ema(dif, signal)
        hist = 2 * (dif - dea)
        return dif, dea, hist

    def detect_hammer(self, ohlc: pd.DataFrame) -> Dict[str, Any]:
        """
        检测金针探底/锤子线

        条件:
          - 下影线 > 实体2倍
          - 上影线很短
          - 处于下跌末端
        """
        if ohlc is None or len(ohlc) < 5:
            return {'detected': False, 'strength': 0}

        recent = ohlc.tail(5)
        open_ = recent['open'].values
        high = recent['high'].values
        low = recent['low'].values
        close = recent['close'].values

        # 检查最近3天是否有锤子线
        for i in range(-3, 0):
            body = abs(close[i] - open_[i])
            lower_shadow = min(open_[i], close[i]) - low[i]
            upper_shadow = high[i] - max(open_[i], close[i])

            if body <= 0:
                continue

            ratio = lower_shadow / body if body > 0 else 0

            # 锤子线: 下影>实体2倍, 上影<实体0.5倍
            if ratio >= 2.0 and upper_shadow < body * 0.5:
                # 前期处于下跌趋势
                prior_trend = close[i-1] < close[max(i-3, -5)] if len(close) > 5 else True
                strength = min(ratio / 3.0, 1.0) * (1.0 if prior_trend else 0.6)
                return {
                    'detected': True,
                    'strength': round(strength, 2),
                    'day_offset': i,
                    'lower_shadow_ratio': round(ratio, 2),
                }

        return {'detected': False, 'strength': 0}

    def detect_bottom_divergence(self, ohlc: pd.DataFrame) -> Dict[str, Any]:
        """
        检测底背离: 价格创新低但RSI/MACD未创新低

        条件:
          - 最近20日内有2个低点
          - 第2个低点价格 < 第1个低点
          - 第2个低点RSI > 第1个低点RSI (底背离)
          - MACD绿柱也未创新低 (共振)
        """
        if ohlc is None or len(ohlc) < 30:
            return {'detected': False, 'strength': 0}

        df = ohlc.tail(60).copy()
        close = df['close'].values
        rsi = self._calc_rsi(close, 14)
        dif, dea, hist = self._calc_macd(close)

        # 找低点 (局部最小值)
        lows = []
        for i in range(2, len(close) - 2):
            if close[i] < close[i-1] and close[i] < close[i-2] and \
               close[i] < close[i+1] and close[i] < close[i+2]:
                lows.append(i)

        if len(lows) < 2:
            return {'detected': False, 'strength': 0}

        # 取最近的两个低点
        idx1, idx2 = lows[-2], lows[-1]
        price1, price2 = close[idx1], close[idx2]
        rsi1, rsi2 = rsi[idx1], rsi[idx2]
        hist1, hist2 = hist[idx1], hist[idx2]

        divergence_count = 0

        # 价格创新低
        if price2 < price1:
            # RSI底背离
            rsi_div = rsi2 > rsi1
            if rsi_div:
                divergence_count += 1

            # MACD绿柱底背离 (hist2 > hist1 即绿柱缩短)
            macd_div = hist2 > hist1
            if macd_div:
                divergence_count += 1

            if divergence_count > 0:
                strength = divergence_count / 2.0
                # 60分钟MACD金叉近似 (日线MACD即将金叉)
                macd_gold_cross = (dif[idx2] > dea[idx2]) and (dif[idx2-1] <= dea[idx2-1])
                if macd_gold_cross:
                    strength = min(strength + 0.3, 1.0)

                return {
                    'detected': True,
                    'strength': round(strength, 2),
                    'rsi_divergence': rsi_div,
                    'macd_divergence': macd_div,
                    'macd_gold_cross': macd_gold_cross,
                    'price_low1': round(price1, 2),
                    'price_low2': round(price2, 2),
                    'rsi_low1': round(rsi1, 1),
                    'rsi_low2': round(rsi2, 1),
                }

        return {'detected': False, 'strength': 0}

    def detect_volume_breakout(self, ohlc: pd.DataFrame) -> Dict[str, Any]:
        """
        检测标志性倍量阳线

        条件:
          - 涨幅 > 5%
          - 成交量 > 前5日均量 × 2
        """
        if ohlc is None or len(ohlc) < 10:
            return {'detected': False, 'strength': 0}

        df = ohlc.tail(10).copy()
        close = df['close'].values
        open_ = df['open'].values
        volume = df['volume'].values if 'volume' in df.columns else None

        if volume is None:
            return {'detected': False, 'strength': 0}

        # 最近3天找倍量阳
        for i in range(-3, 0):
            change_pct = (close[i] - close[i-1]) / close[i-1] * 100 if close[i-1] > 0 else 0
            is_positive = close[i] > open_[i]

            # 前5日均量
            start = max(i - 5, -10)
            avg_vol = np.mean(volume[start:i]) if i > start else 0
            if avg_vol <= 0:
                continue

            vol_ratio = volume[i] / avg_vol

            if change_pct > 5 and vol_ratio >= 2.0 and is_positive:
                strength = min((change_pct / 8.0) * (vol_ratio / 3.0), 1.0)
                return {
                    'detected': True,
                    'strength': round(strength, 2),
                    'change_pct': round(change_pct, 2),
                    'volume_ratio': round(vol_ratio, 2),
                    'day_offset': i,
                }

        return {'detected': False, 'strength': 0}

    def score(self, ohlc: pd.DataFrame) -> Dict[str, Any]:
        """
        计算触底反弹维度得分

        参数:
          ohlc: 含 open/high/low/close/volume 列的K线DataFrame

        返回:
          {'score': 0-100, 'hammer': ..., 'divergence': ..., 'volume_breakout': ..., 'details': str}
        """
        hammer = self.detect_hammer(ohlc)
        divergence = self.detect_bottom_divergence(ohlc)
        vol_breakout = self.detect_volume_breakout(ohlc)

        sub_scores = []

        if hammer['detected']:
            sub_scores.append(60 + hammer['strength'] * 40)
        else:
            sub_scores.append(30)

        if divergence['detected']:
            base = 60 + divergence['strength'] * 40
            sub_scores.append(base)
        else:
            sub_scores.append(25)

        if vol_breakout['detected']:
            sub_scores.append(70 + vol_breakout['strength'] * 30)
        else:
            sub_scores.append(30)

        score = round(np.mean(sub_scores), 1)

        details = (
            f"锤子线={'✓' if hammer['detected'] else '✗'}"
            f"(强度{hammer.get('strength', 0)}), "
            f"底背离={'✓' if divergence['detected'] else '✗'}"
            f"(强度{divergence.get('strength', 0)}), "
            f"倍量阳={'✓' if vol_breakout['detected'] else '✗'}"
            f"(强度{vol_breakout.get('strength', 0)})"
        )

        return {
            'score': score,
            'hammer': hammer,
            'divergence': divergence,
            'volume_breakout': vol_breakout,
            'details': details,
        }


# =====================================================================
# 维度三: 主力资金真流入 (识别伪装)
# =====================================================================

class CapitalFlowDimension:
    """
    主力资金真流入评分

    量化标准:
      1. DDX大单净量: 连续3日飘红且当日值>0.5
      2. 大单成交方向: 主动性买入特大单频繁且股价不跌 (压盘吸筹)
      3. 龙虎榜机构席位: 2家以上机构净买入且无游资做T
    """

    def __init__(self):
        self.name = "capital_flow"
        self.weight = 0.25

    def score(
        self,
        ddx_series: pd.Series = None,
        large_order_net: float = None,
        large_order_ratio: float = None,
        dragon_tiger_list: List[Dict] = None,
        volume: np.ndarray = None,
        close: np.ndarray = None,
        turnover_rate: float = None,
    ) -> Dict[str, Any]:
        """
        计算主力资金维度得分

        参数:
          ddx_series: DDX大单净买入序列(最近5日)
          large_order_net: 大单净买入额(万元)
          large_order_ratio: 大单占比 (0-1)
          dragon_tiger_list: 龙虎榜数据 [{'seat_type':'机构','net_buy':xxx,'is_t':bool}]
          volume: 成交量序列(用于代理计算)
          close: 收盘价序列
          turnover_rate: 换手率

        返回:
          {'score': 0-100, 'ddx_ok': bool, 'large_order_ok': bool, 'dragon_tiger_ok': bool, ...}
        """
        result = {
            'score': 50.0,
            'ddx_score': 50,
            'ddx_consecutive': 0,
            'ddx_ok': False,
            'large_order_score': 50,
            'large_order_ok': False,
            'is_accumulating': False,
            'dragon_tiger_score': 50,
            'dragon_tiger_ok': False,
            'institutional_count': 0,
            'has_hot_money_t': False,
            'details': '',
        }

        sub_scores = []

        # 1. DDX大单净量
        if ddx_series is not None and len(ddx_series) >= 3:
            ddx_values = ddx_series.dropna()
            if len(ddx_values) >= 3:
                # 连续飘红天数
                consecutive = 0
                for v in ddx_values.iloc[::-1]:
                    if v > 0:
                        consecutive += 1
                    else:
                        break

                latest_ddx = ddx_values.iloc[-1]
                result['ddx_consecutive'] = consecutive
                result['ddx_ok'] = consecutive >= 3 and latest_ddx > 0.5

                if consecutive >= 3 and latest_ddx > 0.5:
                    result['ddx_score'] = 90
                elif consecutive >= 2 and latest_ddx > 0:
                    result['ddx_score'] = 70
                elif latest_ddx > 0:
                    result['ddx_score'] = 55
                else:
                    result['ddx_score'] = 30

                sub_scores.append(result['ddx_score'])
        else:
            # 代理计算: 用量价关系近似大单方向
            if volume is not None and close is not None and len(volume) >= 5:
                proxy_ddx = self._proxy_ddx(close, volume)
                result['ddx_consecutive'] = proxy_ddx['consecutive']
                result['ddx_ok'] = proxy_ddx['consecutive'] >= 3
                result['ddx_score'] = proxy_ddx['score']
                result['is_accumulating'] = proxy_ddx.get('accumulating', False)
                sub_scores.append(proxy_ddx['score'])
            else:
                sub_scores.append(40)

        # 2. 大单成交方向
        if large_order_net is not None:
            if large_order_net > 5000:  # >5000万
                result['large_order_score'] = 90
                result['large_order_ok'] = True
            elif large_order_net > 1000:
                result['large_order_score'] = 70
                result['large_order_ok'] = True
            elif large_order_net > 0:
                result['large_order_score'] = 55
            elif large_order_net > -1000:
                result['large_order_score'] = 40
            else:
                result['large_order_score'] = 20
            sub_scores.append(result['large_order_score'])

            # 压盘吸筹判断: 大单净流入但股价未大涨
            if close is not None and len(close) >= 3:
                recent_change = (close[-1] - close[-3]) / close[-3] if close[-3] > 0 else 0
                if large_order_net > 1000 and abs(recent_change) < 0.02:
                    result['is_accumulating'] = True
        else:
            # 用大单占比代理
            if large_order_ratio is not None:
                if large_order_ratio > 0.15:
                    result['large_order_score'] = 80
                    result['large_order_ok'] = True
                elif large_order_ratio > 0.08:
                    result['large_order_score'] = 65
                elif large_order_ratio > 0.03:
                    result['large_order_score'] = 50
                else:
                    result['large_order_score'] = 30
                sub_scores.append(result['large_order_score'])
            else:
                sub_scores.append(45)

        # 3. 龙虎榜
        if dragon_tiger_list is not None and len(dragon_tiger_list) > 0:
            inst_count = sum(1 for s in dragon_tiger_list if s.get('seat_type') == '机构')
            inst_net = sum(s.get('net_buy', 0) for s in dragon_tiger_list if s.get('seat_type') == '机构')
            has_t = any(s.get('is_t', False) for s in dragon_tiger_list)

            result['institutional_count'] = inst_count
            result['has_hot_money_t'] = has_t

            if inst_count >= 2 and not has_t:
                result['dragon_tiger_score'] = 90
                result['dragon_tiger_ok'] = True
            elif inst_count >= 2 and has_t:
                result['dragon_tiger_score'] = 65  # 有游资做T,降低评级
            elif inst_count >= 1:
                result['dragon_tiger_score'] = 60
            elif has_t:
                result['dragon_tiger_score'] = 40  # 纯游资
            else:
                result['dragon_tiger_score'] = 45
            sub_scores.append(result['dragon_tiger_score'])
        else:
            # 无龙虎榜数据,不扣分(中性)
            sub_scores.append(50)

        # 汇总
        if sub_scores:
            result['score'] = round(np.mean(sub_scores), 1)

        result['details'] = (
            f"DDX连续={result['ddx_consecutive']}日({'OK' if result['ddx_ok'] else 'X'}), "
            f"大单={'OK' if result['large_order_ok'] else 'X'}, "
            f"压盘吸筹={'✓' if result['is_accumulating'] else '✗'}, "
            f"龙虎榜机构={result['institutional_count']}家({'OK' if result['dragon_tiger_ok'] else 'X'})"
        )
        return result

    @staticmethod
    def _proxy_ddx(close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """
        代理DDX计算: 无Level-2数据时用量价关系近似

        逻辑:
          - 放量上涨 = 大单买入 (DDX+)
          - 缩量下跌 = 大单卖出 (DDX-)
          - 放量不涨 = 压盘吸筹
        """
        n = min(len(close), len(volume), 5)
        if n < 3:
            return {'consecutive': 0, 'score': 40, 'accumulating': False}

        consecutive = 0
        accumulating = False

        for i in range(-n, 0):
            if i == 0 or i - 1 < -n:
                continue
            price_change = (close[i] - close[i-1]) / close[i-1] if close[i-1] > 0 else 0
            vol_change = (volume[i] - volume[i-1]) / volume[i-1] if volume[i-1] > 0 else 0

            # 放量上涨 = DDX正
            if price_change > 0.005 and vol_change > 0.1:
                consecutive += 1
            # 放量但价格不动 = 压盘吸筹
            elif abs(price_change) < 0.005 and vol_change > 0.3:
                accumulating = True
                consecutive += 1
            # 缩量下跌 = DDX正 (卖压衰竭)
            elif price_change < -0.005 and vol_change < -0.1:
                consecutive += 1
            else:
                break

        score = 40 + min(consecutive * 15, 50)
        return {'consecutive': consecutive, 'score': score, 'accumulating': accumulating}


# =====================================================================
# 维度四: 主升浪启动检测 (加速临界点)
# =====================================================================

class MainRallyDimension:
    """
    主升浪启动检测评分

    量化标准:
      1. 均线粘合后多头发散: MA5/10/20/60粘合后MA5金叉MA10,MA20拐头向上
      2. 成交量梯度递增: 连续5-8日阶梯式温和放大(每日+10%-15%)
      3. 老鸭头/空中加油: 缩量回调不破MA20,放量阳线站上MA5,MACD水上金叉
      4. 筹码断层: 获利比例>85%,单峰密集,上方无压力
    """

    def __init__(self):
        self.name = "main_rally"
        self.weight = 0.25

    @staticmethod
    def _calc_ma(close: np.ndarray, period: int) -> np.ndarray:
        """计算移动平均线"""
        if len(close) < period:
            return np.full(len(close), np.nan)
        ma = np.convolve(close, np.ones(period) / period, mode='valid')
        result = np.full(len(close), np.nan)
        result[period-1:] = ma
        return result

    @staticmethod
    def _calc_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        """计算MACD: DIF, DEA, HIST (复用BottomReversalDimension实现)"""
        if len(close) < slow + signal:
            n = len(close)
            return np.zeros(n), np.zeros(n), np.zeros(n)

        def ema(data, period):
            result = np.zeros(len(data))
            result[0] = data[0]
            k = 2 / (period + 1)
            for i in range(1, len(data)):
                result[i] = data[i] * k + result[i-1] * (1 - k)
            return result

        ema_fast = ema(close, fast)
        ema_slow = ema(close, slow)
        dif = ema_fast - ema_slow
        dea = ema(dif, signal)
        hist = 2 * (dif - dea)
        return dif, dea, hist

    def detect_ma_convergence(self, ohlc: pd.DataFrame) -> Dict[str, Any]:
        """
        检测均线粘合后多头发散

        条件:
          1. MA5/MA10/MA20/MA60 在过去10日内曾高度粘合 (最大偏离<2%)
          2. 当前MA5 > MA10 (金叉)
          3. MA20拐头向上
          4. 四线呈多头排列 (MA5>MA10>MA20>MA60)
        """
        if ohlc is None or len(ohlc) < 70:
            return {'detected': False, 'strength': 0}

        close = ohlc['close'].values
        ma5 = self._calc_ma(close, 5)
        ma10 = self._calc_ma(close, 10)
        ma20 = self._calc_ma(close, 20)
        ma60 = self._calc_ma(close, 60)

        # 检查最近10日是否有粘合
        lookback = 10
        converged = False
        for i in range(-lookback, 0):
            if np.isnan(ma60[i]):
                continue
            vals = [ma5[i], ma10[i], ma20[i], ma60[i]]
            vals = [v for v in vals if not np.isnan(v)]
            if len(vals) < 4:
                continue
            max_val = max(vals)
            min_val = min(vals)
            spread = (max_val - min_val) / min_val if min_val > 0 else 0
            if spread < 0.02:  # 偏离<2%
                converged = True
                break

        # 当前多头发散
        i = -1
        if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]) or np.isnan(ma60[i]):
            return {'detected': False, 'strength': 0}

        golden_cross = ma5[i] > ma10[i]
        ma20_turning_up = ma20[i] > ma20[i-3] if not np.isnan(ma20[i-3]) else False
        bull_alignment = ma5[i] > ma10[i] > ma20[i] > ma60[i]

        # 发散程度
        current_spread = (ma5[i] - ma60[i]) / ma60[i] if ma60[i] > 0 else 0

        if converged and golden_cross and ma20_turning_up:
            strength = 0.6
            if bull_alignment:
                strength = min(strength + 0.3, 1.0)
            if current_spread > 0.05:
                strength = min(strength + 0.1, 1.0)
            return {
                'detected': True,
                'strength': round(strength, 2),
                'converged': converged,
                'golden_cross': golden_cross,
                'ma20_turning_up': ma20_turning_up,
                'bull_alignment': bull_alignment,
                'spread': round(current_spread, 3),
            }

        return {
            'detected': False,
            'strength': 0,
            'converged': converged,
            'golden_cross': golden_cross,
            'ma20_turning_up': ma20_turning_up,
            'bull_alignment': bull_alignment,
        }

    def detect_volume_gradient(self, ohlc: pd.DataFrame) -> Dict[str, Any]:
        """
        检测成交量梯度递增

        条件:
          - 连续5-8个交易日量能阶梯式温和放大
          - 每日量能递增10%-15%
          - 非单日爆量
        """
        if ohlc is None or len(ohlc) < 15:
            return {'detected': False, 'strength': 0}

        volume = ohlc['volume'].values if 'volume' in ohlc.columns else None
        if volume is None:
            return {'detected': False, 'strength': 0}

        # 检查最近5-8天
        for window in [8, 7, 6, 5]:
            if len(volume) < window:
                continue
            recent_vol = volume[-window:]
            increases = 0
            valid_increases = 0

            for i in range(1, len(recent_vol)):
                if recent_vol[i-1] > 0:
                    ratio = (recent_vol[i] - recent_vol[i-1]) / recent_vol[i-1]
                    if 0.05 <= ratio <= 0.25:  # 温和递增5%-25%
                        increases += 1
                        valid_increases += 1
                    elif ratio > 0:
                        increases += 1

            # 至少60%的天数是温和递增的
            if valid_increases >= window * 0.5 and increases >= window * 0.7:
                strength = min(valid_increases / (window - 1), 1.0)
                return {
                    'detected': True,
                    'strength': round(strength, 2),
                    'window': window,
                    'valid_increases': valid_increases,
                    'total_increases': increases,
                }

        return {'detected': False, 'strength': 0}

    def detect_air_refueling(self, ohlc: pd.DataFrame) -> Dict[str, Any]:
        """
        检测老鸭头/空中加油形态

        条件:
          1. 经历一波拉升后缩量回调
          2. 回调不破MA20
          3. 随后放量阳线重新站上MA5
          4. MACD在零轴上方再度金叉 (水上金叉)
        """
        if ohlc is None or len(ohlc) < 40:
            return {'detected': False, 'strength': 0}

        df = ohlc.tail(40).copy()
        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else None
        ma5 = self._calc_ma(close, 5)
        ma20 = self._calc_ma(close, 20)
        dif, dea, hist = self._calc_macd(close)

        # 1. 找前期拉升高点 (前20日内涨幅最大的一段)
        rally_end = None
        for i in range(20, 5, -1):
            if i >= len(close):
                continue
            prior_rally = (close[i] - close[i-10]) / close[i-10] if close[i-10] > 0 else 0
            if prior_rally > 0.08:  # 拉升>8%
                rally_end = i
                break

        if rally_end is None:
            return {'detected': False, 'strength': 0}

        # 2. 回调阶段: 从rally_end到最近
        pullback_start = rally_end
        pullback_end = len(close) - 1

        # 回调幅度
        high_price = close[pullback_start]
        low_in_pullback = min(close[pullback_start:pullback_end+1])
        pullback_pct = (high_price - low_in_pullback) / high_price if high_price > 0 else 0

        # 3. 回调不破MA20
        ma20_at_low = ma20[np.argmin(close[pullback_start:pullback_end+1]) + pullback_start]
        broke_ma20 = low_in_pullback < ma20_at_low if not np.isnan(ma20_at_low) else False

        # 4. 放量阳线站上MA5
        last_idx = pullback_end
        stood_above_ma5 = close[last_idx] > ma5[last_idx] if not np.isnan(ma5[last_idx]) else False
        is_positive = close[last_idx] > df['open'].values[last_idx]

        vol_confirm = False
        if volume is not None and last_idx > 0:
            avg_vol = np.mean(volume[max(0, last_idx-5):last_idx])
            vol_confirm = volume[last_idx] > avg_vol * 1.2 if avg_vol > 0 else False

        # 5. MACD水上金叉
        macd_above_zero = dif[last_idx] > 0 and dea[last_idx] > 0
        macd_golden = (dif[last_idx] > dea[last_idx]) and (dif[last_idx-1] <= dea[last_idx-1])

        # 评分
        conditions_met = 0
        if 0.03 <= pullback_pct <= 0.15:  # 适度回调
            conditions_met += 1
        if not broke_ma20:
            conditions_met += 1
        if stood_above_ma5 and is_positive:
            conditions_met += 1
        if vol_confirm:
            conditions_met += 1
        if macd_above_zero and macd_golden:
            conditions_met += 1

        if conditions_met >= 3:
            strength = conditions_met / 5.0
            return {
                'detected': True,
                'strength': round(strength, 2),
                'pullback_pct': round(pullback_pct, 3),
                'broke_ma20': broke_ma20,
                'stood_above_ma5': stood_above_ma5,
                'vol_confirm': vol_confirm,
                'macd_above_zero': macd_above_zero,
                'macd_golden_cross': macd_golden,
                'conditions_met': conditions_met,
            }

        return {
            'detected': False,
            'strength': 0,
            'conditions_met': conditions_met,
        }

    def detect_chip_distribution(self, ohlc: pd.DataFrame, turnover_rates: np.ndarray = None) -> Dict[str, Any]:
        """
        检测筹码分布状态

        条件:
          1. 获利比例 > 85%
          2. 单峰密集 (筹码集中度高)
          3. 上方无压力 (当前价上方筹码少)

        注: 无真实筹码数据时用历史成交量分布近似
        """
        if ohlc is None or len(ohlc) < 60:
            return {'detected': False, 'strength': 0}

        close = ohlc['close'].values
        volume = ohlc['volume'].values if 'volume' in ohlc.columns else None

        if volume is None:
            return {'detected': False, 'strength': 0}

        current_price = close[-1]

        # 近120日成交量加权价格分布 (筹码近似)
        lookback = min(120, len(close))
        prices = close[-lookback:]
        vols = volume[-lookback:]

        if turnover_rates is not None and len(turnover_rates) >= lookback:
            decay = np.exp(-np.arange(lookback)[::-1] * 0.02)  # 时间衰减
            weights = vols * decay
        else:
            decay = np.exp(-np.arange(lookback)[::-1] * 0.02)
            weights = vols * decay

        total_weight = np.sum(weights)
        if total_weight <= 0:
            return {'detected': False, 'strength': 0}

        # 获利比例
        profit_mask = prices <= current_price
        profit_ratio = np.sum(weights[profit_mask]) / total_weight

        # 筹码集中度 (HHI)
        price_bins = np.linspace(prices.min(), prices.max(), 30)
        bin_weights = np.zeros(30)
        for i in range(len(prices)):
            bin_idx = np.searchsorted(price_bins, prices[i]) - 1
            bin_idx = max(0, min(29, bin_idx))
            bin_weights[bin_idx] += weights[i]

        bin_pct = bin_weights / total_weight
        hhi = np.sum(bin_pct ** 2)  # 赫芬达尔指数

        # 上方压力 (当前价以上的筹码占比)
        above_mask = prices > current_price
        above_pressure = np.sum(weights[above_mask]) / total_weight

        # 单峰检测
        peak_count = 0
        for i in range(1, 29):
            if bin_pct[i] > bin_pct[i-1] and bin_pct[i] > bin_pct[i+1] and bin_pct[i] > 0.05:
                peak_count += 1

        is_single_peak = peak_count == 1

        # 评分
        score_parts = 0
        if profit_ratio > 0.85:
            score_parts += 1
        if hhi > 0.15:  # 较集中
            score_parts += 1
        if above_pressure < 0.20:
            score_parts += 1
        if is_single_peak:
            score_parts += 1

        if score_parts >= 3:
            strength = min(score_parts / 4.0, 1.0)
            return {
                'detected': True,
                'strength': round(strength, 2),
                'profit_ratio': round(profit_ratio, 3),
                'concentration_hhi': round(float(hhi), 3),
                'above_pressure': round(above_pressure, 3),
                'is_single_peak': is_single_peak,
                'peak_count': peak_count,
            }

        return {
            'detected': False,
            'strength': 0,
            'profit_ratio': round(profit_ratio, 3),
            'concentration_hhi': round(float(hhi), 3),
            'above_pressure': round(above_pressure, 3),
            'is_single_peak': is_single_peak,
        }

    def score(self, ohlc: pd.DataFrame, turnover_rates: np.ndarray = None) -> Dict[str, Any]:
        """
        计算主升浪维度得分

        返回:
          {'score': 0-100, 'ma_convergence': ..., 'volume_gradient': ..., 'air_refueling': ..., 'chip': ...}
        """
        ma_conv = self.detect_ma_convergence(ohlc)
        vol_grad = self.detect_volume_gradient(ohlc)
        air_refuel = self.detect_air_refueling(ohlc)
        chip = self.detect_chip_distribution(ohlc, turnover_rates)

        sub_scores = []

        # 均线粘合发散
        if ma_conv['detected']:
            sub_scores.append(60 + ma_conv['strength'] * 40)
        else:
            base = 35
            if ma_conv.get('golden_cross'):
                base += 10
            if ma_conv.get('ma20_turning_up'):
                base += 5
            if ma_conv.get('bull_alignment'):
                base += 10
            sub_scores.append(min(base, 60))

        # 量能梯度
        if vol_grad['detected']:
            sub_scores.append(70 + vol_grad['strength'] * 30)
        else:
            sub_scores.append(30)

        # 空中加油
        if air_refuel['detected']:
            sub_scores.append(65 + air_refuel['strength'] * 35)
        else:
            base = 30 + air_refuel.get('conditions_met', 0) * 5
            sub_scores.append(min(base, 55))

        # 筹码分布
        if chip['detected']:
            sub_scores.append(65 + chip['strength'] * 35)
        else:
            base = 35
            if chip.get('profit_ratio', 0) > 0.7:
                base += 10
            if chip.get('above_pressure', 1) < 0.3:
                base += 5
            sub_scores.append(min(base, 55))

        score = round(np.mean(sub_scores), 1)

        details = (
            f"均线发散={'✓' if ma_conv['detected'] else '✗'}"
            f"(强度{ma_conv.get('strength', 0)}), "
            f"量能递增={'✓' if vol_grad['detected'] else '✗'}"
            f"(强度{vol_grad.get('strength', 0)}), "
            f"空中加油={'✓' if air_refuel['detected'] else '✗'}"
            f"(强度{air_refuel.get('strength', 0)}), "
            f"筹码单峰={'✓' if chip['detected'] else '✗'}"
            f"(获利{chip.get('profit_ratio', 0)})"
        )

        return {
            'score': score,
            'ma_convergence': ma_conv,
            'volume_gradient': vol_grad,
            'air_refueling': air_refuel,
            'chip_distribution': chip,
            'details': details,
        }


# =====================================================================
# 四维综合引擎: 右侧确认买入框架
# =====================================================================

class FourDimensionEngine:
    """
    四维综合选股引擎 v7.4

    将四个维度串联,构建"右侧确认"买入框架:

      1. 筛选: 估值分位<30%且技术面底背离 (备选池)
      2. 确认: 倍量阳线或金针探底 + DDX>0.5
      3. 下单: 均线粘合后首次发散 + 筹码单峰密集 → 尾盘或次日回调买入

    四维共振时给出"强买入"信号。

    v7.4增强:
      - 支持ETF估值适配 (ETFValuationAdapter)
      - 集成右侧确认三步框架 (RightSideConfirmation)
      - 资金维度支持成交额/换手率代理 (无Level-2时)
    """

    def __init__(self, asset_type: str = 'stock'):
        """
        参数:
          asset_type: 'stock' 或 'etf'
            - 'stock': 使用个股PE/PB/PEG估值
            - 'etf': 使用ETF折溢价/价格分位/规模变化估值
        """
        self.asset_type = asset_type
        self.valuation = ValuationDimension()
        self.etf_valuation = ETFValuationAdapter()
        self.bottom_reversal = BottomReversalDimension()
        self.capital_flow = CapitalFlowDimension()
        self.main_rally = MainRallyDimension()
        self.right_side = RightSideConfirmation()

    def analyze(
        self,
        ohlc: pd.DataFrame = None,
        pe: float = None,
        pb: float = None,
        industry_avg_pe: float = None,
        growth_rate: float = None,
        pb_history: pd.Series = None,
        pe_history: pd.Series = None,
        market_cap: float = None,
        leader_market_cap: float = None,
        ddx_series: pd.Series = None,
        large_order_net: float = None,
        large_order_ratio: float = None,
        dragon_tiger_list: List[Dict] = None,
        turnover_rates: np.ndarray = None,
        # v7.4 ETF专用参数
        current_price: float = None,
        discount_rate: float = None,
        index_pe: float = None,
        index_pe_percentile: float = None,
        fund_scale_change: float = None,
        main_inflow: float = None,
        turnover_rate: float = None,
    ) -> Dict[str, Any]:
        """
        执行四维综合分析

        v7.4新增:
          - ETF估值适配 (当asset_type='etf'时)
          - 右侧确认三步框架结果
          - 资金维度支持成交额/换手率代理

        返回:
          {
            'total_score': 0-100,
            'signal': '强买入'|'买入'|'观察'|'观望',
            'valuation': {...},
            'bottom_reversal': {...},
            'capital_flow': {...},
            'main_rally': {...},
            'resonance': int,         # 共振维度数(0-4)
            'resonance_level': str,   # '四维共振'|'三维共振'|'二维共振'|'单维信号'|'无信号'
            'action': str,            # 操作建议
            'stop_loss_rule': str,    # 止损纪律
            'details': str,
            'right_side': {...},      # v7.4: 右侧确认三步结果
          }
        """
        # 维度一: 估值 (根据asset_type选择)
        if self.asset_type == 'etf':
            val_result = self.etf_valuation.score(
                ohlc=ohlc,
                current_price=current_price,
                discount_rate=discount_rate,
                index_pe=index_pe,
                index_pe_percentile=index_pe_percentile,
                fund_scale_change=fund_scale_change,
            )
        else:
            val_result = self.valuation.score(
                pe=pe, industry_avg_pe=industry_avg_pe, growth_rate=growth_rate,
                pb=pb, pb_history=pb_history, pe_history=pe_history,
                market_cap=market_cap, leader_market_cap=leader_market_cap,
            )

        br_result = {'score': 50, 'details': '无K线数据'}
        if ohlc is not None and len(ohlc) > 0:
            br_result = self.bottom_reversal.score(ohlc)

        cf_result = self.capital_flow.score(
            ddx_series=ddx_series,
            large_order_net=large_order_net if large_order_net is not None else main_inflow,
            large_order_ratio=large_order_ratio,
            dragon_tiger_list=dragon_tiger_list,
            volume=ohlc['volume'].values if ohlc is not None and 'volume' in ohlc.columns else None,
            close=ohlc['close'].values if ohlc is not None else None,
            turnover_rate=turnover_rate,
        )

        mr_result = {'score': 50, 'details': '无K线数据'}
        if ohlc is not None and len(ohlc) > 0:
            mr_result = self.main_rally.score(ohlc, turnover_rates)

        # 四维加权综合 (各25%)
        total_score = (
            val_result['score'] * self.valuation.weight +
            br_result['score'] * self.bottom_reversal.weight +
            cf_result['score'] * self.capital_flow.weight +
            mr_result['score'] * self.main_rally.weight
        )

        # 共振检测: 维度得分>=65视为该维度触发
        threshold = 65
        resonance_dims = []
        if val_result['score'] >= threshold:
            resonance_dims.append('估值')
        if br_result['score'] >= threshold:
            resonance_dims.append('拐点')
        if cf_result['score'] >= threshold:
            resonance_dims.append('资金')
        if mr_result['score'] >= threshold:
            resonance_dims.append('主升浪')

        resonance_count = len(resonance_dims)

        if resonance_count >= 4:
            resonance_level = '四维共振'
            signal = '强买入'
            action = '四维全部触发,右侧确认完毕。尾盘(14:50后)或次日回调不破前日阳线实体1/2时,果断介入。'
        elif resonance_count == 3:
            resonance_level = '三维共振'
            signal = '买入'
            action = f'三维共振({"+".join(resonance_dims)}),信号较强。可在尾盘轻仓介入,次日确认后加仓。'
        elif resonance_count == 2:
            resonance_level = '二维共振'
            signal = '观察'
            action = f'二维共振({"+".join(resonance_dims)}),需等待更多维度确认。建议加入备选池密切跟踪。'
        elif resonance_count == 1:
            resonance_level = '单维信号'
            signal = '观察'
            action = f'仅{resonance_dims[0]}维度触发,信号偏弱。持续观察,等待共振。'
        else:
            resonance_level = '无信号'
            signal = '观望'
            action = '四维均未触发,暂不具备短线买入条件。'

        # 止损纪律提醒
        stop_loss_rule = (
            '破5日线即减仓,破10日线即清仓。'
            '主升浪加速赶顶时: 放量滞涨或MACD顶背离需果断止盈。'
        )

        details = (
            f"估值={val_result['score']}({('✓' if val_result['score']>=threshold else '✗')}), "
            f"拐点={br_result['score']}({('✓' if br_result['score']>=threshold else '✗')}), "
            f"资金={cf_result['score']}({('✓' if cf_result['score']>=threshold else '✗')}), "
            f"主升={mr_result['score']}({('✓' if mr_result['score']>=threshold else '✗')}) "
            f"→ {resonance_level}({resonance_count}维)"
        )

        # v7.4: 右侧确认三步分析
        analysis_for_rc = {
            'valuation': val_result,
            'bottom_reversal': br_result,
            'capital_flow': cf_result,
            'main_rally': mr_result,
        }
        right_side_result = self.right_side.execute(analysis_for_rc)

        return {
            'total_score': round(total_score, 1),
            'signal': signal,
            'valuation': val_result,
            'bottom_reversal': br_result,
            'capital_flow': cf_result,
            'main_rally': mr_result,
            'resonance': resonance_count,
            'resonance_level': resonance_level,
            'resonance_dims': resonance_dims,
            'action': action,
            'stop_loss_rule': stop_loss_rule,
            'details': details,
            'right_side': right_side_result,
        }


# =====================================================================
# ETF专用估值适配器 (v7.4)
# =====================================================================

class ETFValuationAdapter:
    """
    ETF/LOF 估值维度适配器

    ETF没有个股的PE/PB/PEG指标, 但可用以下代理:
      1. 折溢价率 (discount/premium): IOPV偏离度, 负折价=折价买入机会
      2. 历史价格分位: 当前价格在近3年价格区间的位置
      3. 跟踪指数估值: 底层指数的PE/PB分位 (如有)
      4. 规模变化: 近期净流入/流出趋势 (规模增长=资金认可)
    """

    def __init__(self):
        self.name = "etf_valuation"

    @staticmethod
    def calc_price_percentile(current_price: float, ohlc: pd.DataFrame) -> float:
        """
        计算当前价格在历史K线中的分位点 (0-1)

        近3年(约750个交易日)的价格分位
        """
        if ohlc is None or len(ohlc) < 20:
            return 0.5
        close = ohlc['close'].values
        clean = close[~np.isnan(close)]
        if len(clean) < 20:
            return 0.5
        # 取近3年数据
        lookback = min(750, len(clean))
        recent = clean[-lookback:]
        pct = float(np.sum(recent < current_price) / len(recent))
        return pct

    @staticmethod
    def calc_discount_rate(price: float, iopv: float = None, prev_close: float = None) -> float:
        """
        计算折溢价率

        优先使用IOPV, 其次用前收盘价
        返回: 正值=溢价, 负值=折价
        """
        ref = iopv if iopv is not None and iopv > 0 else prev_close
        if ref is None or ref <= 0 or price is None or price <= 0:
            return 0.0
        return (price - ref) / ref

    def score(
        self,
        ohlc: pd.DataFrame = None,
        current_price: float = None,
        discount_rate: float = None,
        index_pe: float = None,
        index_pe_percentile: float = None,
        fund_scale_change: float = None,
    ) -> Dict[str, Any]:
        """
        ETF估值评分

        参数:
          ohlc: 历史K线 (用于价格分位)
          current_price: 当前价格
          discount_rate: 折溢价率 (正=溢价, 负=折价)
          index_pe: 底层指数PE (如有)
          index_pe_percentile: 底层指数PE历史分位 (如有)
          fund_scale_change: 近期规模变化率 (正=流入, 负=流出)

        返回:
          {'score': 0-100, 'price_percentile': float, 'discount_rate': float, ...}
        """
        result = {
            'score': 50.0,
            'price_percentile': None,
            'discount_rate': None,
            'index_pe_percentile': None,
            'scale_change': None,
            'details': '',
        }

        sub_scores = []

        # 1. 价格历史分位 (ETF版"历史估值分位")
        if ohlc is not None and current_price is not None:
            pct = self.calc_price_percentile(current_price, ohlc)
            result['price_percentile'] = round(pct, 3)
            if pct < 0.10:
                sub_scores.append(95)
            elif pct < 0.20:
                sub_scores.append(85)
            elif pct < 0.30:
                sub_scores.append(75)
            elif pct < 0.50:
                sub_scores.append(55)
            elif pct < 0.70:
                sub_scores.append(40)
            else:
                sub_scores.append(25)

        # 2. 折溢价率 (ETF版"PE vs 行业")
        if discount_rate is not None:
            result['discount_rate'] = round(discount_rate, 4)
            # 折价买入更优 (负折价=安全垫)
            if discount_rate < -0.02:  # 折价>2%
                sub_scores.append(85)
            elif discount_rate < -0.005:  # 小幅折价
                sub_scores.append(70)
            elif discount_rate < 0.005:  # 平价
                sub_scores.append(55)
            elif discount_rate < 0.02:  # 小幅溢价
                sub_scores.append(40)
            else:  # 高溢价 (风险)
                sub_scores.append(20)

        # 3. 底层指数估值分位 (ETF版"PEG")
        if index_pe_percentile is not None:
            result['index_pe_percentile'] = round(index_pe_percentile, 3)
            if index_pe_percentile < 0.15:
                sub_scores.append(90)
            elif index_pe_percentile < 0.30:
                sub_scores.append(75)
            elif index_pe_percentile < 0.50:
                sub_scores.append(55)
            elif index_pe_percentile < 0.70:
                sub_scores.append(40)
            else:
                sub_scores.append(25)
        elif index_pe is not None and index_pe > 0:
            # 有PE但无分位, 用绝对值粗略判断
            if index_pe < 10:
                sub_scores.append(80)
            elif index_pe < 15:
                sub_scores.append(65)
            elif index_pe < 20:
                sub_scores.append(50)
            elif index_pe < 30:
                sub_scores.append(40)
            else:
                sub_scores.append(30)

        # 4. 规模变化趋势 (资金认可度)
        if fund_scale_change is not None:
            result['scale_change'] = round(fund_scale_change, 3)
            if fund_scale_change > 0.10:  # 规模增长>10%
                sub_scores.append(85)
            elif fund_scale_change > 0.03:
                sub_scores.append(70)
            elif fund_scale_change > -0.03:
                sub_scores.append(50)
            elif fund_scale_change > -0.10:
                sub_scores.append(35)
            else:
                sub_scores.append(20)

        if sub_scores:
            result['score'] = round(np.mean(sub_scores), 1)

        result['details'] = (
            f"价格分位={result['price_percentile']}, "
            f"折溢价={result['discount_rate']}, "
            f"指数PE分位={result['index_pe_percentile']}, "
            f"规模变化={result['scale_change']}"
        )
        return result


# =====================================================================
# 右侧确认买入框架 (终极应用策略 v7.4)
# =====================================================================

class RightSideConfirmation:
    """
    终极应用策略: 右侧确认买入框架

    将四维串联为三步决策:

      步骤1 — 筛选 (备选池):
        估值分位 < 30% 且 技术面底背离 → 进入备选池

      步骤2 — 确认 (触发信号):
        出现倍量阳线或金针探底 → 查看DDX是否>0.5

      步骤3 — 下单 (执行条件):
        均线粘合后首次发散 且 筹码单峰密集
        → 尾盘(14:50后)或次日回调不破前日阳线实体1/2 → 果断介入

    退出纪律:
      - 破5日线 → 减仓50%
      - 破10日线 → 清仓
      - MACD顶背离 / 放量滞涨 → 止盈
    """

    # 步骤1筛选阈值
    SCREEN_VALUATION_MAX = 40   # 估值分位<40% (放宽30%→40%适配ETF)
    SCREEN_DIVERGENCE_MIN = 55  # 拐点得分≥55

    # 步骤2确认阈值
    CONFIRM_BREAKOUT_MIN = 60   # 倍量阳/金针探底得分≥60
    CONFIRM_DDX_MIN = 0.5       # DDX>0.5

    # 步骤3下单阈值
    ORDER_MA_CONVERGE = True    # 均线粘合发散
    ORDER_CHIP_SINGLE_PEAK = True  # 筹码单峰密集

    def __init__(self):
        # 不创建FourDimensionEngine实例,避免循环引用
        # RightSideConfirmation只做纯逻辑判断,接收analysis_result dict
        pass

    def screen(self, analysis_result: Dict[str, Any]) -> bool:
        """
        步骤1: 筛选备选池

        条件: 估值分位低 + 技术面有底背离迹象
        """
        val_score = analysis_result.get('valuation', {}).get('score', 50)
        br_score = analysis_result.get('bottom_reversal', {}).get('score', 50)
        br_div = analysis_result.get('bottom_reversal', {}).get('divergence', {})

        # 估值低位 + 底背离
        val_ok = val_score >= 55  # 估值评分较好
        br_ok = br_score >= self.SCREEN_DIVERGENCE_MIN or br_div.get('detected', False)

        return val_ok and br_ok

    def confirm(self, analysis_result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        步骤2: 确认触发信号

        条件: 倍量阳线或金针探底 + DDX资金确认
        """
        br = analysis_result.get('bottom_reversal', {})
        cf = analysis_result.get('capital_flow', {})

        hammer = br.get('hammer', {})
        vol_breakout = br.get('volume_breakout', {})
        ddx_ok = cf.get('ddx_ok', False)
        ddx_consecutive = cf.get('ddx_consecutive', 0)
        large_order_ok = cf.get('large_order_ok', False)

        # 触发信号: 倍量阳 或 金针探底
        trigger = False
        trigger_type = []

        if vol_breakout.get('detected', False):
            trigger = True
            trigger_type.append('倍量阳')

        if hammer.get('detected', False):
            trigger = True
            trigger_type.append('金针探底')

        if not trigger:
            return False, '无倍量阳/金针探底信号'

        # 资金确认: DDX或大单
        if ddx_ok or large_order_ok or ddx_consecutive >= 2:
            return True, f"触发: {'+'.join(trigger_type)}, 资金确认: DDX连续{ddx_consecutive}日"
        else:
            return False, f"触发: {'+'.join(trigger_type)}, 但资金未确认 (DDX未达标)"

    def order_ready(self, analysis_result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        步骤3: 下单条件检查

        条件: 均线粘合后首次发散 + 筹码单峰密集
        """
        mr = analysis_result.get('main_rally', {})
        ma_conv = mr.get('ma_convergence', {})
        chip = mr.get('chip_distribution', {})

        ma_ok = ma_conv.get('detected', False) or ma_conv.get('golden_cross', False)
        chip_ok = chip.get('detected', False) or chip.get('profit_ratio', 0) > 0.7

        conditions = []
        if ma_ok:
            conditions.append('均线发散')
        if chip_ok:
            conditions.append(f"筹码获利{chip.get('profit_ratio', 0):.0%}")

        if ma_ok and chip_ok:
            return True, f"下单条件满足: {' + '.join(conditions)}"
        elif ma_ok or chip_ok:
            return False, f"部分满足: {' + '.join(conditions)}, 需等待完整确认"
        else:
            return False, '均线未发散, 筹码未密集'

    def execute(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行完整的右侧确认三步分析

        返回:
          {
            'step1_screen': bool,    # 是否通过筛选
            'step2_confirm': bool,   # 是否确认触发
            'step3_order': bool,     # 是否满足下单条件
            'final_action': str,     # 最终操作建议
            'action_detail': str,    # 详细说明
            'entry_timing': str,     # 入场时机建议
            'exit_rule': str,        # 退出纪律
          }
        """
        # 步骤1
        step1 = self.screen(analysis_result)

        # 步骤2
        step2, confirm_msg = self.confirm(analysis_result)

        # 步骤3
        step3, order_msg = self.order_ready(analysis_result)

        # 综合判断
        if step1 and step2 and step3:
            final_action = '果断介入'
            action_detail = (
                f'三步确认全部通过。{confirm_msg}。{order_msg}。'
                f'建议: 当日尾盘(14:50后)或次日开盘回调不破前日阳线实体1/2时介入。'
            )
            entry_timing = '尾盘14:50后 或 次日回调至前日阳线1/2'
        elif step1 and step2:
            final_action = '轻仓试探'
            action_detail = (
                f'筛选+确认通过,但下单条件未完全满足。{confirm_msg}。{order_msg}。'
                f'建议: 可轻仓试探,等待均线发散+筹码密集后加仓。'
            )
            entry_timing = '次日开盘观察,轻仓介入'
        elif step1:
            final_action = '加入备选池'
            action_detail = (
                f'通过筛选(估值低位+底背离),但触发信号未出现。'
                f'建议: 加入备选池密切跟踪,等待倍量阳/金针探底信号。'
            )
            entry_timing = '等待信号出现'
        else:
            final_action = '暂不介入'
            action_detail = '未通过筛选(估值偏高或无底背离),暂不具备短线买入条件。'
            entry_timing = '暂不操作'

        exit_rule = (
            '退出纪律: '
            '1) 破5日线 → 减仓50%; '
            '2) 破10日线 → 清仓; '
            '3) MACD顶背离或放量滞涨 → 止盈离场。'
            '原则: 吃鱼身,弃鱼尾。'
        )

        return {
            'step1_screen': step1,
            'step2_confirm': step2,
            'step3_order': step3,
            'final_action': final_action,
            'action_detail': action_detail,
            'entry_timing': entry_timing,
            'exit_rule': exit_rule,
            'confirm_detail': confirm_msg,
            'order_detail': order_msg,
        }


# =====================================================================
# 批量分析: 对多只标的进行四维评分
# =====================================================================

def batch_analyze(
    funds_data: pd.DataFrame,
    kline_fetcher=None,
    financial_fetcher=None,
) -> List[Dict[str, Any]]:
    """
    批量对场内基金/股票进行四维分析

    参数:
      funds_data: 含 code/name/price/change_pct/amount 等列的DataFrame
      kline_fetcher: 可调用对象, code -> ohlc DataFrame
      financial_fetcher: 可调用对象, code -> {pe, pb, growth_rate, ...}

    返回:
      分析结果列表,每个元素是 FourDimensionEngine.analyze() 的输出
    """
    engine = FourDimensionEngine()
    results = []

    for _, row in funds_data.iterrows():
        code = row.get('code', '')
        name = row.get('name', '')

        # 获取K线
        ohlc = None
        if kline_fetcher is not None:
            try:
                ohlc = kline_fetcher(code)
            except Exception as e:
                logger.warning(f"获取{code}K线失败: {e}")

        # 获取财务数据
        fin_data = {}
        if financial_fetcher is not None:
            try:
                fin_data = financial_fetcher(code) or {}
            except Exception as e:
                logger.warning(f"获取{code}财务数据失败: {e}")

        # 执行分析
        result = engine.analyze(
            ohlc=ohlc,
            pe=fin_data.get('pe'),
            pb=fin_data.get('pb'),
            industry_avg_pe=fin_data.get('industry_avg_pe'),
            growth_rate=fin_data.get('growth_rate'),
            market_cap=fin_data.get('market_cap'),
            leader_market_cap=fin_data.get('leader_market_cap'),
            large_order_net=fin_data.get('large_order_net'),
            large_order_ratio=fin_data.get('large_order_ratio'),
            ddx_series=fin_data.get('ddx_series'),
        )

        result['code'] = code
        result['name'] = name
        results.append(result)

    # 按总分排序
    results.sort(key=lambda x: x['total_score'], reverse=True)
    return results
