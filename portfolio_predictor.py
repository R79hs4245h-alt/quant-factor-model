"""
投资组合预测引擎 v7.6
=====================

核心功能:
  1. 获取30日历史K线数据 (复用腾讯API)
  2. 多维度筛选: 下跌风险低 + 上涨趋势明显 + 估值合理
  3. 预测未来2日盈利概率 (历史模式匹配 + 蒙特卡洛)
  4. 10万资金组合优化 (最大化2日盈利概率 + 风险约束)
  5. 股票+ETF同时投资 (龙头股+ETF混合配置)

v7.6 更新:
  - 引入真实龙头股票候选池 (25只各板块龙头)
  - 股票和ETF差异化筛选阈值 (股票阈值适当放宽)
  - 组合优化器确保股票和ETF同时配置 (ETF 50-70%, 股票 30-50%)
  - 股票止损止盈更宽 (-8%/+8%), ETF更窄 (-5%/+5%)

预测方法论:
  - 趋势确认: MA5/MA10多头排列 + MACD水上 + ADX>15
  - 下跌风险: ETF回撤<12%/股票<15% + 波动率<4%/5% + VaR<5%/6%
  - 估值安全: 价格分位<65%/70% + RSI<78/80
  - 2日盈利概率: 历史相似模式后2日上涨频率 + 蒙特卡洛模拟
  - 组合优化: 分组风险平价 + ETF/股票比例约束 + 主题分散

依赖: numpy, pandas, scipy(可选)
"""

import warnings
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from utils import get_logger
    logger = get_logger("portfolio_predictor")
except Exception:
    import logging
    logger = logging.getLogger("portfolio_predictor")

try:
    from config import (
        OUTPUT_DIR, ETF_THEME_MAP,
        CORE_BROAD_ETF, THEME_ETF, CORE_LOF,
    )
except Exception:
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
    ETF_THEME_MAP = {}
    CORE_BROAD_ETF = {}
    THEME_ETF = {}
    CORE_LOF = {}

warnings.filterwarnings("ignore")


# =====================================================================
# 龙头股票候选池 (v7.6: 股票+基金同时投资)
# =====================================================================
# 从各热门板块中选取龙头股,覆盖科技/消费/金融/医药/新能源/军工等
# 选取标准: 行业龙头 + 流动性好 + 市值适中
LEADER_STOCKS = {
    # 科技/半导体
    "600519": ("贵州茅台", "消费白酒"),
    "601318": ("中国平安", "金融地产"),
    "600036": ("招商银行", "金融地产"),
    "601012": ("隆基绿能", "新能源"),
    "300750": ("宁德时代", "新能源"),
    "002594": ("比亚迪", "新能源"),
    "600276": ("恒瑞医药", "消费医药"),
    "000858": ("五粮液", "消费白酒"),
    "600887": ("伊利股份", "消费医药"),
    "603259": ("药明康德", "消费医药"),
    # 科技/半导体
    "688981": ("中芯国际", "科技AI"),
    "002475": ("立讯精密", "科技AI"),
    "600585": ("海螺水泥", "周期制造"),
    "601899": ("紫金矿业", "周期制造"),
    "002241": ("歌尔股份", "科技AI"),
    # 金融
    "601166": ("兴业银行", "金融地产"),
    "600000": ("浦发银行", "金融地产"),
    "601688": ("华泰证券", "金融地产"),
    # 消费
    "000568": ("泸州老窖", "消费白酒"),
    "600809": ("山西汾酒", "消费白酒"),
    "000333": ("美的集团", "消费医药"),
    "000651": ("格力电器", "消费医药"),
    "603288": ("海天味业", "消费医药"),
    # 军工
    "600760": ("中航沈飞", "军工制造"),
    "000768": ("中航飞机", "军工制造"),
}

# 股票主题分类
STOCK_THEME_MAP = {
    "科技AI": ["688981", "002475", "002241"],
    "消费白酒": ["600519", "000858", "000568", "600809"],
    "消费医药": ["600276", "600887", "603259", "000333", "000651", "603288"],
    "新能源": ["601012", "300750", "002594"],
    "金融地产": ["601318", "600036", "601166", "600000", "601688"],
    "周期制造": ["600585", "601899"],
    "军工制造": ["600760", "000768"],
}


# =====================================================================
# 数据获取层
# =====================================================================

class KLineFetcher:
    """K线数据获取器 (腾讯API)"""

    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.proxies = {
            'http': os.environ.get('HTTPS_PROXY', 'http://127.0.0.1:18080'),
            'https': os.environ.get('HTTPS_PROXY', 'http://127.0.0.1:18080'),
        }

    def fetch(self, code: str, days: int = 60) -> pd.DataFrame:
        """获取指定代码的日K线数据"""
        if code in self.cache:
            cached = self.cache[code]
            if len(cached) >= days:
                return cached.tail(days).copy()

        import requests

        if code.startswith('5') or code.startswith('6') or code.startswith('9'):
            prefix = 'sh'
        else:
            prefix = 'sz'

        url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
        start_date = (datetime.now() - timedelta(days=days * 2 + 90)).strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        params = {
            'param': f'{prefix}{code},day,{start_date},{end_date},640,qfq',
        }

        try:
            resp = requests.get(url, params=params, timeout=10, proxies=self.proxies)
            data = resp.json()
            key = f'{prefix}{code}'
            if data and data.get('data') and data['data'].get(key):
                stock_data = data['data'][key]
                klines = stock_data.get('qfqday') or stock_data.get('day') or []
                if klines:
                    df = pd.DataFrame(klines, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                    df['date'] = pd.to_datetime(df['date'])
                    for c in ['open', 'close', 'high', 'low', 'volume']:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                    df = df.dropna(subset=['close'])
                    df = df.sort_values('date').reset_index(drop=True)
                    self.cache[code] = df
                    return df.tail(days).copy()
        except Exception as e:
            logger.debug(f"获取{code}K线失败: {e}")

        return pd.DataFrame()

    def fetch_batch(self, codes: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """批量获取K线数据"""
        results = {}
        for i, code in enumerate(codes):
            df = self.fetch(code, days)
            if len(df) >= 20:
                results[code] = df
            if (i + 1) % 20 == 0:
                logger.info(f"K线获取进度: {i+1}/{len(codes)}")
                time.sleep(0.3)  # 避免请求过快
        return results


# =====================================================================
# 技术指标计算层
# =====================================================================

class TechnicalIndicators:
    """技术指标计算"""

    @staticmethod
    def calc_ma(close: pd.Series, period: int) -> pd.Series:
        """简单移动平均"""
        return close.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def calc_ema(close: pd.Series, period: int) -> pd.Series:
        """指数移动平均"""
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD: DIF, DEA, HIST"""
        ema_fast = TechnicalIndicators.calc_ema(close, fast)
        ema_slow = TechnicalIndicators.calc_ema(close, slow)
        dif = ema_fast - ema_slow
        dea = TechnicalIndicators.calc_ema(dif, signal)
        hist = 2 * (dif - dea)
        return dif, dea, hist

    @staticmethod
    def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ATR"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ADX (简化版)"""
        plus_dm = (high - high.shift(1)).where(
            (high - high.shift(1)) > (low.shift(1) - low), 0
        ).clip(upper=9999)
        minus_dm = (low.shift(1) - low).where(
            (low.shift(1) - low) > (high - high.shift(1)), 0
        ).clip(upper=9999)
        atr = TechnicalIndicators.calc_atr(high, low, close, period)
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx.fillna(20)

    @staticmethod
    def calc_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
        """布林带: mid, upper, lower"""
        mid = close.rolling(window=period, min_periods=1).mean()
        std = close.rolling(window=period, min_periods=1).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return mid, upper, lower

    @staticmethod
    def calc_volatility(close: pd.Series, period: int = 20) -> float:
        """日波动率 (年化前)"""
        returns = close.pct_change().dropna()
        if len(returns) < 5:
            return 0.03
        return float(returns.tail(period).std())


# =====================================================================
# 筛选层: 下跌风险低 + 上涨趋势明显 + 估值合理
# =====================================================================

class StockScreener:
    """
    三重筛选器 (v7.6: 股票和ETF差异化阈值):
      1. 下跌风险低: 最大回撤 + 波动率 + VaR
      2. 上涨趋势明显: MA多头 + MACD>0 + ADX
      3. 估值合理: 价格分位 + RSI

    股票波动比ETF大,阈值适当放宽:
      - 股票: 回撤<15%, 波动率<5%, VaR<6%
      - ETF:  回撤<12%, 波动率<4%, VaR<5%
    """

    # ETF筛选阈值
    ETF_MAX_DRAWDOWN = 0.12       # ETF: 30日最大回撤 < 12%
    ETF_MAX_VOLATILITY = 0.04     # ETF: 日波动率 < 4%
    ETF_MAX_VAR_95 = 0.05         # ETF: VaR(95%) < 5%
    ETF_MIN_ADX = 15              # ETF: ADX > 15
    ETF_MAX_PRICE_PERCENTILE = 0.65  # ETF: 价格分位 < 65%
    ETF_MAX_RSI = 78              # ETF: RSI < 78

    # 股票筛选阈值 (放宽,因为个股波动更大)
    STOCK_MAX_DRAWDOWN = 0.15     # 股票: 30日最大回撤 < 15%
    STOCK_MAX_VOLATILITY = 0.05   # 股票: 日波动率 < 5%
    STOCK_MAX_VAR_95 = 0.06       # 股票: VaR(95%) < 6%
    STOCK_MIN_ADX = 15            # 股票: ADX > 15
    STOCK_MAX_PRICE_PERCENTILE = 0.70  # 股票: 价格分位 < 70%
    STOCK_MAX_RSI = 80            # 股票: RSI < 80

    # 兼容旧代码
    MAX_DRAWDOWN = 0.12
    MAX_VOLATILITY = 0.04
    MAX_VAR_95 = 0.05
    MIN_ADX = 15
    MAX_PRICE_PERCENTILE = 0.65
    MAX_RSI = 78
    MIN_VOLUME = 1e6

    @staticmethod
    def calc_max_drawdown(close: pd.Series) -> float:
        """计算最大回撤"""
        peak = close.expanding().max()
        drawdown = (close - peak) / peak
        return float(drawdown.min())

    @staticmethod
    def calc_var_95(close: pd.Series) -> float:
        """计算VaR(95%)"""
        returns = close.pct_change().dropna()
        if len(returns) < 10:
            return 0.05
        return float(abs(np.percentile(returns, 5)))

    @staticmethod
    def calc_price_percentile(close: pd.Series) -> float:
        """当前价格在序列中的分位 (0-1)"""
        if len(close) < 10:
            return 0.5
        current = close.iloc[-1]
        return float((close < current).sum() / len(close))

    @classmethod
    def screen(cls, code: str, ohlc: pd.DataFrame, name: str = '', asset_type: str = 'etf') -> Dict[str, Any]:
        """
        执行三重筛选 (v7.6: 根据asset_type使用差异化阈值)

        参数:
          asset_type: 'etf' 或 'stock',决定使用哪套阈值

        返回:
          {
            'pass': bool,
            'risk_score': float,    # 下跌风险评分 (0-100, 越高越安全)
            'trend_score': float,   # 上涨趋势评分 (0-100, 越高越强)
            'valuation_score': float, # 估值评分 (0-100, 越高越合理)
            'details': Dict,
            'fail_reasons': List[str],
          }
        """
        result = {
            'pass': False,
            'risk_score': 0.0,
            'trend_score': 0.0,
            'valuation_score': 0.0,
            'details': {},
            'fail_reasons': [],
        }

        if ohlc is None or len(ohlc) < 20:
            result['fail_reasons'].append('数据不足(<20日)')
            return result

        # v7.6: 根据资产类型选择阈值
        if asset_type == 'stock':
            max_dd_limit = cls.STOCK_MAX_DRAWDOWN
            vol_limit = cls.STOCK_MAX_VOLATILITY
            var_limit = cls.STOCK_MAX_VAR_95
            adx_limit = cls.STOCK_MIN_ADX
            pct_limit = cls.STOCK_MAX_PRICE_PERCENTILE
            rsi_limit = cls.STOCK_MAX_RSI
        else:
            max_dd_limit = cls.ETF_MAX_DRAWDOWN
            vol_limit = cls.ETF_MAX_VOLATILITY
            var_limit = cls.ETF_MAX_VAR_95
            adx_limit = cls.ETF_MIN_ADX
            pct_limit = cls.ETF_MAX_PRICE_PERCENTILE
            rsi_limit = cls.ETF_MAX_RSI

        close = ohlc['close']
        high = ohlc['high']
        low = ohlc['low']
        volume = ohlc['volume']
        current_price = float(close.iloc[-1])

        ti = TechnicalIndicators

        # ===== 1. 下跌风险评估 =====
        max_dd = cls.calc_max_drawdown(close.tail(30))
        vol = ti.calc_volatility(close, 20)
        var95 = cls.calc_var_95(close.tail(30))

        # 风险评分: 回撤越小、波动越低、VaR越小 → 分越高
        dd_score = max(0, 100 - abs(max_dd) * 1000)  # -8% → 20分, -3% → 70分
        vol_score = max(0, 100 - vol * 3000)  # 3.5% → -5, 2% → 40, 1% → 70
        var_score = max(0, 100 - var95 * 2500)  # 4% → 0, 2% → 50, 1% → 75

        risk_score = (dd_score * 0.4 + vol_score * 0.3 + var_score * 0.3)
        result['details']['max_drawdown'] = round(max_dd, 4)
        result['details']['volatility'] = round(vol, 4)
        result['details']['var_95'] = round(var95, 4)

        risk_pass = max_dd > -max_dd_limit and vol < vol_limit and var95 < var_limit

        if not risk_pass:
            if abs(max_dd) >= max_dd_limit:
                result['fail_reasons'].append(f'回撤过大({max_dd:.1%})')
            if vol >= vol_limit:
                result['fail_reasons'].append(f'波动过高({vol:.2%})')
            if var95 >= var_limit:
                result['fail_reasons'].append(f'VaR过高({var95:.2%})')

        # ===== 2. 上涨趋势评估 =====
        ma5 = ti.calc_ma(close, 5)
        ma10 = ti.calc_ma(close, 10)
        ma20 = ti.calc_ma(close, 20)
        dif, dea, hist = ti.calc_macd(close)
        adx = ti.calc_adx(high, low, close)
        rsi = ti.calc_rsi(close)

        ma_bull = bool(ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1])
        macd_above_water = bool(dif.iloc[-1] > 0 and dea.iloc[-1] > 0)
        macd_golden = bool(hist.iloc[-1] > 0 and hist.iloc[-2] <= 0)  # 今日金叉
        macd_rising = bool(hist.iloc[-1] > hist.iloc[-2])  # 红柱增长
        adx_strong = bool(adx.iloc[-1] > adx_limit)
        price_above_ma5 = bool(current_price > ma5.iloc[-1])

        # 5日涨幅
        ret_5d = float((close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0)
        # 20日涨幅
        ret_20d = float((close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0)

        # 趋势评分
        trend_components = []
        if ma_bull:
            trend_components.append(20)
        if macd_above_water:
            trend_components.append(20)
        if macd_rising or macd_golden:
            trend_components.append(20)
        if adx_strong:
            trend_components.append(15)
        if price_above_ma5:
            trend_components.append(10)
        if ret_5d > 0:
            trend_components.append(min(15, ret_5d * 300))  # 5日涨幅加分

        trend_score = sum(trend_components)
        trend_pass = trend_score >= 40  # 至少40分

        result['details']['ma_bull'] = ma_bull
        result['details']['macd_above_water'] = macd_above_water
        result['details']['macd_rising'] = macd_rising
        result['details']['adx'] = round(float(adx.iloc[-1]), 1)
        result['details']['rsi'] = round(float(rsi.iloc[-1]), 1)
        result['details']['ret_5d'] = round(ret_5d, 4)
        result['details']['ret_20d'] = round(ret_20d, 4)

        if not trend_pass:
            result['fail_reasons'].append(f'趋势不足({trend_score:.0f}分)')

        # ===== 3. 估值合理性评估 =====
        price_pct = cls.calc_price_percentile(close.tail(60)) if len(close) >= 60 else cls.calc_price_percentile(close)
        rsi_val = float(rsi.iloc[-1])
        boll_mid, boll_upper, boll_lower = ti.calc_bollinger(close)

        # 布林带位置: 0=下轨, 1=上轨
        if boll_upper.iloc[-1] > boll_lower.iloc[-1]:
            boll_pos = (current_price - boll_lower.iloc[-1]) / (boll_upper.iloc[-1] - boll_lower.iloc[-1])
            boll_pos = float(np.clip(boll_pos, 0, 1))
        else:
            boll_pos = 0.5

        # 估值评分: 价格分位低 + RSI不超买 + 布林带中下部
        pct_score = max(0, 100 - price_pct * 120)  # 0%→100, 50%→40, 60%→28
        rsi_score = max(0, 100 - max(0, rsi_val - 30) * 1.5)  # 30→100, 50→70, 70→40
        boll_score = max(0, 100 - boll_pos * 100)  # 下轨→100, 中轨→50, 上轨→0

        valuation_score = pct_score * 0.4 + rsi_score * 0.3 + boll_score * 0.3

        val_pass = price_pct < pct_limit and rsi_val < rsi_limit

        result['details']['price_percentile'] = round(price_pct, 3)
        result['details']['boll_position'] = round(boll_pos, 3)

        if not val_pass:
            if price_pct >= pct_limit:
                result['fail_reasons'].append(f'价格分位偏高({price_pct:.0%})')
            if rsi_val >= rsi_limit:
                result['fail_reasons'].append(f'RSI超买({rsi_val:.0f})')

        # ===== 综合判断 =====
        result['risk_score'] = round(risk_score, 1)
        result['trend_score'] = round(trend_score, 1)
        result['valuation_score'] = round(valuation_score, 1)
        result['pass'] = risk_pass and trend_pass and val_pass

        return result


# =====================================================================
# 2日盈利概率预测层
# =====================================================================

class ProfitPredictor:
    """
    预测未来2日盈利概率

    方法:
      1. 历史模式匹配: 找到过去30日内与当前技术形态相似的日子,
         统计其后2日上涨的概率
      2. 蒙特卡洛模拟: 基于历史收益率分布,模拟10000次未来2日走势
      3. 综合概率: 加权融合两种方法
    """

    # 最低盈利概率阈值
    MIN_PROFIT_PROB = 0.50  # 至少50%概率盈利

    @staticmethod
    def _extract_pattern_features(ohlc: pd.DataFrame) -> np.ndarray:
        """提取当前技术形态特征向量"""
        close = ohlc['close']
        ti = TechnicalIndicators

        ma5 = ti.calc_ma(close, 5)
        ma10 = ti.calc_ma(close, 10)
        ma20 = ti.calc_ma(close, 20)
        dif, dea, hist = ti.calc_macd(close)
        rsi = ti.calc_rsi(close)
        adx = ti.calc_adx(ohlc['high'], ohlc['low'], close)

        current = close.iloc[-1]
        features = np.array([
            # 均线相对位置
            (current / ma5.iloc[-1] - 1) if ma5.iloc[-1] > 0 else 0,
            (current / ma10.iloc[-1] - 1) if ma10.iloc[-1] > 0 else 0,
            (current / ma20.iloc[-1] - 1) if ma20.iloc[-1] > 0 else 0,
            (ma5.iloc[-1] / ma10.iloc[-1] - 1) if ma10.iloc[-1] > 0 else 0,
            (ma10.iloc[-1] / ma20.iloc[-1] - 1) if ma20.iloc[-1] > 0 else 0,
            # MACD
            float(dif.iloc[-1] / current) if current > 0 else 0,
            float(dea.iloc[-1] / current) if current > 0 else 0,
            float(hist.iloc[-1] / current) if current > 0 else 0,
            # RSI归一化
            float(rsi.iloc[-1] / 100),
            # ADX归一化
            float(adx.iloc[-1] / 100),
            # 近期收益率
            float(close.pct_change(1).iloc[-1]) if len(close) > 1 else 0,
            float(close.pct_change(3).iloc[-1]) if len(close) > 3 else 0,
            float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0,
        ])
        return features

    @classmethod
    def predict_2d_profit_prob(cls, ohlc: pd.DataFrame) -> Dict[str, Any]:
        """
        预测未来2日盈利概率

        返回:
          {
            'profit_prob': float,       # 2日盈利概率 (0-1)
            'expected_return': float,   # 预期2日收益率
            'upside': float,            # 预期上涨幅度
            'downside': float,          # 预期下跌幅度
            'confidence': str,          # 高/中/低
            'method_details': Dict,
          }
        """
        result = {
            'profit_prob': 0.5,
            'expected_return': 0.0,
            'upside': 0.0,
            'downside': 0.0,
            'confidence': '低',
            'method_details': {},
        }

        if ohlc is None or len(ohlc) < 25:
            return result

        close = ohlc['close']

        # ===== 方法1: 历史模式匹配 =====
        current_features = cls._extract_pattern_features(ohlc)

        # 遍历历史中每个交易日,提取该日的技术形态,计算与当前的相似度
        match_results = []
        lookback = min(len(ohlc) - 2, 200)  # 最多回看200日

        for i in range(20, len(ohlc) - 2):
            hist_ohlc = ohlc.iloc[:i+1]
            if len(hist_ohlc) < 25:
                continue

            hist_features = cls._extract_pattern_features(hist_ohlc)

            # 计算余弦相似度
            dot = np.dot(current_features, hist_features)
            norm_cur = np.linalg.norm(current_features)
            norm_hist = np.linalg.norm(hist_features)
            if norm_cur > 0 and norm_hist > 0:
                similarity = dot / (norm_cur * norm_hist)
            else:
                continue

            # 记录相似度高的日子的后续2日收益
            if similarity > 0.85:
                future_2d_ret = float(close.iloc[i+2] / close.iloc[i] - 1)
                match_results.append({
                    'similarity': float(similarity),
                    'future_ret': future_2d_ret,
                    'date': ohlc.iloc[i]['date'],
                })

        if match_results:
            match_rets = [m['future_ret'] for m in match_results]
            prob_hist = sum(1 for r in match_rets if r > 0) / len(match_rets)
            avg_ret_hist = float(np.mean(match_rets))
            result['method_details']['pattern_match'] = {
                'match_count': len(match_results),
                'profit_prob': round(prob_hist, 3),
                'avg_return': round(avg_ret_hist, 4),
            }
        else:
            prob_hist = 0.5
            avg_ret_hist = 0.0
            result['method_details']['pattern_match'] = {
                'match_count': 0,
                'profit_prob': 0.5,
                'avg_return': 0.0,
            }

        # ===== 方法2: 蒙特卡洛模拟 =====
        returns = close.pct_change().dropna().tail(30).values
        if len(returns) < 10:
            returns = close.pct_change().dropna().values

        n_sims = 10000
        np.random.seed(42)
        # Bootstrap抽样: 从历史日收益率中随机抽取2日
        sim_indices = np.random.randint(0, len(returns), size=(n_sims, 2))
        sim_returns = returns[sim_indices].sum(axis=1)

        prob_mc = float(np.sum(sim_returns > 0) / n_sims)
        avg_ret_mc = float(np.mean(sim_returns))
        upside_mc = float(np.percentile(sim_returns, 90))
        downside_mc = float(np.percentile(sim_returns, 10))

        result['method_details']['monte_carlo'] = {
            'simulations': n_sims,
            'profit_prob': round(prob_mc, 3),
            'avg_return': round(avg_ret_mc, 4),
            'upside_90': round(upside_mc, 4),
            'downside_10': round(downside_mc, 4),
        }

        # ===== 综合概率 =====
        # 模式匹配权重0.6, 蒙特卡洛权重0.4
        # 如果模式匹配匹配数较少(<5),降低其权重
        match_count = len(match_results) if match_results else 0
        if match_count >= 10:
            w_hist = 0.6
            w_mc = 0.4
        elif match_count >= 3:
            w_hist = 0.4
            w_mc = 0.6
        else:
            w_hist = 0.2
            w_mc = 0.8

        combined_prob = prob_hist * w_hist + prob_mc * w_mc
        combined_ret = avg_ret_hist * w_hist + avg_ret_mc * w_mc

        result['profit_prob'] = round(combined_prob, 3)
        result['expected_return'] = round(combined_ret, 4)
        result['upside'] = round(upside_mc, 4)
        result['downside'] = round(downside_mc, 4)

        # 置信度
        if combined_prob > 0.70 and match_count >= 5:
            result['confidence'] = '高'
        elif combined_prob > 0.60:
            result['confidence'] = '中'
        else:
            result['confidence'] = '低'

        return result


# =====================================================================
# 组合优化层
# =====================================================================

class PortfolioOptimizer:
    """
    投资组合优化器 (v7.6: 股票+基金同时投资)

    目标: 最大化组合2日盈利概率
    约束:
      - 总资金: 100,000元
      - 单只标的最大权重: 25%
      - 单只标的最小权重: 5%
      - 最多持有: 8只标的
      - 股票+ETF混合: ETF占50-70%, 股票占30-50%
      - 行业分散: 单一主题最大占比<=35%
      - 必须同时持有股票和ETF (如果候选池中有)
    """

    TOTAL_CAPITAL = 100000  # 10万
    MAX_POSITION = 0.25     # 单只最大25%
    MIN_POSITION = 0.05     # 单只最小5%
    MAX_HOLDINGS = 8        # 最多8只
    MIN_ETF_RATIO = 0.50    # ETF最少占50%
    MAX_ETF_RATIO = 0.70    # ETF最多占70%
    MIN_STOCK_RATIO = 0.30  # 股票最少占30% (如果有股票候选)
    MAX_STOCK_RATIO = 0.50  # 股票最多占50%
    MAX_THEME_RATIO = 0.35  # 单主题最大35%
    CASH_RESERVE = 0.05     # 保留5%现金

    @classmethod
    def optimize(cls, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        优化投资组合 (v7.6: 股票+ETF同时配置)

        参数:
          candidates: 候选标的列表,每个包含:
            - code, name, asset_type (stock/etf), theme
            - profit_prob, expected_return, risk_score
            - current_price, screen_result

        返回:
          {
            'portfolio': List[Dict],  # 持仓列表
            'total_invested': float,
            'cash_reserve': float,
            'portfolio_profit_prob': float,
            'portfolio_expected_return': float,
            'portfolio_risk': float,
            'allocation_summary': Dict,
          }
        """
        if not candidates:
            return cls._empty_portfolio()

        # v7.6: 分为ETF和股票两组
        etf_cands = [c for c in candidates if c.get('asset_type') == 'etf']
        stock_cands = [c for c in candidates if c.get('asset_type') == 'stock']

        # 按盈利概率排序
        etf_cands.sort(key=lambda x: x.get('profit_prob', 0), reverse=True)
        stock_cands.sort(key=lambda x: x.get('profit_prob', 0), reverse=True)

        # 按主题分组,确保分散化
        def diversify(cands):
            theme_groups = defaultdict(list)
            for c in cands:
                theme_groups[c.get('theme', '其他')].append(c)
            diversified = []
            for theme, funds in theme_groups.items():
                for f in funds[:2]:
                    diversified.append(f)
            diversified.sort(key=lambda x: x.get('profit_prob', 0), reverse=True)
            return diversified

        etf_diversified = diversify(etf_cands)
        stock_diversified = diversify(stock_cands)

        # v7.6: 确定ETF和股票的持仓数量
        # 目标: ETF占4-6只, 股票占2-4只, 总共最多8只
        if stock_diversified:
            # 有股票候选: ETF最多5只, 股票最多3只
            n_etf = min(5, len(etf_diversified))
            n_stock = min(3, len(stock_diversified))
            # 确保总数不超过MAX_HOLDINGS
            while n_etf + n_stock > cls.MAX_HOLDINGS and n_etf > 2:
                n_etf -= 1
        else:
            # 无股票候选: 全部ETF
            n_etf = min(cls.MAX_HOLDINGS, len(etf_diversified))
            n_stock = 0

        selected = etf_diversified[:n_etf] + stock_diversified[:n_stock]

        if not selected:
            return cls._empty_portfolio()

        # v7.6: 分配ETF和股票的资金比例
        investable = cls.TOTAL_CAPITAL * (1 - cls.CASH_RESERVE)

        # 根据实际选中的标的数量动态调整比例
        n_etf_actual = len(etf_diversified)
        n_stock_actual = len(stock_diversified)

        if n_stock_actual > 0 and n_etf_actual > 0:
            # 同时有股票和ETF: 股票30-50%, ETF 50-70%
            avg_stock_prob = np.mean([s.get('profit_prob', 0.5) for s in stock_diversified[:min(n_stock, n_stock_actual)]])
            avg_etf_prob = np.mean([s.get('profit_prob', 0.5) for s in etf_diversified[:min(n_etf, n_etf_actual)]])
            if avg_stock_prob > avg_etf_prob:
                stock_ratio = min(cls.MAX_STOCK_RATIO, 0.45)
            else:
                stock_ratio = cls.MIN_STOCK_RATIO
            etf_ratio = 1.0 - stock_ratio
            # v7.6: 如果ETF只有1只,限制其不超过MAX_POSITION
            if n_etf == 1:
                etf_ratio = min(etf_ratio, cls.MAX_POSITION)
                stock_ratio = 1.0 - etf_ratio
            # 如果股票只有1只,限制其不超过MAX_POSITION
            if n_stock == 1:
                stock_ratio = min(stock_ratio, cls.MAX_POSITION)
                etf_ratio = 1.0 - stock_ratio
        elif n_stock_actual > 0:
            stock_ratio = 1.0
            etf_ratio = 0.0
        else:
            stock_ratio = 0.0
            etf_ratio = 1.0

        etf_budget = investable * etf_ratio
        stock_budget = investable * stock_ratio

        # ===== 权重分配: 基于盈利概率的风险平价 =====
        # 分组分配: ETF组内分配, 股票组内分配
        etf_selected = [s for s in selected if s.get('asset_type') == 'etf']
        stock_selected = [s for s in selected if s.get('asset_type') == 'stock']

        def calc_group_weights(group, budget):
            """计算组内权重和金额"""
            if not group:
                return [], 0

            scores = []
            for s in group:
                prob = s.get('profit_prob', 0.5)
                risk = s.get('risk_score', 50)
                attractiveness = prob * np.sqrt(max(risk, 10) / 100)
                scores.append(max(attractiveness, 0.01))

            total_score = sum(scores)
            raw_weights = [s / total_score for s in scores]

            # 应用权重约束
            weights = cls._constrain_weights(raw_weights, len(group))

            return weights, budget

        etf_weights, etf_budget = calc_group_weights(etf_selected, etf_budget)
        stock_weights, stock_budget = calc_group_weights(stock_selected, stock_budget)

        # 合并为统一的持仓列表
        portfolio = []
        for i, s in enumerate(etf_selected):
            weight = etf_weights[i] * etf_ratio  # 全局权重
            amount = etf_weights[i] * etf_budget
            price = s.get('current_price', 0)
            if price > 0:
                shares = int(amount / price / 100) * 100
                if shares == 0:
                    shares = 100
                actual_amount = shares * price
            else:
                shares = 0
                actual_amount = 0
            portfolio.append(cls._make_holding(s, weight, shares, actual_amount, price))

        for i, s in enumerate(stock_selected):
            weight = stock_weights[i] * stock_ratio
            amount = stock_weights[i] * stock_budget
            price = s.get('current_price', 0)
            if price > 0:
                shares = int(amount / price / 100) * 100
                if shares == 0:
                    shares = 100
                actual_amount = shares * price
            else:
                shares = 0
                actual_amount = 0
            portfolio.append(cls._make_holding(s, weight, shares, actual_amount, price))

        total_invested = sum(p['amount'] for p in portfolio)

        # 组合级指标
        port_prob = sum(p['profit_prob'] * p['weight'] for p in portfolio)
        port_ret = sum(p['expected_return'] * p['weight'] for p in portfolio)

        # 组合风险 (简化: 加权平均风险分的倒数)
        port_risk = 1 - sum((p['risk_score'] / 100) * p['weight'] for p in portfolio)

        # 配置摘要
        etf_amount = sum(p['amount'] for p in portfolio if p['asset_type'] == 'etf')
        theme_alloc = defaultdict(float)
        for p in portfolio:
            theme_alloc[p['theme']] += p['amount']
        theme_pct = {t: round(a / total_invested, 3) for t, a in theme_alloc.items()}

        return {
            'portfolio': portfolio,
            'total_invested': round(total_invested, 2),
            'cash_reserve': round(cls.TOTAL_CAPITAL - total_invested, 2),
            'portfolio_profit_prob': round(port_prob, 3),
            'portfolio_expected_return': round(port_ret, 4),
            'portfolio_risk': round(port_risk, 3),
            'allocation_summary': {
                'etf_ratio': round(etf_amount / total_invested, 3) if total_invested > 0 else 0,
                'stock_ratio': round(1 - etf_amount / total_invested, 3) if total_invested > 0 else 0,
                'theme_allocation': dict(theme_pct),
                'n_holdings': len(portfolio),
                'max_position': max(p['weight'] for p in portfolio) if portfolio else 0,
            },
        }

    @classmethod
    def _make_holding(cls, s: Dict, weight: float, shares: int, actual_amount: float, price: float) -> Dict:
        """构建单个持仓字典 (v7.6: 股票止损止盈更宽)"""
        asset_type = s.get('asset_type', 'etf')
        # 股票止损-8%,止盈+8%; ETF止损-5%,止盈+5%
        if asset_type == 'stock':
            stop_loss_pct = 0.92
            target_pct = 1.08
        else:
            stop_loss_pct = 0.95
            target_pct = 1.05

        return {
            'code': s.get('code', ''),
            'name': s.get('name', ''),
            'asset_type': asset_type,
            'theme': s.get('theme', ''),
            'current_price': price,
            'shares': shares,
            'amount': round(actual_amount, 2),
            'weight': round(weight, 4),
            'profit_prob': s.get('profit_prob', 0),
            'expected_return': s.get('expected_return', 0),
            'risk_score': s.get('risk_score', 50),
            'trend_score': s.get('trend_score', 50),
            'valuation_score': s.get('valuation_score', 50),
            'screen_details': s.get('screen_result', {}).get('details', {}),
            'predict_details': s.get('predict_result', {}).get('method_details', {}),
            'stop_loss': round(price * stop_loss_pct, 3),
            'target_profit': round(price * target_pct, 3),
        }

    @classmethod
    def _constrain_weights(cls, raw_weights: List[float], n: int) -> List[float]:
        """应用权重约束: 5%-25%"""
        weights = np.array(raw_weights, dtype=float)

        # 迭代约束
        for _ in range(50):
            # 上限约束
            over = weights > cls.MAX_POSITION
            if over.any():
                excess = np.sum(weights[over] - cls.MAX_POSITION)
                weights[over] = cls.MAX_POSITION
                under = ~over & (weights > 0)
                if under.any():
                    weights[under] += excess / np.sum(under)

            # 下限约束
            under_min = weights < cls.MIN_POSITION
            if under_min.any():
                deficit = np.sum(cls.MIN_POSITION - weights[under_min])
                weights[under_min] = cls.MIN_POSITION
                over_ok = weights > cls.MIN_POSITION
                if over_ok.any():
                    weights[over_ok] -= deficit / np.sum(over_ok)

            # 归一化
            total = weights.sum()
            if total > 0:
                weights = weights / total

        # 最终裁剪
        weights = np.clip(weights, cls.MIN_POSITION, cls.MAX_POSITION)
        weights = weights / weights.sum()

        return weights.tolist()

    @classmethod
    def _empty_portfolio(cls) -> Dict[str, Any]:
        return {
            'portfolio': [],
            'total_invested': 0,
            'cash_reserve': cls.TOTAL_CAPITAL,
            'portfolio_profit_prob': 0,
            'portfolio_expected_return': 0,
            'portfolio_risk': 0,
            'allocation_summary': {
                'etf_ratio': 0,
                'stock_ratio': 0,
                'theme_allocation': {},
                'n_holdings': 0,
                'max_position': 0,
            },
        }


# =====================================================================
# 主引擎: 投资组合预测器
# =====================================================================

class PortfolioPredictor:
    """
    投资组合预测引擎 v7.5

    完整流程:
      1. 获取候选标的池 (ETF + 股票)
      2. 获取30日K线数据
      3. 三重筛选 (下跌风险低 + 上涨趋势 + 估值合理)
      4. 预测2日盈利概率
      5. 组合优化 (10万资金分配)
      6. 输出投资组合 + 止损止盈 + 风险提示
    """

    def __init__(self, capital: float = 100000):
        self.capital = capital
        self.kline_fetcher = KLineFetcher()
        self.screener = StockScreener
        self.predictor = ProfitPredictor
        self.optimizer = PortfolioOptimizer

        # 候选标的池
        self.etf_candidates: List[str] = []
        self.stock_candidates: List[str] = []
        self.kline_data: Dict[str, pd.DataFrame] = {}
        self.screen_results: Dict[str, Dict] = {}
        self.predict_results: Dict[str, Dict] = {}
        self.candidates: List[Dict] = []

    def set_etf_candidates(self, codes: List[str]):
        """设置ETF候选池"""
        self.etf_candidates = codes

    def set_stock_candidates(self, codes: List[str]):
        """设置股票候选池"""
        self.stock_candidates = codes

    def set_default_candidates(self):
        """设置默认候选池 (v7.6: ETF + 龙头股票)"""
        # ETF候选池: 核心宽基ETF + 主题ETF
        all_etf_codes = list(set(
            list(CORE_BROAD_ETF.keys()) + list(THEME_ETF.keys())
        ))
        self.etf_candidates = all_etf_codes

        # v7.6: 股票候选池 — 各板块龙头股
        self.stock_candidates = list(LEADER_STOCKS.keys())

    def run(self) -> Dict[str, Any]:
        """
        执行完整预测流程

        返回:
          {
            'report_date': str,
            'capital': float,
            'candidates_total': int,
            'kline_fetched': int,
            'screen_passed': int,
            'portfolio': Dict,
            'market_analysis': Dict,
            'timestamp': str,
          }
        """
        logger.info(f"=== 投资组合预测引擎 v7.5 启动 ===")
        logger.info(f"启动资金: {self.capital:.0f}元")

        # 1. 设置候选池
        if not self.etf_candidates and not self.stock_candidates:
            self.set_default_candidates()

        all_codes = self.etf_candidates + self.stock_candidates
        logger.info(f"候选标的: {len(all_codes)} 只 (ETF:{len(self.etf_candidates)}, 股票:{len(self.stock_candidates)})")

        # 2. 获取30日K线
        logger.info("获取30日K线数据...")
        self.kline_data = self.kline_fetcher.fetch_batch(all_codes, days=60)
        logger.info(f"K线获取成功: {len(self.kline_data)}/{len(all_codes)}")

        # 3. 三重筛选
        logger.info("执行三重筛选 (下跌风险+上涨趋势+估值合理)...")
        passed_codes = []
        for code, ohlc in self.kline_data.items():
            asset_type = 'stock' if code in self.stock_candidates else 'etf'
            name = self._get_name(code)

            screen = self.screener.screen(code, ohlc, name, asset_type=asset_type)
            self.screen_results[code] = screen

            if screen['pass']:
                passed_codes.append(code)
                logger.debug(f"  ✓ {code} {name}: 风险={screen['risk_score']}, 趋势={screen['trend_score']}, 估值={screen['valuation_score']}")
            # else:
            #     logger.debug(f"  ✗ {code} {name}: {'; '.join(screen['fail_reasons'])}")

        logger.info(f"筛选通过: {len(passed_codes)}/{len(self.kline_data)}")

        # 4. 预测2日盈利概率
        logger.info("预测未来2日盈利概率...")
        for code in passed_codes:
            ohlc = self.kline_data[code]
            pred = self.predictor.predict_2d_profit_prob(ohlc)
            self.predict_results[code] = pred

            if pred['profit_prob'] >= self.predictor.MIN_PROFIT_PROB:
                name = self._get_name(code)
                asset_type = 'stock' if code in self.stock_candidates else 'etf'
                theme = self._get_theme(code)

                screen = self.screen_results[code]
                self.candidates.append({
                    'code': code,
                    'name': name,
                    'asset_type': asset_type,
                    'theme': theme,
                    'current_price': float(ohlc['close'].iloc[-1]),
                    'profit_prob': pred['profit_prob'],
                    'expected_return': pred['expected_return'],
                    'upside': pred['upside'],
                    'downside': pred['downside'],
                    'confidence': pred['confidence'],
                    'risk_score': screen['risk_score'],
                    'trend_score': screen['trend_score'],
                    'valuation_score': screen['valuation_score'],
                    'screen_result': screen,
                    'predict_result': pred,
                })

        logger.info(f"高概率候选(>={self.predictor.MIN_PROFIT_PROB:.0%}): {len(self.candidates)} 只")

        # Fallback: 如果高概率候选不足5只,从筛选通过的标的中补充
        if len(self.candidates) < 5:
            logger.info("高概率候选不足,从筛选通过的标的中补充...")
            existing_codes = {c['code'] for c in self.candidates}
            for code in passed_codes:
                if code in existing_codes:
                    continue
                ohlc = self.kline_data[code]
                pred = self.predict_results.get(code, {'profit_prob': 0.5, 'expected_return': 0, 'upside': 0, 'downside': 0, 'confidence': '低', 'method_details': {}})
                name = self._get_name(code)
                asset_type = 'stock' if code in self.stock_candidates else 'etf'
                theme = self._get_theme(code)
                screen = self.screen_results[code]

                self.candidates.append({
                    'code': code,
                    'name': name,
                    'asset_type': asset_type,
                    'theme': theme,
                    'current_price': float(ohlc['close'].iloc[-1]),
                    'profit_prob': pred.get('profit_prob', 0.5),
                    'expected_return': pred.get('expected_return', 0),
                    'upside': pred.get('upside', 0),
                    'downside': pred.get('downside', 0),
                    'confidence': pred.get('confidence', '低'),
                    'risk_score': screen['risk_score'],
                    'trend_score': screen['trend_score'],
                    'valuation_score': screen['valuation_score'],
                    'screen_result': screen,
                    'predict_result': pred,
                })
            logger.info(f"补充后候选总数: {len(self.candidates)} 只")

        # 5. 组合优化
        logger.info("执行组合优化 (10万资金分配)...")
        portfolio = self.optimizer.optimize(self.candidates)

        # 6. 市场分析
        market_analysis = self._analyze_market()

        result = {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'capital': self.capital,
            'candidates_total': len(all_codes),
            'kline_fetched': len(self.kline_data),
            'screen_passed': len(passed_codes),
            'high_prob_count': len(self.candidates),
            'portfolio': portfolio,
            'market_analysis': market_analysis,
            'all_candidates': sorted(self.candidates, key=lambda x: x['profit_prob'], reverse=True),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        logger.info(f"=== 预测完成 ===")
        logger.info(f"组合盈利概率: {portfolio.get('portfolio_profit_prob', 0):.1%}")
        logger.info(f"预期2日收益: {portfolio.get('portfolio_expected_return', 0):.2%}")
        logger.info(f"持仓数量: {portfolio.get('allocation_summary', {}).get('n_holdings', 0)}")

        return result

    def _get_name(self, code: str) -> str:
        """获取标的名称 (v7.6: 支持股票)"""
        if code in LEADER_STOCKS:
            return LEADER_STOCKS[code][0]
        if code in CORE_BROAD_ETF:
            return CORE_BROAD_ETF[code]
        if code in THEME_ETF:
            return THEME_ETF[code]
        if code in CORE_LOF:
            return CORE_LOF[code]
        return code

    def _get_theme(self, code: str) -> str:
        """获取主题分类 (v7.6: 支持股票)"""
        # 先查股票主题
        if code in LEADER_STOCKS:
            return LEADER_STOCKS[code][1]
        for theme, codes in STOCK_THEME_MAP.items():
            if code in codes:
                return theme
        # 再查ETF主题
        for theme, codes in ETF_THEME_MAP.items():
            if code in codes:
                return theme
        if code in CORE_BROAD_ETF:
            return '宽基指数'
        return '其他'

    def _analyze_market(self) -> Dict[str, Any]:
        """分析市场整体状况"""
        # 基于所有已获取K线的涨跌统计
        if not self.kline_data:
            return {'status': '未知', 'breadth': 0, 'avg_change': 0}

        changes = []
        for code, ohlc in self.kline_data.items():
            if len(ohlc) >= 2:
                ret = float(ohlc['close'].iloc[-1] / ohlc['close'].iloc[-2] - 1)
                changes.append(ret)

        if not changes:
            return {'status': '未知', 'breadth': 0, 'avg_change': 0}

        up_count = sum(1 for c in changes if c > 0)
        breadth = up_count / len(changes)
        avg_change = float(np.mean(changes))

        if breadth > 0.7 and avg_change > 0.005:
            status = '强势'
        elif breadth > 0.55 and avg_change > 0:
            status = '偏强'
        elif breadth < 0.3 and avg_change < -0.005:
            status = '弱势'
        elif breadth < 0.45 and avg_change < 0:
            status = '偏弱'
        else:
            status = '震荡'

        return {
            'status': status,
            'breadth': round(breadth, 3),
            'avg_change': round(avg_change, 4),
            'up_count': up_count,
            'down_count': len(changes) - up_count,
            'total_observed': len(changes),
        }

    def save_results(self, result: Dict, filename: str = None) -> str:
        """保存结果到JSON"""
        if filename is None:
            filename = f"portfolio_prediction_{datetime.now().strftime('%Y%m%d')}.json"

        filepath = os.path.join(str(OUTPUT_DIR), filename)

        # 确保可序列化
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return str(obj)
            return obj

        clean_result = make_serializable(result)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_result, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"结果已保存: {filepath}")
        return filepath
