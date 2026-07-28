"""
高级风险管理器 v6.1
功能:
  1. 波动率目标(Vol Targeting)自适应仓位
  2. CVaR(条件VaR)风险预算
  3. 行业/风格暴露实时监控
  4. 动态止损(A-Trailing Stop)
  5. 组合压力测试
  6. Kelly公式仓位优化
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field

from config import (
    MAX_WEIGHT, MAX_INDUSTRY_WEIGHT, MAX_TURNOVER,
    STOP_LOSS_THRESHOLD, DRAWDOWN_ALERT, TOP_N,
    BEAR_MARKET_MAX_POSITION
)
from utils import get_logger

logger = get_logger("risk_manager")


@dataclass
class PositionSizingResult:
    """仓位计算结果"""
    target_weight: float
    vol_target: float
    current_vol: float
    kelly_fraction: float
    confidence: float
    method: str


@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario: str
    portfolio_loss: float
    max_single_loss: float
    n_breach_stop: int
    survival: bool


class AdvancedRiskManager:
    """高级风险管理器"""

    def __init__(self,
                 vol_target: float = 0.15,        # 年化波动目标 15%
                 max_cvar: float = 0.03,          # 单日CVaR上限 3%
                 max_sector_exposure: float = 0.25,
                 min_position: float = 0.005,     # 最小持仓 0.5%
                 max_position: float = 0.05,      # 最大持仓 5%
                 confidence_level: float = 0.95):  # 置信水平
        self.vol_target = vol_target
        self.max_cvar = max_cvar
        self.max_sector_exposure = max_sector_exposure
        self.min_position = min_position
        self.max_position = max_position
        self.confidence_level = confidence_level

        # 历史波动率缓存
        self._vol_history: List[float] = []
        self._drawdown_history: List[float] = []

    def vol_target_sizing(self,
                          stock_vol: float,
                          portfolio_vol: float = 0.15,
                          target_vol: float = None) -> float:
        """
        波动率目标仓位计算
        :param stock_vol: 个股年化波动率
        :param portfolio_vol: 当前组合波动率
        :param target_vol: 目标波动率
        :return: 建议权重
        """
        target = target_vol or self.vol_target

        if stock_vol <= 0:
            return self.max_position

        # 基础权重 = 目标波动 / 个股波动
        base_weight = target / stock_vol

        # 考虑组合波动率的调整
        if portfolio_vol > 0:
            vol_ratio = target / portfolio_vol
            base_weight *= min(vol_ratio, 1.5)

        # 限制在合理范围
        base_weight = np.clip(base_weight, self.min_position, self.max_position)

        return float(base_weight)

    def kelly_criterion(self,
                        win_rate: float,
                        win_loss_ratio: float,
                        max_fraction: float = 0.25) -> float:
        """
        Kelly公式计算最优仓位
        f* = (p*b - q) / b
        其中 p=胜率, b=盈亏比, q=1-p

        :param win_rate: 胜率
        :param win_loss_ratio: 平均盈利/平均亏损
        :param max_fraction: 最大Kelly分数(通常用半Kelly)
        :return: Kelly仓位比例
        """
        p = win_rate
        q = 1 - p
        b = win_loss_ratio

        if b <= 0:
            return 0.0

        kelly = (p * b - q) / b

        # 半Kelly(更保守)
        kelly = kelly * 0.5

        # 限制最大仓位
        kelly = min(kelly, max_fraction)
        kelly = max(kelly, 0.0)

        return float(kelly)

    def calculate_cvar(self, returns: pd.Series,
                       confidence: float = None) -> float:
        """
        计算条件VaR (CVaR / Expected Shortfall)
        :param returns: 收益率序列
        :param confidence: 置信水平
        :return: CVaR值(负数表示损失)
        """
        conf = confidence or self.confidence_level
        if returns.empty:
            return 0.0

        var = np.percentile(returns, (1 - conf) * 100)
        tail = returns[returns <= var]
        if tail.empty:
            return float(var)
        cvar = tail.mean()
        return float(cvar)

    def calculate_var(self, returns: pd.Series,
                      confidence: float = None) -> float:
        """计算VaR"""
        conf = confidence or self.confidence_level
        if returns.empty:
            return 0.0
        return float(np.percentile(returns, (1 - conf) * 100))

    def dynamic_stop_loss(self,
                          entry_price: float,
                          current_price: float,
                          atr: float,
                          trend_strength: str = "up",
                          holding_days: int = 0) -> Dict:
        """
        动态止损(A-Trailing Stop)
        根据ATR和趋势强度动态调整止损线

        :param entry_price: 入场价
        :param current_price: 当前价
        :param atr: ATR值
        :param trend_strength: 趋势强度 up/down/range
        :param holding_days: 持有天数
        :return: {"stop_price": float, "stop_type": str, "reason": str}
        """
        if atr <= 0:
            atr = entry_price * 0.03  # 默认3%

        # 趋势向上: 跟踪止损(2倍ATR)
        if trend_strength == "up":
            stop_price = current_price - 2 * atr
            stop_type = "trailing"
            reason = "上升趋势跟踪止损(2xATR)"

        # 趋势向下: 固定止损(1.5倍ATR)
        elif trend_strength == "down":
            stop_price = entry_price - 1.5 * atr
            stop_type = "fixed"
            reason = "下降趋势固定止损(1.5xATR)"

        # 震荡: 时间止损 + 适中ATR
        else:
            base_stop = entry_price - 2.5 * atr
            # 持有时间越长,止损越紧
            time_factor = max(0.5, 1.0 - holding_days * 0.005)
            stop_price = max(base_stop, current_price * (1 - 0.08 * time_factor))
            stop_type = "time"
            reason = f"震荡时间止损(2.5xATR, 时间因子={time_factor:.2f})"

        # 确保止损不低于绝对止损线
        absolute_stop = entry_price * (1 + STOP_LOSS_THRESHOLD)
        if stop_price < absolute_stop:
            stop_price = absolute_stop
            stop_type = "absolute"
            reason = f"绝对止损线({STOP_LOSS_THRESHOLD:.0%})"

        # 确保止损不高于当前价
        if stop_price >= current_price:
            stop_price = current_price * 0.95
            stop_type = "emergency"
            reason = "紧急止损(5%)"

        return {
            "stop_price": round(stop_price, 2),
            "stop_pct": round((stop_price - current_price) / current_price * 100, 2),
            "stop_type": stop_type,
            "reason": reason,
        }

    def check_sector_exposure(self, weights: pd.Series,
                               industry: pd.Series,
                               max_exposure: float = None) -> Dict:
        """
        检查行业暴露
        :return: {"breach": bool, "over_exposed": [...], "adjusted_weights": pd.Series}
        """
        max_exp = max_exposure or self.max_sector_exposure

        if industry is None or industry.empty:
            return {"breach": False, "over_exposed": [], "adjusted_weights": weights}

        common = weights.index.intersection(industry.index)
        if len(common) == 0:
            return {"breach": False, "over_exposed": [], "adjusted_weights": weights}

        w = weights.loc[common]
        ind = industry.loc[common]

        sector_weights = w.groupby(ind).sum()
        over_exposed = sector_weights[sector_weights > max_exp].index.tolist()

        adjusted = weights.copy()
        if over_exposed:
            for sector in over_exposed:
                sector_codes = common[ind.loc[common] == sector]
                sector_total = adjusted.loc[sector_codes].sum()
                if sector_total > 0:
                    scale = max_exp / sector_total
                    adjusted.loc[sector_codes] *= scale
                    logger.warning(f"行业暴露超标: {sector} "
                                 f"({sector_total:.1%} -> {max_exp:.1%})")

        return {
            "breach": len(over_exposed) > 0,
            "over_exposed": over_exposed,
            "adjusted_weights": adjusted,
        }

    def stress_test(self, weights: pd.Series,
                    returns_history: pd.DataFrame,
                    scenarios: List[str] = None) -> List[StressTestResult]:
        """
        组合压力测试
        :param weights: 当前权重
        :param returns_history: 历史收益率面板(code x date)
        :param scenarios: 压力情景列表
        :return: 压力测试结果列表
        """
        if scenarios is None:
            scenarios = [
                "2008金融危机", "2015股灾", "2020疫情", "极端单日下跌",
                "流动性枯竭", "行业轮动反转"
            ]

        results = []

        for scenario in scenarios:
            try:
                if scenario == "2008金融危机":
                    shock = self._apply_shock(returns_history, multiplier=-3.0, vol_mult=2.5)
                elif scenario == "2015股灾":
                    shock = self._apply_shock(returns_history, multiplier=-2.0, vol_mult=2.0)
                elif scenario == "2020疫情":
                    shock = self._apply_shock(returns_history, multiplier=-1.5, vol_mult=1.8)
                elif scenario == "极端单日下跌":
                    shock = self._apply_shock(returns_history, multiplier=-5.0, vol_mult=1.0)
                elif scenario == "流动性枯竭":
                    shock = self._apply_shock(returns_history, multiplier=-1.0, vol_mult=3.0)
                elif scenario == "行业轮动反转":
                    shock = self._apply_shock(returns_history, multiplier=-1.0, vol_mult=1.5, reverse=True)
                else:
                    continue

                # 计算组合损失
                common = weights.index.intersection(shock.index)
                if len(common) == 0:
                    continue

                w = weights.loc[common]
                stock_losses = shock.loc[common]

                portfolio_loss = float((w * stock_losses).sum())
                max_single_loss = float(stock_losses.min())
                n_breach = int((stock_losses < STOP_LOSS_THRESHOLD).sum())

                survival = portfolio_loss > -DRAWDOWN_ALERT

                results.append(StressTestResult(
                    scenario=scenario,
                    portfolio_loss=portfolio_loss,
                    max_single_loss=max_single_loss,
                    n_breach_stop=n_breach,
                    survival=survival,
                ))

            except Exception as e:
                logger.debug(f"压力测试 {scenario} 异常: {e}")

        return results

    def _apply_shock(self, returns_history: pd.DataFrame,
                     multiplier: float = -1.0,
                     vol_mult: float = 1.5,
                     reverse: bool = False) -> pd.Series:
        """应用压力情景到历史收益"""
        if returns_history.empty:
            return pd.Series(dtype=float)

        # 取最近一期的日收益
        if isinstance(returns_history, pd.DataFrame):
            latest = returns_history.iloc[-1] if len(returns_history) > 0 else pd.Series()
        else:
            latest = returns_history

        # 放大波动
        shocked = latest * vol_mult

        # 叠加冲击
        if reverse:
            # 反转: 之前涨的现在跌
            shocked = -shocked.abs() * np.sign(multiplier)
        else:
            shocked = shocked + multiplier * latest.std()

        return shocked

    def optimize_position_sizes(self,
                                scores: pd.Series,
                                vols: pd.Series,
                                target_vol: float = None,
                                method: str = "vol_target") -> pd.Series:
        """
        优化个股仓位分配
        :param scores: 因子得分
        :param vols: 个股波动率
        :param target_vol: 组合目标波动率
        :param method: vol_target / equal / score_weighted / risk_parity
        :return: 优化后权重
        """
        target = target_vol or self.vol_target

        common = scores.index.intersection(vols.index)
        if len(common) == 0:
            # 无法优化,等权
            n = min(len(scores), TOP_N)
            return pd.Series(1.0/n, index=scores.head(n).index)

        s = scores.loc[common]
        v = vols.loc[common]

        if method == "vol_target":
            # 波动率目标: 高分低波动 -> 更大仓位
            raw_weights = s.clip(lower=0) / (v + 1e-8)
            raw_weights = raw_weights.clip(lower=0)

        elif method == "risk_parity":
            # 风险平价: 每只股票贡献相同风险
            raw_weights = 1.0 / (v + 1e-8)

        elif method == "score_weighted":
            # 得分加权
            raw_weights = s.clip(lower=0)

        else:  # equal
            raw_weights = pd.Series(1.0, index=common)

        # 归一化
        total = raw_weights.sum()
        if total > 0:
            weights = raw_weights / total
        else:
            weights = pd.Series(1.0/len(common), index=common)

        # 限制单股权重
        weights = weights.clip(upper=self.max_position)

        # 再次归一化
        total = weights.sum()
        if total > 0:
            weights = weights / total

        # 只保留TOP_N
        if len(weights) > TOP_N:
            top_codes = weights.nlargest(TOP_N).index
            weights = weights.loc[top_codes]
            weights = weights / weights.sum()

        return weights

    def get_risk_report(self, weights: pd.Series,
                        returns_history: Optional[pd.DataFrame] = None,
                        industry: Optional[pd.Series] = None) -> Dict:
        """生成风险报告"""
        report = {
            "n_holdings": int((weights > 0).sum()),
            "max_weight": float(weights.max()),
            "concentration_hhi": float((weights ** 2).sum()),
            "total_position": float(weights.sum()),
        }

        if returns_history is not None and not returns_history.empty:
            common = weights.index.intersection(returns_history.columns
                                                if isinstance(returns_history, pd.DataFrame)
                                                else returns_history.index)
            if len(common) > 0:
                w = weights.loc[common]
                if isinstance(returns_history, pd.DataFrame):
                    port_returns = (returns_history[common] * w).sum(axis=1)
                else:
                    port_returns = returns_history.loc[common] * w

                report["portfolio_vol"] = float(port_returns.std() * np.sqrt(252))
                report["var_95"] = self.calculate_var(port_returns)
                report["cvar_95"] = self.calculate_cvar(port_returns)
                report["max_drawdown"] = float(
                    ((1 + port_returns).cumprod() /
                     (1 + port_returns).cumprod().cummax() - 1).min()
                )

        if industry is not None:
            sector_check = self.check_sector_exposure(weights, industry)
            report["sector_breach"] = sector_check["breach"]
            report["over_exposed_sectors"] = sector_check["over_exposed"]

        return report
