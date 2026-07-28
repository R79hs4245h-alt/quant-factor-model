"""
选股与风控: 打分排序、行业暴露限制、权重优化、换手率控制
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

from config import (
    TOP_N, MAX_WEIGHT, MAX_INDUSTRY_WEIGHT, MAX_TURNOVER,
    STOP_LOSS_THRESHOLD, DRAWDOWN_ALERT
)
from utils import get_logger

logger = get_logger("selector")


class StockSelector:
    """选股器 + 风险控制"""

    def __init__(self,
                 top_n: int = TOP_N,
                 max_weight: float = MAX_WEIGHT,
                 max_industry_weight: float = MAX_INDUSTRY_WEIGHT,
                 max_turnover: float = MAX_TURNOVER):
        self.top_n = top_n
        self.max_weight = max_weight
        self.max_industry_weight = max_industry_weight
        self.max_turnover = max_turnover

    def select(self, scores: pd.Series,
               industry: Optional[pd.Series] = None,
               prev_weights: Optional[pd.Series] = None) -> pd.Series:
        """
        根据综合得分选股并分配权重
        :param scores: 综合得分
        :param industry: 行业归属
        :param prev_weights: 上期持仓权重(用于换手率控制)
        :return: 目标权重 Series
        """
        # 1. 得分排序选 top_n
        ranked = scores.sort_values(ascending=False)
        selected = ranked.head(self.top_n).index
        weights = pd.Series(0.0, index=scores.index)
        weights.loc[selected] = 1.0 / self.top_n

        # 2. 行业暴露限制
        if industry is not None:
            weights = self._limit_industry(weights, industry)

        # 3. 单股权重上限
        weights = self._clip_weights(weights)

        # 4. 换手率控制
        if prev_weights is not None:
            weights = self._control_turnover(weights, prev_weights)

        # 5. 归一化
        weights = weights / weights.sum()

        logger.info(f"选股完成: 持仓 {int((weights > 0).sum())} 只, "
                    f"最大权重 {weights.max():.2%}")
        return weights

    def _limit_industry(self, weights: pd.Series,
                        industry: pd.Series) -> pd.Series:
        """限制单行业权重"""
        result = weights.copy()
        # 对齐到权重索引,补全行业信息
        ind_aligned = industry.reindex(result.index)
        for ind in ind_aligned.dropna().unique():
            ind_mask = ind_aligned == ind
            ind_codes = result.index[ind_mask.fillna(False)]
            ind_weight = result.loc[ind_codes].sum()
            if ind_weight > self.max_industry_weight and ind_weight > 0:
                # 等比例缩减
                scale = self.max_industry_weight / ind_weight
                result.loc[ind_codes] *= scale
        return result

    def _clip_weights(self, weights: pd.Series) -> pd.Series:
        """单股权重上限"""
        clipped = weights.clip(upper=self.max_weight)
        return clipped

    def _control_turnover(self, weights: pd.Series,
                          prev_weights: pd.Series) -> pd.Series:
        """换手率控制: 若换手超过阈值则混合新旧权重"""
        common = weights.index.intersection(prev_weights.index)
        new_w = weights.loc[common]
        old_w = prev_weights.reindex(common, fill_value=0.0)

        turnover = (new_w - old_w).abs().sum() / 2
        if turnover > self.max_turnover:
            # 混合: 保留部分旧仓位
            retain_ratio = min(0.5, self.max_turnover / (turnover + 1e-10))
            blended = old_w * retain_ratio + new_w * (1 - retain_ratio)
            weights.loc[common] = blended
            logger.info(f"换手率控制: {turnover:.2%} -> "
                        f"{(weights.loc[common] - old_w).abs().sum()/2:.2%}")
        return weights

    def apply_stop_loss(self, current_weights: pd.Series,
                        current_returns: pd.Series) -> pd.Series:
        """个股止损: 跌破止损线则清仓"""
        result = current_weights.copy()
        stop_mask = current_returns < STOP_LOSS_THRESHOLD
        n_stopped = stop_mask.sum()
        if n_stopped > 0:
            result[stop_mask] = 0
            # 重新分配到剩余持仓
            remaining = result[result > 0]
            if remaining.sum() > 0:
                result[result > 0] = remaining / remaining.sum()
            logger.warning(f"止损触发: {n_stopped} 只股票清仓")
        return result

    def check_drawdown(self, portfolio_value: pd.Series) -> bool:
        """检查组合回撤是否超预警"""
        peak = portfolio_value.cummax()
        drawdown = (portfolio_value - peak) / peak
        max_dd = drawdown.min()
        if max_dd < -DRAWDOWN_ALERT:
            logger.warning(f"回撤预警: 当前最大回撤 {max_dd:.2%} "
                          f"超过阈值 {-DRAWDOWN_ALERT:.2%}")
            return True
        return False

    def get_portfolio_summary(self, weights: pd.Series,
                              industry: Optional[pd.Series] = None) -> Dict:
        """获取组合摘要统计"""
        active = weights[weights > 0]
        summary = {
            "n_holdings": len(active),
            "max_weight": active.max() if len(active) > 0 else 0,
            "mean_weight": active.mean() if len(active) > 0 else 0,
            "concentration": (active ** 2).sum() if len(active) > 0 else 0,  # HHI
        }
        if industry is not None:
            common = active.index.intersection(industry.index)
            ind_weights = industry.loc[common].map(active).groupby(
                industry.loc[common]).sum().sort_values(ascending=False)
            summary["top_industries"] = ind_weights.head(5).to_dict()
        return summary
