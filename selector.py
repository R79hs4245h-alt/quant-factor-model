"""
选股与风控 v6.2: 打分排序、行业暴露限制、权重优化、换手率控制
v6.2修复:
  1. [P0] _limit_sector_count: 按得分排序后保留最高分股票(之前未排序)
  2. [P1] clip_weights后重新归一化
  3. [P1] 换手率控制后重新检查权重上限
  4. [P1] 择时减仓后重新归一化剩余权重
  5. [P1] 集成risk_manager的仓位优化方法
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

from config import (
    TOP_N, MAX_WEIGHT, MAX_INDUSTRY_WEIGHT, MAX_TURNOVER,
    STOP_LOSS_THRESHOLD, DRAWDOWN_ALERT, MAX_SECTOR_STOCKS,
    MARKET_TIMING_ENABLED, BEAR_MARKET_MAX_POSITION, MIN_AMOUNT
)
from utils import get_logger

logger = get_logger("selector")


class StockSelector:
    """选股器 + 风险控制 v6.2"""

    def __init__(self,
                 top_n: int = TOP_N,
                 max_weight: float = MAX_WEIGHT,
                 max_industry_weight: float = MAX_INDUSTRY_WEIGHT,
                 max_turnover: float = MAX_TURNOVER,
                 max_sector_stocks: int = MAX_SECTOR_STOCKS):
        self.top_n = top_n
        self.max_weight = max_weight
        self.max_industry_weight = max_industry_weight
        self.max_turnover = max_turnover
        self.max_sector_stocks = max_sector_stocks
        self.market_timing: Optional[Dict] = None

    def set_market_timing(self, timing: Dict):
        """设置大盘择时信号"""
        self.market_timing = timing
        if timing and not timing.get('bullish', True):
            logger.warning(f"大盘择时: 熊市信号,最大仓位限制 {BEAR_MARKET_MAX_POSITION:.0%}")

    def select(self, scores: pd.Series,
               industry: Optional[pd.Series] = None,
               prev_weights: Optional[pd.Series] = None,
               risk_scores: Optional[pd.Series] = None,
               regime_adjustments: Optional[Dict] = None) -> pd.Series:
        """
        根据综合得分选股并分配权重
        v6.2新增: regime_adjustments参数,接收市场状态调整信号
        
        :param scores: 综合得分
        :param industry: 行业归属
        :param prev_weights: 上期持仓权重(用于换手率控制)
        :param risk_scores: 个股风险评分(0-100,越高风险越大)
        :param regime_adjustments: 市场状态调整信息(目标仓位等)
        :return: 目标权重 Series
        """
        # 0. 风险过滤: 排除高风险个股
        if risk_scores is not None:
            high_risk = risk_scores[risk_scores > 70].index
            if len(high_risk) > 0:
                scores = scores.drop(high_risk, errors='ignore')
                logger.info(f"风险过滤: 排除 {len(high_risk)} 只高风险个股")

        # 1. 得分排序选 top_n
        ranked = scores.sort_values(ascending=False)
        selected = ranked.head(self.top_n).index
        weights = pd.Series(0.0, index=scores.index)
        weights.loc[selected] = 1.0 / self.top_n

        # 2. 行业暴露限制
        if industry is not None:
            weights = self._limit_industry(weights, industry)

        # 3. v6.2修复: 板块集中度控制(按得分排序保留最高分)
        if industry is not None:
            weights = self._limit_sector_count(weights, industry, ranked)

        # 4. 单股权重上限 + 重新归一化
        weights = self._clip_weights(weights)
        weights = self._normalize(weights)  # v6.2: clip后重新归一化

        # 5. 换手率控制
        if prev_weights is not None:
            weights = self._control_turnover(weights, prev_weights)
            # v6.2: 换手率控制后重新检查权重上限
            weights = self._clip_weights(weights)
            weights = self._normalize(weights)

        # 6. v6.2: 市场状态仓位调整(优先于简单择时)
        if regime_adjustments and 'target_position' in regime_adjustments:
            target_pos = regime_adjustments['target_position']
            weights = self._apply_target_position(weights, target_pos)
        elif self.market_timing and not self.market_timing.get('bullish', True):
            weights = self._apply_market_timing(weights)
        
        # 7. 最终归一化
        weights = self._normalize(weights)

        logger.info(f"选股完成: 持仓 {int((weights > 0).sum())} 只, "
                    f"最大权重 {weights.max():.2%}, "
                    f"总仓位 {weights.sum():.0%}")
        return weights

    def _normalize(self, weights: pd.Series) -> pd.Series:
        """v6.2: 归一化权重"""
        total = weights.sum()
        if total > 0:
            return weights / total
        return weights

    def _limit_industry(self, weights: pd.Series,
                        industry: pd.Series) -> pd.Series:
        """限制单行业权重"""
        result = weights.copy()
        ind_aligned = industry.reindex(result.index)
        for ind in ind_aligned.dropna().unique():
            ind_mask = ind_aligned == ind
            ind_codes = result.index[ind_mask.fillna(False)]
            ind_weight = result.loc[ind_codes].sum()
            if ind_weight > self.max_industry_weight and ind_weight > 0:
                scale = self.max_industry_weight / ind_weight
                result.loc[ind_codes] *= scale
        return result

    def _limit_sector_count(self, weights: pd.Series,
                           industry: pd.Series,
                           ranked: pd.Series) -> pd.Series:
        """
        v6.2修复: 限制单板块入选股票数量
        按得分排序后保留得分最高的max_sector_stocks只
        """
        result = weights.copy()
        ind_aligned = industry.reindex(result.index)
        active = result[result > 0].index
        for ind in ind_aligned.loc[active].dropna().unique():
            ind_codes = result.index[(ind_aligned == ind) & (result > 0)]
            if len(ind_codes) > self.max_sector_stocks:
                # v6.2修复: 按得分排序,保留得分最高的
                ind_scores = ranked.loc[ranked.index.intersection(ind_codes)]
                ind_scores = ind_scores.sort_values(ascending=False)
                keep = ind_scores.head(self.max_sector_stocks).index
                drop = [c for c in ind_codes if c not in keep]
                result.loc[drop] = 0
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
            retain_ratio = min(0.5, self.max_turnover / (turnover + 1e-10))
            blended = old_w * retain_ratio + new_w * (1 - retain_ratio)
            weights.loc[common] = blended
            logger.info(f"换手率控制: {turnover:.2%} -> "
                        f"{(weights.loc[common] - old_w).abs().sum()/2:.2%}")
        return weights

    def _apply_market_timing(self, weights: pd.Series) -> pd.Series:
        """大盘择时: 熊市减仓"""
        total = weights.sum()
        if total > 0:
            target_total = BEAR_MARKET_MAX_POSITION
            scale = target_total / total
            weights = weights * scale
            logger.info(f"大盘择时减仓: {total:.0%} -> {weights.sum():.0%}")
        return weights

    def _apply_target_position(self, weights: pd.Series,
                                target_position: float) -> pd.Series:
        """v6.2: 根据市场状态目标仓位调整"""
        total = weights.sum()
        if total > 0 and target_position < 1.0:
            scale = target_position / total
            weights = weights * scale
            logger.info(f"市场状态仓位调整: 目标仓位 {target_position:.0%}")
        return weights

    def apply_stop_loss(self, current_weights: pd.Series,
                        current_returns: pd.Series) -> pd.Series:
        """个股止损: 跌破止损线则清仓"""
        result = current_weights.copy()
        stop_mask = current_returns < STOP_LOSS_THRESHOLD
        n_stopped = stop_mask.sum()
        if n_stopped > 0:
            result[stop_mask] = 0
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
            "concentration": (active ** 2).sum() if len(active) > 0 else 0,
            "total_position": weights.sum(),
        }
        if industry is not None:
            common = active.index.intersection(industry.index)
            ind_weights = industry.loc[common].map(active).groupby(
                industry.loc[common]).sum().sort_values(ascending=False)
            summary["top_industries"] = ind_weights.head(5).to_dict()
        return summary


def assess_stock_risk(quote: dict, tech: dict, fund_flow: Optional[dict] = None) -> Dict:
    """
    个股风险等级评估(0-100分,越高风险越大)
    整合自 v5.0 模型的风险评估逻辑
    """
    risk_score = 0
    risks = []

    if quote:
        pe = quote.get('pe_ttm', 0)
        if pe > 80:
            risk_score += 20
            risks.append(f"PE过高({pe:.1f})")
        elif pe > 50:
            risk_score += 10
            risks.append(f"PE偏高({pe:.1f})")

        turnover = quote.get('turnover', 0)
        if turnover > 15:
            risk_score += 10
            risks.append(f"换手率过高({turnover:.1f}%)")

        amplitude = quote.get('amplitude', 0)
        if amplitude > 8:
            risk_score += 10
            risks.append(f"振幅过大({amplitude:.1f}%)")

    if tech:
        if tech.get('trend') == '空头':
            risk_score += 25
            risks.append("均线空头排列")
        if tech.get('rsi', 50) > 80:
            risk_score += 15
            risks.append(f"RSI超买({tech['rsi']:.1f})")
        if tech.get('macd_signal') == '死叉':
            risk_score += 10
            risks.append("MACD死叉")
        if tech.get('boll_pos') == '上轨':
            risk_score += 10
            risks.append("触及布林上轨")
        if tech.get('vol_ratio', 1) > 2.5 and tech.get('macd_signal') == '死叉':
            risk_score += 15
            risks.append("放量下跌")

    if fund_flow:
        if fund_flow['main_net'] < 0 and fund_flow['main_5d'] < 0:
            risk_score += 20
            risks.append("主力持续净流出")

    level = '安全' if risk_score < 30 else ('关注' if risk_score < 50 else ('警告' if risk_score < 70 else '紧急撤离'))
    return {
        'score': min(risk_score, 100),
        'level': level,
        'risks': risks if risks else ['未发现明显风险信号']
    }
