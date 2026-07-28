"""
因子合成: 等权 / IC 加权 / 截面回归
IC/IR 评估因子有效性
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from config import FACTOR_COMBINE_METHOD, FACTOR_CONFIG
from utils import get_logger

logger = get_logger("combine")


class FactorCombiner:
    """因子合成器"""

    def __init__(self, method: str = FACTOR_COMBINE_METHOD):
        self.method = method
        self.directions = {k: v["direction"] for k, v in FACTOR_CONFIG.items()}
        self.ic_history: Dict[str, List[float]] = defaultdict(list)

    def combine(self, factors: pd.DataFrame,
                forward_returns: Optional[pd.Series] = None) -> pd.Series:
        """
        合成多因子综合得分
        :param factors: 预处理后的因子矩阵
        :param forward_returns: 下期收益(用于IC计算),可选
        :return: 综合得分 Series
        """
        # 应用因子方向
        adjusted = factors.copy()
        for col, direction in self.directions.items():
            if col in adjusted.columns:
                adjusted[col] = adjusted[col] * direction

        # 计算 IC(若提供下期收益)
        if forward_returns is not None:
            self._update_ic(adjusted, forward_returns)

        if self.method == "equal_weight":
            score = self._equal_weight(adjusted)
        elif self.method == "ic_weight":
            score = self._ic_weight(adjusted)
        elif self.method == "regression":
            score = self._regression(adjusted, forward_returns)
        else:
            score = self._equal_weight(adjusted)

        logger.info(f"因子合成完成 (方法={self.method}): {len(score)} 只股票得分")
        return score

    def _equal_weight(self, factors: pd.DataFrame) -> pd.Series:
        """等权合成"""
        return factors.mean(axis=1)

    def _ic_weight(self, factors: pd.DataFrame) -> pd.Series:
        """IC 加权合成: 用历史 IC 均值作为权重"""
        weights = {}
        for col in factors.columns:
            ic_list = self.ic_history.get(col, [])
            if ic_list:
                ic_mean = np.mean(ic_list[-12:])  # 取最近12期
                weights[col] = ic_mean
            else:
                weights[col] = 1.0 / len(factors.columns)  # 默认等权

        # 归一化权重(保证正权重)
        total = sum(abs(w) for w in weights.values()) or 1
        weights = {k: max(v, 0) / total for k, v in weights.items()}
        # 若全为0则退化为等权
        if sum(weights.values()) == 0:
            return self._equal_weight(factors)

        score = pd.Series(0.0, index=factors.index)
        for col, w in weights.items():
            if col in factors.columns and w > 0:
                score += factors[col].fillna(0) * w
        return score

    def _regression(self, factors: pd.DataFrame,
                    forward_returns: Optional[pd.Series]) -> pd.Series:
        """截面回归合成"""
        if forward_returns is None:
            return self._equal_weight(factors)

        from sklearn.linear_model import LinearRegression

        common_idx = factors.index.intersection(forward_returns.index)
        if len(common_idx) < 50:
            return self._equal_weight(factors)

        X = factors.loc[common_idx].fillna(0).values
        y = forward_returns.loc[common_idx].values

        try:
            model = LinearRegression()
            model.fit(X, y)
            score = pd.Series(model.predict(X), index=common_idx)
            return score
        except Exception:
            return self._equal_weight(factors)

    def _update_ic(self, factors: pd.DataFrame, forward_returns: pd.Series):
        """更新各因子的 IC 值(Spearman 秩相关)"""
        common_idx = factors.index.intersection(forward_returns.index)
        if len(common_idx) < 30:
            return
        for col in factors.columns:
            try:
                ic = factors.loc[common_idx, col].corr(
                    forward_returns.loc[common_idx], method="spearman"
                )
                if not np.isnan(ic):
                    self.ic_history[col].append(ic)
            except Exception:
                pass

    def get_ic_summary(self) -> pd.DataFrame:
        """获取 IC 统计摘要"""
        rows = []
        for factor, ic_list in self.ic_history.items():
            if not ic_list:
                continue
            ic_series = pd.Series(ic_list)
            rows.append({
                "factor": factor,
                "ic_mean": ic_series.mean(),
                "ic_std": ic_series.std(),
                "ir": ic_series.mean() / (ic_series.std() + 1e-10),  # IR = IC均值/IC标准差
                "ic_positive_ratio": (ic_series > 0).mean(),
                "n_periods": len(ic_series),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("ic_mean", ascending=False)
        return df
