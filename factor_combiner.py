"""
因子合成 v6.2: 等权 / IC 加权 / 截面回归(无前瞻偏差)
v6.2修复:
  1. [P0] 修复regression方法数据泄露: 使用历史模型预测当期,不再用当期forward_returns训练
  2. [P1] IC加权使用指数衰减(半衰期),增强近期IC权重
  3. [P1] 负IC因子保留反向预测能力,不再截断为0
  4. [P1] 新增IC_IR加权方法
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from config import FACTOR_COMBINE_METHOD, FACTOR_CONFIG
from utils import get_logger

logger = get_logger("combine")


class FactorCombiner:
    """因子合成器 v6.2"""

    def __init__(self, method: str = FACTOR_COMBINE_METHOD):
        self.method = method
        self.directions = {k: v["direction"] for k, v in FACTOR_CONFIG.items()}
        self.ic_history: Dict[str, List[float]] = defaultdict(list)
        
        # v6.2: 历史回归模型缓存(防数据泄露)
        self._regression_model = None
        self._regression_features = None
        
        # v6.2: IC半衰期
        self.ic_half_life = 6  # 6期半衰期

    def combine(self, factors: pd.DataFrame,
                forward_returns: Optional[pd.Series] = None) -> pd.Series:
        """
        合成多因子综合得分
        :param factors: 预处理后的因子矩阵
        :param forward_returns: 下期收益(仅用于事后IC更新,不用于当期训练)
        :return: 综合得分 Series
        """
        # 应用因子方向
        adjusted = factors.copy()
        for col, direction in self.directions.items():
            if col in adjusted.columns:
                adjusted[col] = adjusted[col] * direction

        # v6.2修复: 先用历史模型预测当期,再用当期forward_returns更新IC(事后)
        if self.method == "equal_weight":
            score = self._equal_weight(adjusted)
        elif self.method == "ic_weight":
            score = self._ic_weight(adjusted)
        elif self.method == "ic_ir_weight":
            score = self._ic_ir_weight(adjusted)
        elif self.method == "regression":
            score = self._regression(adjusted)
        else:
            score = self._equal_weight(adjusted)

        # v6.2修复: 在预测完成后再更新IC(纯事后评估,不影响当期预测)
        if forward_returns is not None:
            self._update_ic(adjusted, forward_returns)
            self._train_regression_model(adjusted, forward_returns)

        logger.info(f"因子合成完成 (方法={self.method}): {len(score)} 只股票得分")
        return score

    def _equal_weight(self, factors: pd.DataFrame) -> pd.Series:
        return factors.mean(axis=1)

    def _ic_weight(self, factors: pd.DataFrame) -> pd.Series:
        weights = {}
        for col in factors.columns:
            ic_list = self.ic_history.get(col, [])
            if ic_list:
                weights[col] = self._weighted_ic_mean(ic_list)
            else:
                weights[col] = 0.0
        total = sum(abs(w) for w in weights.values())
        if total == 0:
            return self._equal_weight(factors)
        weights = {k: w / total for k, w in weights.items()}
        score = pd.Series(0.0, index=factors.index)
        for col, w in weights.items():
            if col in factors.columns:
                score += factors[col].fillna(0) * w
        return score

    def _ic_ir_weight(self, factors: pd.DataFrame) -> pd.Series:
        weights = {}
        has_history = False
        for col in factors.columns:
            ic_list = self.ic_history.get(col, [])
            if len(ic_list) >= 3:
                ic_series = pd.Series(ic_list[-12:])
                ic_mean = self._weighted_ic_mean(ic_list)
                ic_std = ic_series.std() + 1e-10
                ir = ic_mean / ic_std
                weights[col] = max(ir, 0)
                has_history = True
            else:
                weights[col] = 0.0
        if not has_history or sum(weights.values()) == 0:
            return self._equal_weight(factors)
        total = sum(weights.values())
        weights = {k: w / total for k, w in weights.items()}
        score = pd.Series(0.0, index=factors.index)
        for col, w in weights.items():
            if col in factors.columns and w > 0:
                score += factors[col].fillna(0) * w
        return score

    def _weighted_ic_mean(self, ic_list: List[float]) -> float:
        if not ic_list:
            return 0.0
        recent = ic_list[-12:]
        n = len(recent)
        alpha = np.log(2) / self.ic_half_life
        weights = np.exp(-alpha * np.arange(n - 1, -1, -1))
        weights = weights / weights.sum()
        return float(np.average(recent, weights=weights))

    def _regression(self, factors: pd.DataFrame) -> pd.Series:
        if self._regression_model is None or self._regression_features is None:
            if self.ic_history and any(len(v) > 0 for v in self.ic_history.values()):
                return self._ic_weight(factors)
            return self._equal_weight(factors)
        try:
            feature_cols = [c for c in self._regression_features if c in factors.columns]
            if len(feature_cols) != len(self._regression_features):
                return self._ic_weight(factors)
            X = factors[feature_cols].fillna(0).values
            predictions = self._regression_model.predict(X)
            return pd.Series(predictions, index=factors.index)
        except Exception:
            return self._ic_weight(factors)

    def _train_regression_model(self, factors: pd.DataFrame, forward_returns: pd.Series):
        from sklearn.linear_model import Ridge
        common_idx = factors.index.intersection(forward_returns.index)
        if len(common_idx) < 50:
            return
        try:
            X = factors.loc[common_idx].fillna(0).values
            y = forward_returns.loc[common_idx].values
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X, y)
            self._regression_model = model
            self._regression_features = list(factors.columns)
        except Exception:
            pass

    def _update_ic(self, factors: pd.DataFrame, forward_returns: pd.Series):
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
                    if len(self.ic_history[col]) > 60:
                        self.ic_history[col] = self.ic_history[col][-60:]
            except Exception:
                pass

    def get_ic_summary(self) -> pd.DataFrame:
        rows = []
        for factor, ic_list in self.ic_history.items():
            if not ic_list:
                continue
            ic_series = pd.Series(ic_list)
            rows.append({
                "factor": factor,
                "ic_mean": ic_series.mean(),
                "ic_std": ic_series.std(),
                "ir": ic_series.mean() / (ic_series.std() + 1e-10),
                "ic_positive_ratio": (ic_series > 0).mean(),
                "n_periods": len(ic_series),
                "latest_ic": ic_series.iloc[-1],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("ic_mean", ascending=False)
        return df
