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
        # 顺序很重要: 预测 -> 更新IC, 不能反过来
        
        if self.method == "equal_weight":
            score = self._equal_weight(adjusted)
        elif self.method == "ic_weight":
            score = self._ic_weight(adjusted)
        elif self.method == "ic_ir_weight":
            score = self._ic_ir_weight(adjusted)
        elif self.method == "regression":
            score = self._regression(adjusted)  # v6.2: 不再传forward_returns
        else:
            score = self._equal_weight(adjusted)

        # v6.2修复: 在预测完成后再更新IC(纯事后评估,不影响当期预测)
        if forward_returns is not None:
            self._update_ic(adjusted, forward_returns)
            # 训练下一期用的回归模型(用当期数据训练,下一期使用)
            self._train_regression_model(adjusted, forward_returns)

        logger.info(f"因子合成完成 (方法={self.method}): {len(score)} 只股票得分")
        return score

    def _equal_weight(self, factors: pd.DataFrame) -> pd.Series:
        """等权合成"""
        return factors.mean(axis=1)

    def _ic_weight(self, factors: pd.DataFrame) -> pd.Series:
        """IC 加权合成: 用历史 IC 指数衰减加权"""
        weights = {}
        for col in factors.columns:
            ic_list = self.ic_history.get(col, [])
            if ic_list:
                # v6.2: 指数衰减加权(半衰期)
                ic_mean = self._weighted_ic_mean(ic_list)
                weights[col] = ic_mean
            else:
                weights[col] = 1.0 / len(factors.columns)

        # v6.2修复: 保留负IC因子的反向预测能力
        # 不再截断为0,而是保留符号,按绝对值归一化
        total = sum(abs(w) for w in weights.values()) or 1
        weights = {k: w / total for k, v in weights.items() for w in [weights[k]]}
        
        # 如果IC历史不足,退化为等权
        if not self.ic_history or all(len(v) == 0 for v in self.ic_history.values()):
            return self._equal_weight(factors)

        # 重新计算(上面的字典推导有bug,直接用循环)
        weights = {}
        for col in factors.columns:
            ic_list = self.ic_history.get(col, [])
            if ic_list:
                weights[col] = self._weighted_ic_mean(ic_list)
            else:
                weights[col] = 0.0  # 无IC历史的因子给0权重
        
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
        """v6.2新增: IC_IR加权 (IC均值/IC标准差)"""
        weights = {}
        has_history = False
        for col in factors.columns:
            ic_list = self.ic_history.get(col, [])
            if len(ic_list) >= 3:
                ic_series = pd.Series(ic_list[-12:])
                ic_mean = self._weighted_ic_mean(ic_list)
                ic_std = ic_series.std() + 1e-10
                ir = ic_mean / ic_std
                # 只保留IR为正的因子(有稳定预测能力)
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
        """v6.2: 指数衰减IC加权均值"""
        if not ic_list:
            return 0.0
        recent = ic_list[-12:]  # 最多取最近12期
        n = len(recent)
        # 指数衰减权重
        alpha = np.log(2) / self.ic_half_life
        weights = np.exp(-alpha * np.arange(n - 1, -1, -1))
        weights = weights / weights.sum()
        return float(np.average(recent, weights=weights))

    def _regression(self, factors: pd.DataFrame) -> pd.Series:
        """
        v6.2修复: 截面回归合成 - 使用历史训练的模型预测当期
        不再使用当期forward_returns训练(消除数据泄露)
        """
        if self._regression_model is None or self._regression_features is None:
            # 无历史模型,退化为IC加权或等权
            if self.ic_history and any(len(v) > 0 for v in self.ic_history.values()):
                return self._ic_weight(factors)
            return self._equal_weight(factors)

        # 使用历史训练的模型预测当期
        try:
            # 确保特征列对齐
            feature_cols = [c for c in self._regression_features if c in factors.columns]
            if len(feature_cols) != len(self._regression_features):
                logger.warning("特征列不匹配,退化为IC加权")
                return self._ic_weight(factors)

            X = factors[feature_cols].fillna(0).values
            predictions = self._regression_model.predict(X)
            return pd.Series(predictions, index=factors.index)
        except Exception as e:
            logger.debug(f"回归预测失败: {e}, 退化为IC加权")
            return self._ic_weight(factors)

    def _train_regression_model(self, factors: pd.DataFrame,
                                 forward_returns: pd.Series):
        """
        v6.2新增: 用当期数据训练模型,供下一期使用
        这是正确的时间顺序: t期训练 -> t+1期使用
        """
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
            logger.debug(f"回归模型已训练(供下期使用), 样本数={len(common_idx)}")
        except Exception as e:
            logger.debug(f"回归模型训练失败: {e}")

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
                    # 限制历史长度
                    if len(self.ic_history[col]) > 60:
                        self.ic_history[col] = self.ic_history[col][-60:]
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
                "ir": ic_series.mean() / (ic_series.std() + 1e-10),
                "ic_positive_ratio": (ic_series > 0).mean(),
                "n_periods": len(ic_series),
                "latest_ic": ic_series.iloc[-1],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("ic_mean", ascending=False)
        return df
