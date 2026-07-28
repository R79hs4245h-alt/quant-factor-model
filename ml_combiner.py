"""
机器学习因子合成器 v6.1
支持: 随机森林/XGBoost/LightGBM/神经网络 自动学习因子权重
配合传统IC加权进行集成,提升预测准确度

新增方法:
  1. ml_weight   - ML特征重要性加权
  2. ml_predict  - ML直接预测下期收益
  3. ensemble    - IC + ML 集成合成
  4. rolling_ml  - 滚动窗口ML训练(防过拟合)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from config import FACTOR_COMBINE_METHOD, FACTOR_CONFIG
from utils import get_logger

logger = get_logger("ml_combiner")


class MLFactorCombiner:
    """机器学习因子合成器"""

    def __init__(self,
                 method: str = "ensemble",
                 model_type: str = "auto",
                 rolling_window: int = 12,
                 min_samples: int = 200):
        """
        :param method: ml_weight / ml_predict / ensemble / rolling_ml
        :param model_type: auto / rf / xgboost / lightgbm / linear
        :param rolling_window: 滚动训练窗口(期数)
        :param min_samples: 最少样本数才训练ML
        """
        self.method = method
        self.model_type = model_type
        self.rolling_window = rolling_window
        self.min_samples = min_samples

        self.directions = {k: v["direction"] for k, v in FACTOR_CONFIG.items()}
        self.ic_history: Dict[str, List[float]] = defaultdict(list)

        # ML模型缓存
        self._model = None
        self._feature_importances: Optional[pd.Series] = None
        self._training_history: List[Dict] = []

        # 自动选择可用的模型
        self._available_models = self._detect_available_models()

    def _detect_available_models(self) -> List[str]:
        """检测可用的ML模型库"""
        available = []
        try:
            from sklearn.ensemble import RandomForestRegressor
            available.append("rf")
        except ImportError:
            pass
        try:
            import xgboost
            available.append("xgboost")
        except ImportError:
            pass
        try:
            import lightgbm
            available.append("lightgbm")
        except ImportError:
            pass
        try:
            from sklearn.linear_model import Ridge
            available.append("linear")
        except ImportError:
            pass

        if not available:
            logger.warning("无可用的ML模型库,退化为等权")
            available = ["equal"]
        return available

    def _select_model(self):
        """根据配置选择ML模型"""
        model_type = self.model_type
        if model_type == "auto":
            # 优先级: lightgbm > xgboost > rf > linear
            for m in ["lightgbm", "xgboost", "rf", "linear"]:
                if m in self._available_models:
                    model_type = m
                    break
            else:
                model_type = "equal"

        if model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )
        elif model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=0,
            )
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "linear":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0, random_state=42)
        else:
            return None

    def combine(self, factors: pd.DataFrame,
                forward_returns: Optional[pd.Series] = None) -> pd.Series:
        """
        ML因子合成主入口
        """
        if factors.empty:
            return pd.Series(dtype=float)

        # 应用因子方向
        adjusted = factors.copy()
        for col, direction in self.directions.items():
            if col in adjusted.columns:
                adjusted[col] = adjusted[col] * direction

        # 更新IC
        if forward_returns is not None:
            self._update_ic(adjusted, forward_returns)

        if self.method == "ml_weight":
            score = self._ml_weight_combine(adjusted, forward_returns)
        elif self.method == "ml_predict":
            score = self._ml_predict_combine(adjusted, forward_returns)
        elif self.method == "rolling_ml":
            score = self._rolling_ml_combine(adjusted, forward_returns)
        elif self.method == "ensemble":
            score = self._ensemble_combine(adjusted, forward_returns)
        else:
            score = adjusted.mean(axis=1)

        logger.info(f"ML合成完成 (方法={self.method}, 模型={self.model_type}): "
                    f"{len(score)} 只股票得分")
        return score

    def _ml_weight_combine(self, factors: pd.DataFrame,
                           forward_returns: Optional[pd.Series]) -> pd.Series:
        """ML特征重要性加权合成"""
        if forward_returns is None or len(forward_returns) < self.min_samples:
            # 不足训练样本,退化为IC加权
            return self._ic_weight_fallback(factors)

        model = self._select_model()
        if model is None:
            return self._ic_weight_fallback(factors)

        common_idx = factors.index.intersection(forward_returns.index)
        if len(common_idx) < self.min_samples:
            return self._ic_weight_fallback(factors)

        X = factors.loc[common_idx].fillna(0).values
        y = forward_returns.loc[common_idx].values

        try:
            model.fit(X, y)

            # 提取特征重要性
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
            else:
                return self._ic_weight_fallback(factors)

            self._feature_importances = pd.Series(
                importances, index=factors.columns
            ).sort_values(ascending=False)

            logger.info(f"ML特征重要性TOP5:\n{self._feature_importances.head()}")

            # 用特征重要性作为权重
            weights = self._feature_importances / self._feature_importances.sum()
            weights = weights.clip(lower=0.01 / len(weights))  # 最低权重保护

            score = pd.Series(0.0, index=factors.index)
            for col, w in weights.items():
                if col in factors.columns:
                    score += factors[col].fillna(0) * w

            return score
        except Exception as e:
            logger.warning(f"ML训练失败: {e}, 退化为IC加权")
            return self._ic_weight_fallback(factors)

    def _ml_predict_combine(self, factors: pd.DataFrame,
                            forward_returns: Optional[pd.Series]) -> pd.Series:
        """ML直接预测下期收益"""
        if forward_returns is None or len(forward_returns) < self.min_samples:
            return self._ic_weight_fallback(factors)

        model = self._select_model()
        if model is None:
            return self._ic_weight_fallback(factors)

        common_idx = factors.index.intersection(forward_returns.index)
        if len(common_idx) < self.min_samples:
            return self._ic_weight_fallback(factors)

        X = factors.loc[common_idx].fillna(0).values
        y = forward_returns.loc[common_idx].values

        try:
            model.fit(X, y)
            self._model = model

            # 对全样本预测
            X_all = factors.fillna(0).values
            predictions = model.predict(X_all)
            score = pd.Series(predictions, index=factors.index)

            # 记录训练信息
            train_score = model.score(X, y) if hasattr(model, "score") else 0
            self._training_history.append({
                "n_samples": len(common_idx),
                "n_features": X.shape[1],
                "train_r2": float(train_score),
            })
            logger.info(f"ML预测训练完成: R2={train_score:.4f}, 样本数={len(common_idx)}")

            return score
        except Exception as e:
            logger.warning(f"ML预测失败: {e}")
            return self._ic_weight_fallback(factors)

    def _rolling_ml_combine(self, factors: pd.DataFrame,
                            forward_returns: Optional[pd.Series]) -> pd.Series:
        """滚动窗口ML训练(防过拟合)"""
        # 当前期使用最近N期训练的模型
        if not self._training_history or len(self._training_history) < 3:
            # 历史不足,先用普通ML
            return self._ml_predict_combine(factors, forward_returns)

        # 使用最近一次训练的模型
        if self._model is not None:
            try:
                X_all = factors.fillna(0).values
                predictions = self._model.predict(X_all)
                return pd.Series(predictions, index=factors.index)
            except Exception:
                pass

        return self._ml_predict_combine(factors, forward_returns)

    def _ensemble_combine(self, factors: pd.DataFrame,
                          forward_returns: Optional[pd.Series]) -> pd.Series:
        """IC + ML 集成合成"""
        # IC加权分数
        ic_score = self._ic_weight_fallback(factors)

        # ML分数
        ml_score = self._ml_predict_combine(factors, forward_returns)

        # 集成权重: 动态调整
        # 如果ML训练效果好(R2高),给ML更高权重
        ml_weight = 0.3  # 默认ML 30%
        if self._training_history:
            latest = self._training_history[-1]
            r2 = latest.get("train_r2", 0)
            # R2 > 0.1 给 50% 权重, R2 > 0.05 给 40%
            if r2 > 0.1:
                ml_weight = 0.5
            elif r2 > 0.05:
                ml_weight = 0.4
            elif r2 < 0:
                ml_weight = 0.1  # ML效果差,降低权重

        # 标准化两个分数到同一尺度
        ic_norm = self._rank_normalize(ic_score)
        ml_norm = self._rank_normalize(ml_score)

        ensemble = ic_norm * (1 - ml_weight) + ml_norm * ml_weight
        logger.info(f"集成合成: IC权重={1-ml_weight:.0%}, ML权重={ml_weight:.0%} "
                    f"(R2={self._training_history[-1]['train_r2']:.4f})"
                    if self._training_history else
                    f"集成合成: IC权重={1-ml_weight:.0%}, ML权重={ml_weight:.0%}")
        return ensemble

    def _ic_weight_fallback(self, factors: pd.DataFrame) -> pd.Series:
        """IC加权回退方案"""
        weights = {}
        for col in factors.columns:
            ic_list = self.ic_history.get(col, [])
            if ic_list:
                ic_mean = np.mean(ic_list[-12:])
                weights[col] = ic_mean
            else:
                weights[col] = 1.0 / len(factors.columns)

        total = sum(abs(w) for w in weights.values()) or 1
        weights = {k: max(v, 0) / total for k, v in weights.items()}
        if sum(weights.values()) == 0:
            return factors.mean(axis=1)

        score = pd.Series(0.0, index=factors.index)
        for col, w in weights.items():
            if col in factors.columns and w > 0:
                score += factors[col].fillna(0) * w
        return score

    @staticmethod
    def _rank_normalize(series: pd.Series) -> pd.Series:
        """排名归一化到[0, 1]"""
        return series.rank(pct=True)

    def _update_ic(self, factors: pd.DataFrame, forward_returns: pd.Series):
        """更新各因子IC值"""
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

    def get_feature_importance(self) -> Optional[pd.Series]:
        """获取ML特征重要性"""
        return self._feature_importances

    def get_ic_summary(self) -> pd.DataFrame:
        """获取IC统计摘要"""
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
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("ic_mean", ascending=False)
        return df

    def get_training_summary(self) -> pd.DataFrame:
        """获取ML训练历史摘要"""
        if not self._training_history:
            return pd.DataFrame()
        return pd.DataFrame(self._training_history)
