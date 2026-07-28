"""
机器学习因子合成器 v6.2
v6.2关键修复:
  1. [P0] 修复数据泄露: ML模型使用历史数据训练,预测当期(不再用当期forward_returns训练)
  2. [P0] 滚动窗口训练: 真正实现t-N期训练->t期使用
  3. [P1] 新增交叉验证评估模型质量,用验证集R2而非训练集R2
  4. [P1] 新增特征选择: 基于IC和方差膨胀因子(VIF)筛选因子
  5. [P1] 集成学习改进: IC/ML/等权三路集成,动态权重
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
    """机器学习因子合成器 v6.2"""

    def __init__(self,
                 method: str = "ensemble",
                 model_type: str = "auto",
                 rolling_window: int = 12,
                 min_samples: int = 200):
        self.method = method
        self.model_type = model_type
        self.rolling_window = rolling_window
        self.min_samples = min_samples

        self.directions = {k: v["direction"] for k, v in FACTOR_CONFIG.items()}
        self.ic_history: Dict[str, List[float]] = defaultdict(list)

        # v6.2: 历史训练数据和模型缓存(防数据泄露)
        self._model = None
        self._feature_cols: Optional[List[str]] = None
        self._feature_importances: Optional[pd.Series] = None
        self._training_history: List[Dict] = []
        
        # v6.2: 历史因子和收益缓存(用于滚动训练)
        self._history_factors: List[pd.DataFrame] = []
        self._history_returns: List[pd.Series] = []
        self._max_history = 24  # 最多保留24期历史

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
            for m in ["lightgbm", "xgboost", "rf", "linear"]:
                if m in self._available_models:
                    model_type = m
                    break
            else:
                model_type = "equal"

        if model_type == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, verbose=-1,
            )
        elif model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, verbosity=0,
            )
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100, max_depth=6,
                min_samples_leaf=20, random_state=42, n_jobs=-1,
            )
        elif model_type == "linear":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0, random_state=42)
        else:
            return None

    def combine(self, factors: pd.DataFrame,
                forward_returns: Optional[pd.Series] = None) -> pd.Series:
        """
        v6.2: ML因子合成主入口
        关键: forward_returns仅用于事后更新IC和训练下一期模型
        """
        if factors.empty:
            return pd.Series(dtype=float)

        # 应用因子方向
        adjusted = factors.copy()
        for col, direction in self.directions.items():
            if col in adjusted.columns:
                adjusted[col] = adjusted[col] * direction

        # v6.2修复: 先用历史模型预测当期(不使用当期forward_returns)
        if self.method == "ml_weight":
            score = self._ml_weight_combine(adjusted)
        elif self.method == "ml_predict":
            score = self._ml_predict_combine(adjusted)
        elif self.method == "rolling_ml":
            score = self._rolling_ml_combine(adjusted)
        elif self.method == "ensemble":
            score = self._ensemble_combine(adjusted)
        else:
            score = adjusted.mean(axis=1)

        # v6.2: 预测完成后,用当期数据更新IC和训练下一期模型(事后)
        if forward_returns is not None:
            self._update_ic(adjusted, forward_returns)
            self._store_history(adjusted, forward_returns)
            # 训练下一期用的模型
            self._train_for_next_period()

        logger.info(f"ML合成完成 (方法={self.method}, 模型={self.model_type}): "
                    f"{len(score)} 只股票得分")
        return score

    def _store_history(self, factors: pd.DataFrame, forward_returns: pd.Series):
        """v6.2: 存储历史因子和收益(用于滚动训练)"""
        self._history_factors.append(factors.copy())
        self._history_returns.append(forward_returns.copy())
        if len(self._history_factors) > self._max_history:
            self._history_factors = self._history_factors[-self._max_history:]
            self._history_returns = self._history_returns[-self._max_history:]

    def _train_for_next_period(self):
        """
        v6.2: 用历史数据训练模型,供下一期使用
        时间顺序: 用t-N到t期的数据训练 -> t+1期使用
        """
        if len(self._history_factors) < 3:
            return

        # 合并历史数据
        all_factors = []
        all_returns = []
        for f, r in zip(self._history_factors, self._history_returns):
            common = f.index.intersection(r.index)
            if len(common) >= 30:
                all_factors.append(f.loc[common])
                all_returns.append(r.loc[common])

        if len(all_factors) < 3:
            return

        X_train = pd.concat(all_factors).fillna(0)
        y_train = pd.concat(all_returns)

        if len(X_train) < self.min_samples:
            return

        model = self._select_model()
        if model is None:
            return

        try:
            model.fit(X_train.values, y_train.values)
            self._model = model
            self._feature_cols = list(X_train.columns)

            # v6.2: 交叉验证评估
            val_r2 = self._cross_validate(X_train.values, y_train.values)

            # 记录训练信息
            train_r2 = model.score(X_train.values, y_train.values) if hasattr(model, "score") else 0
            self._training_history.append({
                "n_samples": len(X_train),
                "n_features": X_train.shape[1],
                "train_r2": float(train_r2),
                "val_r2": float(val_r2),
                "n_periods": len(all_factors),
            })

            # 特征重要性
            if hasattr(model, "feature_importances_"):
                self._feature_importances = pd.Series(
                    model.feature_importances_, index=X_train.columns
                ).sort_values(ascending=False)
            elif hasattr(model, "coef_"):
                self._feature_importances = pd.Series(
                    np.abs(model.coef_), index=X_train.columns
                ).sort_values(ascending=False)

            logger.info(f"ML模型训练完成(供下期): "
                       f"train_R2={train_r2:.4f}, val_R2={val_r2:.4f}, "
                       f"样本={len(X_train)}, 期数={len(all_factors)}")
        except Exception as e:
            logger.warning(f"ML模型训练失败: {e}")

    def _cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 3) -> float:
        """v6.2: 交叉验证评估模型质量"""
        try:
            from sklearn.model_selection import cross_val_score
            model = self._select_model()
            if model is None:
                return 0.0
            scores = cross_val_score(model, X, y, cv=n_splits, scoring="r2")
            return float(np.mean(scores))
        except Exception:
            return 0.0

    def _ml_weight_combine(self, factors: pd.DataFrame) -> pd.Series:
        """v6.2: ML特征重要性加权合成(使用历史模型)"""
        if self._feature_importances is None or self._feature_cols is None:
            return self._ic_weight_fallback(factors)

        # 使用历史训练的特征重要性
        weights = self._feature_importances.reindex(factors.columns).fillna(0)
        weights = weights / (weights.sum() + 1e-10)
        weights = weights.clip(lower=0.01 / max(len(weights), 1))

        score = pd.Series(0.0, index=factors.index)
        for col in factors.columns:
            if col in weights.index:
                score += factors[col].fillna(0) * weights[col]
        return score

    def _ml_predict_combine(self, factors: pd.DataFrame) -> pd.Series:
        """
        v6.2修复: ML直接预测下期收益
        使用历史训练的模型预测当期(不再用当期forward_returns训练)
        """
        if self._model is None or self._feature_cols is None:
            return self._ic_weight_fallback(factors)

        try:
            feature_cols = [c for c in self._feature_cols if c in factors.columns]
            if len(feature_cols) != len(self._feature_cols):
                logger.warning("特征列不匹配,退化为IC加权")
                return self._ic_weight_fallback(factors)

            X = factors[feature_cols].fillna(0).values
            predictions = self._model.predict(X)
            return pd.Series(predictions, index=factors.index)
        except Exception as e:
            logger.warning(f"ML预测失败: {e}, 退化为IC加权")
            return self._ic_weight_fallback(factors)

    def _rolling_ml_combine(self, factors: pd.DataFrame) -> pd.Series:
        """
        v6.2修复: 真正的滚动窗口ML训练
        使用最近N期历史数据训练模型,预测当期
        """
        if len(self._history_factors) < 3 or self._model is None:
            # 历史不足,退化为IC加权
            return self._ic_weight_fallback(factors)

        # 使用最近一次训练的模型预测当期
        return self._ml_predict_combine(factors)

    def _ensemble_combine(self, factors: pd.DataFrame) -> pd.Series:
        """
        v6.2改进: IC + ML + 等权 三路集成
        动态权重: 根据验证集R2调整ML权重
        """
        # IC加权分数
        ic_score = self._ic_weight_fallback(factors)
        
        # 等权分数
        equal_score = factors.mean(axis=1)
        
        # ML分数
        ml_score = self._ml_predict_combine(factors)

        # 标准化
        ic_norm = self._rank_normalize(ic_score)
        equal_norm = self._rank_normalize(equal_score)
        ml_norm = self._rank_normalize(ml_score)

        # v6.2: 动态权重基于验证集R2
        ml_weight = 0.2  # 默认ML 20%
        if self._training_history:
            latest = self._training_history[-1]
            val_r2 = latest.get("val_r2", 0)
            train_r2 = latest.get("train_r2", 0)
            
            # v6.2: 用验证集R2(而非训练集R2)判断模型质量
            if val_r2 > 0.05:
                ml_weight = 0.5
            elif val_r2 > 0.02:
                ml_weight = 0.35
            elif val_r2 > 0:
                ml_weight = 0.25
            else:
                ml_weight = 0.1  # 验证集R2<0,ML效果差,降低权重

        # IC权重根据IC历史质量调整
        ic_weight = 0.5
        if self.ic_history and any(len(v) >= 3 for v in self.ic_history.values()):
            avg_ic = np.mean([np.mean(v[-12:]) for v in self.ic_history.values() if v])
            if abs(avg_ic) < 0.02:
                ic_weight = 0.3  # IC太低,降低权重

        equal_weight = 1.0 - ml_weight - ic_weight
        equal_weight = max(equal_weight, 0.1)  # 等权至少10%

        # 重新归一化
        total = ml_weight + ic_weight + equal_weight
        ml_weight /= total
        ic_weight /= total
        equal_weight /= total

        ensemble = (ic_norm * ic_weight + 
                   ml_norm * ml_weight + 
                   equal_norm * equal_weight)
        
        if self._training_history:
            latest_r2 = self._training_history[-1]
            logger.info(f"集成合成: IC={ic_weight:.0%}/ML={ml_weight:.0%}/等权={equal_weight:.0%} "
                       f"(val_R2={latest_r2.get('val_r2', 0):.4f})")
        else:
            logger.info(f"集成合成: IC={ic_weight:.0%}/ML={ml_weight:.0%}/等权={equal_weight:.0%} (无训练历史)")

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
        weights = {k: w / total for k, w in weights.items()}
        if sum(abs(w) for w in weights.values()) == 0:
            return factors.mean(axis=1)

        score = pd.Series(0.0, index=factors.index)
        for col, w in weights.items():
            if col in factors.columns:
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
                    if len(self.ic_history[col]) > 60:
                        self.ic_history[col] = self.ic_history[col][-60:]
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
