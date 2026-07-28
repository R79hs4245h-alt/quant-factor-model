"""
v7.1 预测引擎模块
==================

在 v7.0 因子体系基础上，引入三层预测增强:

  1. 因子动量 (Factor Momentum):
     - 跟踪各因子IC的时间序列
     - 用IC动量预测下期因子有效性
     - IC动量正 = 因子近期表现好，增加权重

  2. 截面回归预测 (Cross-Sectional Regression):
     - 每期用历史因子值对收益做截面回归
     - 用回归系数预测下期收益
     - 类似Fama-MacBeth回归

  3. IC衰减加权 (IC Decay Weighting):
     - 不同因子IC衰减速度不同
     - 衰减慢的因子赋予更高权重
     - 适配调仓频率

依赖: numpy, pandas, scipy
"""

import warnings
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy import stats

from utils import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("prediction_engine")


class FactorMomentumTracker:
    """因子动量追踪器 v7.1"""

    def __init__(self, max_history=60, short_window=3, long_window=12):
        self.max_history = max_history
        self.short_window = short_window
        self.long_window = long_window
        self._ic_history = defaultdict(lambda: deque(maxlen=max_history))

    def update(self, factor_name, ic_value):
        if not np.isnan(ic_value):
            self._ic_history[factor_name].append(ic_value)

    def update_batch(self, ic_dict):
        for name, ic in ic_dict.items():
            self.update(name, ic)

    def get_factor_momentum(self, factor_name):
        history = list(self._ic_history.get(factor_name, []))
        if len(history) < self.short_window:
            return 0.0
        short_ic = np.mean(history[-self.short_window:])
        long_ic = np.mean(history[-min(self.long_window, len(history)):])
        return float(short_ic - long_ic)

    def get_momentum_scores(self):
        return {name: self.get_factor_momentum(name) for name in self._ic_history}

    def get_momentum_weights(self, min_weight=0.05):
        weights = {}
        for name in self._ic_history:
            history = list(self._ic_history[name])
            if len(history) < self.short_window:
                continue
            recent_ic = np.mean(history[-self.short_window:])
            momentum = self.get_factor_momentum(name)
            adjusted_ic = recent_ic + 0.5 * momentum
            if adjusted_ic > 0:
                weights[name] = adjusted_ic
        if not weights:
            return {}
        total = sum(weights.values())
        if total <= 0:
            return {}
        weights = {k: v / total for k, v in weights.items()}
        weights = {k: max(v, min_weight) for k, v in weights.items()}
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        return weights

    def get_ic_stability(self, factor_name):
        history = list(self._ic_history.get(factor_name, []))
        if len(history) < 5:
            return 0.0
        ic_mean = np.mean(history)
        ic_std = np.std(history, ddof=1)
        if ic_std < 1e-10:
            return 0.0
        return float(ic_mean / ic_std)

    def get_all_ic_stats(self):
        rows = []
        for name in self._ic_history:
            history = list(self._ic_history[name])
            if len(history) < 2:
                continue
            ic_arr = np.array(history)
            rows.append({
                "factor": name,
                "ic_mean": float(np.mean(ic_arr)),
                "ic_std": float(np.std(ic_arr, ddof=1)),
                "ic_ir": float(np.mean(ic_arr) / (np.std(ic_arr, ddof=1) + 1e-10)),
                "momentum": self.get_factor_momentum(name),
                "n_periods": len(history),
                "positive_ratio": float(np.mean(ic_arr > 0)),
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("factor")
        return df.sort_values("ic_mean", ascending=False)


class CrossSectionalRegressor:
    """截面回归预测器 v7.1 - Fama-MacBeth风格"""

    def __init__(self, min_periods=6, max_periods=24):
        self.min_periods = min_periods
        self.max_periods = max_periods
        self._factor_history = []
        self._return_history = []
        self._factor_premium = None
        self._factor_premium_std = None
        self._period_factor_returns = []

    def add_period(self, factors, forward_returns):
        common = factors.index.intersection(forward_returns.dropna().index)
        if len(common) < 30:
            return
        self._factor_history.append(factors.loc[common].copy())
        self._return_history.append(forward_returns.loc[common].copy())
        if len(self._factor_history) > self.max_periods:
            self._factor_history = self._factor_history[-self.max_periods:]
            self._return_history = self._return_history[-self.max_periods:]
        self._update_factor_premium()

    def _update_factor_premium(self):
        if len(self._factor_history) < self.min_periods:
            return
        period_betas = []
        for factors, returns in zip(self._factor_history, self._return_history):
            try:
                X = factors.fillna(0).values
                y = returns.values
                if X.shape[0] < X.shape[1] + 1:
                    continue
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_std = scaler.fit_transform(X)
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0, fit_intercept=True)
                model.fit(X_std, y)
                betas = pd.Series(model.coef_, index=factors.columns)
                period_betas.append(betas)
            except Exception:
                continue
        if len(period_betas) < self.min_periods:
            return
        beta_df = pd.DataFrame(period_betas)
        self._factor_premium = beta_df.mean()
        self._factor_premium_std = beta_df.std()
        self._period_factor_returns = period_betas

    def predict(self, current_factors):
        if self._factor_premium is None:
            return pd.Series(0.0, index=current_factors.index)
        common_cols = current_factors.columns.intersection(self._factor_premium.index)
        if len(common_cols) == 0:
            return pd.Series(0.0, index=current_factors.index)
        X = current_factors[common_cols].fillna(0)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        try:
            X_std = pd.DataFrame(scaler.fit_transform(X.values), index=X.index, columns=X.columns)
        except Exception:
            X_std = X
        premium = self._factor_premium.reindex(common_cols)
        predicted = X_std.multiply(premium, axis=1).sum(axis=1)
        return predicted

    def get_factor_premium(self):
        return self._factor_premium

    def get_premium_significance(self):
        if self._factor_premium is None or self._factor_premium_std is None:
            return None
        n = len(self._period_factor_returns)
        if n < 2:
            return None
        t_stat = self._factor_premium / (self._factor_premium_std / np.sqrt(n))
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 1))
        df = pd.DataFrame({
            "premium": self._factor_premium,
            "std": self._factor_premium_std,
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.10,
        })
        return df.sort_values("premium", ascending=False)

    @property
    def is_ready(self):
        return self._factor_premium is not None


class ICDecayAnalyzer:
    """IC衰减分析器 v7.1"""

    def __init__(self, max_lag=6):
        self.max_lag = max_lag
        self._lagged_ics = defaultdict(dict)

    def compute_decay(self, factor_panel, return_panel, lags=None):
        if lags is None:
            lags = [1, 2, 3, 6]
        dates = sorted(factor_panel.index.unique())
        if len(dates) < max(lags) + 2:
            return {}
        decay_results = {}
        for factor_name in factor_panel.columns:
            ic_by_lag = {}
            for lag in lags:
                ics = []
                for i in range(len(dates) - lag):
                    curr_date = dates[i]
                    future_date = dates[i + lag]
                    if curr_date not in factor_panel.index or future_date not in return_panel.index:
                        continue
                    fvals = factor_panel.loc[curr_date, factor_name]
                    if isinstance(fvals, pd.DataFrame):
                        fvals = fvals.iloc[0]
                    frets = return_panel.loc[future_date]
                    if isinstance(frets, pd.DataFrame):
                        frets = frets.iloc[0]
                    common = fvals.dropna().index.intersection(frets.dropna().index)
                    if len(common) < 10:
                        continue
                    ic, _ = stats.spearmanr(fvals.loc[common], frets.loc[common])
                    if not np.isnan(ic):
                        ics.append(ic)
                ic_by_lag[lag] = float(np.mean(ics)) if ics else np.nan
            decay_results[factor_name] = pd.Series(ic_by_lag, name=factor_name)
        return decay_results

    def get_half_life(self, decay_series):
        if len(decay_series) < 2:
            return np.nan
        initial_ic = decay_series.iloc[0]
        if initial_ic <= 0:
            return np.nan
        half_ic = initial_ic / 2
        for lag, ic in decay_series.items():
            if ic <= half_ic:
                prev_lag = 0
                for l, v in decay_series.items():
                    if v > half_ic:
                        prev_lag = l
                prev_ic = decay_series.get(prev_lag, initial_ic)
                if prev_ic > half_ic and ic <= half_ic:
                    ratio = (prev_ic - half_ic) / (prev_ic - ic + 1e-10)
                    return float(prev_lag + ratio)
        return float(decay_series.index[-1])

    def recommend_rebalance_freq(self, decay_results):
        recommendations = {}
        for factor_name, decay in decay_results.items():
            half_life = self.get_half_life(decay)
            if np.isnan(half_life):
                recommendations[factor_name] = "月度"
            elif half_life >= 4:
                recommendations[factor_name] = "月度"
            elif half_life >= 2:
                recommendations[factor_name] = "双周"
            else:
                recommendations[factor_name] = "周度"
        return recommendations


class PredictionEngine:
    """统一预测引擎 v7.1 - 融合因子动量+截面回归+IC加权"""

    def __init__(self, momentum_weight=0.30, regression_weight=0.40, ic_weight=0.30, min_periods=6):
        self.momentum_weight = momentum_weight
        self.regression_weight = regression_weight
        self.ic_weight = ic_weight
        self.min_periods = min_periods
        self.momentum_tracker = FactorMomentumTracker()
        self.regressor = CrossSectionalRegressor(min_periods=min_periods, max_periods=24)
        self._ic_history = defaultdict(list)
        self._factor_directions = {}
        self._n_predictions = 0

    def set_directions(self, directions):
        self._factor_directions = directions.copy()

    def update(self, factors, forward_returns):
        if factors.empty or forward_returns is None or forward_returns.empty:
            return
        common = factors.index.intersection(forward_returns.dropna().index)
        if len(common) < 30:
            return
        f_aligned = factors.loc[common]
        r_aligned = forward_returns.loc[common]
        ic_dict = {}
        for col in f_aligned.columns:
            try:
                ic, _ = stats.spearmanr(f_aligned[col].dropna(), r_aligned.reindex(f_aligned[col].dropna().index).dropna())
                if not np.isnan(ic):
                    ic_dict[col] = ic
                    self._ic_history[col].append(ic)
                    if len(self._ic_history[col]) > 60:
                        self._ic_history[col] = self._ic_history[col][-60:]
            except Exception:
                pass
        self.momentum_tracker.update_batch(ic_dict)
        self.regressor.add_period(f_aligned, r_aligned)

    def predict(self, current_factors):
        if current_factors.empty:
            return pd.Series(dtype=float)
        adjusted = current_factors.copy()
        for col, direction in self._factor_directions.items():
            if col in adjusted.columns:
                adjusted[col] = adjusted[col] * direction
        scores = []
        momentum_weights = self.momentum_tracker.get_momentum_weights()
        if momentum_weights:
            available = [c for c in momentum_weights if c in adjusted.columns]
            if available:
                w = pd.Series({c: momentum_weights[c] for c in available})
                w = w / w.sum()
                momentum_score = pd.Series(0.0, index=adjusted.index)
                for col in available:
                    momentum_score += adjusted[col].fillna(0.5).rank(pct=True) * w[col]
                scores.append(("momentum", momentum_score, self.momentum_weight))
        if self.regressor.is_ready:
            reg_pred = self.regressor.predict(adjusted)
            if not reg_pred.empty:
                reg_score = reg_pred.rank(pct=True)
                scores.append(("regression", reg_score, self.regression_weight))
        if self._ic_history:
            ic_weights = {}
            for col in adjusted.columns:
                ic_list = self._ic_history.get(col, [])
                if ic_list:
                    ic_weights[col] = np.mean(ic_list[-12:])
            if ic_weights:
                total = sum(abs(v) for v in ic_weights.values()) or 1
                ic_weights = {k: v / total for k, v in ic_weights.items()}
                ic_score = pd.Series(0.0, index=adjusted.index)
                for col, w in ic_weights.items():
                    if col in adjusted.columns:
                        ic_score += adjusted[col].fillna(0.5).rank(pct=True) * w
                scores.append(("ic_weight", ic_score, self.ic_weight))
        if not scores:
            return adjusted.rank(pct=True).mean(axis=1)
        total_weight = sum(w for _, _, w in scores)
        if total_weight <= 0:
            return adjusted.rank(pct=True).mean(axis=1)
        final_score = pd.Series(0.0, index=adjusted.index)
        for name, score, weight in scores:
            aligned = score.reindex(adjusted.index).fillna(0.5)
            final_score += aligned * (weight / total_weight)
        self._n_predictions += 1
        return final_score

    def get_prediction_report(self):
        ic_stats = self.momentum_tracker.get_all_ic_stats()
        report = {
            "n_predictions": self._n_predictions,
            "n_factors_tracked": len(self._ic_history),
            "momentum_tracker": {"n_factors": len(self.momentum_tracker._ic_history)},
            "regressor": {"is_ready": self.regressor.is_ready, "n_periods": len(self.regressor._factor_history)},
            "ic_stats": ic_stats.to_dict("index") if not ic_stats.empty else {},
        }
        if self.regressor.is_ready:
            premium = self.regressor.get_factor_premium()
            if premium is not None:
                report["factor_premium"] = premium.to_dict()
            significance = self.regressor.get_premium_significance()
            if significance is not None:
                report["premium_significance"] = significance.to_dict("index")
        return report

    @property
    def is_ready(self):
        return len(self._ic_history) >= 3 or self.regressor.is_ready or len(self.momentum_tracker._ic_history) >= 3

    @property
    def n_updates(self):
        if not self._ic_history:
            return 0
        return max(len(v) for v in self._ic_history.values())
