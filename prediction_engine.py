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


# ============================================================
#  1. FactorMomentumTracker — 因子动量追踪器
# ============================================================
class FactorMomentumTracker:
    """
    因子动量追踪器 v7.1

    核心思想: 因子IC本身具有持续性(动量效应)，
    近期IC表现好的因子，下期大概率继续有效。

    功能:
      - 记录各因子IC的时间序列
      - 计算IC动量 (近M期IC均值 - 近N期IC均值)
      - 输出因子动量得分，用于动态调整因子权重

    使用方式:
        tracker = FactorMomentumTracker()
        tracker.update("ep", ic_value=0.05)
        momentum = tracker.get_factor_momentum("ep")
        weights = tracker.get_momentum_weights()
    """

    def __init__(self, max_history: int = 60,
                 short_window: int = 3,
                 long_window: int = 12):
        """
        初始化因子动量追踪器

        :param max_history: 最大保留IC历史期数
        :param short_window: 短期IC均值窗口
        :param long_window: 长期IC均值窗口
        """
        self.max_history = max_history
        self.short_window = short_window
        self.long_window = long_window
        self._ic_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    def update(self, factor_name: str, ic_value: float) -> None:
        """
        更新某个因子的IC值

        :param factor_name: 因子名称
        :param ic_value: 当期IC值
        """
        if not np.isnan(ic_value):
            self._ic_history[factor_name].append(ic_value)

    def update_batch(self, ic_dict: Dict[str, float]) -> None:
        """
        批量更新多个因子的IC值

        :param ic_dict: {因子名: IC值} 字典
        """
        for name, ic in ic_dict.items():
            self.update(name, ic)

    def get_factor_momentum(self, factor_name: str) -> float:
        """
        计算单个因子的IC动量

        IC动量 = mean(近short_window期IC) - mean(近long_window期IC)
        正值表示因子近期表现优于长期平均(动量向上)

        :param factor_name: 因子名称
        :return: IC动量值; 若历史不足返回0
        """
        history = list(self._ic_history.get(factor_name, []))
        if len(history) < self.short_window:
            return 0.0

        short_ic = np.mean(history[-self.short_window:])
        long_ic = np.mean(history[-min(self.long_window, len(history)):])

        return float(short_ic - long_ic)

    def get_momentum_scores(self) -> Dict[str, float]:
        """
        获取所有因子的IC动量得分

        :return: {因子名: 动量值} 字典
        """
        return {
            name: self.get_factor_momentum(name)
            for name in self._ic_history
        }

    def get_momentum_weights(self, min_weight: float = 0.05) -> Dict[str, float]:
        """
        基于IC动量计算因子权重

        策略:
          1. 计算各因子近期IC均值作为基础权重
          2. 用IC动量调整: 动量为正的因子增加权重
          3. 只保留正IC因子
          4. 归一化，设最低权重下限

        :param min_weight: 单因子最低权重
        :return: {因子名: 权重} 字典
        """
        weights = {}
        for name in self._ic_history:
            history = list(self._ic_history[name])
            if len(history) < self.short_window:
                continue

            # 近期IC均值
            recent_ic = np.mean(history[-self.short_window:])

            # IC动量调整
            momentum = self.get_factor_momentum(name)
            adjusted_ic = recent_ic + 0.5 * momentum  # 动量半权重

            if adjusted_ic > 0:
                weights[name] = adjusted_ic

        if not weights:
            return {}

        # 归一化
        total = sum(weights.values())
        if total <= 0:
            return {}

        weights = {k: v / total for k, v in weights.items()}

        # 最低权重保护
        n = len(weights)
        weights = {k: max(v, min_weight) for k, v in weights.items()}
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_ic_stability(self, factor_name: str) -> float:
        """
        计算因子IC的稳定性 (IC_IR = mean(IC) / std(IC))

        IC_IR越高，因子预测能力越稳定可靠。

        :param factor_name: 因子名称
        :return: IC信息比率; 若历史不足返回0
        """
        history = list(self._ic_history.get(factor_name, []))
        if len(history) < 5:
            return 0.0

        ic_mean = np.mean(history)
        ic_std = np.std(history, ddof=1)

        if ic_std < 1e-10:
            return 0.0

        return float(ic_mean / ic_std)

    def get_all_ic_stats(self) -> pd.DataFrame:
        """
        获取所有因子的IC统计信息

        :return: DataFrame, 每行一个因子, 列包含:
            ic_mean, ic_std, ic_ir, momentum, n_periods, positive_ratio
        """
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


# ============================================================
#  2. CrossSectionalRegressor — 截面回归预测器
# ============================================================
class CrossSectionalRegressor:
    """
    截面回归预测器 v7.1

    实现 Fama-MacBeth 风格的两步截面回归:
      Step 1: 对每期截面数据，用因子值回归收益率，得到因子收益
      Step 2: 对因子收益序列取均值，得到长期因子风险溢价
      Step 3: 用最新因子值 × 因子风险溢价 = 预测收益

    这种方法的优势:
      - 纯统计驱动，不依赖人工设定因子方向
      - 自动处理因子间的相关性
      - 可解释性强(每个因子有明确的收益贡献)

    使用方式:
        regressor = CrossSectionalRegressor()
        regressor.fit(period_factors_list, period_returns_list)
        predicted = regressor.predict(current_factors)
    """

    def __init__(self, min_periods: int = 6, max_periods: int = 24):
        """
        初始化截面回归预测器

        :param min_periods: 最少需要多少期数据才开始预测
        :param max_periods: 最多使用多少期历史数据(滚动窗口)
        """
        self.min_periods = min_periods
        self.max_periods = max_periods

        # 存储历史截面数据
        self._factor_history: List[pd.DataFrame] = []
        self._return_history: List[pd.Series] = []

        # 因子风险溢价 (时间序列均值)
        self._factor_premium: Optional[pd.Series] = None
        self._factor_premium_std: Optional[pd.Series] = None

        # 每期因子收益 (用于分析)
        self._period_factor_returns: List[pd.Series] = []

    def add_period(self, factors: pd.DataFrame, forward_returns: pd.Series) -> None:
        """
        添加一期的截面数据

        :param factors: 当期因子值 (index=code, columns=factors)
        :param forward_returns: 下期收益 (index=code)
        """
        # 对齐数据
        common = factors.index.intersection(forward_returns.dropna().index)
        if len(common) < 30:
            logger.debug(f"截面回归: 有效样本不足({len(common)}), 跳过")
            return

        self._factor_history.append(factors.loc[common].copy())
        self._return_history.append(forward_returns.loc[common].copy())

        # 滚动窗口
        if len(self._factor_history) > self.max_periods:
            self._factor_history = self._factor_history[-self.max_periods:]
            self._return_history = self._return_history[-self.max_periods:]

        # 重新计算因子溢价
        self._update_factor_premium()

    def _update_factor_premium(self) -> None:
        """
        更新因子风险溢价

        对每期截面做回归: return = alpha + beta_1*factor_1 + ... + beta_n*factor_n
        然后对各期beta取时间序列均值，得到因子风险溢价。
        """
        if len(self._factor_history) < self.min_periods:
            return

        period_betas = []

        for factors, returns in zip(self._factor_history, self._return_history):
            try:
                # 准备回归数据
                X = factors.fillna(0).values
                y = returns.values

                if X.shape[0] < X.shape[1] + 1:
                    continue  # 样本不足

                # 标准化因子 (截面z-score)
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_std = scaler.fit_transform(X)

                # OLS回归
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0, fit_intercept=True)
                model.fit(X_std, y)

                betas = pd.Series(
                    model.coef_, index=factors.columns
                )
                period_betas.append(betas)

            except Exception as e:
                logger.debug(f"截面回归异常: {e}")
                continue

        if len(period_betas) < self.min_periods:
            return

        # 因子溢价 = 各期beta的时间序列均值
        beta_df = pd.DataFrame(period_betas)
        self._factor_premium = beta_df.mean()
        self._factor_premium_std = beta_df.std()
        self._period_factor_returns = period_betas

        logger.info(
            f"截面回归: {len(period_betas)} 期, "
            f"因子溢价范围 [{self._factor_premium.min():.4f}, "
            f"{self._factor_premium.max():.4f}]"
        )

    def predict(self, current_factors: pd.DataFrame) -> pd.Series:
        """
        用因子风险溢价预测下期收益

        predicted_return = sum(factor_premium_i * standardized_factor_i)

        :param current_factors: 当期因子值 (index=code, columns=factors)
        :return: 预测收益 Series (index=code)
        """
        if self._factor_premium is None:
            logger.warning("截面回归: 因子溢价未计算, 返回中性预测")
            return pd.Series(0.0, index=current_factors.index)

        # 对齐因子列
        common_cols = current_factors.columns.intersection(
            self._factor_premium.index
        )
        if len(common_cols) == 0:
            return pd.Series(0.0, index=current_factors.index)

        X = current_factors[common_cols].fillna(0)

        # 截面标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        try:
            X_std = pd.DataFrame(
                scaler.fit_transform(X.values),
                index=X.index, columns=X.columns
            )
        except Exception:
            X_std = X

        # 预测收益 = 因子值 × 因子溢价
        premium = self._factor_premium.reindex(common_cols)
        predicted = X_std.multiply(premium, axis=1).sum(axis=1)

        return predicted

    def get_factor_premium(self) -> Optional[pd.Series]:
        """
        获取因子风险溢价

        :return: 因子溢价 Series; 若未计算返回None
        """
        return self._factor_premium

    def get_premium_significance(self) -> Optional[pd.DataFrame]:
        """
        获取因子溢价的显著性统计

        :return: DataFrame, 包含 premium, std, t_stat, p_value
        """
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
    def is_ready(self) -> bool:
        """是否已有足够数据做出预测"""
        return self._factor_premium is not None


# ============================================================
#  3. ICDecayAnalyzer — IC衰减分析器
# ============================================================
class ICDecayAnalyzer:
    """
    IC衰减分析器 v7.1

    分析因子IC随持有期的衰减速度:
      - IC衰减慢 -> 因子信号持久, 适合低频调仓
      - IC衰减快 -> 因子信号短暂, 需要高频调仓

    根据IC衰减速度为不同因子推荐最佳调仓频率。
    """

    def __init__(self, max_lag: int = 6):
        """
        初始化IC衰减分析器

        :param max_lag: 最大分析滞后期(月)
        """
        self.max_lag = max_lag
        self._lagged_ics: Dict[int, Dict[str, float]] = defaultdict(dict)

    def compute_decay(
        self,
        factor_panel: pd.DataFrame,
        return_panel: pd.DataFrame,
        lags: List[int] = None
    ) -> Dict[str, pd.Series]:
        """
        计算各因子在不同滞后期下的IC

        :param factor_panel: 因子面板 (index=日期, columns=因子)
                              每个元素是截面Series (index=code)
        :param return_panel: 收益面板 (index=日期, columns=收益)
                              每个元素是截面Series (index=code)
        :param lags: 滞后期列表(月), 默认[1,2,3,6]
        :return: {因子名: IC衰减Series(index=lag)} 字典
        """
        if lags is None:
            lags = [1, 2, 3, 6]

        dates = sorted(factor_panel.index.unique())
        if len(dates) < max(lags) + 2:
            logger.warning(f"IC衰减分析: 数据期数({len(dates)})不足")
            return {}

        decay_results = {}
        factor_names = factor_panel.columns

        for factor_name in factor_names:
            ic_by_lag = {}
            for lag in lags:
                ics = []
                for i in range(len(dates) - lag):
                    curr_date = dates[i]
                    future_date = dates[i + lag]

                    if curr_date not in factor_panel.index or \
                       future_date not in return_panel.index:
                        continue

                    fvals = factor_panel.loc[curr_date, factor_name]
                    if isinstance(fvals, pd.DataFrame):
                        fvals = fvals.iloc[0]
                    frets = return_panel.loc[future_date]
                    if isinstance(frets, pd.DataFrame):
                        frets = frets.iloc[0]

                    common = fvals.dropna().index.intersection(
                        frets.dropna().index
                    )
                    if len(common) < 10:
                        continue

                    ic, _ = stats.spearmanr(
                        fvals.loc[common], frets.loc[common]
                    )
                    if not np.isnan(ic):
                        ics.append(ic)

                ic_by_lag[lag] = float(np.mean(ics)) if ics else np.nan

            decay_results[factor_name] = pd.Series(
                ic_by_lag, name=factor_name
            )
            self._lagged_ics = {
                lag: {**self._lagged_ics.get(lag, {}), factor_name: ic_by_lag[lag]}
                for lag, ic_by_lag in [(lag, ic_by_lag) for lag in lags]
            }

        return decay_results

    def get_half_life(self, decay_series: pd.Series) -> float:
        """
        计算IC半衰期: IC从初始值衰减到一半所需的期数

        :param decay_series: IC衰减Series (index=lag, values=IC)
        :return: 半衰期(月); 若无法计算返回NaN
        """
        if len(decay_series) < 2:
            return np.nan

        initial_ic = decay_series.iloc[0]
        if initial_ic <= 0:
            return np.nan

        half_ic = initial_ic / 2

        for lag, ic in decay_series.items():
            if ic <= half_ic:
                # 线性插值
                prev_lag = decay_series.index[decay_series.values.tolist().index(
                    decay_series[decay_series > half_ic].iloc[-1]
                )] if (decay_series > half_ic).any() else 0
                prev_ic = decay_series.get(prev_lag, initial_ic)
                if prev_ic > half_ic and ic <= half_ic:
                    ratio = (prev_ic - half_ic) / (prev_ic - ic + 1e-10)
                    return float(prev_lag + ratio)

        # 如果IC始终未衰减到一半，返回最大lag
        return float(decay_series.index[-1])

    def recommend_rebalance_freq(self, decay_results: Dict[str, pd.Series]) -> Dict[str, str]:
        """
        根据IC衰减为各因子推荐调仓频率

        规则:
          - 半衰期 >= 4月 -> 月度调仓
          - 半衰期 2-4月 -> 双周调仓
          - 半衰期 < 2月 -> 周度调仓

        :param decay_results: compute_decay()的返回值
        :return: {因子名: 推荐频率} 字典
        """
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


# ============================================================
#  4. PredictionEngine — 统一预测引擎
# ============================================================
class PredictionEngine:
    """
    统一预测引擎 v7.1

    整合三种预测方法:
      1. 因子动量追踪 (FactorMomentumTracker)
      2. 截面回归预测 (CrossSectionalRegressor)
      3. IC衰减加权 (ICDecayAnalyzer)

    提供统一的预测接口，融合多源信号:
      final_score = w1 * momentum_score + w2 * regression_score + w3 * ic_weight_score

    使用方式:
        engine = PredictionEngine()
        # 每期更新
        engine.update(factors, forward_returns)
        # 预测
        predicted = engine.predict(current_factors)
    """

    def __init__(
        self,
        momentum_weight: float = 0.30,
        regression_weight: float = 0.40,
        ic_weight: float = 0.30,
        min_periods: int = 6,
    ):
        """
        初始化统一预测引擎

        :param momentum_weight: 因子动量得分权重
        :param regression_weight: 截面回归预测权重
        :param ic_weight: IC加权得分权重
        :param min_periods: 最少历史期数
        """
        self.momentum_weight = momentum_weight
        self.regression_weight = regression_weight
        self.ic_weight = ic_weight
        self.min_periods = min_periods

        # 子模块
        self.momentum_tracker = FactorMomentumTracker()
        self.regressor = CrossSectionalRegressor(
            min_periods=min_periods, max_periods=24
        )

        # IC历史 (与momentum_tracker共享)
        self._ic_history: Dict[str, List[float]] = defaultdict(list)

        # 因子方向
        self._factor_directions: Dict[str, int] = {}

        # 预测次数
        self._n_predictions = 0

        logger.info(
            f"PredictionEngine v7.1 初始化 | "
            f"动量权重={momentum_weight:.0%} | "
            f"回归权重={regression_weight:.0%} | "
            f"IC权重={ic_weight:.0%}"
        )

    def set_directions(self, directions: Dict[str, int]) -> None:
        """
        设置因子方向

        :param directions: {因子名: direction(1或-1)}
        """
        self._factor_directions = directions.copy()

    def update(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> None:
        """
        更新预测引擎(每期调用)

        1. 计算各因子IC并更新动量追踪器
        2. 添加截面数据到回归器
        3. 更新IC历史

        :param factors: 当期因子值
        :param forward_returns: 下期收益(事后)
        """
        if factors.empty or forward_returns is None or forward_returns.empty:
            return

        # 对齐
        common = factors.index.intersection(forward_returns.dropna().index)
        if len(common) < 30:
            return

        f_aligned = factors.loc[common]
        r_aligned = forward_returns.loc[common]

        # 计算各因子IC
        ic_dict = {}
        for col in f_aligned.columns:
            try:
                ic, _ = stats.spearmanr(
                    f_aligned[col].dropna(),
                    r_aligned.reindex(f_aligned[col].dropna().index).dropna()
                )
                if not np.isnan(ic):
                    ic_dict[col] = ic
                    self._ic_history[col].append(ic)
                    if len(self._ic_history[col]) > 60:
                        self._ic_history[col] = self._ic_history[col][-60:]
            except Exception:
                pass

        # 更新动量追踪器
        self.momentum_tracker.update_batch(ic_dict)

        # 更新截面回归器
        self.regressor.add_period(f_aligned, r_aligned)

    def predict(self, current_factors: pd.DataFrame) -> pd.Series:
        """
        生成综合预测得分

        融合三种预测信号:
          1. 因子动量加权得分
          2. 截面回归预测收益
          3. IC加权得分

        :param current_factors: 当期因子值
        :return: 综合得分 Series (index=code)
        """
        if current_factors.empty:
            return pd.Series(dtype=float)

        # 应用因子方向
        adjusted = current_factors.copy()
        for col, direction in self._factor_directions.items():
            if col in adjusted.columns:
                adjusted[col] = adjusted[col] * direction

        scores = []

        # 1. 因子动量加权得分
        momentum_weights = self.momentum_tracker.get_momentum_weights()
        if momentum_weights:
            available = [c for c in momentum_weights if c in adjusted.columns]
            if available:
                w = pd.Series(
                    {c: momentum_weights[c] for c in available}
                )
                w = w / w.sum()
                momentum_score = pd.Series(0.0, index=adjusted.index)
                for col in available:
                    momentum_score += adjusted[col].fillna(0.5).rank(pct=True) * w[col]
                scores.append(("momentum", momentum_score, self.momentum_weight))

        # 2. 截面回归预测
        if self.regressor.is_ready:
            reg_pred = self.regressor.predict(adjusted)
            if not reg_pred.empty:
                # 转为排名得分
                reg_score = reg_pred.rank(pct=True)
                scores.append(("regression", reg_score, self.regression_weight))

        # 3. IC加权得分
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

        # 融合
        if not scores:
            # 无历史数据，用等权排名
            logger.debug("预测引擎: 无足够历史, 使用等权排名")
            return adjusted.rank(pct=True).mean(axis=1)

        # 加权融合
        total_weight = sum(w for _, _, w in scores)
        if total_weight <= 0:
            return adjusted.rank(pct=True).mean(axis=1)

        final_score = pd.Series(0.0, index=adjusted.index)
        for name, score, weight in scores:
            aligned = score.reindex(adjusted.index).fillna(0.5)
            final_score += aligned * (weight / total_weight)

        self._n_predictions += 1
        logger.info(
            f"预测引擎: 融合 {len(scores)} 个信号 "
            f"({', '.join(n for n, _, _ in scores)}), "
            f"预测 {len(final_score)} 只股票"
        )

        return final_score

    def get_prediction_report(self) -> Dict:
        """
        生成预测引擎状态报告

        :return: 状态信息字典
        """
        ic_stats = self.momentum_tracker.get_all_ic_stats()

        report = {
            "n_predictions": self._n_predictions,
            "n_factors_tracked": len(self._ic_history),
            "momentum_tracker": {
                "n_factors": len(self.momentum_tracker._ic_history),
            },
            "regressor": {
                "is_ready": self.regressor.is_ready,
                "n_periods": len(self.regressor._factor_history),
            },
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
    def is_ready(self) -> bool:
        """是否有足够数据做出预测"""
        return (
            len(self._ic_history) >= 3 or
            self.regressor.is_ready or
            len(self.momentum_tracker._ic_history) >= 3
        )

    @property
    def n_updates(self) -> int:
        """已更新次数(用于动态调整融合权重)"""
        # 用IC历史中最长的序列长度作为更新次数的代理
        if not self._ic_history:
            return 0
        return max(len(v) for v in self._ic_history.values())


# ============================================================
#  模块自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("v7.1 预测引擎 - 模块自测")
    print("=" * 60)

    np.random.seed(42)

    # 生成模拟数据
    n_stocks = 100
    n_periods = 15
    factor_names = ["ep", "mom_20d", "vol_20d", "roe"]

    # 生成多期因子和收益
    all_factors = []
    all_returns = []
    for t in range(n_periods):
        codes = [f"S{str(i).zfill(4)}" for i in range(n_stocks)]
        factors = pd.DataFrame(
            np.random.randn(n_stocks, len(factor_names)),
            index=codes, columns=factor_names
        )
        # 收益与因子有相关性
        true_beta = np.array([0.3, 0.5, -0.2, 0.4])
        returns = pd.Series(
            factors.values @ true_beta + np.random.randn(n_stocks) * 0.05,
            index=codes
        )
        all_factors.append(factors)
        all_returns.append(returns)

    # 测试因子动量追踪器
    print("\n--- 因子动量追踪器测试 ---")
    tracker = FactorMomentumTracker(short_window=3, long_window=6)
    for factors, returns in zip(all_factors, all_returns):
        for col in factors.columns:
            ic, _ = stats.spearmanr(factors[col], returns)
            tracker.update(col, ic)

    momentum = tracker.get_momentum_scores()
    print(f"  因子动量: {momentum}")
    weights = tracker.get_momentum_weights()
    print(f"  动量权重: {weights}")
    print(f"  IC统计:\n{tracker.get_all_ic_stats()}")

    # 测试截面回归预测器
    print("\n--- 截面回归预测器测试 ---")
    regressor = CrossSectionalRegressor(min_periods=6)
    for factors, returns in zip(all_factors, all_returns):
        regressor.add_period(factors, returns)

    print(f"  就绪: {regressor.is_ready}")
    if regressor.is_ready:
        premium = regressor.get_factor_premium()
        print(f"  因子溢价: {premium.to_dict()}")
        significance = regressor.get_premium_significance()
        print(f"  显著性:\n{significance}")

        # 预测
        pred = regressor.predict(all_factors[-1])
        print(f"  预测范围: [{pred.min():.4f}, {pred.max():.4f}]")

    # 测试统一预测引擎
    print("\n--- 统一预测引擎测试 ---")
    engine = PredictionEngine()
    engine.set_directions({f: 1 for f in factor_names})

    for factors, returns in zip(all_factors[:-1], all_returns[:-1]):
        engine.update(factors, returns)

    print(f"  就绪: {engine.is_ready}")
    pred = engine.predict(all_factors[-1])
    print(f"  预测范围: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"  TOP5: {pred.nlargest(5).to_dict()}")

    report = engine.get_prediction_report()
    print(f"  预测次数: {report['n_predictions']}")
    print(f"  追踪因子数: {report['n_factors_tracked']}")

    print("\n" + "=" * 60)
    print("模块自测完成")
    print("=" * 60)
