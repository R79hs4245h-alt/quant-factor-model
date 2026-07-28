"""
performance_v7.py - v7.0 增强版绩效分析引擎

在原有 PerformanceAnalyzer 基础上，新增以下高级绩效指标:
  1.  Information Ratio (IR)             — Alpha / Tracking Error
  2.  Tracking Error                    — 超额收益年化标准差
  3.  Treynor Ratio                     — (年化收益 - 无风险利率) / Beta
  4.  Jensen's Alpha                     — CAPM 残差年化
  5.  Rolling Sharpe Ratio               — 12个月 / 36个月滚动窗口
  6.  Rolling Max Drawdown               — 滚动最大回撤序列
  7.  Bootstrap CI for Sharpe Ratio      — 1000次重采样, 95% 置信区间
  8.  Up/Down Market Capture Ratios       — 上涨/下跌市场捕获率
  9.  Hit Rate vs Benchmark              — 月度跑赢基准频率
 10.  Factor Attribution (Brinson)       — 多因子归因分析
 11.  Annual/Monthly Return Decomposition — 年度/月度收益分解表
 12.  Calmar Ratio (Improved)            — 使用滚动最大回撤
 13.  Omega Ratio                        — 累积分布收益比
 14.  Tail Ratio                         — 5%分位收益比

依赖: numpy, scipy, pandas
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path

from performance import PerformanceAnalyzer
from config import OUTPUT_DIR
from utils import get_logger

logger = get_logger("performance_v7")

# ──────────────────────── 常量 ────────────────────────
RISK_FREE_RATE = 0.025
TRADING_DAYS = 252


class EnhancedPerformanceAnalyzer(PerformanceAnalyzer):
    """
    v7.0 增强版绩效分析器

    继承 PerformanceAnalyzer 的全部功能，并扩展以下高级指标:
    - 信息比率 (IR) 与跟踪误差
    - Treynor 比率与 Jensen's Alpha
    - 滚动夏普比率 (12M / 36M)
    - 滚动最大回撤序列
    - Bootstrap 夏普比率置信区间
    - 上涨/下跌市场捕获率
    - 月度跑赢基准频率 (Hit Rate)
    - Brinson 多因子归因分析
    - 年度/月度收益分解表
    - 改进版 Calmar 比率 (滚动最大回撤)
    - Omega 比率
    - Tail 比率 (5% 分位)
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        """
        初始化增强版绩效分析器

        Parameters
        ----------
        risk_free_rate : float
            年化无风险利率，默认 2.5%
        """
        super().__init__(risk_free_rate=risk_free_rate)

    # ────────────────────────────────────────────────
    # 1. Information Ratio (IR)
    # ────────────────────────────────────────────────
    def information_ratio(self, portfolio_returns: pd.Series,
                          benchmark_returns: pd.Series) -> float:
        """
        计算信息比率 (Information Ratio)

        IR = Alpha / Tracking Error，衡量主动管理组合每承担一单位跟踪误差
        所获得的超额收益。

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        benchmark_returns : pd.Series
            基准日收益率序列

        Returns
        -------
        float
            信息比率
        """
        te = self.tracking_error(portfolio_returns, benchmark_returns)
        alpha = self._annualized_active_return(
            portfolio_returns, benchmark_returns
        )
        if te == 0 or np.isnan(te):
            return 0.0
        return alpha / te

    # ────────────────────────────────────────────────
    # 2. Tracking Error
    # ────────────────────────────────────────────────
    def tracking_error(self, portfolio_returns: pd.Series,
                       benchmark_returns: pd.Series) -> float:
        """
        计算跟踪误差 (Tracking Error)

        定义为组合收益率与基准收益率之差的年化标准差。
        TE = std(portfolio_returns - benchmark_returns) * sqrt(252)

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        benchmark_returns : pd.Series
            基准日收益率序列

        Returns
        -------
        float
            年化跟踪误差
        """
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) < 30:
            logger.warning("计算跟踪误差: 共同交易日不足30天，返回NaN")
            return np.nan
        active = portfolio_returns.loc[common] - benchmark_returns.loc[common]
        return active.std() * np.sqrt(TRADING_DAYS)

    def _annualized_active_return(self, portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> float:
        """
        计算年化主动收益 (超额收益)

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        benchmark_returns : pd.Series
            基准日收益率序列

        Returns
        -------
        float
            年化主动收益
        """
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) < 30:
            return 0.0
        active = portfolio_returns.loc[common] - benchmark_returns.loc[common]
        n_days = len(active)
        total_active = (1 + active).prod() - 1
        return (1 + total_active) ** (TRADING_DAYS / max(n_days, 1)) - 1

    # ────────────────────────────────────────────────
    # 3. Treynor Ratio
    # ────────────────────────────────────────────────
    def treynor_ratio(self, portfolio_returns: pd.Series,
                      benchmark_returns: pd.Series,
                      risk_free_rate: Optional[float] = None) -> float:
        """
        计算 Treynor 比率

        Treynor = (年化收益 - 无风险利率) / Beta
        衡量每单位系统性风险所获得的超额收益。

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        benchmark_returns : pd.Series
            基准日收益率序列
        risk_free_rate : float, optional
            无风险利率，默认使用初始化值

        Returns
        -------
        float
            Treynor 比率；若 Beta 为零或无法计算则返回 NaN
        """
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        beta = self._calc_beta(portfolio_returns, benchmark_returns)
        if beta is None or beta == 0:
            logger.warning("Treynor比率: Beta为0或无法计算，返回NaN")
            return np.nan

        n_days = len(portfolio_returns)
        nav = (1 + portfolio_returns).cumprod()
        total_return = nav.iloc[-1] - 1
        annual_return = (1 + total_return) ** (TRADING_DAYS / max(n_days, 1)) - 1

        return (annual_return - rf) / beta

    # ────────────────────────────────────────────────
    # 4. Jensen's Alpha
    # ────────────────────────────────────────────────
    def jensens_alpha(self, portfolio_returns: pd.Series,
                      benchmark_returns: pd.Series,
                      risk_free_rate: Optional[float] = None) -> float:
        """
        计算 Jensen's Alpha

        基于 CAPM 模型:
        Alpha = annual_return - (rf + beta * (bench_annual_return - rf))
        衡量经系统性风险调整后的超额收益。

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        benchmark_returns : pd.Series
            基准日收益率序列
        risk_free_rate : float, optional
            无风险利率，默认使用初始化值

        Returns
        -------
        float
            Jensen's Alpha (年化)；若 Beta 无法计算则返回 NaN
        """
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        beta = self._calc_beta(portfolio_returns, benchmark_returns)
        if beta is None:
            logger.warning("Jensen's Alpha: Beta无法计算，返回NaN")
            return np.nan

        n_days_p = len(portfolio_returns)
        nav_p = (1 + portfolio_returns).cumprod()
        annual_return = (nav_p.iloc[-1]) ** (TRADING_DAYS / max(n_days_p, 1)) - 1

        n_days_b = len(benchmark_returns)
        nav_b = (1 + benchmark_returns).cumprod()
        bench_annual_return = (nav_b.iloc[-1]) ** (TRADING_DAYS / max(n_days_b, 1)) - 1

        alpha = annual_return - (rf + beta * (bench_annual_return - rf))
        return alpha

    # ────────────────────────────────────────────────
    # 5. Rolling Sharpe Ratio
    # ────────────────────────────────────────────────
    def rolling_sharpe_ratio(self, portfolio_returns: pd.Series,
                             windows: Optional[List[int]] = None
                             ) -> Dict[str, pd.Series]:
        """
        计算滚动夏普比率

        使用 pandas rolling 窗口计算不同周期的滚动夏普比率。
        年化: mean(excess) / std * sqrt(252)

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        windows : list of int, optional
            窗口列表(交易日)，默认 [252, 756] (约12个月和36个月)

        Returns
        -------
        dict
            键为窗口描述(如 '12M', '36M')，值为对应的滚动夏普比率 Series
        """
        if windows is None:
            windows = [252, 756]

        excess = portfolio_returns - self.rf_daily
        result = {}

        for w in windows:
            if len(portfolio_returns) < w:
                logger.warning(
                    f"滚动夏普比率: 数据长度({len(portfolio_returns)}) "
                    f"不足窗口({w})，跳过"
                )
                continue

            roll_mean = excess.rolling(window=w, min_periods=w).mean()
            roll_std = portfolio_returns.rolling(window=w, min_periods=w).std()
            roll_sharpe = (roll_mean / (roll_std + 1e-10)) * np.sqrt(TRADING_DAYS)
            label = f"{w // 21}M" if w % 21 == 0 else f"{w}D"
            result[label] = roll_sharpe

        return result

    # ────────────────────────────────────────────────
    # 6. Rolling Max Drawdown
    # ────────────────────────────────────────────────
    def rolling_max_drawdown(self, portfolio_returns: pd.Series,
                             window: int = 252) -> pd.Series:
        """
        计算滚动最大回撤序列

        在给定的滚动窗口内计算累计最大回撤。
        滚动窗口默认为252个交易日(约一年)。

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        window : int
            滚动窗口大小(交易日数)，默认 252

        Returns
        -------
        pd.Series
            滚动最大回撤序列，值域为 [-1, 0]
        """
        nav = (1 + portfolio_returns).cumprod()

        def _max_dd_in_window(x):
            """在窗口内计算最大回撤"""
            peak = np.maximum.accumulate(x)
            drawdown = (x - peak) / peak
            return drawdown.min()

        rolling_dd = nav.rolling(window=window, min_periods=window).apply(
            _max_dd_in_window, raw=True
        )
        return rolling_dd

    # ────────────────────────────────────────────────
    # 7. Bootstrap Confidence Interval for Sharpe Ratio
    # ────────────────────────────────────────────────
    def bootstrap_sharpe_ci(self, portfolio_returns: pd.Series,
                            n_bootstrap: int = 1000,
                            confidence_level: float = 0.95,
                            random_seed: Optional[int] = None
                            ) -> Dict[str, float]:
        """
        通过 Bootstrap 重采样计算夏普比率的置信区间

        对日收益率进行有放回抽样，计算每次抽样的夏普比率，
        最终得到指定置信水平下的置信区间。

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        n_bootstrap : int
            Bootstrap 重采样次数，默认 1000
        confidence_level : float
            置信水平，默认 0.95
        random_seed : int, optional
            随机种子，用于可复现结果

        Returns
        -------
        dict
            包含以下键:
            - 'sharpe_original': 原始夏普比率
            - 'sharpe_mean': Bootstrap 均值
            - 'ci_lower': 置信区间下界
            - 'ci_upper': 置信区间上界
            - 'n_bootstrap': 重采样次数
        """
        returns = portfolio_returns.dropna().values
        n = len(returns)

        if n < 30:
            logger.warning("Bootstrap: 数据量不足30条，结果可能不可靠")

        excess = returns - self.rf_daily
        sharpe_original = np.mean(excess) / (np.std(returns) + 1e-10) * np.sqrt(TRADING_DAYS)

        rng = np.random.RandomState(random_seed)
        bootstrap_sharpes = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            sample_indices = rng.randint(0, n, size=n)
            sample = returns[sample_indices]
            sample_excess = sample - self.rf_daily
            bootstrap_sharpes[i] = (
                np.mean(sample_excess) / (np.std(sample) + 1e-10) * np.sqrt(TRADING_DAYS)
            )

        alpha_half = (1 - confidence_level) / 2
        ci_lower = np.percentile(bootstrap_sharpes, alpha_half * 100)
        ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha_half) * 100)

        result = {
            "sharpe_original": sharpe_original,
            "sharpe_mean": float(np.mean(bootstrap_sharpes)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "n_bootstrap": n_bootstrap,
        }

        logger.info(
            f"Bootstrap 夏普比率 CI({confidence_level:.0%}): "
            f"[{ci_lower:.3f}, {ci_upper:.3f}], 原始={sharpe_original:.3f}"
        )
        return result

    # ────────────────────────────────────────────────
    # 8. Up/Down Market Capture Ratios
    # ────────────────────────────────────────────────
    def market_capture_ratios(self, portfolio_returns: pd.Series,
                              benchmark_returns: pd.Series
                              ) -> Dict[str, float]:
        """
        计算上涨/下跌市场捕获率 (Up/Down Market Capture Ratios)

        上涨捕获率 = 组合在基准上涨期间的收益 / 基准同期收益
        下跌捕获率 = 组合在基准下跌期间的收益 / 基准同期收益

        - 上涨捕获率 > 100%: 组合在牛市中跑赢基准
        - 下跌捕获率 < 100%: 组合在熊市中跌幅小于基准
        - 理想组合: 上涨捕获率高，下跌捕获率低

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        benchmark_returns : pd.Series
            基准日收益率序列

        Returns
        -------
        dict
            包含以下键:
            - 'up_capture': 上涨市场捕获率
            - 'down_capture': 下跌市场捕获率
            - 'up_days': 上涨天数
            - 'down_days': 下跌天数
        """
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        p_ret = portfolio_returns.loc[common]
        b_ret = benchmark_returns.loc[common]

        up_mask = b_ret >= 0
        down_mask = b_ret < 0

        up_days = up_mask.sum()
        down_days = down_mask.sum()

        # 上涨捕获率
        if up_days > 0:
            up_capture = p_ret[up_mask].sum() / b_ret[up_mask].sum() if b_ret[up_mask].sum() != 0 else np.nan
        else:
            up_capture = np.nan

        # 下跌捕获率
        if down_days > 0:
            down_capture = p_ret[down_mask].sum() / b_ret[down_mask].sum() if b_ret[down_mask].sum() != 0 else np.nan
        else:
            down_capture = np.nan

        result = {
            "up_capture": float(up_capture),
            "down_capture": float(down_capture),
            "up_days": int(up_days),
            "down_days": int(down_days),
        }

        logger.info(
            f"市场捕获率: 上涨={up_capture:.2%}, 下跌={down_capture:.2%}"
        )
        return result

    # ────────────────────────────────────────────────
    # 9. Hit Rate vs Benchmark (月度跑赢基准频率)
    # ────────────────────────────────────────────────
    def hit_rate_vs_benchmark(self, portfolio_returns: pd.Series,
                              benchmark_returns: pd.Series
                              ) -> Dict[str, float]:
        """
        计算月度跑赢基准频率 (Hit Rate)

        将日收益率重采样为月度收益率，统计组合跑赢基准的月份占比。
        Hit Rate 越高，说明组合持续跑赢基准的能力越强。

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        benchmark_returns : pd.Series
            基准日收益率序列

        Returns
        -------
        dict
            包含以下键:
            - 'hit_rate': 月度跑赢频率 (0~1)
            - 'total_months': 总月数
            - 'win_months': 跑赢月数
            - 'lose_months': 未跑赢月数
        """
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        p_ret = portfolio_returns.loc[common]
        b_ret = benchmark_returns.loc[common]

        # 月度收益
        p_monthly = p_ret.resample("M").apply(lambda x: (1 + x).prod() - 1)
        b_monthly = b_ret.resample("M").apply(lambda x: (1 + x).prod() - 1)

        common_months = p_monthly.index.intersection(b_monthly.index)
        p_m = p_monthly.loc[common_months]
        b_m = b_monthly.loc[common_months]

        win = (p_m > b_m).sum()
        total = len(common_months)
        hit_rate = win / total if total > 0 else np.nan

        result = {
            "hit_rate": float(hit_rate),
            "total_months": int(total),
            "win_months": int(win),
            "lose_months": int(total - win),
        }

        logger.info(
            f"月度Hit Rate: {hit_rate:.2%} ({win}/{total}个月跑赢)"
        )
        return result

    # ────────────────────────────────────────────────
    # 10. Factor Attribution (Brinson 多因子归因)
    # ────────────────────────────────────────────────
    def factor_attribution(self,
                           portfolio_returns: pd.Series,
                           factor_exposures: pd.DataFrame,
                           factor_returns: pd.DataFrame) -> Dict[str, dict]:
        """
        Brinson 风格多因子归因分析

        将组合超额收益分解为各因子的贡献:
        contribution_i = exposure_i * factor_return_i
        总归因收益 = sum(contribution_i) + residual

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列，index 为日期
        factor_exposures : pd.DataFrame
            因子暴露矩阵，index 为日期，columns 为因子名称
            每行代表某个时点的因子暴露值(如持仓加权因子暴露)
        factor_returns : pd.DataFrame
            因子收益率序列，index 为日期，columns 为因子名称
            每行代表某日各因子的收益率

        Returns
        -------
        dict
            包含以下键:
            - 'factor_contributions': 各因子的累计贡献 (dict, 因子名 -> 累计贡献值)
            - 'total_factor_return': 因子解释的总收益
            - 'residual': 残差收益(因子无法解释的部分)
            - 'daily_contributions': 日频因子贡献 DataFrame
            - 'r_squared': 因子模型解释度 R^2
        """
        common_idx = factor_exposures.index.intersection(factor_returns.index)
        common_idx = common_idx.intersection(portfolio_returns.index)

        if len(common_idx) < 30:
            logger.warning("因子归因: 共同时点不足30，结果可能不可靠")

        exposures = factor_exposures.loc[common_idx]
        f_returns = factor_returns.loc[common_idx]
        p_ret = portfolio_returns.loc[common_idx]

        # 仅保留共同因子
        common_factors = exposures.columns.intersection(f_returns.columns)
        exposures = exposures[common_factors]
        f_returns = f_returns[common_factors]

        # 日频因子贡献: exposure * factor_return
        daily_contrib = exposures * f_returns

        # 累计因子贡献
        factor_contributions = {}
        for col in daily_contrib.columns:
            factor_contributions[col] = float(daily_contrib[col].sum())

        total_factor_return = float(daily_contrib.sum(axis=1).sum())
        total_portfolio_return = float(p_ret.sum())
        residual = total_portfolio_return - total_factor_return

        # R^2: 因子贡献 vs 实际收益的方差解释度
        predicted_daily = daily_contrib.sum(axis=1)
        ss_res = ((p_ret - predicted_daily) ** 2).sum()
        ss_tot = ((p_ret - p_ret.mean()) ** 2).sum()
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        result = {
            "factor_contributions": factor_contributions,
            "total_factor_return": total_factor_return,
            "residual": float(residual),
            "daily_contributions": daily_contrib,
            "r_squared": float(r_squared),
        }

        logger.info(
            f"因子归因: R^2={r_squared:.4f}, "
            f"因子解释收益={total_factor_return:.4%}, "
            f"残差={residual:.4%}"
        )
        return result

    # ────────────────────────────────────────────────
    # 11. Annual/Monthly Return Decomposition
    # ────────────────────────────────────────────────
    def return_decomposition(self, portfolio_returns: pd.Series,
                            benchmark_returns: Optional[pd.Series] = None
                            ) -> Dict[str, Union[pd.DataFrame, dict]]:
        """
        生成年度/月度收益分解表

        输出一个二维表格，行=年份，列=月份，值为该月收益率。
        附加年度汇总行(累计年收益)和列汇总(月度平均收益)。
        如果提供基准，同时生成基准和超额收益分解表。

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        benchmark_returns : pd.Series, optional
            基准日收益率序列

        Returns
        -------
        dict
            包含以下键:
            - 'portfolio_pivot': 组合月度收益透视表 (DataFrame)
            - 'yearly_summary': 年度收益汇总 Series
            - 'benchmark_pivot': 基准月度收益透视表 (若有基准, DataFrame)
            - 'excess_pivot': 超额收益透视表 (若有基准, DataFrame)
        """
        monthly = portfolio_returns.resample("M").apply(
            lambda x: (1 + x).prod() - 1
        )

        # 透视表
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
            "date": monthly.index
        })
        pivot = monthly_df.pivot(index="year", columns="month", values="return")

        # 年度汇总
        yearly_summary = monthly_df.groupby("year")["return"].apply(
            lambda x: (1 + x).prod() - 1
        )

        result = {
            "portfolio_pivot": pivot,
            "yearly_summary": yearly_summary,
        }

        if benchmark_returns is not None and not benchmark_returns.empty:
            bench_monthly = benchmark_returns.resample("M").apply(
                lambda x: (1 + x).prod() - 1
            )
            bench_df = pd.DataFrame({
                "year": bench_monthly.index.year,
                "month": bench_monthly.index.month,
                "return": bench_monthly.values,
            })
            bench_pivot = bench_df.pivot(index="year", columns="month", values="return")
            result["benchmark_pivot"] = bench_pivot

            # 超额收益透视表
            excess_pivot = pivot - bench_pivot
            result["excess_pivot"] = excess_pivot

        logger.info(f"收益分解表: 覆盖{len(pivot)}个年度")
        return result

    # ────────────────────────────────────────────────
    # 12. Calmar Ratio (Improved — 滚动最大回撤)
    # ────────────────────────────────────────────────
    def calmar_ratio_rolling(self, portfolio_returns: pd.Series,
                              window: int = 252) -> Dict[str, Union[float, pd.Series]]:
        """
        改进版 Calmar 比率 — 使用滚动最大回撤

        原版 Calmar 仅使用历史最大回撤，改进版在滚动窗口内计算:
        - 滚动窗口年化收益 / 滚动窗口最大回撤
        - 最终值取最近一个窗口的结果

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        window : int
            滚动窗口大小(交易日数)，默认 252 (约一年)

        Returns
        -------
        dict
            包含以下键:
            - 'calmar_rolling_latest': 最近窗口的滚动 Calmar 比率
            - 'calmar_original': 原版 Calmar 比率
            - 'rolling_calmar_series': 滚动 Calmar 比率序列
            - 'rolling_max_dd_series': 滚动最大回撤序列
        """
        nav = (1 + portfolio_returns).cumprod()

        # 滚动窗口年化收益
        rolling_nav_end = nav.rolling(window=window, min_periods=window).apply(lambda x: x[-1], raw=True)
        rolling_nav_start = nav.shift(window)
        rolling_annual_ret = (rolling_nav_end / rolling_nav_start) ** (TRADING_DAYS / window) - 1

        # 滚动最大回撤
        rolling_max_dd = self.rolling_max_drawdown(portfolio_returns, window=window)

        # 滚动 Calmar
        rolling_calmar = rolling_annual_ret / (rolling_max_dd.abs() + 1e-10)

        calmar_original = super().analyze(portfolio_returns).get("calmar_ratio", np.nan)

        latest_calmar = float(rolling_calmar.dropna().iloc[-1]) if not rolling_calmar.dropna().empty else np.nan

        result = {
            "calmar_rolling_latest": latest_calmar,
            "calmar_original": float(calmar_original),
            "rolling_calmar_series": rolling_calmar,
            "rolling_max_dd_series": rolling_max_dd,
        }

        logger.info(
            f"改进版Calmar(滚动{window}日): {latest_calmar:.3f}, "
            f"原版: {calmar_original:.3f}"
        )
        return result

    # ────────────────────────────────────────────────
    # 13. Omega Ratio
    # ────────────────────────────────────────────────
    def omega_ratio(self, portfolio_returns: pd.Series,
                   threshold: Optional[float] = None
                   ) -> float:
        """
        计算 Omega 比率

        Omega = sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))
        即收益高于阈值的累积量与低于阈值的累积量之比。
        阈值默认使用日化无风险利率。

        Omega > 1: 收益高于阈值的总量大于亏损
        Omega < 1: 亏损总量更大

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        threshold : float, optional
            收益阈值(日化)，默认使用日化无风险利率

        Returns
        -------
        float
            Omega 比率
        """
        if threshold is None:
            threshold = self.rf_daily

        gains = portfolio_returns - threshold
        sum_positive = gains[gains > 0].sum()
        sum_negative = -gains[gains < 0].sum()

        if sum_negative == 0:
            return np.inf if sum_positive > 0 else 1.0

        omega = sum_positive / sum_negative
        logger.info(f"Omega 比率: {omega:.3f} (阈值={threshold:.6f})")
        return float(omega)

    # ────────────────────────────────────────────────
    # 14. Tail Ratio (5% 分位收益比)
    # ────────────────────────────────────────────────
    def tail_ratio(self, portfolio_returns: pd.Series,
                   lower_pct: float = 0.05,
                   upper_pct: float = 0.95) -> float:
        """
        计算 Tail 比率 (尾部收益比)

        Tail Ratio = abs(quantile(upper_pct)) / abs(quantile(lower_pct))
        衡量收益分布右尾(好尾)与左尾(坏尾)的厚度比。
        - Tail > 1: 右尾更厚，极端正收益的概率更高
        - Tail < 1: 左尾更厚，极端负收益的风险更大

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        lower_pct : float
            下分位数，默认 0.05 (5%)
        upper_pct : float
            上分位数，默认 0.95 (95%)

        Returns
        -------
        float
            Tail 比率
        """
        lower = portfolio_returns.quantile(lower_pct)
        upper = portfolio_returns.quantile(upper_pct)

        if lower == 0:
            return np.inf if upper != 0 else 1.0

        tail = abs(upper) / abs(lower)

        logger.info(
            f"Tail 比率 ({lower_pct*100:.0f}%/{upper_pct*100:.0f}%): "
            f"{tail:.3f} (上尾={upper:.4%}, 下尾={lower:.4%})"
        )
        return float(tail)

    # ────────────────────────────────────────────────
    # 综合分析: enhanced_analyze()
    # ────────────────────────────────────────────────
    def enhanced_analyze(self, portfolio_returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         factor_exposures: Optional[pd.DataFrame] = None,
                         factor_returns: Optional[pd.DataFrame] = None,
                         bootstrap: bool = True,
                         n_bootstrap: int = 1000,
                         confidence_level: float = 0.95,
                         random_seed: Optional[int] = None
                         ) -> Dict:
        """
        增强版综合绩效分析

        在原有 PerformanceAnalyzer.analyze() 的基础上，计算全部14项高级指标，
        并整合为一个统一的字典返回。

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        benchmark_returns : pd.Series, optional
            基准日收益率序列
        factor_exposures : pd.DataFrame, optional
            因子暴露矩阵 (用于因子归因)
        factor_returns : pd.DataFrame, optional
            因子收益率序列 (用于因子归因)
        bootstrap : bool
            是否执行 Bootstrap 夏普比率置信区间，默认 True
        n_bootstrap : int
            Bootstrap 重采样次数，默认 1000
        confidence_level : float
            Bootstrap 置信水平，默认 0.95
        random_seed : int, optional
            随机种子

        Returns
        -------
        dict
            包含原有指标 + 全部高级指标的字典
        """
        # 先执行父类分析
        metrics = super().analyze(portfolio_returns, benchmark_returns)

        # ── 仅需要组合收益的指标 ──
        # 5. 滚动夏普比率
        metrics["rolling_sharpe"] = self.rolling_sharpe_ratio(portfolio_returns)

        # 6. 滚动最大回撤
        metrics["rolling_max_drawdown"] = self.rolling_max_drawdown(portfolio_returns)

        # 7. Bootstrap CI
        if bootstrap:
            metrics["bootstrap_sharpe_ci"] = self.bootstrap_sharpe_ci(
                portfolio_returns,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_seed=random_seed,
            )

        # 12. 改进版 Calmar
        metrics["calmar_rolling"] = self.calmar_ratio_rolling(portfolio_returns)

        # 13. Omega 比率
        metrics["omega_ratio"] = self.omega_ratio(portfolio_returns)

        # 14. Tail 比率
        metrics["tail_ratio"] = self.tail_ratio(portfolio_returns)

        # 11. 收益分解表
        metrics["return_decomposition"] = self.return_decomposition(
            portfolio_returns, benchmark_returns
        )

        # ── 需要基准的指标 ──
        if benchmark_returns is not None and not benchmark_returns.empty:
            common = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common) >= 30:
                # 1. 信息比率
                metrics["information_ratio"] = self.information_ratio(
                    portfolio_returns, benchmark_returns
                )

                # 2. 跟踪误差
                metrics["tracking_error"] = self.tracking_error(
                    portfolio_returns, benchmark_returns
                )

                # 3. Treynor 比率
                metrics["treynor_ratio"] = self.treynor_ratio(
                    portfolio_returns, benchmark_returns
                )

                # 4. Jensen's Alpha
                metrics["jensens_alpha"] = self.jensens_alpha(
                    portfolio_returns, benchmark_returns
                )

                # 8. 市场捕获率
                metrics["market_capture"] = self.market_capture_ratios(
                    portfolio_returns, benchmark_returns
                )

                # 9. Hit Rate
                metrics["hit_rate"] = self.hit_rate_vs_benchmark(
                    portfolio_returns, benchmark_returns
                )

        # ── 因子归因 (需要因子暴露和因子收益) ──
        if (factor_exposures is not None and not factor_exposures.empty
                and factor_returns is not None and not factor_returns.empty):
            metrics["factor_attribution"] = self.factor_attribution(
                portfolio_returns, factor_exposures, factor_returns
            )

        logger.info("增强版综合绩效分析完成")
        return metrics

    # ────────────────────────────────────────────────
    # 增强版报告生成
    # ────────────────────────────────────────────────
    def generate_enhanced_report(self, metrics: Dict,
                                 save_path: Optional[Path] = None) -> str:
        """
        生成增强版文本绩效报告

        在原有报告基础上，增加全部高级指标的展示。

        Parameters
        ----------
        metrics : dict
            enhanced_analyze() 返回的指标字典
        save_path : Path, optional
            报告保存路径

        Returns
        -------
        str
            文本格式的绩效报告
        """
        # 先调用父类报告
        report = super().generate_report(metrics, save_path=None)

        lines = [report, "", "=" * 50, "v7.0 增强版指标", "=" * 50, ""]

        # 跟踪误差与信息比率
        if "tracking_error" in metrics:
            lines.append("【跟踪误差与信息比率】")
            lines.append(f"  跟踪误差(年化): {metrics['tracking_error']:.4%}")
            lines.append(f"  信息比率 (IR):  {metrics['information_ratio']:.4f}")
            lines.append("")

        # Treynor 与 Jensen's Alpha
        if "treynor_ratio" in metrics:
            lines.append("【CAPM 风险调整收益】")
            lines.append(f"  Treynor 比率:   {metrics['treynor_ratio']:.4f}")
            lines.append(f"  Jensen's Alpha: {metrics['jensens_alpha']:.4%}")
            lines.append("")

        # 改进版 Calmar
        if "calmar_rolling" in metrics:
            cr = metrics["calmar_rolling"]
            lines.append("【改进版 Calmar 比率】")
            lines.append(f"  滚动Calmar(最新): {cr['calmar_rolling_latest']:.3f}")
            lines.append(f"  原版Calmar:       {cr['calmar_original']:.3f}")
            lines.append("")

        # Omega 与 Tail
        lines.append("【收益分布指标】")
        if "omega_ratio" in metrics:
            lines.append(f"  Omega 比率:     {metrics['omega_ratio']:.4f}")
        if "tail_ratio" in metrics:
            lines.append(f"  Tail 比率 (5%):  {metrics['tail_ratio']:.4f}")
        lines.append("")

        # 市场捕获率
        if "market_capture" in metrics:
            mc = metrics["market_capture"]
            lines.append("【市场捕获率】")
            lines.append(f"  上涨捕获率: {mc['up_capture']:.2%} "
                         f"(共{mc['up_days']}个上涨日)")
            lines.append(f"  下跌捕获率: {mc['down_capture']:.2%} "
                         f"(共{mc['down_days']}个下跌日)")
            lines.append("")

        # Hit Rate
        if "hit_rate" in metrics:
            hr = metrics["hit_rate"]
            lines.append("【月度跑赢基准】")
            lines.append(f"  Hit Rate: {hr['hit_rate']:.2%} "
                         f"({hr['win_months']}/{hr['total_months']}个月)")
            lines.append("")

        # Bootstrap CI
        if "bootstrap_sharpe_ci" in metrics:
            bs = metrics["bootstrap_sharpe_ci"]
            lines.append("【Bootstrap 夏普比率置信区间】")
            lines.append(f"  原始夏普比率:    {bs['sharpe_original']:.4f}")
            lines.append(f"  Bootstrap 均值:  {bs['sharpe_mean']:.4f}")
            lines.append(f"  95% 置信区间:    [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")
            lines.append(f"  重采样次数:      {bs['n_bootstrap']}")
            lines.append("")

        # 收益分解表摘要
        if "return_decomposition" in metrics:
            rd = metrics["return_decomposition"]
            ys = rd["yearly_summary"]
            lines.append("【年度收益汇总】")
            for year, ret in ys.items():
                lines.append(f"  {int(year)}: {ret:.2%}")
            lines.append("")

        # 因子归因摘要
        if "factor_attribution" in metrics:
            fa = metrics["factor_attribution"]
            lines.append("【因子归因摘要】")
            lines.append(f"  因子模型 R^2:   {fa['r_squared']:.4f}")
            lines.append(f"  因子解释收益:   {fa['total_factor_return']:.4%}")
            lines.append(f"  残差收益:       {fa['residual']:.4%}")
            lines.append("  各因子贡献:")
            for factor_name, contrib in fa["factor_contributions"].items():
                lines.append(f"    {factor_name}: {contrib:.4%}")
            lines.append("")

        final_report = "\n".join(lines)

        if save_path:
            save_path = Path(save_path)
            save_path.write_text(final_report, encoding="utf-8")
            logger.info(f"增强版报告已保存: {save_path}")

        return final_report

    # ────────────────────────────────────────────────
    # 内部辅助方法
    # ────────────────────────────────────────────────
    def _calc_beta(self, portfolio_returns: pd.Series,
                   benchmark_returns: pd.Series) -> Optional[float]:
        """
        计算组合相对基准的 Beta 系数

        Parameters
        ----------
        portfolio_returns : pd.Series
            组合日收益率序列
        benchmark_returns : pd.Series
            基准日收益率序列

        Returns
        -------
        float or None
            Beta 系数；若共同交易日不足30则返回 None
        """
        from scipy import stats

        common = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common) < 30:
            logger.warning("_calc_beta: 共同交易日不足30")
            return None

        slope, _, _, _, _ = stats.linregress(
            benchmark_returns.loc[common],
            portfolio_returns.loc[common]
        )
        return float(slope)
