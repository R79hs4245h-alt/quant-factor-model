"""
绩效分析: 年化收益、最大回撤、夏普比率、索提诺比率、胜率
可视化: 净值曲线、回撤曲线、月度收益热力图
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from config import OUTPUT_DIR
from utils import get_logger

logger = get_logger("performance")

# 无风险利率(年化),可调整
RISK_FREE_RATE = 0.025
TRADING_DAYS = 252


class PerformanceAnalyzer:
    """绩效分析器"""

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self.rf_daily = risk_free_rate / TRADING_DAYS

    def analyze(self, portfolio_returns: pd.Series,
                benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        计算全部绩效指标
        :return: 指标字典
        """
        if portfolio_returns is None or portfolio_returns.empty:
            return {}

        # 净值曲线
        nav = (1 + portfolio_returns).cumprod()
        bench_nav = None
        if benchmark_returns is not None and not benchmark_returns.empty:
            bench_nav = (1 + benchmark_returns).cumprod()

        # 基本指标
        n_days = len(portfolio_returns)
        total_return = nav.iloc[-1] - 1
        annual_return = (1 + total_return) ** (TRADING_DAYS / max(n_days, 1)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS)

        # 风险调整收益
        excess = portfolio_returns - self.rf_daily
        sharpe = excess.mean() / (portfolio_returns.std() + 1e-10) * np.sqrt(TRADING_DAYS)
        downside = portfolio_returns[portfolio_returns < 0]
        sortino = excess.mean() / (downside.std() + 1e-10) * np.sqrt(TRADING_DAYS)

        # 回撤
        peak = nav.cummax()
        drawdown = (nav - peak) / peak
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin()

        # 胜率与盈亏比
        win_days = (portfolio_returns > 0).sum()
        win_rate = win_days / n_days if n_days > 0 else 0
        avg_win = portfolio_returns[portfolio_returns > 0].mean() if win_days > 0 else 0
        avg_loss = portfolio_returns[portfolio_returns < 0].mean()
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        # 换手率(近似:从权重变化计算)
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf

        # 相对基准
        alpha = beta = None
        if benchmark_returns is not None and not benchmark_returns.empty:
            common = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common) > 30:
                from scipy import stats
                slope, intercept, r_value, _, _ = stats.linregress(
                    benchmark_returns.loc[common],
                    portfolio_returns.loc[common]
                )
                beta = slope
                alpha = intercept * TRADING_DAYS  # 年化 alpha

        result = {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "max_drawdown_date": max_dd_date,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "alpha": alpha,
            "beta": beta,
            "n_days": n_days,
            "nav": nav,
            "benchmark_nav": bench_nav,
            "drawdown": drawdown,
        }

        logger.info(f"绩效分析完成: 年化 {annual_return:.2%}, "
                    f"夏普 {sharpe:.2f}, 最大回撤 {max_drawdown:.2%}")
        return result

    def generate_report(self, metrics: Dict, save_path: Optional[Path] = None) -> str:
        """生成文本绩效报告"""
        lines = ["=" * 50, "多因子选股策略回测绩效报告", "=" * 50, ""]

        # 收益指标
        lines.append("【收益指标】")
        lines.append(f"  累计收益率:     {metrics['total_return']:.2%}")
        lines.append(f"  年化收益率:     {metrics['annual_return']:.2%}")
        lines.append(f"  交易日数:       {metrics['n_days']}")
        lines.append("")

        # 风险指标
        lines.append("【风险指标】")
        lines.append(f"  年化波动率:     {metrics['annual_volatility']:.2%}")
        lines.append(f"  最大回撤:       {metrics['max_drawdown']:.2%}")
        lines.append(f"  最大回撤日期:   {metrics.get('max_drawdown_date', '-')}")
        lines.append("")

        # 风险调整收益
        lines.append("【风险调整收益】")
        lines.append(f"  夏普比率:       {metrics['sharpe_ratio']:.3f}")
        lines.append(f"  索提诺比率:     {metrics['sortino_ratio']:.3f}")
        lines.append(f"  卡玛比率:       {metrics['calmar_ratio']:.3f}")
        lines.append("")

        # 基准对比
        if metrics.get('alpha') is not None:
            lines.append("【基准对比】")
            lines.append(f"  Alpha(年化):   {metrics['alpha']:.2%}")
            lines.append(f"  Beta:          {metrics['beta']:.3f}")
            lines.append("")

        # 交易统计
        lines.append("【交易统计】")
        lines.append(f"  日胜率:         {metrics['win_rate']:.2%}")
        lines.append(f"  盈亏比:         {metrics['profit_loss_ratio']:.2f}")
        lines.append("")

        report = "\n".join(lines)

        if save_path:
            save_path = Path(save_path)
            save_path.write_text(report, encoding="utf-8")
            logger.info(f"报告已保存: {save_path}")

        return report

    def plot_equity_curve(self, metrics: Dict, save_path: Optional[Path] = None):
        """绘制净值曲线与回撤"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # 中文字体
            plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei"]
            plt.rcParams["axes.unicode_minus"] = False

            fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

            # 净值曲线
            nav = metrics["nav"]
            axes[0].plot(nav.index, nav.values, label="组合净值", linewidth=1.5, color="#2563eb")
            if metrics.get("benchmark_nav") is not None:
                bench = metrics["benchmark_nav"]
                common_idx = nav.index.intersection(bench.index)
                axes[0].plot(common_idx, bench.loc[common_idx].values,
                            label="基准净值", linewidth=1, color="#94a3b8", alpha=0.7)
            axes[0].set_title("组合净值曲线 vs 基准", fontsize=13)
            axes[0].legend(loc="upper left")
            axes[0].grid(True, alpha=0.3)

            # 回撤曲线
            dd = metrics["drawdown"]
            axes[1].fill_between(dd.index, dd.values, 0, color="#e63946", alpha=0.4)
            axes[1].set_title("回撤曲线", fontsize=11)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            if save_path is None:
                save_path = OUTPUT_DIR / "equity_curve.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"净值图已保存: {save_path}")
        except Exception as e:
            logger.warning(f"绘图失败: {e}")

    def plot_monthly_returns(self, portfolio_returns: pd.Series,
                             save_path: Optional[Path] = None):
        """绘制月度收益热力图"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei"]
            plt.rcParams["axes.unicode_minus"] = False

            # 月度收益
            monthly = portfolio_returns.resample("M").apply(
                lambda x: (1 + x).prod() - 1
            )
            monthly_df = pd.DataFrame({
                "year": monthly.index.year,
                "month": monthly.index.month,
                "return": monthly.values
            })
            pivot = monthly_df.pivot(index="year", columns="month", values="return")

            fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.6)))
            im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                          vmin=-0.1, vmax=0.1)

            ax.set_xticks(range(12))
            ax.set_xticklabels([f"{m}月" for m in range(1, 13)])
            ax.set_yticks(range(len(pivot)))
            ax.set_yticklabels(pivot.index)
            ax.set_title("月度收益热力图", fontsize=13)

            # 标注数值
            for i in range(len(pivot)):
                for j in range(12):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                               fontsize=8, color="black")

            plt.colorbar(im, ax=ax, label="月度收益率")
            plt.tight_layout()
            if save_path is None:
                save_path = OUTPUT_DIR / "monthly_returns.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"月度收益图已保存: {save_path}")
        except Exception as e:
            logger.warning(f"月度图绘制失败: {e}")
