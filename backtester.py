"""
回测引擎 v6.1: 按月调仓、组合收益计算、基准对比
v6.1新增: 交易成本(佣金+印花税+滑点)、涨跌停限制、ML合成器集成、市场状态自适应
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm

from config import (
    BACKTEST_START, BACKTEST_END, REBALANCE_FREQ, BENCHMARK,
    MIN_LIST_DAYS, FACTOR_COMBINE_METHOD
)
from utils import get_logger, get_rebalance_dates, get_trade_dates
from data_loader import get_panel_data, get_benchmark_data, get_stock_industry
from factors import FactorCalculator, get_factor_directions
from preprocessing import FactorPreprocessor
from factor_combiner import FactorCombiner
from selector import StockSelector

logger = get_logger("backtest")

# ============ 交易成本参数 ============
COMMISSION_RATE = 0.0003     # 佣金费率 万3
STAMP_TAX_RATE = 0.001       # 印花税 千1(卖出)
SLIPPAGE_RATE = 0.002        # 滑点 0.2%
MIN_COMMISSION = 5.0         # 最低佣金 5元


class TransactionCostModel:
    """交易成本模型 v6.1"""

    def __init__(self,
                 commission_rate: float = COMMISSION_RATE,
                 stamp_tax_rate: float = STAMP_TAX_RATE,
                 slippage: float = SLIPPAGE_RATE,
                 min_commission: float = MIN_COMMISSION):
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.min_commission = min_commission

    def calculate_cost(self, turnover: float, portfolio_value: float = 1e6) -> float:
        """
        计算交易成本(占组合价值比例)
        :param turnover: 换手率(0-1, 单边)
        :param portfolio_value: 组合总价值
        :return: 交易成本(绝对值)
        """
        traded_value = turnover * portfolio_value

        # 佣金(买卖都收)
        commission = max(traded_value * self.commission_rate, self.min_commission)

        # 印花税(仅卖出)
        stamp_tax = traded_value * 0.5 * self.stamp_tax_rate  # 换手率是双边的,卖出一半

        # 滑点
        slippage_cost = traded_value * self.slippage

        return commission + stamp_tax + slippage_cost

    def apply_to_returns(self, daily_return: float, turnover: float) -> float:
        """将交易成本应用到日收益"""
        cost_pct = turnover * (self.commission_rate + self.stamp_tax_rate * 0.5 + self.slippage)
        return daily_return - cost_pct


class Backtester:
    """多因子回测引擎 v6.1"""

    def __init__(self,
                 start_date: str = BACKTEST_START,
                 end_date: str = BACKTEST_END,
                 rebalance_freq: str = REBALANCE_FREQ,
                 benchmark: str = BENCHMARK,
                 use_ml: bool = False,
                 use_cost_model: bool = True):
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_freq = rebalance_freq
        self.benchmark = benchmark
        self.use_ml = use_ml
        self.use_cost_model = use_cost_model

        self.factor_calc = FactorCalculator()
        self.preprocessor = FactorPreprocessor()

        # v6.1: 可选ML合成器
        if use_ml:
            try:
                from ml_combiner import MLFactorCombiner
                self.combiner = MLFactorCombiner(method="ensemble", model_type="auto")
                logger.info("使用ML因子合成器(ensemble)")
            except Exception as e:
                logger.warning(f"ML合成器加载失败,退化为传统IC加权: {e}")
                self.combiner = FactorCombiner()
        else:
            self.combiner = FactorCombiner()

        self.selector = StockSelector()
        self.cost_model = TransactionCostModel() if use_cost_model else None

        # 结果存储
        self.weights_history: List[pd.Series] = []
        self.rebalance_dates: List[str] = []
        self.portfolio_returns: Optional[pd.Series] = None
        self.benchmark_returns: Optional[pd.Series] = None
        self.holdings_history: List[Dict] = []
        self.turnover_history: List[float] = []
        self.cost_history: List[float] = []

    def run(self, panel: pd.DataFrame,
            financial: pd.DataFrame,
            codes: List[str],
            industry: Optional[pd.Series] = None) -> Dict:
        """
        执行回测
        :param panel: 全部股票历史行情
        :param financial: 财务数据
        :param codes: 股票池
        :param industry: 行业归属
        :return: 回测结果字典
        """
        logger.info(f"开始回测: {self.start_date} ~ {self.end_date}")
        logger.info(f"股票池: {len(codes)} 只, 调仓频率: {self.rebalance_freq}")

        # 获取调仓日
        rebalance_dates = get_rebalance_dates(
            self.start_date, self.end_date, self.rebalance_freq
        )
        if not rebalance_dates:
            logger.error("无法获取调仓日,请检查日期范围")
            return {}

        logger.info(f"调仓日: {len(rebalance_dates)} 个")

        # 获取交易日
        trade_dates = get_trade_dates(self.start_date, self.end_date)
        logger.info(f"交易日: {len(trade_dates)} 个")

        # 逐期调仓
        prev_weights = None
        all_daily_returns = []

        for i, rb_date in enumerate(tqdm(rebalance_dates, desc="回测进度")):
            # 截面计算因子
            factors = self.factor_calc.compute_all_factors(
                panel, financial, rb_date
            )
            if factors.empty:
                continue

            # 预处理
            market_cap = panel[panel["date"] == pd.to_datetime(rb_date, format="%Y%m%d")]
            mc_series = market_cap.set_index("code")["close"] * 1e8 if not market_cap.empty else None
            processed = self.preprocessor.process(
                factors, industry=industry, market_cap=mc_series
            )

            # 计算 forward returns(用于IC)
            next_rb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
            forward_returns = self._compute_forward_returns(panel, rb_date, next_rb)

            # 因子合成
            scores = self.combiner.combine(processed, forward_returns)

            # 选股+风控
            weights = self.selector.select(scores, industry, prev_weights)

            # v6.1: 计算换手率和交易成本
            if prev_weights is not None and self.cost_model is not None:
                common = weights.index.intersection(prev_weights.index)
                turnover = float(
                    (weights.loc[common] - prev_weights.reindex(common, fill_value=0.0)).abs().sum() / 2
                )
                self.turnover_history.append(turnover)
            else:
                self.turnover_history.append(0.0)

            # 记录
            self.weights_history.append(weights)
            self.rebalance_dates.append(rb_date)
            self.holdings_history.append({
                "date": rb_date,
                "holdings": weights[weights > 0].to_dict(),
                "n_holdings": int((weights > 0).sum()),
                "turnover": self.turnover_history[-1],
            })

            prev_weights = weights

        # 计算每日组合收益
        self.portfolio_returns = self._compute_portfolio_returns(
            panel, rebalance_dates, trade_dates
        )

        # v6.1: 应用交易成本
        if self.cost_model is not None and self.portfolio_returns is not None:
            self._apply_transaction_costs(rebalance_dates)

        # 基准收益
        self.benchmark_returns = self._compute_benchmark_returns()

        # 输出统计
        if self.turnover_history:
            avg_turnover = np.mean(self.turnover_history)
            total_cost = sum(self.cost_history) if self.cost_history else 0
            logger.info(f"回测完成 | 平均换手率: {avg_turnover:.2%} | "
                       f"总交易成本: {total_cost:.2%}")

        return {
            "portfolio_returns": self.portfolio_returns,
            "benchmark_returns": self.benchmark_returns,
            "weights_history": self.weights_history,
            "holdings_history": self.holdings_history,
            "rebalance_dates": self.rebalance_dates,
            "turnover_history": self.turnover_history,
            "cost_history": self.cost_history,
        }

    def _apply_transaction_costs(self, rebalance_dates: List[str]):
        """将交易成本应用到组合收益"""
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            return

        for i, rb_date in enumerate(rebalance_dates[:-1]):
            if i >= len(self.turnover_history):
                break

            turnover = self.turnover_history[i]
            if turnover <= 0:
                self.cost_history.append(0.0)
                continue

            next_rb = rebalance_dates[i + 1]
            rb_dt = pd.to_datetime(rb_date, format="%Y%m%d")
            next_rb_dt = pd.to_datetime(next_rb, format="%Y%m%d")

            # 在调仓日扣除交易成本
            mask = (self.portfolio_returns.index > rb_dt) & (self.portfolio_returns.index <= next_rb_dt)
            if mask.any():
                cost_pct = turnover * (
                    self.cost_model.commission_rate
                    + self.cost_model.stamp_tax_rate * 0.5
                    + self.cost_model.slippage
                )
                # 在调仓后第一天扣除
                first_day = self.portfolio_returns.index[mask][0]
                self.portfolio_returns.loc[first_day] -= cost_pct
                self.cost_history.append(cost_pct)
            else:
                self.cost_history.append(0.0)

    def _compute_forward_returns(self, panel: pd.DataFrame,
                                  rb_date: str, next_rb: Optional[str]) -> Optional[pd.Series]:
        """计算前瞻收益(用于IC)"""
        if next_rb is None:
            return None
        try:
            td = pd.to_datetime(rb_date, format="%Y%m%d")
            ntd = pd.to_datetime(next_rb, format="%Y%m%d")
            mask = (panel["date"] > td) & (panel["date"] <= ntd)
            period = panel[mask]
            if period.empty:
                return None
            ret = period.groupby("code")["pct_chg"].apply(
                lambda x: (1 + x / 100).prod() - 1
            )
            return ret
        except Exception:
            return None

    def _compute_portfolio_returns(self, panel: pd.DataFrame,
                                    rebalance_dates: List[str],
                                    trade_dates: List[str]) -> pd.Series:
        """计算组合每日收益"""
        daily_returns = []

        for j in range(len(rebalance_dates) - 1):
            rb_date = rebalance_dates[j]
            next_rb = rebalance_dates[j + 1]
            weights = self.weights_history[j]

            # 该期间交易日
            td = pd.to_datetime(rb_date, format="%Y%m%d")
            ntd = pd.to_datetime(next_rb, format="%Y%m%d")
            period_dates = [d for d in trade_dates
                           if td < pd.to_datetime(d, format="%Y%m%d") <= ntd]

            for d in period_dates:
                d_dt = pd.to_datetime(d, format="%Y%m%d")
                day_data = panel[panel["date"] == d_dt]
                if day_data.empty:
                    daily_returns.append({"date": d, "return": 0.0})
                    continue
                day_ret = day_data.set_index("code")["pct_chg"] / 100
                common = weights.index.intersection(day_ret.index)
                if len(common) == 0:
                    daily_returns.append({"date": d, "return": 0.0})
                    continue
                port_ret = (weights.loc[common] * day_ret.loc[common]).sum()
                daily_returns.append({"date": d, "return": port_ret})

        df = pd.DataFrame(daily_returns)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df = df.set_index("date").sort_index()
            return df["return"]
        return pd.Series(dtype=float)

    def _compute_benchmark_returns(self) -> pd.Series:
        """获取基准收益"""
        bench = get_benchmark_data(
            self.benchmark,
            self.start_date.replace("-", ""),
            self.end_date.replace("-", "")
        )
        if bench.empty:
            return pd.Series(dtype=float)
        bench["date"] = pd.to_datetime(bench["date"])
        bench = bench.set_index("date").sort_index()
        bench_ret = bench["close"].pct_change().dropna()
        return bench_ret
