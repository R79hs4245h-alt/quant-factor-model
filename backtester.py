"""
回测引擎: 按月调仓、组合收益计算、基准对比
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm

from config import (
    BACKTEST_START, BACKTEST_END, REBALANCE_FREQ, BENCHMARK,
    MIN_LIST_DAYS
)
from utils import get_logger, get_rebalance_dates, get_trade_dates
from data_loader import get_panel_data, get_benchmark_data, get_stock_industry
from factors import FactorCalculator, get_factor_directions
from preprocessing import FactorPreprocessor
from factor_combiner import FactorCombiner
from selector import StockSelector

logger = get_logger("backtest")


class Backtester:
    """多因子回测引擎"""

    def __init__(self,
                 start_date: str = BACKTEST_START,
                 end_date: str = BACKTEST_END,
                 rebalance_freq: str = REBALANCE_FREQ,
                 benchmark: str = BENCHMARK):
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_freq = rebalance_freq
        self.benchmark = benchmark

        self.factor_calc = FactorCalculator()
        self.preprocessor = FactorPreprocessor()
        self.combiner = FactorCombiner()
        self.selector = StockSelector()

        # 结果存储
        self.weights_history: List[pd.Series] = []
        self.rebalance_dates: List[str] = []
        self.portfolio_returns: Optional[pd.Series] = None
        self.benchmark_returns: Optional[pd.Series] = None
        self.holdings_history: List[Dict] = []

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

            # 记录
            self.weights_history.append(weights)
            self.rebalance_dates.append(rb_date)
            self.holdings_history.append({
                "date": rb_date,
                "holdings": weights[weights > 0].to_dict(),
                "n_holdings": int((weights > 0).sum()),
            })

            prev_weights = weights

        # 计算每日组合收益
        self.portfolio_returns = self._compute_portfolio_returns(
            panel, rebalance_dates, trade_dates
        )

        # 基准收益
        self.benchmark_returns = self._compute_benchmark_returns()

        logger.info("回测完成")

        return {
            "portfolio_returns": self.portfolio_returns,
            "benchmark_returns": self.benchmark_returns,
            "weights_history": self.weights_history,
            "holdings_history": self.holdings_history,
            "rebalance_dates": self.rebalance_dates,
        }

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
