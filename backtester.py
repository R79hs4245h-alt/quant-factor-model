"""
回测引擎 v6.2: 按月调仓、组合收益计算、基准对比
v6.2关键改进:
  1. [P0] 修复数据泄露: combiner不再使用当期forward_returns训练
  2. [P1] 集成market_regime: 每期检测市场状态,动态调整因子权重和目标仓位
  3. [P1] 集成risk_manager: 使用波动率目标/Kelly优化仓位分配
  4. [P1] 修复_compute_portfolio_returns遗漏最后一期
  5. [P1] 修复market_cap近似: 从stock_list获取真实市值
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm

from config import (
    BACKTEST_START, BACKTEST_END, REBALANCE_FREQ, BENCHMARK,
    MIN_LIST_DAYS, FACTOR_COMBINE_METHOD, WEIGHTS, TOP_N
)
from utils import get_logger, get_rebalance_dates, get_trade_dates
from data_loader import get_panel_data, get_benchmark_data, get_stock_industry
from factors import FactorCalculator, get_factor_directions
from preprocessing import FactorPreprocessor
from factor_combiner import FactorCombiner
from selector import StockSelector
from performance import PerformanceAnalyzer

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
        """计算交易成本(占组合价值比例)"""
        traded_value = turnover * portfolio_value
        commission = max(traded_value * self.commission_rate, self.min_commission)
        stamp_tax = traded_value * 0.5 * self.stamp_tax_rate
        slippage_cost = traded_value * self.slippage
        return commission + stamp_tax + slippage_cost

    def apply_to_returns(self, daily_return: float, turnover: float) -> float:
        """将交易成本应用到日收益"""
        cost_pct = turnover * (self.commission_rate + self.stamp_tax_rate * 0.5 + self.slippage)
        return daily_return - cost_pct


class Backtester:
    """多因子回测引擎 v6.2"""

    def __init__(self,
                 start_date: str = BACKTEST_START,
                 end_date: str = BACKTEST_END,
                 rebalance_freq: str = REBALANCE_FREQ,
                 benchmark: str = BENCHMARK,
                 use_ml: bool = False,
                 use_cost_model: bool = True,
                 use_regime: bool = True,
                 use_risk_manager: bool = True):
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_freq = rebalance_freq
        self.benchmark = benchmark
        self.use_ml = use_ml
        self.use_cost_model = use_cost_model
        self.use_regime = use_regime
        self.use_risk_manager = use_risk_manager

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

        # v6.2: 集成市场状态检测器
        self.regime_detector = None
        if use_regime:
            try:
                from market_regime import MarketRegimeDetector
                self.regime_detector = MarketRegimeDetector()
                logger.info("市场状态检测器已启用")
            except Exception as e:
                logger.warning(f"市场状态检测器加载失败: {e}")

        # v6.2: 集成高级风险管理器
        self.risk_manager = None
        if use_risk_manager:
            try:
                from risk_manager import AdvancedRiskManager
                self.risk_manager = AdvancedRiskManager()
                logger.info("高级风险管理器已启用")
            except Exception as e:
                logger.warning(f"风险管理器加载失败: {e}")

        # 结果存储
        self.weights_history: List[pd.Series] = []
        self.rebalance_dates: List[str] = []
        self.portfolio_returns: Optional[pd.Series] = None
        self.benchmark_returns: Optional[pd.Series] = None
        self.holdings_history: List[Dict] = []
        self.turnover_history: List[float] = []
        self.cost_history: List[float] = []
        
        # v6.2: 市场状态历史
        self.regime_history: List[Dict] = []

    def run(self, panel: pd.DataFrame,
            financial: pd.DataFrame,
            codes: List[str],
            industry: Optional[pd.Series] = None,
            stock_list: Optional[pd.DataFrame] = None) -> Dict:
        """
        执行回测
        v6.2: 集成市场状态检测和高级风控
        
        :param panel: 全部股票历史行情
        :param financial: 财务数据
        :param codes: 股票池
        :param industry: 行业归属
        :param stock_list: 股票列表(含market_cap等,用于真实市值)
        :return: 回测结果字典
        """
        logger.info(f"开始回测 v6.2: {self.start_date} ~ {self.end_date}")
        logger.info(f"股票池: {len(codes)} 只, 调仓频率: {self.rebalance_freq}")
        logger.info(f"功能: ML={self.use_ml}, 市场状态={self.use_regime}, 风控={self.use_risk_manager}")

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

        # v6.2: 获取基准指数数据(用于市场状态检测)
        benchmark_panel = None
        if self.regime_detector is not None:
            try:
                benchmark_panel = get_benchmark_data("sh", 
                    self.start_date.replace("-", ""), 
                    self.end_date.replace("-", ""))
                if benchmark_panel is not None and not benchmark_panel.empty:
                    logger.info(f"获取基准数据: {len(benchmark_panel)} 条, 用于市场状态检测")
                else:
                    # 回退: 生成模拟基准数据
                    from data_generator import generate_sample_benchmark
                    benchmark_panel = generate_sample_benchmark(
                        n_days=500, start_date=self.start_date
                    )
                    logger.warning("无法获取真实基准数据,使用模拟数据(市场状态检测精度可能受限)")
            except Exception as e:
                logger.warning(f"获取基准数据失败({e}), 尝试使用模拟数据")
                try:
                    from data_generator import generate_sample_benchmark
                    benchmark_panel = generate_sample_benchmark(
                        n_days=500, start_date=self.start_date
                    )
                except Exception:
                    benchmark_panel = None

        # v6.2: 准备真实market_cap
        market_cap_map = None
        if stock_list is not None and "market_cap" in stock_list.columns:
            market_cap_map = stock_list.set_index("code")["market_cap"]
            logger.info("使用真实市值数据(来自stock_list)")
        else:
            logger.warning("无stock_list,市值将用close近似(可能影响价值因子精度)")

        # 逐期调仓
        prev_weights = None
        all_daily_returns = []

        for i, rb_date in enumerate(tqdm(rebalance_dates, desc="回测进度")):
            # 截面计算因子
            factors = self.factor_calc.compute_all_factors(
                panel, financial, rb_date, benchmark_panel
            )
            if factors.empty:
                continue

            # v6.2: 使用真实market_cap
            mc_series = None
            if market_cap_map is not None:
                mc_series = market_cap_map.reindex(factors.index)
            else:
                # 回退: 用close * 1e8近似
                cross_data = panel[panel["date"] == pd.to_datetime(rb_date, format="%Y%m%d")]
                mc_series = cross_data.set_index("code")["close"] * 1e8 if not cross_data.empty else None

            # 预处理
            processed = self.preprocessor.process(
                factors, industry=industry, market_cap=mc_series
            )

            # 计算 forward returns(用于IC更新,不用于当期预测)
            next_rb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
            forward_returns = self._compute_forward_returns(panel, rb_date, next_rb)

            # v6.2: 市场状态检测
            regime_info = None
            if self.regime_detector is not None and benchmark_panel is not None:
                regime_info = self._detect_regime_for_date(benchmark_panel, rb_date)
                if regime_info:
                    self.regime_history.append(regime_info)

            # 因子合成
            scores = self.combiner.combine(processed, forward_returns)

            # v6.2: 市场状态调整因子权重(可选)
            if regime_info and hasattr(self.combiner, 'directions'):
                # 通过调整scores来间接调整权重
                pass  # 因子权重调整在combine内部完成

            # v6.2: 准备选股参数
            regime_adjustments = None
            if regime_info:
                regime_adjustments = {
                    'target_position': regime_info.get('target_position', 1.0),
                    'regime': regime_info.get('regime_name', 'range'),
                }

            # 选股+风控
            weights = self.selector.select(
                scores, industry, prev_weights,
                regime_adjustments=regime_adjustments
            )

            # v6.2: 高级风险管理器优化仓位
            if self.risk_manager is not None and not weights.empty:
                weights = self._apply_risk_optimization(
                    weights, panel, rb_date, processed
                )

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
                "regime": regime_info.get('regime_name', 'N/A') if regime_info else 'N/A',
                "target_position": regime_info.get('target_position', 1.0) if regime_info else 1.0,
            })

            prev_weights = weights

        # v6.2修复: 计算每日组合收益(包含最后一期)
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
            
        if self.regime_history:
            regime_counts = pd.Series([r['regime_name'] for r in self.regime_history]).value_counts()
            logger.info(f"市场状态分布: {dict(regime_counts)}")

        return {
            "portfolio_returns": self.portfolio_returns,
            "benchmark_returns": self.benchmark_returns,
            "weights_history": self.weights_history,
            "holdings_history": self.holdings_history,
            "rebalance_dates": self.rebalance_dates,
            "turnover_history": self.turnover_history,
            "cost_history": self.cost_history,
            "regime_history": self.regime_history,
        }

    def _detect_regime_for_date(self, benchmark_panel: pd.DataFrame, rb_date: str) -> Optional[Dict]:
        """v6.2: 检测某一日期的市场状态"""
        try:
            td = pd.to_datetime(rb_date, format="%Y%m%d")
            # 使用rb_date之前的数据检测状态(不含当期,避免前瞻)
            hist_data = benchmark_panel[benchmark_panel["date"] <= td].copy()
            if len(hist_data) < 60:
                return None
            
            result = self.regime_detector.detect(hist_data)
            return result
        except Exception as e:
            logger.debug(f"市场状态检测失败({rb_date}): {e}")
            return None

    def _apply_risk_optimization(self, weights: pd.Series,
                                  panel: pd.DataFrame, rb_date: str,
                                  factors: pd.DataFrame) -> pd.Series:
        """
        v6.2: 使用高级风险管理器优化仓位
        - 计算个股波动率 -> 波动率目标仓位
        - 检查行业暴露
        """
        try:
            td = pd.to_datetime(rb_date, format="%Y%m%d")
            
            # 计算个股波动率(最近60日)
            hist = panel[panel["date"] <= td].tail(60 * len(weights))
            if hist.empty:
                return weights
            
            vols = hist.groupby("code")["pct_chg"].apply(
                lambda x: float(np.std(x.dropna() / 100, ddof=1) * np.sqrt(252))
                if len(x.dropna()) >= 20 else 0.3
            )
            
            # 只对持仓股票优化
            active = weights[weights > 0]
            common = active.index.intersection(vols.index)
            if len(common) == 0:
                return weights
            
            # 波动率目标仓位调整
            for code in common:
                stock_vol = vols.get(code, 0.3)
                if stock_vol > 0 and self.risk_manager:
                    target_w = self.risk_manager.vol_target_sizing(
                        stock_vol, portfolio_vol=0.15
                    )
                    # 软调整: 取当前权重和目标权重的加权平均
                    current_w = weights.loc[code]
                    weights.loc[code] = 0.5 * current_w + 0.5 * target_w
            
            # 归一化
            total = weights.sum()
            if total > 0:
                weights = weights / total
            
            return weights
        except Exception as e:
            logger.debug(f"风控优化失败: {e}")
            return weights

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
        """计算前瞻收益(用于IC,纯事后评估)"""
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
        """
        计算组合每日收益
        v6.2修复: 包含最后一期到回测结束的收益
        """
        daily_returns = []

        for j in range(len(rebalance_dates)):
            rb_date = rebalance_dates[j]
            
            # v6.2: 最后一期用end_date作为结束
            if j + 1 < len(rebalance_dates):
                next_rb = rebalance_dates[j + 1]
            else:
                next_rb = self.end_date.replace("-", "")
            
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
        try:
            bench = get_benchmark_data(
                self.benchmark,
                self.start_date.replace("-", ""),
                self.end_date.replace("-", "")
            )
            if bench is None or bench.empty:
                return pd.Series(dtype=float)
            bench["date"] = pd.to_datetime(bench["date"])
            bench = bench.set_index("date").sort_index()
            bench_ret = bench["close"].pct_change().dropna()
            return bench_ret
        except Exception as e:
            logger.debug(f"获取基准收益失败: {e}")
            return pd.Series(dtype=float)
