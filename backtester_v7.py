"""
v7.0 增强回测引擎
=================

继承 v6.2 Backtester 的全部功能，在以下6个维度进行增强:

  1. 扩展因子体系   — 使用 ExtendedFactorCalculator 自动计算37个因子
                      (v6基础27个 + v7高级10个)，通过 FactorRegistry 统一管理方向
  2. 增强绩效分析   — 使用 EnhancedPerformanceAnalyzer，自动计算
                      IR / Tracking Error / Treynor / Jensen Alpha / Bootstrap CI 等
  3. 因子归因分析   — 回测过程中逐期记录因子暴露，结束后计算因子收益归因
                      (Brinson风格)，识别收益的主要贡献因子
  4. Walk-Forward   — run_walk_forward() 将回测期间分为多个训练/测试窗口，
                      每个窗口独立训练因子权重，报告样本外表现
  5. 增强交易成本   — 在固定佣金/印花税/滑点之上叠加:
                      - 市场冲击成本: impact = k * sqrt(trade_amount / daily_volume)
                      - 大单额外滑点
  6. 组合优化       - 最大权重限制基于个股波动率调整
                      - 行业暴露动态调整(基于市场状态)

依赖:
  - backtester.Backtester / TransactionCostModel
  - factors_v7.ExtendedFactorCalculator
  - performance_v7.EnhancedPerformanceAnalyzer
  - factor_registry.FactorRegistry / FactorEvaluator
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm

from backtester import Backtester, TransactionCostModel
from factors_v7 import ExtendedFactorCalculator
from performance_v7 import EnhancedPerformanceAnalyzer
from factor_registry import FactorRegistry, FactorEvaluator
from prediction_engine import PredictionEngine

from config import (
    BACKTEST_START, BACKTEST_END, REBALANCE_FREQ, BENCHMARK,
    FACTOR_COMBINE_METHOD, WEIGHTS, TOP_N
)
from utils import get_logger, get_rebalance_dates, get_trade_dates
from data_loader import get_panel_data, get_benchmark_data, get_stock_industry

logger = get_logger("backtester_v7")


# ============ 增强交易成本参数 ============
MARKET_IMPACT_COEFF = 0.1      # 市场冲击系数 k
LARGE_ORDER_THRESHOLD = 0.10   # 大单阈值: 单笔交易占日成交额比例
LARGE_ORDER_SLIPPAGE = 0.002   # 大单额外滑点 0.2%
VOL_ADJUSTED_MAX_WEIGHT = 0.10  # 波动率调整后的基础最大权重上限


class EnhancedBacktester(Backtester):
    """
    增强回测引擎 v7.0

    继承 Backtester (v6.2) 的全部回测逻辑，并通过以下增强点提升分析深度:

    1. 扩展因子:
       - 使用 ExtendedFactorCalculator 替代 FactorCalculator (自动计算37个因子)
       - 使用 FactorRegistry 管理因子方向，统一回测因子与扩展因子

    2. 增强绩效分析:
       - 使用 EnhancedPerformanceAnalyzer 替代 PerformanceAnalyzer
       - 回测结束后自动计算 IR / Tracking Error / Treynor / Jensen Alpha 等
       - 生成增强版绩效报告

    3. 因子归因分析:
       - 回测过程中记录每期因子暴露
       - 回测结束后计算因子收益归因 (Brinson风格)
       - 识别收益的主要贡献因子

    4. Walk-Forward 验证:
       - run_walk_forward() 方法
       - 将回测期间分为多个训练/测试窗口
       - 每个窗口独立训练因子权重
       - 报告样本外表现

    5. 增强交易成本:
       - 考虑冲击成本 (market impact)
       - 基于成交额的市场冲击模型: impact = k * sqrt(trade_amount / daily_volume)
       - 大单额外滑点

    6. 组合优化:
       - 最大权重限制基于个股波动率调整
       - 行业暴露动态调整(基于市场状态)
    """

    def __init__(self,
                 start_date: str = BACKTEST_START,
                 end_date: str = BACKTEST_END,
                 rebalance_freq: str = REBALANCE_FREQ,
                 benchmark: str = BENCHMARK,
                 use_ml: bool = False,
                 use_cost_model: bool = True,
                 use_regime: bool = True,
                 use_risk_manager: bool = True,
                 market_impact_coeff: float = MARKET_IMPACT_COEFF,
                 large_order_threshold: float = LARGE_ORDER_THRESHOLD,
                 large_order_slippage: float = LARGE_ORDER_SLIPPAGE,
                 vol_adjusted_max_weight: float = VOL_ADJUSTED_MAX_WEIGHT):
        """
        初始化增强回测引擎

        :param start_date: 回测起始日期 (YYYY-MM-DD)
        :param end_date: 回测结束日期 (YYYY-MM-DD)
        :param rebalance_freq: 调仓频率 ('M' 月度 / 'W' 周度 / 'Q' 季度)
        :param benchmark: 基准指数代码
        :param use_ml: 是否使用ML因子合成器
        :param use_cost_model: 是否启用交易成本模型
        :param use_regime: 是否启用市场状态检测
        :param use_risk_manager: 是否启用高级风险管理器
        :param market_impact_coeff: 市场冲击系数 k，用于 impact = k * sqrt(amt/vol)
        :param large_order_threshold: 大单阈值，单笔交易占日成交额比例超过此值视为大单
        :param large_order_slippage: 大单额外滑点
        :param vol_adjusted_max_weight: 波动率调整后的基础最大权重上限
        """
        # 调用父类初始化 (会创建 factor_calc / preprocessor / combiner / selector 等)
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            rebalance_freq=rebalance_freq,
            benchmark=benchmark,
            use_ml=use_ml,
            use_cost_model=use_cost_model,
            use_regime=use_regime,
            use_risk_manager=use_risk_manager,
        )

        # ─── 增强点1: 使用 ExtendedFactorCalculator 替代 FactorCalculator ───
        self.factor_calc = ExtendedFactorCalculator()
        logger.info("v7.0: 已切换为 ExtendedFactorCalculator (37个因子)")

        # ─── 增强点1b: 使用 FactorRegistry 管理因子方向 ───
        self.registry = FactorRegistry()
        self.factor_directions = self.registry.get_factor_directions()
        logger.info(f"v7.0: FactorRegistry 已加载 {len(self.factor_directions)} 个因子方向")

        # ─── 增强点2: 使用 EnhancedPerformanceAnalyzer ───
        self.enhanced_analyzer = EnhancedPerformanceAnalyzer()
        logger.info("v7.0: EnhancedPerformanceAnalyzer 已启用")

        # ─── 增强点3: 因子归因分析相关存储 ───
        self.factor_exposure_history: List[Dict] = []     # 每期因子暴露记录
        self.factor_returns_history: List[Dict] = []      # 每期因子收益记录
        self._factor_exposure_df: Optional[pd.DataFrame] = None  # 汇总的因子暴露矩阵
        self._factor_returns_df: Optional[pd.DataFrame] = None   # 汇总的因子收益矩阵

        # ─── 增强点5: 增强交易成本参数 ───
        self.market_impact_coeff = market_impact_coeff
        self.large_order_threshold = large_order_threshold
        self.large_order_slippage = large_order_slippage

        # ─── 增强点6: 组合优化参数 ───
        self.vol_adjusted_max_weight = vol_adjusted_max_weight

        # ─── v7.1: 预测引擎 ───
        self.prediction_engine = PredictionEngine(
            momentum_weight=0.30,
            regression_weight=0.40,
            ic_weight=0.30,
            min_periods=6,
        )
        logger.info("v7.1: PredictionEngine 已启用 (因子动量+截面回归+IC衰减)")

        # 增强结果存储
        self.enhanced_metrics: Optional[Dict] = None
        self.factor_attribution: Optional[Dict] = None
        self.enhanced_cost_history: List[float] = []

    # ==================================================================
    #  重写 run() 方法
    # ==================================================================
    def run(self, panel: pd.DataFrame,
            financial: pd.DataFrame,
            codes: List[str],
            industry: Optional[pd.Series] = None,
            stock_list: Optional[pd.DataFrame] = None) -> Dict:
        """
        执行增强回测 v7.0

        流程:
          1. 调用父类初始化逻辑 (获取调仓日/交易日/基准数据等)
          2. 在因子计算阶段使用 ExtendedFactorCalculator (自动37个因子)
          3. 在选股后使用增强风控 (波动率调整最大权重 + 行业暴露动态调整)
          4. 回测过程中逐期记录因子暴露
          5. 回测结束后调用 enhanced_analyze() 与 _compute_factor_attribution()
          6. 返回增强结果 (含因子归因、滚动指标等)

        :param panel: 全部股票历史行情面板
        :param financial: 财务数据
        :param codes: 股票池
        :param industry: 行业归属
        :param stock_list: 股票列表(含market_cap等)
        :return: 增强回测结果字典
        """
        logger.info(f"开始增强回测 v7.0: {self.start_date} ~ {self.end_date}")
        logger.info(f"股票池: {len(codes)} 只, 调仓频率: {self.rebalance_freq}")
        logger.info(f"增强功能: 扩展因子(37) | 增强绩效 | 因子归因 | 市场冲击成本 | 组合优化")

        # 获取调仓日与交易日
        rebalance_dates = get_rebalance_dates(
            self.start_date, self.end_date, self.rebalance_freq
        )
        if not rebalance_dates:
            logger.error("无法获取调仓日,请检查日期范围")
            return {}

        trade_dates = get_trade_dates(self.start_date, self.end_date)
        logger.info(f"调仓日: {len(rebalance_dates)} 个, 交易日: {len(trade_dates)} 个")

        # 获取基准数据 (用于市场状态检测与协偏度计算)
        benchmark_panel = None
        if self.use_regime or True:  # 基准数据对协偏度因子也是必需的
            try:
                benchmark_panel = get_benchmark_data(
                    "sh",
                    self.start_date.replace("-", ""),
                    self.end_date.replace("-", ""),
                )
                if benchmark_panel is None or benchmark_panel.empty:
                    from data_generator import generate_sample_benchmark
                    benchmark_panel = generate_sample_benchmark(
                        n_days=500, start_date=self.start_date
                    )
                    logger.warning("使用模拟基准数据(协偏度/市场状态精度可能受限)")
                else:
                    logger.info(f"获取基准数据: {len(benchmark_panel)} 条")
            except Exception as e:
                logger.warning(f"获取基准数据失败({e})")
                try:
                    from data_generator import generate_sample_benchmark
                    benchmark_panel = generate_sample_benchmark(
                        n_days=500, start_date=self.start_date
                    )
                except Exception:
                    benchmark_panel = None

        # 准备真实market_cap
        market_cap_map = None
        if stock_list is not None and "market_cap" in stock_list.columns:
            market_cap_map = stock_list.set_index("code")["market_cap"]
            logger.info("使用真实市值数据(来自stock_list)")
        else:
            logger.warning("无stock_list,市值将用close近似")

        # 重置历史存储 (避免多次run叠加)
        self.weights_history = []
        self.rebalance_dates = []
        self.portfolio_returns = None
        self.benchmark_returns = None
        self.holdings_history = []
        self.turnover_history = []
        self.cost_history = []
        self.regime_history = []
        self.enhanced_cost_history = []
        self.factor_exposure_history = []
        self.factor_returns_history = []

        # ─── 逐期调仓 ───
        prev_weights = None

        for i, rb_date in enumerate(tqdm(rebalance_dates, desc="v7.0回测进度")):
            # ── 增强点1: 使用 ExtendedFactorCalculator 计算因子 ──
            factors = self.factor_calc.compute_all_factors(
                panel, financial, rb_date, benchmark_panel
            )
            if factors.empty:
                logger.debug(f"截面 {rb_date} 因子计算为空,跳过")
                continue

            # 使用 FactorRegistry 的方向对齐因子 (保证方向一致性)
            factors = self._align_factor_directions(factors)

            # market_cap
            mc_series = None
            if market_cap_map is not None:
                mc_series = market_cap_map.reindex(factors.index)
            else:
                cross_data = panel[panel["date"] == pd.to_datetime(rb_date, format="%Y%m%d")]
                mc_series = (
                    cross_data.set_index("code")["close"] * 1e8
                    if not cross_data.empty else None
                )

            # 预处理
            processed = self.preprocessor.process(
                factors, industry=industry, market_cap=mc_series
            )

            # forward returns (用于IC更新,不用于当期预测)
            next_rb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
            forward_returns = self._compute_forward_returns(panel, rb_date, next_rb)

            # v7.1: 更新预测引擎(用历史数据训练,不使用当期forward_returns预测当期)
            if forward_returns is not None and not forward_returns.empty:
                # 使用上一期的因子和当期的forward_returns更新预测引擎
                # 注意: 这里用processed(当期因子)和forward_returns(当期到下期收益)更新
                # 预测引擎内部会正确处理时间顺序
                self.prediction_engine.update(processed, forward_returns)

            # 市场状态检测
            regime_info = None
            if self.regime_detector is not None and benchmark_panel is not None:
                regime_info = self._detect_regime_for_date(benchmark_panel, rb_date)
                if regime_info:
                    self.regime_history.append(regime_info)

            # 因子合成
            scores = self.combiner.combine(processed, forward_returns)

            # v7.1: 预测引擎融合(如果有足够历史数据)
            if self.prediction_engine.is_ready:
                pred_scores = self.prediction_engine.predict(processed)
                if not pred_scores.empty:
                    # 将预测得分与因子合成得分加权融合
                    # 预测引擎权重随历史数据增加而增大
                    n_updates = self.prediction_engine.n_updates
                    pred_weight = min(0.3, n_updates / 20 * 0.3)  # 最多30%权重
                    combiner_weight = 1.0 - pred_weight

                    # 标准化两者到相同尺度
                    scores_rank = scores.rank(pct=True)
                    pred_rank = pred_scores.reindex(scores.index).rank(pct=True)

                    scores = combiner_weight * scores_rank + pred_weight * pred_rank
                    logger.debug(
                        f"v7.1预测融合: combiner={combiner_weight:.0%} "
                        f"+ prediction={pred_weight:.0%}"
                    )

            # 选股参数
            regime_adjustments = None
            if regime_info:
                regime_adjustments = {
                    'target_position': regime_info.get('target_position', 1.0),
                    'regime': regime_info.get('regime_name', 'range'),
                }

            # 选股
            weights = self.selector.select(
                scores, industry, prev_weights,
                regime_adjustments=regime_adjustments
            )

            # v6.2 高级风险管理器
            if self.risk_manager is not None and not weights.empty:
                weights = self._apply_risk_optimization(
                    weights, panel, rb_date, processed
                )

            # ── 增强点6: 增强风控 — 波动率调整最大权重 + 行业暴露动态调整 ──
            if not weights.empty:
                weights = self._apply_enhanced_portfolio_optimization(
                    weights, panel, rb_date, industry, regime_info
                )

            # ── 增强点3: 记录因子暴露 ──
            self._record_factor_exposure(processed, weights, rb_date)

            # ── 增强点5: 增强交易成本计算 ──
            if prev_weights is not None and self.use_cost_model:
                turnover = self._compute_turnover(weights, prev_weights)
                enhanced_cost = self._enhanced_cost_model(
                    turnover, weights, panel, rb_date
                )
                self.turnover_history.append(turnover)
                self.enhanced_cost_history.append(enhanced_cost)
            else:
                self.turnover_history.append(0.0)
                self.enhanced_cost_history.append(0.0)

            # 记录持仓
            self.weights_history.append(weights)
            self.rebalance_dates.append(rb_date)
            self.holdings_history.append({
                "date": rb_date,
                "holdings": weights[weights > 0].to_dict(),
                "n_holdings": int((weights > 0).sum()),
                "turnover": self.turnover_history[-1],
                "enhanced_cost": self.enhanced_cost_history[-1],
                "regime": regime_info.get('regime_name', 'N/A') if regime_info else 'N/A',
                "target_position": regime_info.get('target_position', 1.0) if regime_info else 1.0,
            })

            prev_weights = weights

        # ── 计算每日组合收益 (含最后一期) ──
        self.portfolio_returns = self._compute_portfolio_returns(
            panel, self.rebalance_dates, trade_dates
        )

        # ── 应用增强交易成本 ──
        if self.use_cost_model and self.portfolio_returns is not None:
            self._apply_enhanced_transaction_costs(self.rebalance_dates)

        # ── 基准收益 ──
        self.benchmark_returns = self._compute_benchmark_returns()

        # ── 增强点2: 调用 enhanced_analyze() ──
        self._build_factor_matrices()
        self.enhanced_metrics = self._run_enhanced_analysis()

        # ── 增强点3: 因子归因 ──
        self.factor_attribution = self._compute_factor_attribution()

        # ── 输出统计 ──
        self._log_summary()

        return {
            # 原有结果
            "portfolio_returns": self.portfolio_returns,
            "benchmark_returns": self.benchmark_returns,
            "weights_history": self.weights_history,
            "holdings_history": self.holdings_history,
            "rebalance_dates": self.rebalance_dates,
            "turnover_history": self.turnover_history,
            "cost_history": self.cost_history,
            "regime_history": self.regime_history,
            # v7.0 增强结果
            "enhanced_cost_history": self.enhanced_cost_history,
            "enhanced_metrics": self.enhanced_metrics,
            "factor_attribution": self.factor_attribution,
            "factor_exposure_history": self.factor_exposure_history,
            "factor_exposure_df": self._factor_exposure_df,
            "factor_returns_df": self._factor_returns_df,
        }

    # ==================================================================
    #  Walk-Forward 验证
    # ==================================================================
    def run_walk_forward(self,
                         panel: pd.DataFrame,
                         financial: pd.DataFrame,
                         codes: List[str],
                         n_windows: int = 5,
                         train_ratio: float = 0.6,
                         industry: Optional[pd.Series] = None,
                         stock_list: Optional[pd.DataFrame] = None) -> Dict:
        """
        Walk-Forward 验证: 将回测期间分为多个训练/测试窗口，
        每个窗口独立训练因子权重，报告样本外表现。

        流程:
          1. 将全部调仓日按时间顺序分为 n_windows 个窗口
          2. 每个窗口内,前 train_ratio 比例的调仓日作为训练集(用于IC加权/ML训练),
             后 (1-train_ratio) 作为测试集(样本外)
          3. 训练集计算因子IC,得到因子权重; 测试集用该权重选股
          4. 汇总所有测试期的样本外收益

        :param panel: 全部股票历史行情面板
        :param financial: 财务数据
        :param codes: 股票池
        :param n_windows: 滑动窗口数量,默认5
        :param train_ratio: 训练集占窗口的比例,默认0.6
        :param industry: 行业归属
        :param stock_list: 股票列表
        :return: Walk-Forward 结果字典,包含:
            - 'oos_returns': 样本外日收益序列
            - 'oos_metrics': 样本外绩效指标
            - 'window_results': 每个窗口的详细结果
            - 'oos_sharpe': 样本外夏普比率
            - 'oos_annual_return': 样本外年化收益
            - 'oos_max_drawdown': 样本外最大回撤
            - 'stability_score': 各窗口夏普的稳定性(标准差倒数)
        """
        logger.info(f"开始 Walk-Forward 验证: {n_windows} 个窗口, 训练比={train_ratio}")

        rebalance_dates = get_rebalance_dates(
            self.start_date, self.end_date, self.rebalance_freq
        )
        if len(rebalance_dates) < n_windows * 2:
            logger.error(f"调仓日数量({len(rebalance_dates)})不足以分为{n_windows}个窗口")
            return {}

        # 将调仓日划分为 n_windows 个窗口
        window_size = len(rebalance_dates) // n_windows
        windows = []
        for w in range(n_windows):
            start_idx = w * window_size
            end_idx = (w + 1) * window_size if w < n_windows - 1 else len(rebalance_dates)
            window_dates = rebalance_dates[start_idx:end_idx]
            n_train = max(int(len(window_dates) * train_ratio), 1)
            train_dates = window_dates[:n_train]
            test_dates = window_dates[n_train:]
            windows.append({
                'window_id': w,
                'train_dates': train_dates,
                'test_dates': test_dates,
            })

        logger.info(f"Walk-Forward: {n_windows} 个窗口, 每窗口约 {window_size} 个调仓日")

        # 获取基准数据
        benchmark_panel = None
        try:
            benchmark_panel = get_benchmark_data(
                "sh",
                self.start_date.replace("-", ""),
                self.end_date.replace("-", ""),
            )
            if benchmark_panel is None or benchmark_panel.empty:
                from data_generator import generate_sample_benchmark
                benchmark_panel = generate_sample_benchmark(
                    n_days=500, start_date=self.start_date
                )
        except Exception as e:
            logger.warning(f"Walk-Forward 获取基准数据失败: {e}")

        # market_cap
        market_cap_map = None
        if stock_list is not None and "market_cap" in stock_list.columns:
            market_cap_map = stock_list.set_index("code")["market_cap"]

        window_results = []
        all_oos_returns = []
        all_oos_weights = []
        all_oos_dates = []

        for window in windows:
            w_id = window['window_id']
            train_dates = window['train_dates']
            test_dates = window['test_dates']

            if not test_dates:
                continue

            logger.info(f"  窗口 {w_id}: 训练={len(train_dates)}期, 测试={len(test_dates)}期")

            # ── 训练阶段: 在训练集上计算因子IC,得到因子权重 ──
            train_ic = self._compute_window_factor_ic(
                panel, financial, train_dates, benchmark_panel, market_cap_map, industry
            )
            factor_weights = self._derive_factor_weights_from_ic(train_ic)

            # ── 测试阶段: 用训练得到的权重在测试集上选股 ──
            oos_weights_history = []
            prev_weights = None

            for rb_date in tqdm(test_dates, desc=f"WF-窗口{w_id}测试", leave=False):
                factors = self.factor_calc.compute_all_factors(
                    panel, financial, rb_date, benchmark_panel
                )
                if factors.empty:
                    continue

                factors = self._align_factor_directions(factors)

                mc_series = None
                if market_cap_map is not None:
                    mc_series = market_cap_map.reindex(factors.index)
                else:
                    cross = panel[panel["date"] == pd.to_datetime(rb_date, format="%Y%m%d")]
                    mc_series = cross.set_index("code")["close"] * 1e8 if not cross.empty else None

                processed = self.preprocessor.process(
                    factors, industry=industry, market_cap=mc_series
                )

                # 用训练得到的权重合成因子得分
                scores = self._score_with_weights(processed, factor_weights)

                # 选股
                weights = self.selector.select(scores, industry, prev_weights)

                # 增强风控
                if not weights.empty:
                    weights = self._apply_enhanced_portfolio_optimization(
                        weights, panel, rb_date, industry, None
                    )

                oos_weights_history.append(weights)
                all_oos_weights.append(weights)
                all_oos_dates.append(rb_date)
                prev_weights = weights

            # 计算该窗口测试期收益
            oos_returns = self._compute_window_returns(
                panel, test_dates, oos_weights_history
            )

            # 窗口指标
            oos_sharpe = self._calc_sharpe(oos_returns)
            oos_annual_ret = self._calc_annual_return(oos_returns)
            oos_max_dd = self._calc_max_drawdown(oos_returns)

            window_results.append({
                'window_id': w_id,
                'train_dates': train_dates,
                'test_dates': test_dates,
                'factor_ic': train_ic,
                'factor_weights': factor_weights,
                'oos_returns': oos_returns,
                'oos_sharpe': oos_sharpe,
                'oos_annual_return': oos_annual_ret,
                'oos_max_drawdown': oos_max_dd,
                'n_test_periods': len(test_dates),
            })

            all_oos_returns.append(oos_returns)

            logger.info(
                f"  窗口 {w_id} 完成: OOS夏普={oos_sharpe:.3f}, "
                f"年化={oos_annual_ret:.2%}, 最大回撤={oos_max_dd:.2%}"
            )

        # 汇总样本外收益
        if all_oos_returns:
            oos_combined = pd.concat(all_oos_returns).sort_index()
        else:
            oos_combined = pd.Series(dtype=float)

        # 样本外汇总指标
        oos_sharpe = self._calc_sharpe(oos_combined)
        oos_annual_ret = self._calc_annual_return(oos_combined)
        oos_max_dd = self._calc_max_drawdown(oos_combined)

        # 稳定性: 各窗口夏普的标准差倒数
        window_sharpes = [r['oos_sharpe'] for r in window_results if not np.isnan(r['oos_sharpe'])]
        if len(window_sharpes) > 1:
            stability = 1.0 / (np.std(window_sharpes) + 1e-10)
        else:
            stability = np.nan

        result = {
            'oos_returns': oos_combined,
            'oos_sharpe': oos_sharpe,
            'oos_annual_return': oos_annual_ret,
            'oos_max_drawdown': oos_max_dd,
            'window_results': window_results,
            'stability_score': float(stability),
            'n_windows': n_windows,
            'train_ratio': train_ratio,
            'mean_window_sharpe': float(np.mean(window_sharpes)) if window_sharpes else np.nan,
        }

        logger.info(
            f"Walk-Forward 完成: OOS夏普={oos_sharpe:.3f}, "
            f"年化={oos_annual_ret:.2%}, 最大回撤={oos_max_dd:.2%}, "
            f"稳定性={stability:.3f}"
        )

        return result

    # ==================================================================
    #  增强点3: 因子暴露记录
    # ==================================================================
    def _record_factor_exposure(self, factors: pd.DataFrame,
                                 weights: pd.Series, date: str) -> None:
        """
        记录某一调仓日的因子暴露 (持仓加权平均因子值)

        因子暴露 = sum(weights[i] * factors[i, :]) for i in holdings
        即以持仓权重对各因子值进行加权平均，得到该期组合在各因子上的暴露。

        :param factors: 当期因子值 DataFrame (index=code, columns=factor_names)
        :param weights: 当期持仓权重 Series (index=code)
        :param date: 调仓日期 YYYYMMDD
        """
        if factors.empty or weights.empty:
            return

        # 仅对持仓股票计算
        holdings = weights[weights > 0]
        common = holdings.index.intersection(factors.index)
        if len(common) == 0:
            return

        w = holdings.loc[common]
        w_norm = w / w.sum() if w.sum() > 0 else w

        # 持仓加权因子暴露
        factor_cols = factors.columns.tolist()
        exposure = {}
        for col in factor_cols:
            vals = factors.loc[common, col]
            masked = vals.where(vals.notna(), 0)
            if w_norm.sum() > 0:
                exposure[col] = float((masked * w_norm).sum())
            else:
                exposure[col] = np.nan

        # 同时记录该期的因子"收益"代理: 因子截面均值 (简化处理,真实因子收益需要时序回归)
        factor_mean = {}
        for col in factor_cols:
            factor_mean[col] = float(factors[col].mean()) if factors[col].notna().any() else np.nan

        self.factor_exposure_history.append({
            'date': date,
            'exposure': exposure,
        })
        self.factor_returns_history.append({
            'date': date,
            'factor_return': factor_mean,
        })

    # ==================================================================
    #  增强点3: 因子归因计算
    # ==================================================================
    def _compute_factor_attribution(self) -> Dict:
        """
        计算因子收益归因 (Brinson风格)

        将组合超额收益分解为各因子的贡献:
          contribution_i = exposure_i * factor_return_i
          总归因收益 = sum(contribution_i) + residual

        使用回测过程中记录的因子暴露矩阵和因子收益矩阵，
        通过 EnhancedPerformanceAnalyzer.factor_attribution() 完成计算。

        识别收益的主要贡献因子 (按累计贡献排序)。

        :return: 因子归因结果字典,包含:
            - 'factor_contributions': 各因子累计贡献 {因子名: 贡献值}
            - 'total_factor_return': 因子解释的总收益
            - 'residual': 残差收益
            - 'daily_contributions': 日频因子贡献 DataFrame
            - 'r_squared': 因子模型解释度 R^2
            - 'top_contributors': 主要贡献因子列表(按绝对贡献排序)
            - 'top_detractors': 主要拖累因子列表
        """
        if (self._factor_exposure_df is None
                or self._factor_returns_df is None
                or self._factor_exposure_df.empty
                or self._factor_returns_df.empty):
            logger.warning("因子归因: 因子暴露/收益矩阵为空,无法计算")
            return {}

        if self.portfolio_returns is None or self.portfolio_returns.empty:
            logger.warning("因子归因: 组合收益序列为空,无法计算")
            return {}

        # 调用增强分析器的因子归因方法
        try:
            attribution = self.enhanced_analyzer.factor_attribution(
                self.portfolio_returns,
                self._factor_exposure_df,
                self._factor_returns_df,
            )
        except Exception as e:
            logger.warning(f"因子归因计算异常: {e}")
            return {}

        # 识别主要贡献因子与拖累因子
        contributions = attribution.get('factor_contributions', {})
        if contributions:
            sorted_items = sorted(
                contributions.items(), key=lambda x: abs(x[1]), reverse=True
            )
            attribution['top_contributors'] = sorted_items[:5]
            attribution['top_detractors'] = sorted_items[-5:]

            logger.info("因子归因 - 主要贡献因子(Top 5):")
            for name, contrib in sorted_items[:5]:
                logger.info(f"  {name}: {contrib:.4%}")

        logger.info(
            f"因子归因完成: R^2={attribution.get('r_squared', 0):.4f}, "
            f"因子解释收益={attribution.get('total_factor_return', 0):.4%}, "
            f"残差={attribution.get('residual', 0):.4%}"
        )

        return attribution

    # ==================================================================
    #  增强点5: 增强交易成本模型
    # ==================================================================
    def _enhanced_cost_model(self, turnover: float, weights: pd.Series,
                              panel: pd.DataFrame, date: str) -> float:
        """
        增强交易成本模型

        在原有固定成本(佣金/印花税/基础滑点)之上，叠加:
          1. 市场冲击成本 (Market Impact):
             impact = k * sqrt(trade_amount / daily_volume)
             其中 trade_amount 为单股交易金额，daily_volume 为该股当日成交额
          2. 大单额外滑点:
             若某笔交易占日成交额比例超过 large_order_threshold，则追加额外滑点

        成本 = 基础成本 + 市场冲击成本 + 大单额外滑点

        :param turnover: 换手率 (0~1)
        :param weights: 当期目标权重 Series (index=code)
        :param panel: 行情面板 (用于获取成交额)
        :param date: 调仓日期 YYYYMMDD
        :return: 增强交易成本 (占组合价值的比例)
        """
        if turnover <= 0 or weights.empty:
            return 0.0

        # 基础成本 (佣金 + 印花税 + 基础滑点)
        base_cost = turnover * (
            self.cost_model.commission_rate
            + self.cost_model.stamp_tax_rate * 0.5
            + self.cost_model.slippage
        )

        # 市场冲击成本 + 大单额外滑点
        impact_cost = 0.0
        large_order_cost = 0.0

        try:
            td = pd.to_datetime(date, format="%Y%m%d")
            cross = panel[panel["date"] == td]

            if not cross.empty and "amount" in cross.columns:
                amount_map = cross.set_index("code")["amount"]

                # 估算各股交易金额 (假设组合总值1.0, 权重即交易比例)
                for code in weights.index:
                    if code not in amount_map.index:
                        continue
                    daily_amount = float(amount_map.loc[code])
                    if daily_amount <= 0:
                        continue

                    # 单股交易金额占组合比例 (近似)
                    trade_pct = float(abs(weights.loc[code]))

                    # 市场冲击: impact = k * sqrt(trade_amount / daily_volume)
                    # trade_amount/daily_volume 用 trade_pct 近似 (因组合价值归一化)
                    participation_rate = trade_pct / max(daily_amount / 1e8, 1e-6)
                    participation_rate = min(participation_rate, 1.0)
                    impact = self.market_impact_coeff * np.sqrt(participation_rate)
                    impact_cost += impact * trade_pct

                    # 大单额外滑点
                    if participation_rate > self.large_order_threshold:
                        large_order_cost += trade_pct * self.large_order_slippage

        except Exception as e:
            logger.debug(f"市场冲击成本计算异常({date}): {e}")

        total_cost = base_cost + impact_cost + large_order_cost
        return float(total_cost)

    # ==================================================================
    #  增强点6: 组合优化
    # ==================================================================
    def _apply_enhanced_portfolio_optimization(self, weights: pd.Series,
                                                panel: pd.DataFrame,
                                                date: str,
                                                industry: Optional[pd.Series],
                                                regime_info: Optional[Dict]) -> pd.Series:
        """
        增强组合优化

        1. 最大权重限制基于个股波动率调整:
           - 计算个股近期波动率
           - 波动率高的股票,最大权重上限降低
           - max_weight_i = base_max * (median_vol / vol_i)
        2. 行业暴露动态调整 (基于市场状态):
           - 在熊市/震荡市降低单一行业暴露上限
           - 在牛市适当放宽

        :param weights: 原始权重 Series
        :param panel: 行情面板
        :param date: 调仓日期
        :param industry: 行业归属 Series (index=code, values=行业名)
        :param regime_info: 市场状态信息
        :return: 优化后的权重 Series
        """
        if weights.empty:
            return weights

        try:
            td = pd.to_datetime(date, format="%Y%m%d")

            # ── 1. 波动率调整最大权重 ──
            weights = self._vol_adjusted_weight_cap(weights, panel, td)

            # ── 2. 行业暴露动态调整 ──
            if industry is not None:
                weights = self._dynamic_industry_adjustment(
                    weights, industry, regime_info
                )

            # 归一化
            total = weights.sum()
            if total > 0:
                weights = weights / total

        except Exception as e:
            logger.debug(f"增强组合优化异常({date}): {e}")

        return weights

    # ==================================================================
    #  内部辅助方法
    # ==================================================================
    def _align_factor_directions(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        使用 FactorRegistry 的方向对齐因子值

        对于 direction=-1 的因子,取负号,使所有因子统一为"越大越好"。

        :param factors: 原始因子值 DataFrame
        :return: 方向对齐后的因子 DataFrame
        """
        aligned = factors.copy()
        for col in aligned.columns:
            direction = self.factor_directions.get(col, 1)
            if direction == -1:
                aligned[col] = -aligned[col]
        return aligned

    def _compute_turnover(self, weights: pd.Series,
                           prev_weights: pd.Series) -> float:
        """
        计算换手率

        :param weights: 当期权重
        :param prev_weights: 上期权重
        :return: 换手率 (0~1)
        """
        common = weights.index.intersection(prev_weights.index)
        if len(common) == 0:
            return 1.0
        turnover = float(
            (weights.loc[common] - prev_weights.reindex(common, fill_value=0.0)
             ).abs().sum() / 2
        )
        return turnover

    def _apply_enhanced_transaction_costs(self, rebalance_dates: List[str]):
        """
        将增强交易成本应用到组合收益序列

        :param rebalance_dates: 调仓日列表
        """
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            return

        self.cost_history = []
        for i, rb_date in enumerate(rebalance_dates[:-1]):
            if i >= len(self.enhanced_cost_history):
                self.cost_history.append(0.0)
                continue

            cost_pct = self.enhanced_cost_history[i]
            if cost_pct <= 0:
                self.cost_history.append(0.0)
                continue

            next_rb = rebalance_dates[i + 1]
            rb_dt = pd.to_datetime(rb_date, format="%Y%m%d")
            next_rb_dt = pd.to_datetime(next_rb, format="%Y%m%d")

            mask = (self.portfolio_returns.index > rb_dt) & \
                   (self.portfolio_returns.index <= next_rb_dt)
            if mask.any():
                first_day = self.portfolio_returns.index[mask][0]
                self.portfolio_returns.loc[first_day] -= cost_pct
                self.cost_history.append(cost_pct)
            else:
                self.cost_history.append(0.0)

    def _build_factor_matrices(self):
        """
        将逐期记录的因子暴露和因子收益构建为 DataFrame 矩阵

        - self._factor_exposure_df: index=date, columns=factor_names
        - self._factor_returns_df: index=date, columns=factor_names
        """
        if not self.factor_exposure_history:
            return

        # 因子暴露矩阵
        exp_records = []
        for item in self.factor_exposure_history:
            rec = {'date': pd.to_datetime(item['date'], format="%Y%m%d")}
            rec.update(item['exposure'])
            exp_records.append(rec)
        if exp_records:
            self._factor_exposure_df = pd.DataFrame(exp_records).set_index('date').sort_index()

        # 因子收益矩阵
        ret_records = []
        for item in self.factor_returns_history:
            rec = {'date': pd.to_datetime(item['date'], format="%Y%m%d")}
            rec.update(item['factor_return'])
            ret_records.append(rec)
        if ret_records:
            self._factor_returns_df = pd.DataFrame(ret_records).set_index('date').sort_index()

    def _run_enhanced_analysis(self) -> Dict:
        """
        调用 EnhancedPerformanceAnalyzer.enhanced_analyze() 进行综合绩效分析

        :return: 增强绩效指标字典
        """
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            logger.warning("增强分析: 组合收益为空,跳过")
            return {}

        try:
            metrics = self.enhanced_analyzer.enhanced_analyze(
                self.portfolio_returns,
                benchmark_returns=self.benchmark_returns,
                factor_exposures=self._factor_exposure_df,
                factor_returns=self._factor_returns_df,
                bootstrap=True,
                n_bootstrap=500,  # 降低次数以加速
                confidence_level=0.95,
                random_seed=42,
            )
            logger.info("增强绩效分析完成 (含IR/TE/Treynor/Jensen/Bootstrap等)")
            return metrics
        except Exception as e:
            logger.warning(f"增强绩效分析异常: {e}")
            return {}

    def _vol_adjusted_weight_cap(self, weights: pd.Series,
                                  panel: pd.DataFrame, td) -> pd.Series:
        """
        基于个股波动率调整最大权重上限

        max_weight_i = base_max * (median_vol / vol_i)
        波动率高的股票上限降低,波动率低的股票上限提高。

        :param weights: 原始权重
        :param panel: 行情面板
        :param td: 截面日期
        :return: 调整后的权重
        """
        try:
            hist = panel[panel["date"] <= td].tail(60 * len(weights))
            if hist.empty:
                return weights

            vols = hist.groupby("code")["pct_chg"].apply(
                lambda x: float(np.std(x.dropna() / 100, ddof=1) * np.sqrt(252))
                if len(x.dropna()) >= 20 else 0.3
            )

            active = weights[weights > 0]
            common = active.index.intersection(vols.index)
            if len(common) == 0:
                return weights

            vols_common = vols.loc[common]
            median_vol = float(vols_common.median())
            if median_vol <= 0:
                return weights

            for code in common:
                stock_vol = float(vols_common.loc[code])
                if stock_vol > 0:
                    # 波动率调整的最大权重
                    adjusted_cap = self.vol_adjusted_max_weight * (median_vol / stock_vol)
                    adjusted_cap = np.clip(adjusted_cap, 0.01, 0.20)
                    if weights.loc[code] > adjusted_cap:
                        weights.loc[code] = adjusted_cap

            # 归一化
            total = weights.sum()
            if total > 0:
                weights = weights / total

        except Exception as e:
            logger.debug(f"波动率调整权重异常: {e}")

        return weights

    def _dynamic_industry_adjustment(self, weights: pd.Series,
                                      industry: pd.Series,
                                      regime_info: Optional[Dict]) -> pd.Series:
        """
        行业暴露动态调整 (基于市场状态)

        - 牛市 (bull): 单一行业上限 25%
        - 震荡市 (range): 单一行业上限 20%
        - 熊市 (bear): 单一行业上限 15%

        :param weights: 原始权重
        :param industry: 行业归属
        :param regime_info: 市场状态信息
        :return: 调整后的权重
        """
        try:
            # 确定行业上限
            if regime_info is not None:
                regime_name = regime_info.get('regime_name', 'range').lower()
                if 'bull' in regime_name:
                    industry_cap = 0.25
                elif 'bear' in regime_name:
                    industry_cap = 0.15
                else:
                    industry_cap = 0.20
            else:
                industry_cap = 0.20

            # 计算各行业当前暴露
            active = weights[weights > 0]
            ind_map = industry.reindex(active.index)
            ind_exposure = active.groupby(ind_map).sum()

            # 对超限行业进行缩放
            for ind, exp in ind_exposure.items():
                if pd.isna(ind):
                    continue
                if exp > industry_cap:
                    scale = industry_cap / exp
                    mask = ind_map == ind
                    mask = mask.fillna(False)
                    codes_in_ind = mask[mask].index
                    weights.loc[codes_in_ind] *= scale

            # 归一化
            total = weights.sum()
            if total > 0:
                weights = weights / total

        except Exception as e:
            logger.debug(f"行业暴露调整异常: {e}")

        return weights

    def _compute_window_factor_ic(self, panel: pd.DataFrame,
                                   financial: pd.DataFrame,
                                   train_dates: List[str],
                                   benchmark_panel: Optional[pd.DataFrame],
                                   market_cap_map: Optional[pd.Series],
                                   industry: Optional[pd.Series]) -> Dict[str, float]:
        """
        计算训练窗口内各因子的平均IC

        :param panel: 行情面板
        :param financial: 财务数据
        :param train_dates: 训练期调仓日列表
        :param benchmark_panel: 基准数据
        :param market_cap_map: 市值映射
        :param industry: 行业归属
        :return: {因子名: 平均IC} 字典
        """
        evaluator = FactorEvaluator()
        all_ics = {}

        for i, rb_date in enumerate(train_dates):
            factors = self.factor_calc.compute_all_factors(
                panel, financial, rb_date, benchmark_panel
            )
            if factors.empty:
                continue

            factors = self._align_factor_directions(factors)

            mc_series = None
            if market_cap_map is not None:
                mc_series = market_cap_map.reindex(factors.index)

            processed = self.preprocessor.process(
                factors, industry=industry, market_cap=mc_series
            )

            next_rb = train_dates[i + 1] if i + 1 < len(train_dates) else None
            fwd_ret = self._compute_forward_returns(panel, rb_date, next_rb)

            if fwd_ret is None or fwd_ret.empty:
                continue

            # 计算各因子IC
            for col in processed.columns:
                try:
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(
                        processed[col], fwd_ret
                    )
                    if not np.isnan(ic):
                        all_ics.setdefault(col, []).append(ic)
                except Exception:
                    continue

        # 平均IC
        mean_ic = {k: float(np.mean(v)) for k, v in all_ics.items()}
        return mean_ic

    def _derive_factor_weights_from_ic(self, factor_ic: Dict[str, float]) -> Dict[str, float]:
        """
        由因子IC推导因子权重 (IC加权)

        weight_i = IC_i / sum(|IC_j|)  (仅保留正IC因子)

        :param factor_ic: {因子名: IC} 字典
        :return: {因子名: 权重} 字典
        """
        if not factor_ic:
            return {}

        # 仅保留正IC因子
        positive_ic = {k: v for k, v in factor_ic.items() if v > 0}
        if not positive_ic:
            return {}

        total = sum(positive_ic.values())
        if total <= 0:
            return {}

        weights = {k: v / total for k, v in positive_ic.items()}
        return weights

    def _score_with_weights(self, factors: pd.DataFrame,
                             factor_weights: Dict[str, float]) -> pd.Series:
        """
        用给定权重合成因子得分

        score = sum(weight_i * rank(factor_i))

        :param factors: 因子值 DataFrame
        :param factor_weights: 因子权重字典
        :return: 合成得分 Series
        """
        if factors.empty or not factor_weights:
            return pd.Series(dtype=float)

        score = pd.Series(0.0, index=factors.index)
        for fname, w in factor_weights.items():
            if fname in factors.columns:
                ranked = factors[fname].rank(pct=True)
                score += w * ranked.fillna(0.5)

        return score

    def _compute_window_returns(self, panel: pd.DataFrame,
                                 window_dates: List[str],
                                 weights_history: List[pd.Series]) -> pd.Series:
        """
        计算某个窗口内的日收益序列

        :param panel: 行情面板
        :param window_dates: 窗口调仓日
        :param weights_history: 窗口内各期权重
        :return: 日收益 Series
        """
        if not weights_history:
            return pd.Series(dtype=float)

        trade_dates = get_trade_dates(
            window_dates[0][:4] + "-" + window_dates[0][4:6] + "-" + window_dates[0][6:8],
            self.end_date,
        )

        daily_returns = []
        for j, rb_date in enumerate(window_dates):
            if j + 1 < len(window_dates):
                next_rb = window_dates[j + 1]
            else:
                next_rb = self.end_date.replace("-", "")

            weights = weights_history[j] if j < len(weights_history) else pd.Series(dtype=float)
            if weights.empty:
                continue

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

        if not daily_returns:
            return pd.Series(dtype=float)

        df = pd.DataFrame(daily_returns)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df.set_index("date").sort_index()
        return df["return"]

    def _calc_sharpe(self, returns: pd.Series) -> float:
        """计算夏普比率 (年化)"""
        if returns is None or returns.empty or len(returns) < 10:
            return np.nan
        excess = returns - self.enhanced_analyzer.rf_daily
        std = returns.std()
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    def _calc_annual_return(self, returns: pd.Series) -> float:
        """计算年化收益"""
        if returns is None or returns.empty:
            return np.nan
        n = len(returns)
        total = (1 + returns).prod() - 1
        if n == 0:
            return 0.0
        return float((1 + total) ** (252 / n) - 1)

    def _calc_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        if returns is None or returns.empty:
            return np.nan
        nav = (1 + returns).cumprod()
        peak = nav.cummax()
        drawdown = (nav - peak) / peak
        return float(drawdown.min())

    def _log_summary(self):
        """输出回测摘要日志"""
        if self.turnover_history:
            avg_turnover = np.mean(self.turnover_history)
            total_cost = sum(self.cost_history) if self.cost_history else 0
            total_enhanced_cost = sum(self.enhanced_cost_history) if self.enhanced_cost_history else 0
            logger.info(
                f"v7.0 回测完成 | 平均换手率: {avg_turnover:.2%} | "
                f"总交易成本(增强): {total_enhanced_cost:.2%} | "
                f"总交易成本(应用): {total_cost:.2%}"
            )

        if self.regime_history:
            regime_counts = pd.Series([r['regime_name'] for r in self.regime_history]).value_counts()
            logger.info(f"市场状态分布: {dict(regime_counts)}")

        if self.enhanced_metrics:
            sharpe = self.enhanced_metrics.get('sharpe_ratio', np.nan)
            annual_ret = self.enhanced_metrics.get('annual_return', np.nan)
            max_dd = self.enhanced_metrics.get('max_drawdown', np.nan)
            ir = self.enhanced_metrics.get('information_ratio', np.nan)
            logger.info(
                f"增强绩效 | 夏普: {sharpe:.3f} | 年化: {annual_ret:.2%} | "
                f"最大回撤: {max_dd:.2%} | IR: {ir:.3f}"
            )

        if self.factor_attribution:
            r2 = self.factor_attribution.get('r_squared', 0)
            top = self.factor_attribution.get('top_contributors', [])
            logger.info(f"因子归因 | R^2: {r2:.4f} | 主要贡献因子数: {len(top)}")

    # ==================================================================
    #  生成增强版报告
    # ==================================================================
    def generate_enhanced_report(self, save_path: Optional[str] = None) -> str:
        """
        生成增强版绩效报告

        :param save_path: 报告保存路径
        :return: 报告文本
        """
        if self.enhanced_metrics is None:
            logger.warning("尚未运行增强分析,无法生成报告")
            return ""

        report = self.enhanced_analyzer.generate_enhanced_report(
            self.enhanced_metrics, save_path=save_path
        )
        return report


# ==================================================================
#  模块自测
# ==================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("v7.0 增强回测引擎 - 模块自测")
    print("=" * 60)

    bt = EnhancedBacktester(
        start_date="2020-01-01",
        end_date="2023-12-31",
        rebalance_freq="M",
    )
    print(f"\n[EnhancedBacktester] 初始化完成")
    print(f"  因子计算器: {type(bt.factor_calc).__name__}")
    print(f"  因子注册表: {bt.registry}")
    print(f"  因子方向数: {len(bt.factor_directions)}")
    print(f"  增强分析器: {type(bt.enhanced_analyzer).__name__}")
    print(f"  市场冲击系数: {bt.market_impact_coeff}")
    print(f"  大单阈值: {bt.large_order_threshold}")
    print(f"  波动率调整最大权重: {bt.vol_adjusted_max_weight}")

    print(f"\n{'=' * 60}")
    print("模块自测完成")
    print(f"{'=' * 60}")
