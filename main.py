"""
多因子选股量化框架 v7.0 - 主入口
整合增强回测引擎 + 增强推荐引擎 + 37因子体系 + ML管线 + 统一因子注册表

v7.0新增:
  - 扩展因子体系: 10个高级Alpha因子(F-Score/Z-Score/应计质量/SUE等)
  - 增强绩效分析: IR/Tracking Error/Treynor/Jensen Alpha/Bootstrap CI/因子归因
  - 统一因子注册表: 桥接回测37因子与推荐16维因子
  - 增强ML管线: VIF特征选择/Walk-Forward分析/模型版本管理/多模型集成
  - 增强回测引擎: Walk-Forward验证/市场冲击成本/因子归因分析
  - 增强推荐引擎: ML预测融合/实时风险监控/板块轮动/多时间框架/智能止损

v6.x保留:
  - ML因子合成器(随机森林/XGBoost/LightGBM)
  - 市场状态判别器(牛市/熊市/震荡自适应)
  - 高级风险管理器(波动率目标/Kelly/CVaR/压力测试)
  - 交易成本模型(佣金+印花税+滑点)
  - 动态因子权重(随市场状态调整)

用法:
    python main.py                        # 完整回测(v7.0增强)
    python main.py --mode screen          # 实时选股(基础因子)
    python main.py --mode recommend       # 增强推荐(v7.0: ML融合+风险+止损)
    python main.py --mode backtest        # v7.0增强回测
    python main.py --mode backtest-ml     # ML增强回测
    python main.py --mode backtest-v7     # v7.0增强回测(含Walk-Forward)
    python main.py --mode recommend-v7    # v7.0增强推荐(含ML+多时间框架)
    python main.py --mode regime          # 市场状态分析
    python main.py --mode risk            # 风险评估
    python main.py --mode mine            # Alpha因子挖掘
    python main.py --mode tune            # 自动调参
    python main.py --mode factor-eval     # 因子有效性评估
    python main.py --mode all             # 全部运行
    python main.py --codes 000001,600519  # 指定股票池
    python main.py --recommend-n 30       # 推荐N只股票
    python main.py --ml                   # 启用ML合成
    python main.py --walk-forward         # 启用Walk-Forward验证
"""
import argparse
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

from config import (
    BACKTEST_START, BACKTEST_END, REBALANCE_FREQ, BENCHMARK,
    TOP_N, OUTPUT_DIR, CACHE_DIR, FACTOR_COMBINE_METHOD,
    MARKET_TIMING_ENABLED, WEIGHTS
)
from utils import get_logger, get_rebalance_dates
from data_loader import (
    get_stock_list, get_panel_data, get_batch_financial,
    get_realtime_quotes, get_stock_industry, get_benchmark_data,
    get_index_daily, get_market_overview, get_sector_heatmap,
    get_fund_flow_ranking, get_northbound_flow
)
from factors import FactorCalculator
from preprocessing import FactorPreprocessor
from factor_combiner import FactorCombiner
from selector import StockSelector, assess_stock_risk
from backtester import Backtester
from performance import PerformanceAnalyzer

logger = get_logger("main")


def run_backtest(codes=None, start=None, end=None, use_ml=False):
    """运行完整回测"""
    start = start or BACKTEST_START
    end = end or BACKTEST_END

    version_str = "v6.1 (ML增强)" if use_ml else "v6.1"
    print("\n" + "=" * 60)
    print(f"  多因子选股量化框架 {version_str} - 回测模式")
    print("=" * 60)
    print(f"  回测区间:   {start} ~ {end}")
    print(f"  调仓频率:   {REBALANCE_FREQ}")
    print(f"  基准:       {BENCHMARK}")
    print(f"  持仓数量:   {TOP_N}")
    print(f"  合成方法:   {'ML ensemble' if use_ml else FACTOR_COMBINE_METHOD}")
    print(f"  交易成本:   已启用(佣金万3+印花税千1+滑点0.2%)")
    print("=" * 60 + "\n")

    # 1. 获取股票池
    if codes is None:
        logger.info("步骤 1/5: 获取全A股股票池...")
        stock_list = get_stock_list()
        codes = stock_list["code"].tolist()
        logger.info(f"  股票池: {len(codes)} 只")

    # 2. 获取历史数据
    logger.info("步骤 2/5: 下载历史行情数据...")
    panel = get_panel_data(codes, start.replace("-", ""), end.replace("-", ""))
    if panel.empty:
        logger.error("无法获取行情数据,请检查网络连接")
        return None

    # 3. 获取财务数据
    logger.info("步骤 3/5: 下载财务数据...")
    financial = get_batch_financial(codes)

    # 4. 获取行业归属
    logger.info("步骤 4/5: 获取行业归属...")
    industry = get_stock_industry(codes)

    # 5. 回测
    logger.info(f"步骤 5/5: 执行回测{'(ML增强)' if use_ml else ''}...")
    bt = Backtester(start, end, REBALANCE_FREQ, BENCHMARK,
                    use_ml=use_ml, use_cost_model=True)
    results = bt.run(panel, financial, codes, industry)

    if not results:
        logger.error("回测失败")
        return None

    # 绩效分析
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(
        results["portfolio_returns"],
        results.get("benchmark_returns")
    )

    if not metrics:
        logger.error("绩效分析失败")
        return None

    # 生成报告
    report = analyzer.generate_report(metrics)
    print("\n" + report)

    # v6.1: 交易成本统计
    if results.get("turnover_history"):
        avg_turnover = np.mean(results["turnover_history"])
        total_cost = sum(results.get("cost_history", [0]))
        print(f"\n【交易成本统计】")
        print(f"  平均换手率:     {avg_turnover:.2%}")
        print(f"  总交易成本:     {total_cost:.2%}")
        print(f"  调仓次数:       {len(results['turnover_history'])}")

    # v6.1: ML训练摘要
    if use_ml and hasattr(bt.combiner, 'get_training_summary'):
        train_summary = bt.combiner.get_training_summary()
        if not train_summary.empty:
            print(f"\n【ML训练摘要】")
            print(f"  训练次数:       {len(train_summary)}")
            print(f"  最新R2:         {train_summary.iloc[-1]['train_r2']:.4f}")
            print(f"  平均R2:         {train_summary['train_r2'].mean():.4f}")

        feature_imp = bt.combiner.get_feature_importance()
        if feature_imp is not None:
            print(f"\n【因子重要性TOP10】")
            print(feature_imp.head(10).to_string())

    # 保存报告
    report_path = OUTPUT_DIR / "backtest_report.txt"
    analyzer.generate_report(metrics, report_path)

    # 绘图
    analyzer.plot_equity_curve(metrics)
    analyzer.plot_monthly_returns(results["portfolio_returns"])

    # IC 分析
    ic_summary = bt.combiner.get_ic_summary()
    if not ic_summary.empty:
        print("\n【因子 IC 分析】")
        print(ic_summary.to_string(index=False))
        ic_summary.to_csv(OUTPUT_DIR / "ic_summary.csv", index=False)

    # 保存持仓历史
    holdings_df = pd.DataFrame([
        {"date": h["date"], "n_holdings": h["n_holdings"],
         "turnover": h.get("turnover", 0),
         "top5": str(dict(list(h["holdings"].items())[:5]))}
        for h in results["holdings_history"]
    ])
    holdings_df.to_csv(OUTPUT_DIR / "holdings_history.csv", index=False)

    logger.info(f"\n所有结果已保存至: {OUTPUT_DIR}")
    return metrics


def run_screen(codes=None):
    """实时选股模式: 基于最新数据筛选(基础因子)"""
    print("\n" + "=" * 60)
    print("  多因子选股量化框架 v6.0 - 实时选股模式")
    print("=" * 60 + "\n")

    # 获取实时行情
    logger.info("获取实时行情...")
    realtime = get_realtime_quotes(codes)
    if realtime.empty:
        logger.error("无法获取实时行情")
        return None

    print(f"当前股票池: {len(realtime)} 只\n")

    # 实时选股(基于实时数据简化版因子)
    factors_rt = pd.DataFrame(index=realtime["code"])

    # 实时价值因子
    if "pe_ttm" in realtime.columns:
        factors_rt["ep"] = 1 / realtime.set_index("code")["pe_ttm"].replace(0, np.nan)
    if "pb" in realtime.columns:
        factors_rt["bp"] = 1 / realtime.set_index("code")["pb"].replace(0, np.nan)

    # 实时动量(用涨跌幅近似)
    if "pct_chg" in realtime.columns:
        factors_rt["mom_1d"] = realtime.set_index("code")["pct_chg"]

    # 实时流动性
    if "amount" in realtime.columns:
        factors_rt["amount"] = realtime.set_index("code")["amount"]
    if "turnover" in realtime.columns:
        factors_rt["turnover"] = -realtime.set_index("code")["turnover"]

    # 预处理
    preprocessor = FactorPreprocessor()
    processed = preprocessor.process(factors_rt)

    # 合成
    combiner = FactorCombiner(method="equal_weight")
    scores = combiner.combine(processed)

    # 选股
    selector = StockSelector()
    weights = selector.select(scores)

    # 输出结果
    selected = weights[weights > 0].sort_values(ascending=False)
    result_df = pd.DataFrame({
        "code": selected.index,
        "weight": selected.values,
    })
    result_df = result_df.merge(
        realtime[["code", "name", "close", "pct_chg", "amount"]],
        on="code", how="left"
    )

    print("\n【实时选股结果】")
    print(f"筛选出 {len(result_df)} 只股票:\n")
    print(result_df.to_string(index=False))

    # 保存
    output_file = OUTPUT_DIR / f"screen_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
    result_df.to_csv(output_file, index=False)
    logger.info(f"\n选股结果已保存: {output_file}")

    return result_df


def run_recommend(codes=None, top_n=30):
    """
    17维因子实时推荐模式(生产级)
    融合 v5.0 模型的17维因子体系和 v6.0 的学术化预处理框架
    """
    print("\n" + "=" * 60)
    print("  多因子选股量化框架 v6.0 - 17维因子实时推荐")
    print("=" * 60)

    # 打印因子权重
    print("\n  【17维因子权重配置】")
    for dim, weight in sorted(WEIGHTS.items(), key=lambda x: -x[1]):
        bar = "█" * int(weight * 100)
        print(f"  {dim:25s} {weight*100:5.1f}% {bar}")
    print("=" * 60 + "\n")

    try:
        from recommender import RealtimeRecommender
        recommender = RealtimeRecommender(codes)
        recommendations = recommender.recommend(top_n=top_n)

        if not recommendations:
            logger.error("推荐失败,请检查数据获取是否正常")
            return None

        # 生成报告
        report = recommender.generate_report(recommendations)
        print(report)

        # 保存结果
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        json_file = OUTPUT_DIR / f"recommend_{timestamp}.json"
        csv_file = OUTPUT_DIR / f"recommend_{timestamp}.csv"

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, ensure_ascii=False, indent=2, default=str)

        # CSV 格式
        rows = []
        for r in recommendations:
            row = {
                'rank': r.get('rank'),
                'code': r.get('code'),
                'name': r.get('name'),
                'price': r.get('price'),
                'total_score': r.get('total_score'),
                'risk_level': r.get('risk_assessment', {}).get('level', ''),
                'stop_loss': r.get('stop_loss'),
            }
            for dim in WEIGHTS:
                row[dim] = r.get('factor_scores', {}).get(dim, 0)
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv_file, index=False, encoding='utf-8-sig')

        logger.info(f"\n推荐结果已保存:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  CSV:  {csv_file}")

        return recommendations
    except Exception as e:
        logger.error(f"推荐引擎运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_market_analysis():
    """大盘市场分析(择时+板块+资金流)"""
    print("\n" + "=" * 60)
    print("  多因子选股量化框架 v6.0 - 市场分析")
    print("=" * 60 + "\n")

    # 大盘总览
    overview = get_market_overview()
    if overview:
        print("【大盘总览】")
        for name, data in overview.items():
            print(f"  {name}: {data['price']:.2f}  {data['change_pct']:+.2f}%  成交额: {data['amount']/1e8:.0f}亿")
        print()

    # 板块热度
    sector = get_sector_heatmap()
    if sector:
        print("【行业板块涨幅TOP5】")
        for s in sector['top5']:
            print(f"  {s['name']:12s} {s['pct']:+.2f}%")
        print("\n【行业板块跌幅TOP5】")
        for s in sector['bottom5']:
            print(f"  {s['name']:12s} {s['pct']:+.2f}%")
        print()

    # 北向资金
    nb = get_northbound_flow()
    if nb:
        print(f"【北向资金】 {nb['latest_date']}  净{nb['trend']}: {nb['latest_net']/1e8:.2f}亿")
        print()

    # 大盘择时信号
    try:
        from recommender import RealtimeRecommender
        recommender = RealtimeRecommender()
        timing = recommender._assess_market_timing()
        if timing:
            print(f"【大盘择时信号】")
            print(f"  趋势: {'多头( bullish)' if timing.get('bullish') else '空头(bearish)'}")
            print(f"  评分: {timing.get('score', 0):.1f}/100")
            if timing.get('ma20'):
                print(f"  上证MA20: {timing['ma20']:.2f}")
            if timing.get('ma60'):
                print(f"  上证MA60: {timing['ma60']:.2f}")
            if timing.get('rsi'):
                print(f"  RSI(14): {timing['rsi']:.1f}")
            print(f"  MACD: {timing.get('macd_signal', 'N/A')}")
    except Exception as e:
        logger.debug(f"择时信号获取失败: {e}")

    print("\n" + "=" * 60)


def run_regime_analysis():
    """市场状态分析模式 v6.1"""
    print("\n" + "=" * 60)
    print("  多因子选股量化框架 v6.1 - 市场状态分析")
    print("=" * 60 + "\n")

    try:
        from market_regime import MarketRegimeDetector, MarketRegime
        detector = MarketRegimeDetector()

        # 获取上证指数数据
        index_data = get_benchmark_data("sh", "20200101", "20260728")
        if index_data is None or index_data.empty:
            logger.error("无法获取指数数据")
            return

        result = detector.detect(index_data)

        print(f"【市场状态判别】")
        print(f"  当前状态:   {result['regime_name'].upper()}")
        print(f"  信心度:     {result['confidence']:.0%}")
        print(f"  目标仓位:   {result['target_position']:.0%}")
        print(f"\n  状态得分:")
        for state, score in result['scores'].items():
            bar = "█" * int(score * 30)
            print(f"    {state:12s} {score:.0%} {bar}")

        print(f"\n  关键指标:")
        for name, value in result.get('indicators', {}).items():
            if isinstance(value, float):
                print(f"    {name:25s}: {value:.4f}")

        # 动态权重
        dynamic_weights = detector.get_dynamic_weights(WEIGHTS)
        print(f"\n  动态因子权重(基于当前市场状态):")
        for dim, weight in sorted(dynamic_weights.items(), key=lambda x: -x[1])[:10]:
            bar = "█" * int(weight * 100)
            print(f"    {dim:25s} {weight*100:5.1f}% {bar}")

    except Exception as e:
        logger.error(f"市场状态分析失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


def run_risk_assessment(codes=None):
    """风险评估模式 v6.1"""
    print("\n" + "=" * 60)
    print("  多因子选股量化框架 v6.1 - 风险评估")
    print("=" * 60 + "\n")

    try:
        from risk_manager import AdvancedRiskManager
        risk_mgr = AdvancedRiskManager()

        # 获取当前推荐组合
        from recommender import RealtimeRecommender
        recommender = RealtimeRecommender(codes)
        recommendations = recommender.recommend(top_n=30)

        if recommendations is None or recommendations.empty:
            logger.error("无法获取推荐组合")
            return

        # 构建权重
        weights = pd.Series(
            1.0 / len(recommendations),
            index=recommendations["code"].tolist()
        )

        # 获取历史收益
        index_data = get_benchmark_data("sh", "20250101", "20260728")

        print(f"【组合风险评估】")
        print(f"  持仓数量:   {len(weights)}")
        print(f"  最大权重:   {weights.max():.2%}")
        print(f"  集中度HHI:  {(weights**2).sum():.4f}")
        print(f"  总仓位:     {weights.sum():.0%}")

        # VaR / CVaR
        if index_data is not None and not index_data.empty:
            returns = index_data["close"].pct_change().dropna()
            var_95 = risk_mgr.calculate_var(returns)
            cvar_95 = risk_mgr.calculate_cvar(returns)
            print(f"\n  VaR(95%):   {var_95:.2%}")
            print(f"  CVaR(95%):  {cvar_95:.2%}")

        # Kelly 仓位
        kelly = risk_mgr.kelly_criterion(win_rate=0.55, win_loss_ratio=1.5)
        print(f"\n  Kelly仓位(半Kelly):  {kelly:.2%}")
        print(f"  波动率目标:          {risk_mgr.vol_target:.1%}")

        # 动态止损示例
        if len(recommendations) > 0:
            top_stock = recommendations.iloc[0]
            stop = risk_mgr.dynamic_stop_loss(
                entry_price=top_stock.get("close", 10),
                current_price=top_stock.get("close", 10),
                atr=top_stock.get("close", 10) * 0.03,
                trend_strength="up",
                holding_days=0
            )
            print(f"\n  TOP1动态止损:")
            print(f"    止损价:   {stop['stop_price']}")
            print(f"    止损幅度: {stop['stop_pct']:.2f}%")
            print(f"    止损类型: {stop['stop_type']}")

        print(f"\n  压力测试:")
        print(f"    (需要历史收益数据,当前仅展示框架)")

    except Exception as e:
        logger.error(f"风险评估失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


def run_alpha_mining(codes=None, top_n=20):
    """v6.2新增: Alpha因子挖掘模式"""
    print("\n" + "=" * 60)
    print("  多因子选股量化框架 v6.2 - Alpha因子挖掘")
    print("=" * 60 + "\n")

    try:
        from alpha_factor_mining import AlphaFactorMiner, FactorDecayTracker
        miner = AlphaFactorMiner()
        
        # 获取数据
        if codes is None:
            stock_list = get_stock_list()
            codes = stock_list["code"].tolist()[:200]  # 挖掘用200只
        
        panel = get_panel_data(codes, "20230101", "20260728")
        if panel.empty:
            # 使用模拟数据
            from data_generator import generate_sample_panel, generate_sample_financial
            panel = generate_sample_panel(n_days=300)
            financial = generate_sample_financial()
        else:
            financial = get_batch_financial(codes)
        
        # 计算基础因子
        td = panel["date"].max()
        base_factors = FactorCalculator().compute_all_factors(panel, financial, td.strftime("%Y%m%d"))
        
        # 计算forward_returns
        fwd = panel[panel["date"] > td].groupby("code")["pct_chg"].apply(
            lambda x: (1 + x / 100).prod() - 1
        )
        
        if base_factors.empty or fwd.empty:
            print("数据不足,无法挖掘因子")
            return
        
        # 挖掘
        new_factors = miner.mine_factors(base_factors, fwd)
        report = miner.get_factor_report()
        
        print(f"\n【挖掘结果】")
        print(f"  基础因子数:   {base_factors.shape[1]}")
        print(f"  新增有效因子: {new_factors.shape[1]}")
        print(f"\n{report.to_string(index=False)}")
        
    except Exception as e:
        logger.error(f"Alpha因子挖掘失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


def run_auto_tune(codes=None):
    """v6.2新增: 自动调参模式"""
    print("\n" + "=" * 60)
    print("  多因子选股量化框架 v6.2 - 自动调参")
    print("=" * 60 + "\n")

    try:
        from auto_tuner import GridSearchTuner, BayesianTuner, DEFAULT_PARAM_SPECS, save_best_params
        
        def evaluate_fn(params):
            """简化的评估函数: 用模拟数据快速评估"""
            try:
                from data_generator import generate_sample_panel, generate_sample_financial
                panel = generate_sample_panel(n_days=300)
                financial = generate_sample_financial()
                
                codes_list = [c for c, _, _ in [
                    ("600519", "贵州茅台", "食品饮料"),
                    ("000858", "五粮液", "食品饮料"),
                    ("600036", "招商银行", "银行"),
                    ("000333", "美的集团", "家用电器"),
                    ("600276", "恒瑞医药", "医药生物"),
                ]]
                
                bt = Backtester(
                    start_date="2022-01-01",
                    end_date="2023-12-31",
                    use_ml=False,
                    use_cost_model=True,
                    use_regime=True,
                    use_risk_manager=True,
                )
                bt.selector = StockSelector(
                    top_n=params.get("TOP_N", 50),
                    max_weight=params.get("MAX_WEIGHT", 0.05),
                    max_industry_weight=params.get("MAX_INDUSTRY_WEIGHT", 0.30),
                    max_turnover=params.get("MAX_TURNOVER", 0.50),
                )
                
                results = bt.run(panel, financial, codes_list)
                if not results or results["portfolio_returns"] is None:
                    return {"sharpe_ratio": 0.0, "annual_return": 0.0, "max_drawdown": -1.0}
                
                analyzer = PerformanceAnalyzer()
                metrics = analyzer.analyze(
                    results["portfolio_returns"],
                    results.get("benchmark_returns")
                )
                return metrics or {"sharpe_ratio": 0.0}
            except Exception:
                return {"sharpe_ratio": 0.0}
        
        print("【网格搜索】")
        grid_tuner = GridSearchTuner(DEFAULT_PARAM_SPECS, metric="sharpe_ratio")
        grid_best = grid_tuner.search(evaluate_fn, n_trials=30)
        print(f"  最优参数: {grid_best['params']}")
        print(f"  最优Sharpe: {grid_best['score']:.4f}")
        
        print("\n【贝叶斯优化】")
        bayes_tuner = BayesianTuner(DEFAULT_PARAM_SPECS, metric="sharpe_ratio")
        bayes_best = bayes_tuner.search(evaluate_fn, n_trials=20)
        print(f"  最优参数: {bayes_best['params']}")
        print(f"  最优Sharpe: {bayes_best['score']:.4f}")
        
        # 保存最优参数
        best = grid_best if grid_best["score"] >= bayes_best["score"] else bayes_best
        save_best_params(best, str(OUTPUT_DIR / "best_params.json"))
        print(f"\n最优参数已保存至: {OUTPUT_DIR / 'best_params.json'}")
        
    except Exception as e:
        logger.error(f"自动调参失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


def run_backtest_v7(codes=None, start=None, end=None, use_ml=False, walk_forward=False):
    """v7.0增强回测: 37因子 + 增强绩效 + 因子归因 + Walk-Forward"""
    from backtester_v7 import EnhancedBacktester
    from data_generator import generate_sample_panel, generate_sample_financial, generate_sample_industry

    start = start or BACKTEST_START
    end = end or BACKTEST_END

    print("\n" + "=" * 60)
    print(f"  多因子选股量化框架 v7.0 - 增强回测{'(Walk-Forward)' if walk_forward else ''}")
    print("=" * 60)
    print(f"  回测区间:   {start} ~ {end}")
    print(f"  因子数量:   37 (27基础 + 10高级Alpha)")
    print(f"  ML合成:     {'启用' if use_ml else '否'}")
    print(f"  Walk-Forward: {'启用' if walk_forward else '否'}")
    print(f"  增强绩效:   IR/TE/Treynor/Jensen/Bootstrap/因子归因")
    print("=" * 60 + "\n")

    # 获取数据
    if codes is None:
        from data_generator import generate_sample_panel, generate_sample_financial, generate_sample_industry
        panel = generate_sample_panel(n_days=500)
        financial = generate_sample_financial()
        industry = generate_sample_industry()
        codes = panel["code"].unique().tolist()
        logger.info(f"使用模拟数据: {len(codes)} 只股票")
    else:
        panel = get_panel_data(codes, start.replace("-", ""), end.replace("-", ""))
        financial = get_batch_financial(codes)
        industry = get_stock_industry(codes)

    if panel.empty:
        logger.error("无法获取行情数据")
        return None

    # 创建增强回测器
    bt = EnhancedBacktester(
        start_date=start, end_date=end,
        use_ml=use_ml, use_cost_model=True,
        use_regime=True, use_risk_manager=True,
    )

    results = bt.run(panel, financial, codes, industry)

    if not results:
        logger.error("回测失败")
        return None

    # 增强绩效分析
    enhanced_metrics = results.get("enhanced_metrics", {})
    if enhanced_metrics:
        report = bt.generate_enhanced_report(enhanced_metrics)
        print(report)

    # 因子归因
    factor_attr = results.get("factor_attribution", {})
    if factor_attr:
        print(f"\n【因子归因分析】")
        top_contrib = factor_attr.get("top_contributors", [])
        if top_contrib:
            print(f"  收益贡献TOP5因子:")
            for item in top_contrib[:5]:
                print(f"    {item['factor']:20s} 贡献: {item['contribution']:.4f}")
        top_drag = factor_attr.get("top_draggers", [])
        if top_drag:
            print(f"  收益拖累TOP5因子:")
            for item in top_drag[:5]:
                print(f"    {item['factor']:20s} 贡献: {item['contribution']:.4f}")

    # Walk-Forward验证
    if walk_forward:
        print(f"\n【Walk-Forward 验证】")
        wf_results = bt.run_walk_forward(panel, financial, codes, n_windows=4)
        if wf_results:
            print(f"  样本外Sharpe:   {wf_results.get('oof_sharpe', 0):.4f}")
            print(f"  样本外年化收益: {wf_results.get('oof_annual_return', 0):.2%}")
            print(f"  样本外最大回撤: {wf_results.get('oof_max_drawdown', 0):.2%}")
            print(f"  过拟合差距:     {wf_results.get('overfit_gap', 0):.4f}")
            window_details = wf_results.get("window_details", [])
            for i, w in enumerate(window_details):
                print(f"  窗口{i+1}: train_IC={w.get('train_ic', 0):.4f} test_IC={w.get('test_ic', 0):.4f}")

    # 保存结果
    report_path = OUTPUT_DIR / "backtest_report_v7.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(bt.generate_enhanced_report(enhanced_metrics) if enhanced_metrics else "无增强指标")
    logger.info(f"\nv7.0增强报告已保存至: {report_path}")

    return enhanced_metrics


def run_recommend_v7(codes=None, top_n=30, use_ml=False):
    """v7.0增强推荐: ML融合 + 实时风险 + 板块轮动 + 多时间框架 + 智能止损"""
    print("\n" + "=" * 60)
    print("  多因子选股量化框架 v7.0 - 增强推荐引擎")
    print("=" * 60)
    print(f"  ML预测融合:   {'启用' if use_ml else '否'}")
    print(f"  实时风险监控: 启用(VaR/CVaR/行业暴露)")
    print(f"  板块轮动信号: 启用")
    print(f"  多时间框架:   短期(5d)/中期(60d)/长期(120d)")
    print(f"  智能止损:     ATR+支撑位+时间止损")
    print("=" * 60)

    try:
        from recommender_v7 import EnhancedRecommender

        if codes is None:
            # 默认使用模拟股票池
            from data_generator import SAMPLE_STOCKS
            codes = [s[0] for s in SAMPLE_STOCKS]

        recommender = EnhancedRecommender(codes)
        recommendations = recommender.recommend(top_n=top_n, use_ml=use_ml)

        if recommendations is None or recommendations.empty:
            logger.error("推荐失败")
            return None

        # 生成报告
        report = recommender.generate_report(recommendations)
        print(report)

        # 保存
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        csv_file = OUTPUT_DIR / f"recommend_v7_{timestamp}.csv"
        json_file = OUTPUT_DIR / f"recommend_v7_{timestamp}.json"

        recommendations.to_csv(csv_file, index=False, encoding='utf-8-sig')

        # JSON格式(含额外信息)
        result_dict = {
            "timestamp": timestamp,
            "n_recommendations": len(recommendations),
            "recommendations": recommendations.to_dict('records'),
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"\n推荐结果已保存:")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  JSON: {json_file}")

        return recommendations

    except Exception as e:
        logger.error(f"v7.0推荐引擎运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_factor_evaluation(codes=None):
    """v7.0新增: 因子有效性评估"""
    print("\n" + "=" * 60)
    print("  多因子选股量化框架 v7.0 - 因子有效性评估")
    print("=" * 60 + "\n")

    try:
        from factor_registry import FactorRegistry, FactorEvaluator, UNIFIED_WEIGHTS
        from factors_v7 import ExtendedFactorCalculator

        # 获取数据
        if codes is None:
            from data_generator import generate_sample_panel, generate_sample_financial
            panel = generate_sample_panel(n_days=500)
            financial = generate_sample_financial()
            codes = panel["code"].unique().tolist()
        else:
            panel = get_panel_data(codes, "20220101", "20260728")
            financial = get_batch_financial(codes)

        if panel.empty:
            logger.error("无法获取数据")
            return

        # 计算因子
        td = panel["date"].max()
        calc = ExtendedFactorCalculator()
        factors = calc.compute_all_factors(panel, financial, td.strftime("%Y%m%d"))

        # 计算forward returns
        fwd = panel[panel["date"] > td].groupby("code")["pct_chg"].apply(
            lambda x: (1 + x / 100).prod() - 1
        )

        if factors.empty or fwd.empty:
            print("数据不足")
            return

        # 评估所有因子
        evaluator = FactorEvaluator()
        eval_results = evaluator.evaluate_all(factors, fwd)

        print(f"【因子有效性评估结果】")
        print(f"  评估因子数: {len(eval_results)}")
        print(f"\n{eval_results.to_string(index=False)}")

        # 因子注册表统计
        registry = FactorRegistry()
        print(f"\n【因子注册表统计】")
        print(f"  总因子数: {len(registry.get_all_factor_names())}")
        for source in ["backtest", "extended", "recommend"]:
            factors_by_source = registry.get_factors_by_source(source)
            print(f"  {source:12s}: {len(factors_by_source)} 个")

        print(f"\n【因子映射关系(推荐维度 -> 回测因子)】")
        mapping = registry.get_mapping()
        for dim, factor_list in sorted(mapping.items()):
            print(f"  {dim:25s} -> {factor_list}")

        # 保存
        eval_path = OUTPUT_DIR / "factor_evaluation_v7.csv"
        eval_results.to_csv(eval_path, index=False)
        logger.info(f"\n评估结果已保存: {eval_path}")

    except Exception as e:
        logger.error(f"因子评估失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


# ============================================================
#  v7.1 新增功能入口
# ============================================================

def run_backtest_v7(codes=None, start=None, end=None, use_ml=False,
                    walk_forward=False, use_prediction=True):
    """
    v7.1: 增强回测 — 使用ExtendedFactorCalculator + PredictionEngine

    参数
    ----------
    codes : list, optional
        股票池
    start : str, optional
        起始日期
    end : str, optional
        结束日期
    use_ml : bool
        是否启用ML合成
    walk_forward : bool
        是否运行Walk-Forward验证
    use_prediction : bool
        是否启用v7.1预测引擎
    """
    start = start or BACKTEST_START
    end = end or BACKTEST_END

    version_str = "v7.1 (预测引擎)" if use_prediction else "v7.0"
    print("\n" + "=" * 70)
    print(f"  多因子选股量化框架 {version_str} - 增强回测模式")
    print("=" * 70)
    print(f"  回测区间:   {start} ~ {end}")
    print(f"  调仓频率:   {REBALANCE_FREQ}")
    print(f"  基准:       {BENCHMARK}")
    print(f"  ML合成:     {'启用' if use_ml else '关闭'}")
    print(f"  预测引擎:   {'启用' if use_prediction else '关闭'}")
    print(f"  Walk-Forward: {'启用' if walk_forward else '关闭'}")
    print("=" * 70 + "\n")

    try:
        from backtester_v7 import EnhancedBacktester
        from data_loader import get_panel_data, get_batch_financial, get_stock_industry

        # 1. 获取股票池
        if codes is None:
            logger.info("步骤 1/5: 获取全A股股票池...")
            stock_list = get_stock_list()
            codes = stock_list["code"].tolist()
        else:
            stock_list = None

        # 2. 获取历史数据
        logger.info("步骤 2/5: 下载历史行情数据...")
        panel = get_panel_data(codes, start.replace("-", ""), end.replace("-", ""))
        if panel.empty:
            logger.error("无法获取行情数据")
            return None

        # 3. 获取财务数据
        logger.info("步骤 3/5: 下载财务数据...")
        financial = get_batch_financial(codes)

        # 4. 获取行业归属
        logger.info("步骤 4/5: 获取行业归属...")
        industry = get_stock_industry(codes)

        # 5. 回测
        logger.info(f"步骤 5/5: 执行v7.1增强回测...")
        bt = EnhancedBacktester(
            start, end, REBALANCE_FREQ, BENCHMARK,
            use_ml=use_ml, use_cost_model=True,
            use_regime=True, use_risk_manager=True,
        )
        results = bt.run(panel, financial, codes, industry, stock_list)

        if not results:
            logger.error("回测失败")
            return None

        # 输出增强绩效
        if results.get("enhanced_metrics"):
            metrics = results["enhanced_metrics"]
            print(f"\n{'=' * 50}")
            print(f"  v7.1 增强绩效分析")
            print(f"{'=' * 50}")
            print(f"  年化收益:     {metrics.get('annual_return', 0):.2%}")
            print(f"  夏普比率:     {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  最大回撤:     {metrics.get('max_drawdown', 0):.2%}")
            print(f"  信息比率(IR): {metrics.get('information_ratio', 0):.3f}")
            print(f"  跟踪误差:     {metrics.get('tracking_error', 0):.2%}")
            print(f"  Treynor比率:  {metrics.get('treynor_ratio', 0):.3f}")
            print(f"  Jensen Alpha: {metrics.get('jensen_alpha', 0):.2%}")

            if metrics.get("bootstrap_ci"):
                ci = metrics["bootstrap_ci"]
                print(f"\n  Bootstrap 95% CI:")
                print(f"    夏普: [{ci.get('sharpe_lower', 0):.3f}, {ci.get('sharpe_upper', 0):.3f}]")

        # 因子归因
        if results.get("factor_attribution"):
            attr = results["factor_attribution"]
            print(f"\n{'=' * 50}")
            print(f"  因子归因分析")
            print(f"{'=' * 50}")
            print(f"  R²:           {attr.get('r_squared', 0):.4f}")
            print(f"  因子解释收益: {attr.get('total_factor_return', 0):.2%}")
            print(f"  残差收益:     {attr.get('residual', 0):.2%}")

            top = attr.get("top_contributors", [])
            if top:
                print(f"\n  主要贡献因子:")
                for name, contrib in top[:5]:
                    print(f"    {name:25s}: {contrib:+.4%}")

        # 预测引擎状态
        if use_prediction and hasattr(bt, 'prediction_engine'):
            pe = bt.prediction_engine
            report = pe.get_prediction_report()
            print(f"\n{'=' * 50}")
            print(f"  v7.1 预测引擎状态")
            print(f"{'=' * 50}")
            print(f"  更新次数:     {report.get('n_updates', 0)}")
            print(f"  预测次数:     {report.get('n_predictions', 0)}")
            print(f"  IC历史因子数: {report.get('n_factors_tracked', 0)}")

            if report.get("momentum_weights"):
                print(f"\n  因子动量权重 (Top 5):")
                sorted_w = sorted(report["momentum_weights"].items(),
                                  key=lambda x: x[1], reverse=True)
                for name, w in sorted_w[:5]:
                    print(f"    {name:25s}: {w:.4f}")

            if report.get("factor_premium"):
                print(f"\n  因子风险溢价 (Top 5):")
                sorted_p = sorted(report["factor_premium"].items(),
                                  key=lambda x: abs(x[1]), reverse=True)
                for name, p in sorted_p[:5]:
                    print(f"    {name:25s}: {p:+.6f}")

        # Walk-Forward
        if walk_forward:
            print(f"\n{'=' * 50}")
            print(f"  Walk-Forward 验证")
            print(f"{'=' * 50}")
            wf_results = bt.run_walk_forward(
                panel, financial, codes, n_windows=5, train_ratio=0.6,
                industry=industry, stock_list=stock_list,
            )
            if wf_results:
                print(f"  样本外夏普:   {wf_results.get('oos_sharpe', 0):.3f}")
                print(f"  样本外年化:   {wf_results.get('oos_annual_return', 0):.2%}")
                print(f"  样本外最大回撤: {wf_results.get('oos_max_drawdown', 0):.2%}")
                print(f"  稳定性得分:   {wf_results.get('stability_score', 0):.3f}")

        # 保存报告
        report = bt.generate_enhanced_report()
        if report:
            report_path = OUTPUT_DIR / "backtest_v71_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n报告已保存: {report_path}")

    except Exception as e:
        logger.error(f"v7.1回测失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)


def run_recommend_v7(codes=None, top_n=30, use_ml=False, use_prediction=False):
    """
    v7.1: 增强实时推荐 — 使用EnhancedRecommender + 向量化批处理

    参数
    ----------
    codes : list, optional
        股票池
    top_n : int
        推荐数量
    use_ml : bool
        是否启用ML融合
    use_prediction : bool
        是否启用预测引擎
    """
    print("\n" + "=" * 70)
    print(f"  多因子选股量化框架 v7.1 - 增强推荐模式")
    print("=" * 70)
    print(f"  推荐数量:   {top_n}")
    print(f"  ML融合:     {'启用' if use_ml else '关闭'}")
    print(f"  预测引擎:   {'启用' if use_prediction else '关闭'}")
    print("=" * 70 + "\n")

    try:
        from recommender_v7 import EnhancedRecommender

        # 获取股票池
        if codes is None:
            logger.info("获取全A股股票池...")
            stock_list = get_stock_list()
            # 过滤: 排除ST、上市不足60日、市值过小
            codes = stock_list[stock_list.get("market_cap", 0) > 5e8]["code"].tolist()
            codes = codes[:500]  # 限制500只避免过慢

        logger.info(f"股票池: {len(codes)} 只")

        # 初始化推荐引擎
        ml_combiner = None
        if use_ml:
            try:
                from ml_pipeline import EnhancedMLCombiner
                ml_combiner = EnhancedMLCombiner(
                    method="enhanced_ensemble",
                    enable_feature_selection=True,
                    enable_walk_forward=False,  # 推荐时不需要Walk-Forward
                    enable_version_control=False,
                )
                logger.info("EnhancedMLCombiner 已加载")
            except Exception as e:
                logger.warning(f"ML合成器加载失败: {e}")

        recommender = EnhancedRecommender(
            universe=codes,
            ml_combiner=ml_combiner,
        )

        # 执行推荐
        results = recommender.recommend(top_n=top_n, use_ml=use_ml)

        if results is None or results.empty:
            logger.error("推荐失败")
            return None

        # 打印结果
        print(f"\n{'=' * 100}")
        print(f"  v7.1 增强推荐结果 (Top {len(results)})")
        print(f"{'=' * 100}")

        # 显示主要列
        display_cols = ["code", "name", "total_score", "signal_strength",
                       "risk_level", "stop_loss_price", "stop_loss_type",
                       "sector_rotation"]
        available_cols = [c for c in display_cols if c in results.columns]
        print(results[available_cols].to_string(index=False))

        # 板块轮动信息
        if recommender._sector_rotation_cache:
            sr = recommender._sector_rotation_cache
            print(f"\n{'=' * 50}")
            print(f"  板块轮动信号")
            print(f"{'=' * 50}")
            print(f"  轮动方向: {sr.get('rotation_direction', '中性')}")
            strong = sr.get("strong_sectors", [])
            if strong:
                print(f"  强势板块: {', '.join(s['name'] for s in strong[:5])}")
            weak = sr.get("weak_sectors", [])
            if weak:
                print(f"  弱势板块: {', '.join(s['name'] for s in weak[:5])}")

        # 组合风险
        if recommender._portfolio_risk_cache:
            pr = recommender._portfolio_risk_cache
            print(f"\n{'=' * 50}")
            print(f"  组合风险评估")
            print(f"{'=' * 50}")
            print(f"  组合VaR(1日): {pr.get('portfolio_var', 0):.2%}")
            print(f"  组合CVaR:     {pr.get('portfolio_cvar', 0):.2%}")
            print(f"  最大行业暴露: {pr.get('max_industry', '未知')} "
                  f"({pr.get('max_industry_exposure', 0):.2%})")
            if pr.get("drawdown_alert"):
                print(f"  [预警] 组合回撤风险较高!")

        # 保存结果
        output_path = OUTPUT_DIR / "recommend_v71.csv"
        results.to_csv(output_path, index=False)
        print(f"\n推荐结果已保存: {output_path}")

        # 生成文本报告
        text_report = recommender.generate_enhanced_report(results)
        report_path = OUTPUT_DIR / "recommend_v71_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text_report)
        print(f"文本报告已保存: {report_path}")

    except Exception as e:
        logger.error(f"v7.1推荐失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)


def run_factor_evaluation(codes=None):
    """
    v7.1: 因子有效性评估 — 使用FactorRegistry + FactorEvaluator

    评估全部37个因子的IC、ICIR、单调性、换手率等指标
    """
    print("\n" + "=" * 70)
    print(f"  多因子选股量化框架 v7.1 - 因子有效性评估")
    print("=" * 70 + "\n")

    try:
        from factor_registry import FactorRegistry, FactorEvaluator
        from factors_v7 import ExtendedFactorCalculator, get_extended_factor_config

        # 获取股票池
        if codes is None:
            stock_list = get_stock_list()
            codes = stock_list["code"].tolist()[:200]

        # 获取数据
        panel = get_panel_data(codes, BACKTEST_START.replace("-", ""),
                               BACKTEST_END.replace("-", ""))
        financial = get_batch_financial(codes)

        if panel.empty:
            logger.error("无法获取行情数据")
            return

        # 初始化
        calc = ExtendedFactorCalculator()
        registry = FactorRegistry()
        evaluator = FactorEvaluator()

        # 获取调仓日
        rebalance_dates = get_rebalance_dates(
            BACKTEST_START, BACKTEST_END, REBALANCE_FREQ
        )

        logger.info(f"评估 {len(rebalance_dates)} 个调仓日的因子有效性...")

        # 逐期计算因子IC
        all_ics = {}
        for rb_date in rebalance_dates:
            factors = calc.compute_all_factors(panel, financial, rb_date)
            if factors.empty:
                continue

            factors = registry.align_directions(factors)

            # 计算forward returns
            idx = rebalance_dates.index(rb_date)
            next_rb = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else None
            if next_rb is None:
                continue

            fwd_ret = _compute_forward_returns_simple(panel, rb_date, next_rb)
            if fwd_ret is None or fwd_ret.empty:
                continue

            for col in factors.columns:
                try:
                    from scipy.stats import spearmanr
                    common = factors[col].dropna().index.intersection(fwd_ret.dropna().index)
                    if len(common) < 20:
                        continue
                    ic, _ = spearmanr(factors.loc[common, col], fwd_ret.loc[common])
                    if not np.isnan(ic):
                        all_ics.setdefault(col, []).append(ic)
                except Exception:
                    continue

        # 汇总
        print(f"\n{'=' * 80}")
        print(f"  因子有效性评估结果 ({len(all_ics)} 个因子)")
        print(f"{'=' * 80}")
        print(f"{'因子名':<25} {'IC均值':>10} {'IC标准差':>10} {'ICIR':>10} {'IC>0占比':>10}")
        print("-" * 80)

        eval_results = []
        for name, ic_list in sorted(all_ics.items()):
            ic_mean = np.mean(ic_list)
            ic_std = np.std(ic_list)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            pos_ratio = np.mean(np.array(ic_list) > 0)

            print(f"{name:<25} {ic_mean:>10.4f} {ic_std:>10.4f} {icir:>10.3f} {pos_ratio:>10.1%}")

            eval_results.append({
                "factor": name,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "ic_positive_ratio": pos_ratio,
                "n_periods": len(ic_list),
            })

        # 保存
        eval_df = pd.DataFrame(eval_results)
        eval_path = OUTPUT_DIR / "factor_evaluation_v71.csv"
        eval_df.to_csv(eval_path, index=False)
        print(f"\n评估结果已保存: {eval_path}")

    except Exception as e:
        logger.error(f"因子评估失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)


def run_prediction_evaluation(codes=None):
    """
    v7.1: 预测引擎评估 — 单独评估PredictionEngine的预测能力

    测试因子动量、截面回归、IC衰减三种预测方法的效果
    """
    print("\n" + "=" * 70)
    print(f"  多因子选股量化框架 v7.1 - 预测引擎评估")
    print("=" * 70 + "\n")

    try:
        from prediction_engine import (
            PredictionEngine, FactorMomentumTracker,
            CrossSectionalRegressor, ICDecayAnalyzer,
        )
        from factors_v7 import ExtendedFactorCalculator

        # 获取股票池
        if codes is None:
            stock_list = get_stock_list()
            codes = stock_list["code"].tolist()[:200]

        # 获取数据
        panel = get_panel_data(codes, BACKTEST_START.replace("-", ""),
                               BACKTEST_END.replace("-", ""))
        financial = get_batch_financial(codes)

        if panel.empty:
            logger.error("无法获取行情数据")
            return

        # 初始化
        calc = ExtendedFactorCalculator()
        engine = PredictionEngine(
            momentum_weight=0.30,
            regression_weight=0.40,
            ic_weight=0.30,
        )

        # 获取调仓日
        rebalance_dates = get_rebalance_dates(
            BACKTEST_START, BACKTEST_END, REBALANCE_FREQ
        )

        logger.info(f"评估 {len(rebalance_dates)} 个调仓日的预测能力...")

        # 逐期更新和预测
        predictions = []
        actuals = []

        for i, rb_date in enumerate(rebalance_dates):
            factors = calc.compute_all_factors(panel, financial, rb_date)
            if factors.empty:
                continue

            # 预测(使用历史数据)
            if engine.is_ready:
                pred = engine.predict(factors)
                if not pred.empty:
                    predictions.append((rb_date, pred))

            # 计算forward returns
            next_rb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
            if next_rb is None:
                continue

            fwd_ret = _compute_forward_returns_simple(panel, rb_date, next_rb)
            if fwd_ret is not None and not fwd_ret.empty:
                actuals.append((rb_date, fwd_ret))
                # 更新引擎
                engine.update(factors, fwd_ret)

        # 评估预测能力
        if len(predictions) < 3:
            logger.error("预测数据不足,无法评估")
            return

        print(f"\n{'=' * 60}")
        print(f"  预测引擎评估结果")
        print(f"{'=' * 60}")

        # 计算预测IC
        pred_ics = []
        for (pred_date, pred_scores), (act_date, act_returns) in zip(predictions, actuals):
            if pred_date != act_date:
                continue
            common = pred_scores.index.intersection(act_returns.dropna().index)
            if len(common) < 20:
                continue
            from scipy.stats import spearmanr
            ic, _ = spearmanr(pred_scores.loc[common], act_returns.loc[common])
            if not np.isnan(ic):
                pred_ics.append(ic)

        if pred_ics:
            ic_mean = np.mean(pred_ics)
            ic_std = np.std(pred_ics)
            icir = ic_mean / ic_std if ic_std > 0 else 0

            print(f"  预测IC均值:   {ic_mean:.4f}")
            print(f"  预测IC标准差: {ic_std:.4f}")
            print(f"  预测ICIR:     {icir:.3f}")
            print(f"  IC>0占比:     {np.mean(np.array(pred_ics) > 0):.1%}")
            print(f"  评估期数:     {len(pred_ics)}")

        # 预测引擎报告
        report = engine.get_prediction_report()
        print(f"\n  更新次数:     {report.get('n_updates', 0)}")
        print(f"  预测次数:     {report.get('n_predictions', 0)}")
        print(f"  追踪因子数:   {report.get('n_factors_tracked', 0)}")

        if report.get("momentum_weights"):
            print(f"\n  因子动量权重 (Top 5):")
            for name, w in sorted(report["momentum_weights"].items(),
                                  key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {name:25s}: {w:.4f}")

        if report.get("factor_premium"):
            print(f"\n  因子风险溢价 (Top 5):")
            for name, p in sorted(report["factor_premium"].items(),
                                  key=lambda x: abs(x[1]), reverse=True)[:5]:
                print(f"    {name:25s}: {p:+.6f}")

        # 保存结果
        results = {
            "pred_ic_mean": ic_mean if pred_ics else 0,
            "pred_ic_std": ic_std if pred_ics else 0,
            "pred_icir": icir if pred_ics else 0,
            "n_periods": len(pred_ics),
            "pred_ics": pred_ics,
        }
        import json
        eval_path = OUTPUT_DIR / "prediction_evaluation_v71.json"
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n评估结果已保存: {eval_path}")

    except Exception as e:
        logger.error(f"预测引擎评估失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)


def _compute_forward_returns_simple(panel, rb_date, next_rb):
    """简化版forward returns计算(供评估使用)"""
    try:
        td = pd.to_datetime(rb_date, format="%Y%m%d")
        ntd = pd.to_datetime(next_rb, format="%Y%m%d")

        before = panel[panel["date"] <= td].groupby("code").last()
        after = panel[panel["date"] <= ntd].groupby("code").last()

        common = before.index.intersection(after.index)
        if len(common) < 20:
            return None

        fwd_ret = (after.loc[common, "close"] / before.loc[common, "close"] - 1)
        fwd_ret = fwd_ret.dropna()
        return fwd_ret
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="多因子选股量化框架 v7.1")
    parser.add_argument("--mode", choices=["backtest", "backtest-ml", "backtest-v7",
                                           "backtest-v71",
                                           "screen", "recommend", "recommend-v7",
                                           "recommend-v71",
                                           "market", "regime", "risk", "mine", "tune",
                                           "factor-eval", "predict-eval", "all"],
                       default="backtest-v71", help="运行模式")
    parser.add_argument("--codes", type=str, default=None,
                       help="指定股票池(逗号分隔)")
    parser.add_argument("--start", type=str, default=None, help="回测起始日期")
    parser.add_argument("--end", type=str, default=None, help="回测结束日期")
    parser.add_argument("--recommend-n", type=int, default=30,
                       help="推荐股票数量(默认30)")
    parser.add_argument("--ml", action="store_true", help="启用ML因子合成")
    parser.add_argument("--no-regime", action="store_true", help="禁用市场状态检测")
    parser.add_argument("--no-risk", action="store_true", help="禁用高级风控")
    args = parser.parse_args()

    codes = args.codes.split(",") if args.codes else None

    if args.mode == "backtest":
        run_backtest(codes, args.start, args.end, use_ml=args.ml)

    elif args.mode == "backtest-ml":
        run_backtest(codes, args.start, args.end, use_ml=True)

    elif args.mode == "screen":
        run_screen(codes)

    elif args.mode == "recommend":
        run_recommend(codes, args.recommend_n)

    elif args.mode == "market":
        run_market_analysis()

    elif args.mode == "regime":
        run_regime_analysis()

    elif args.mode == "risk":
        run_risk_assessment(codes)

    elif args.mode == "mine":
        run_alpha_mining(codes)

    elif args.mode == "tune":
        run_auto_tune(codes)

    elif args.mode == "backtest-v7":
        run_backtest_v7(codes, args.start, args.end, use_ml=args.ml,
                        walk_forward=False)

    elif args.mode == "backtest-v71":
        run_backtest_v7(codes, args.start, args.end, use_ml=args.ml,
                        walk_forward=False, use_prediction=True)

    elif args.mode == "recommend-v7":
        run_recommend_v7(codes, args.recommend_n, use_ml=args.ml)

    elif args.mode == "recommend-v71":
        run_recommend_v7(codes, args.recommend_n, use_ml=args.ml,
                         use_prediction=True)

    elif args.mode == "factor-eval":
        run_factor_evaluation(codes)

    elif args.mode == "predict-eval":
        run_prediction_evaluation(codes)

    elif args.mode == "all":
        run_backtest(codes, args.start, args.end, use_ml=args.ml)
        run_screen(codes)
        run_recommend(codes, args.recommend_n)
        run_market_analysis()
        run_regime_analysis()
        run_alpha_mining(codes)


if __name__ == "__main__":
    main()
