"""
多因子选股量化框架 v6.1 - 主入口
整合回测引擎 + 实时17维因子推荐 + 大盘择时 + ML合成 + 高级风控

v6.1新增:
  - ML因子合成器(随机森林/XGBoost/LightGBM)
  - 市场状态判别器(牛市/熊市/震荡自适应)
  - 高级风险管理器(波动率目标/Kelly/CVaR/压力测试)
  - 交易成本模型(佣金+印花税+滑点)
  - 动态因子权重(随市场状态调整)

用法:
    python main.py                        # 完整回测(含交易成本)
    python main.py --mode screen          # 实时选股(基础因子)
    python main.py --mode recommend       # 17维因子实时推荐(生产级)
    python main.py --mode backtest        # 仅回测
    python main.py --mode backtest-ml     # ML增强回测
    python main.py --mode regime          # 市场状态分析
    python main.py --mode risk            # 风险评估
    python main.py --mode all             # 回测+选股+推荐
    python main.py --codes 000001,600519  # 指定股票池
    python main.py --recommend-n 30       # 推荐N只股票
    python main.py --ml                   # 启用ML合成
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


def main():
    parser = argparse.ArgumentParser(description="多因子选股量化框架 v6.1")
    parser.add_argument("--mode", choices=["backtest", "backtest-ml", "screen", "recommend",
                                           "market", "regime", "risk", "all"],
                       default="backtest", help="运行模式")
    parser.add_argument("--codes", type=str, default=None,
                       help="指定股票池(逗号分隔)")
    parser.add_argument("--start", type=str, default=None, help="回测起始日期")
    parser.add_argument("--end", type=str, default=None, help="回测结束日期")
    parser.add_argument("--recommend-n", type=int, default=30,
                       help="推荐股票数量(默认30)")
    parser.add_argument("--ml", action="store_true", help="启用ML因子合成")
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

    elif args.mode == "all":
        run_backtest(codes, args.start, args.end, use_ml=args.ml)
        run_screen(codes)
        run_recommend(codes, args.recommend_n)
        run_market_analysis()
        run_regime_analysis()


if __name__ == "__main__":
    main()
