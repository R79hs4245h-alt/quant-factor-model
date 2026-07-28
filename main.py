"""
多因子选股量化框架 - 主入口
用法:
    python main.py                    # 完整回测
    python main.py --mode screen      # 实时选股(基于最新数据)
    python main.py --mode backtest    # 仅回测
    python main.py --codes 000001,600519  # 指定股票池
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from config import (
    BACKTEST_START, BACKTEST_END, REBALANCE_FREQ, BENCHMARK,
    TOP_N, OUTPUT_DIR, CACHE_DIR, FACTOR_COMBINE_METHOD
)
from utils import get_logger, get_rebalance_dates
from data_loader import (
    get_stock_list, get_panel_data, get_batch_financial,
    get_realtime_quotes, get_stock_industry, get_benchmark_data
)
from factors import FactorCalculator
from preprocessing import FactorPreprocessor
from factor_combiner import FactorCombiner
from selector import StockSelector
from backtester import Backtester
from performance import PerformanceAnalyzer

logger = get_logger("main")


def run_backtest(codes=None, start=None, end=None):
    """运行完整回测"""
    start = start or BACKTEST_START
    end = end or BACKTEST_END

    print("\n" + "=" * 60)
    print("  多因子选股量化框架 - 回测模式")
    print("=" * 60)
    print(f"  回测区间:   {start} ~ {end}")
    print(f"  调仓频率:   {REBALANCE_FREQ}")
    print(f"  基准:       {BENCHMARK}")
    print(f"  持仓数量:   {TOP_N}")
    print(f"  合成方法:   {FACTOR_COMBINE_METHOD}")
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
    logger.info("步骤 5/5: 执行回测...")
    bt = Backtester(start, end, REBALANCE_FREQ, BENCHMARK)
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
         "top5": str(dict(list(h["holdings"].items())[:5]))}
        for h in results["holdings_history"]
    ])
    holdings_df.to_csv(OUTPUT_DIR / "holdings_history.csv", index=False)

    logger.info(f"\n所有结果已保存至: {OUTPUT_DIR}")
    return metrics


def run_screen(codes=None):
    """实时选股模式: 基于最新数据筛选"""
    print("\n" + "=" * 60)
    print("  多因子选股量化框架 - 实时选股模式")
    print("=" * 60 + "\n")

    # 获取实时行情
    logger.info("获取实时行情...")
    realtime = get_realtime_quotes(codes)
    if realtime.empty:
        logger.error("无法获取实时行情")
        return None

    print(f"当前股票池: {len(realtime)} 只\n")

    # 实时选股(基于实时数据简化版因子)
    # 这里用实时指标做快速筛选
    from config import TOP_N
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
        factors_rt["turnover"] = -realtime.set_index("code")["turnover"]  # 低换手溢价

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


import numpy as np  # 用于 run_screen 中的 np.nan


def main():
    parser = argparse.ArgumentParser(description="多因子选股量化框架")
    parser.add_argument("--mode", choices=["backtest", "screen", "all"],
                       default="backtest", help="运行模式")
    parser.add_argument("--codes", type=str, default=None,
                       help="指定股票池(逗号分隔)")
    parser.add_argument("--start", type=str, default=None, help="回测起始日期")
    parser.add_argument("--end", type=str, default=None, help="回测结束日期")
    args = parser.parse_args()

    codes = args.codes.split(",") if args.codes else None

    if args.mode in ("backtest", "all"):
        run_backtest(codes, args.start, args.end)

    if args.mode in ("screen", "all"):
        run_screen(codes)


if __name__ == "__main__":
    main()
