"""
离线数据生成器: 生成模拟 A 股面板数据与财务数据
用于无网络环境下的框架演示与测试
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

from utils import get_logger

logger = get_logger("generator")


# 模拟股票池(30 只知名 A 股,覆盖 8 个行业)
SAMPLE_STOCKS = [
    ("600519", "贵州茅台", "食品饮料"),
    ("000858", "五粮液",   "食品饮料"),
    ("600887", "伊利股份", "食品饮料"),
    ("601318", "中国平安", "非银金融"),
    ("600036", "招商银行", "银行"),
    ("000001", "平安银行", "银行"),
    ("601166", "兴业银行", "银行"),
    ("000333", "美的集团", "家用电器"),
    ("000651", "格力电器", "家用电器"),
    ("600276", "恒瑞医药", "医药生物"),
    ("000538", "云南白药", "医药生物"),
    ("300015", "爱尔眼科", "医药生物"),
    ("002415", "海康威视", "电子"),
    ("300750", "宁德时代", "电力设备"),
    ("601012", "隆基绿能", "电力设备"),
    ("002714", "牧原股份", "农林牧渔"),
    ("600585", "海螺水泥", "建筑材料"),
    ("601888", "中国中免", "商贸零售"),
    ("600690", "海尔智家", "家用电器"),
    ("002230", "科大讯飞", "计算机"),
    ("000063", "中兴通讯", "通信"),
    ("600031", "三一重工", "机械设备"),
    ("601633", "长城汽车", "汽车"),
    ("600104", "上汽集团", "汽车"),
    ("601857", "中国石油", "石油石化"),
    ("600028", "中国石化", "石油石化"),
    ("601088", "中国神华", "煤炭"),
    ("600900", "长江电力", "公用事业"),
    ("601398", "工商银行", "银行"),
    ("601939", "建设银行", "银行"),
]


def generate_sample_panel(
    n_days: int = 500,
    start_date: str = "2022-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成模拟行情面板数据
    使用 GBM + 波动率聚类 + 行业beta,模拟真实市场
    """
    np.random.seed(seed)

    dates = pd.bdate_range(start_date, periods=n_days)
    stocks = SAMPLE_STOCKS

    # 行业 beta (相对市场)
    industry_beta = {
        "食品饮料": 0.85, "非银金融": 1.25, "银行": 0.90,
        "家用电器": 0.95, "医药生物": 0.80, "电子": 1.15,
        "电力设备": 1.30, "农林牧渔": 1.10, "建筑材料": 1.05,
        "商贸零售": 1.00, "计算机": 1.20, "通信": 1.10,
        "机械设备": 1.15, "汽车": 1.10, "石油石化": 0.75,
        "煤炭": 1.00, "公用事业": 0.60,
    }

    # 生成市场收益(带趋势和波动)
    market_drift = np.random.normal(0.03 / 252, 0.001)  # 年化约3%
    market_vol_base = 0.012
    market_returns = np.zeros(n_days)
    for t in range(1, n_days):
        # 波动率聚类 (GARCH-like)
        vol = market_vol_base * (1 + 0.3 * abs(market_returns[t-1]) / market_vol_base)
        market_returns[t] = np.random.normal(market_drift, vol)

    rows = []
    for code, name, industry in stocks:
        beta = industry_beta.get(industry, 1.0)
        # 个股特质参数
        stock_drift = np.random.normal(0, 0.0005)
        base_vol = np.random.uniform(0.015, 0.035)
        base_price = np.random.uniform(8, 120)
        shares = np.random.uniform(1e8, 5e9)

        # 生成价格路径
        prices = np.zeros(n_days)
        prices[0] = base_price
        volatilities = np.zeros(n_days)
        volatilities[0] = base_vol

        for t in range(1, n_days):
            # GARCH(1,1) 波动率
            volatilities[t] = np.sqrt(
                0.02 + 0.08 * (prices[t-1] / prices[max(t-2, 0)] - 1)**2 +
                0.90 * volatilities[t-1]**2
            )
            volatilities[t] = max(volatilities[t], 0.005)

            # 收益 = 市场分量(beta) + 个股分量
            stock_ret = beta * market_returns[t] + \
                        np.random.normal(stock_drift, volatilities[t])
            prices[t] = prices[t-1] * (1 + stock_ret)

        # 生成交易量(与波动率正相关)
        base_volume = np.random.uniform(5e5, 5e7)
        volumes = base_volume * (1 + 2 * np.abs(np.diff(prices, prepend=prices[0]) / prices)) * np.random.uniform(0.7, 1.3, n_days)

        for t in range(n_days):
            pct_chg = (prices[t] / prices[t-1] - 1) * 100 if t > 0 else 0
            amount = volumes[t] * prices[t]
            turnover = (volumes[t] / shares) * 100

            rows.append({
                "date": dates[t],
                "code": code,
                "open": prices[t] * np.random.uniform(0.995, 1.005),
                "close": prices[t],
                "high": prices[t] * np.random.uniform(1.0, 1.02),
                "low": prices[t] * np.random.uniform(0.98, 1.0),
                "volume": volumes[t],
                "amount": amount,
                "turnover": turnover,
                "pct_chg": pct_chg,
            })

    panel = pd.DataFrame(rows)
    logger.info(f"生成面板数据: {panel.shape}, "
                f"{len(stocks)} 只股票, {n_days} 个交易日")
    return panel


def generate_sample_financial(n_periods: int = 8, seed: int = 42) -> pd.DataFrame:
    """生成模拟财务数据"""
    np.random.seed(seed + 1)

    stocks = SAMPLE_STOCKS
    rows = []
    start = datetime(2020, 1, 1)

    for code, name, industry in stocks:
        # 基础财务参数
        revenue_base = np.random.uniform(1e9, 5e10)
        profit_margin = np.random.uniform(0.05, 0.35)
        asset_base = revenue_base * np.random.uniform(2, 8)
        debt_ratio_base = np.random.uniform(0.25, 0.65)

        prev_revenue = revenue_base
        prev_profit = revenue_base * profit_margin
        prev_eps = np.random.uniform(0.3, 3.0)

        for q in range(n_periods):
            qdate = start + timedelta(days=90 * q)

            # 季度增长
            growth = np.random.uniform(-0.1, 0.25)
            revenue = prev_revenue * (1 + growth)
            profit = revenue * profit_margin * np.random.uniform(0.8, 1.2)
            net_assets = asset_base * (1 + 0.03 * q)
            cashflow = profit * np.random.uniform(0.5, 1.5)
            eps = prev_eps * (1 + growth * 0.5)
            debt_ratio = debt_ratio_base + np.random.uniform(-0.05, 0.05)

            rows.append({
                "date": qdate,
                "code": code,
                "净利润": float(profit),
                "净资产": float(net_assets),
                "营业收入": float(revenue),
                "经营现金流": float(cashflow),
                "净资产收益率(%)": float(profit / net_assets * 100)
                               if net_assets > 0 else 0.0,
                "总资产报酬率(%)": float(profit / (asset_base * (1 + 0.03 * q)) * 100),
                "销售毛利率(%)": float(np.random.uniform(15, 65)),
                "资产负债率(%)": float(debt_ratio * 100),
                "基本每股收益": float(eps),
            })

            prev_revenue = revenue
            prev_profit = profit
            prev_eps = eps

    financial = pd.DataFrame(rows)
    logger.info(f"生成财务数据: {financial.shape}, "
                f"{len(stocks)} 只股票, {n_periods} 期")
    return financial


def generate_sample_industry() -> pd.Series:
    """生成行业归属"""
    data = {code: industry for code, _, industry in SAMPLE_STOCKS}
    return pd.Series(data, name="industry")


def generate_sample_benchmark(
    n_days: int = 500,
    start_date: str = "2022-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """生成模拟基准(沪深300)数据"""
    np.random.seed(seed + 2)

    dates = pd.bdate_range(start_date, periods=n_days)
    base = 4000.0
    prices = [base]

    for t in range(1, n_days):
        ret = np.random.normal(0.02 / 252, 0.011)
        prices.append(prices[-1] * (1 + ret))

    df = pd.DataFrame({
        "date": dates,
        "close": prices,
        "open": [p * np.random.uniform(0.998, 1.002) for p in prices],
        "high": [p * np.random.uniform(1.0, 1.008) for p in prices],
        "low": [p * np.random.uniform(0.992, 1.0) for p in prices],
        "volume": [np.random.uniform(1e8, 5e8) for _ in prices],
    })
    logger.info(f"生成基准数据: {df.shape}")
    return df
