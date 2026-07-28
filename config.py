"""
多因子选股量化框架 - 全局配置
"""
import os
from pathlib import Path

# ============ 路径配置 ============
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "data_cache"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ============ 回测参数 ============
BACKTEST_START = "2020-01-01"      # 回测起始日期
BACKTEST_END = "2026-07-25"       # 回测结束日期
REBALANCE_FREQ = "M"              # 调仓频率: M=月度, Q=季度, W=周
BENCHMARK = "hs300"               # 基准指数: hs300/zz500/cyb

# ============ 选股参数 ============
TOP_N = 50                        # 持仓股票数量
MAX_WEIGHT = 0.05                 # 单股最大权重 5%
MAX_INDUSTRY_WEIGHT = 0.30        # 单行业最大权重 30%
MAX_TURNOVER = 0.50               # 单次调仓最大换手率 50%

# ============ 股票池过滤 ============
EXCLUDE_ST = True                 # 排除 ST/*ST 股票
EXCLUDE_NEW_DAYS = 60             # 排除上市不足 60 天的次新股
MIN_LIST_DAYS = 252               # 最少上市交易日(用于计算因子)
MIN_MARKET_CAP = 5e8              # 最小市值 5 亿(排除超小盘)
EXCLUDE_BOARD = []                # 排除板块: 可选 ["科创","创业板"]

# ============ 因子配置 ============
# 因子名称: (方向) direction=1 越大越好, direction=-1 越小越好
FACTOR_CONFIG = {
    # 价值因子
    "ep":       {"direction": 1,  "category": "value",     "desc": "市盈率倒数 EP"},
    "bp":       {"direction": 1,  "category": "value",     "desc": "市净率倒数 BP"},
    "sp":       {"direction": 1,  "category": "value",     "desc": "市销率倒数 SP"},
    "cfp":      {"direction": 1,  "category": "value",     "desc": "现金流收益率"},
    # 成长因子
    "revenue_growth":  {"direction": 1, "category": "growth", "desc": "营收同比增长率"},
    "profit_growth":   {"direction": 1, "category": "growth", "desc": "净利润同比增长率"},
    "eps_growth":      {"direction": 1, "category": "growth", "desc": "EPS同比增长率"},
    # 动量因子
    "mom_20d":  {"direction": 1,  "category": "momentum",  "desc": "20日动量"},
    "mom_60d":  {"direction": 1,  "category": "momentum",  "desc": "60日动量"},
    "mom_120d": {"direction": 1,  "category": "momentum",  "desc": "120日动量"},
    # 反转因子
    "reversal_5d":  {"direction": -1, "category": "reversal", "desc": "5日反转"},
    "reversal_20d": {"direction": -1, "category": "reversal", "desc": "20日反转"},
    # 质量因子
    "roe":          {"direction": 1, "category": "quality", "desc": "净资产收益率 ROE"},
    "roa":          {"direction": 1, "category": "quality", "desc": "总资产收益率 ROA"},
    "gross_margin": {"direction": 1, "category": "quality", "desc": "毛利率"},
    "debt_ratio":   {"direction": -1,"category": "quality", "desc": "资产负债率(越低越好)"},
    # 波动率因子(低波动溢价)
    "vol_20d":  {"direction": -1, "category": "volatility", "desc": "20日波动率"},
    "vol_60d":  {"direction": -1, "category": "volatility", "desc": "60日波动率"},
    # 流动性因子
    "turnover_20d": {"direction": -1, "category": "liquidity", "desc": "20日平均换手率(低流动性溢价)"},
    "amount_20d":   {"direction": 1,  "category": "liquidity", "desc": "20日平均成交额"},
}

# 因子合成方法: equal_weight / ic_weight / regression
FACTOR_COMBINE_METHOD = "ic_weight"

# ============ 因子预处理 ============
WINSORIZE_QUANTILE = 0.025   # 去极值分位数
NEUTRALIZE_INDUSTRY = True   # 行业中性化
NEUTRALIZE_SIZE = True       # 市值中性化

# ============ 缓存配置 ============
CACHE_EXPIRE_HOURS = 12      # 缓存过期时间(小时)
USE_CACHE = True             # 是否使用缓存

# ============ 风险控制 ============
STOP_LOSS_THRESHOLD = -0.10  # 个股止损线 -10%
DRAWDOWN_ALERT = 0.15       # 组合最大回撤预警 15%

# ============ 数据源 ============
DATA_SOURCE = "akshare"      # 数据源
REALTIME_SOURCE = "eastmoney_direct"  # 实时行情源
