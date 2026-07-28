"""
多因子选股量化框架 v7.2 - 全局配置
整合17维因子体系、多源数据容错、大盘择时、组合风控
v7.2: 推荐标的限定为场内基金(ETF+LOF),不再推荐场外基金
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
BACKTEST_END = "2026-07-28"        # 回测结束日期
REBALANCE_FREQ = "M"              # 调仓频率: M=月度, Q=季度, W=周
BENCHMARK = "hs300"               # 基准指数: hs300/zz500/cyb/sh

# ============ 选股参数 ============
TOP_N = 50                        # 持仓股票数量
MAX_WEIGHT = 0.05                 # 单股最大权重 5%
MAX_INDUSTRY_WEIGHT = 0.30        # 单行业最大权重 30%
MAX_TURNOVER = 0.50               # 单次调仓最大换手率 50%
MAX_SECTOR_STOCKS = 3             # 实时推荐: 单板块最多选入股票数

# ============ 股票池过滤 ============
EXCLUDE_ST = True                 # 排除 ST/*ST 股票
EXCLUDE_NEW_DAYS = 60             # 排除上市不足 60 天的次新股
MIN_LIST_DAYS = 252               # 最少上市交易日(用于计算因子)
MIN_MARKET_CAP = 5e8              # 最小市值 5 亿(排除超小盘)
MIN_AMOUNT = 1e6                  # 实时推荐: 最小日成交额 100万
EXCLUDE_BOARD = []                # 排除板块: 可选 ["科创","创业板"]

# ============ 基础因子配置(回测用) ============
# 因子名称: direction=1 越大越好, direction=-1 越小越好
FACTOR_CONFIG = {
    # --- 价值因子 ---
    "ep":       {"direction": 1,  "category": "value",     "desc": "市盈率倒数 EP"},
    "bp":       {"direction": 1,  "category": "value",     "desc": "市净率倒数 BP"},
    "sp":       {"direction": 1,  "category": "value",     "desc": "市销率倒数 SP"},
    "cfp":      {"direction": 1,  "category": "value",     "desc": "现金流收益率"},
    # --- 成长因子 ---
    "revenue_growth":  {"direction": 1, "category": "growth", "desc": "营收同比增长率"},
    "profit_growth":   {"direction": 1, "category": "growth", "desc": "净利润同比增长率"},
    "eps_growth":      {"direction": 1, "category": "growth", "desc": "EPS同比增长率"},
    # --- 动量因子 ---
    "mom_20d":  {"direction": 1,  "category": "momentum",  "desc": "20日动量"},
    "mom_60d":  {"direction": 1,  "category": "momentum",  "desc": "60日动量"},
    "mom_120d": {"direction": 1,  "category": "momentum",  "desc": "120日动量"},
    # --- 反转因子 ---
    "reversal_5d":  {"direction": -1, "category": "reversal", "desc": "5日反转"},
    "reversal_20d": {"direction": -1, "category": "reversal", "desc": "20日反转"},
    # --- 质量因子 ---
    "roe":          {"direction": 1, "category": "quality", "desc": "净资产收益率 ROE"},
    "roa":          {"direction": 1, "category": "quality", "desc": "总资产收益率 ROA"},
    "gross_margin": {"direction": 1, "category": "quality", "desc": "毛利率"},
    "debt_ratio":   {"direction": -1,"category": "quality", "desc": "资产负债率(越低越好)"},
    # --- 波动率因子(低波动溢价) ---
    "vol_20d":  {"direction": -1, "category": "volatility", "desc": "20日波动率"},
    "vol_60d":  {"direction": -1, "category": "volatility", "desc": "60日波动率"},
    # --- 流动性因子 ---
    "turnover_20d": {"direction": -1, "category": "liquidity", "desc": "20日平均换手率(低流动性溢价)"},
    "amount_20d":   {"direction": 1,  "category": "liquidity", "desc": "20日平均成交额"},
    # --- 扩展因子(v6.0新增) ---
    "mom_3m":   {"direction": 1,  "category": "momentum",  "desc": "3月动量"},
    "mom_6m":   {"direction": 1,  "category": "momentum",  "desc": "6月动量"},
    "rps":      {"direction": 1,  "category": "momentum",  "desc": "相对强度RPS"},
    "amihud":   {"direction": -1, "category": "liquidity", "desc": "Amihud非流动性比率"},
    "atr_pct":  {"direction": -1, "category": "volatility", "desc": "ATR波动率百分比"},
    "up_ratio": {"direction": 1,  "category": "quality",   "desc": "上涨日占比"},
    "beta":     {"direction": 0,  "category": "quality",   "desc": "Beta系数(中性)"},
}

# ============ 17维推荐因子权重(实时推荐用) ============
# 总和100, 融合 v5.0 生产级因子体系
WEIGHTS = {
    'value':               0.07,   # 价值 7分(含PEG/股息率)
    'growth':              0.05,   # 成长 5分
    'quality':             0.06,   # 质量 6分(含Beta/Piotroski F-Score)
    'fund_flow':           0.14,   # 资金流 14分(含龙虎榜)
    'technical':           0.09,   # 技术形态 9分(含MFI/威廉指标/OBV/RSI)
    'sector':              0.03,   # 板块热度 3分
    'sentiment':           0.04,   # 情绪事件 4分
    'fundamental':         0.08,   # 基本面 8分(ROE/毛利率/资产负债率/营收增速)
    'timing':              0.03,   # 大盘择时 3分(MA60/MACD/RSI)
    'risk':                0.09,   # 风险因子 9分(融资融券/解禁/股东人数/商誉/现金流/控盘度)
    'momentum':            0.07,   # 动量/反转 7分(3/6月动量+RPS+短期反转)
    'liquidity':           0.04,   # 流动性 4分(Amihud+成交额+换手率)
    'chip_distribution':   0.07,   # 筹码分布 7分(获利盘/集中度/筹码峰/低位密集) NEW v6.0
    'trend_strength':      0.05,   # 趋势强度 5分(ADX/+DI/-DI/趋势持续性) NEW v6.0
    'volume_price':        0.05,   # 量价背离 5分(MACD背离/量价配合/OBV一致性) NEW v6.0
    'industry_prosperity': 0.04,   # 行业景气度 4分(行业排名/资金流向/估值分位) NEW v6.0
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
RT_CACHE_SECONDS = 300       # 实时数据内存缓存 5分钟

# ============ 组合风控 ============
STOP_LOSS_THRESHOLD = -0.10  # 个股止损线 -10%
DRAWDOWN_ALERT = 0.15       # 组合最大回撤预警 15%

# ============ 大盘择时 ============
MARKET_TIMING_ENABLED = True  # 启用大盘择时(熊市减仓/空仓)
TIMING_MA_SHORT = 20          # 短期均线
TIMING_MA_LONG = 60           # 长期均线
TIMING_RSI_PERIOD = 14        # RSI周期
BEAR_MARKET_MAX_POSITION = 0.30  # 熊市最大仓位 30%

# ============ 数据源 ============
DATA_SOURCE = "akshare"            # 主数据源
REALTIME_SOURCE = "eastmoney_direct"  # 实时行情源
BACKUP_SOURCE_ENABLED = True        # 启用多源备份(腾讯/新浪)
DATA_FETCH_TIMEOUT = 8              # 数据获取超时(秒)
DATA_FETCH_RETRIES = 2              # 失败重试次数

# ============ 热门题材关键词(情绪因子用) ============
HOT_THEMES = [
    "AI", "人工智能", "芯片", "半导体", "新能源", "光伏", "储能",
    "军工", "机器人", "数据要素", "算力", "华为", "自动驾驶",
    "创新药", "CXO", "消费复苏", "国产替代", "信创",
]

# ============ 场内基金配置 v7.2 ============
# 推荐标的限定为场内基金(ETF+LOF),不再推荐场外基金
# 场内基金优势: 可在交易所实时买卖,费率低,透明度高,流动性好
FUND_TYPE = "exchange_traded"  # exchange_traded=场内基金(ETF+LOF), open_ended=场外基金
EXCLUDE_OTC_FUNDS = True       # 排除场外基金(仅保留ETF+LOF)

# 场内基金筛选条件
ETF_MIN_AMOUNT = 5e6           # ETF最小日成交额 500万(流动性筛选)
LOF_MIN_AMOUNT = 1e6           # LOF最小日成交额 100万
ETF_MIN_SCALE = 1e8            # ETF最小规模 1亿
LOF_MIN_SCALE = 5e7            # LOF最小规模 5000万

# 核心宽基ETF代码池(用于基准对比和推荐)
CORE_BROAD_ETF = {
    "510300": "沪深300ETF",
    "510050": "上证50ETF",
    "159915": "创业板ETF",
    "512100": "中证1000ETF",
    "588000": "科创50ETF",
    "512500": "中证500ETF",
    "588200": "科创芯片ETF",
    "159901": "深100ETF",
    "510310": "沪深300ETF易方达",
    "510500": "中证500ETF南方",
}

# 主题/行业ETF代码池(覆盖主要赛道)
THEME_ETF = {
    # 科技/AI
    "515000": "科技ETF",
    "515880": "通信ETF",
    "512480": "半导体ETF",
    "159995": "芯片ETF",
    "515790": "光伏ETF",
    "515030": "新能源车ETF",
    # 消费/医药
    "512690": "酒ETF",
    "510630": "消费ETF",
    "512170": "医疗ETF",
    "512010": "医药ETF",
    "159992": "创新药ETF",
    "515170": "食品饮料ETF",
    "515650": "消费50ETF",
    # 金融
    "512800": "银行ETF",
    "512000": "券商ETF",
    "560670": "银行ETF华泰柏瑞",
    # 军工/制造
    "512660": "军工ETF",
    "159798": "消费ETF易方达",
    "159512": "汽车ETF",
    # 地产/周期
    "159707": "地产ETF",
    "516900": "食品饮料ETF华安",
    # 宽基/其他
    "159533": "中证2000ETF",
    "159515": "国企红利ETF",
}

# LOF基金代码池(高流动性LOF)
CORE_LOF = {
    "161725": "白酒基金LOF",
    "160632": "酒LOF",
    "161831": "恒生国企LOF",
    "501090": "消费龙头LOF",
    "160631": "银行LOF基金",
    "501089": "消费红利增强LOF",
    "161121": "银行LOF易方达",
    "160717": "H股LOF",
    "160218": "房地产LOF",
    "160222": "食品LOF",
    "501087": "交银瑞丰LOF",
    "160613": "鹏华盛世创新LOF",
    "161029": "银行龙头LOF",
    "501093": "华夏翔阳LOF",
    "160918": "中小盘LOF",
    "160629": "传媒LOF",
    "161729": "招商瑞利LOF",
    "160910": "创新成长LOF",
    "160628": "地产LOF",
}

# 场内基金主题分类
ETF_THEME_MAP = {
    "科技AI": ["515000", "515880", "512480", "159995"],
    "半导体芯片": ["512480", "159995", "588200"],
    "消费医药": ["512690", "510630", "512170", "512010", "159992", "515170", "515650"],
    "新能源": ["515790", "515030"],
    "金融地产": ["512800", "512000", "560670", "159707", "160631", "161121", "161029"],
    "宽基指数": ["510300", "510050", "159915", "512100", "588000", "512500", "159533"],
    "港股相关": ["161831", "160717"],
    "消费白酒": ["161725", "160632", "510630", "515170", "515650"],
}

# 场内基金推荐数量配置
ETF_RECOMMEND_TOP = 15        # ETF推荐数量
LOF_RECOMMEND_TOP = 10        # LOF推荐数量
TOTAL_EXCHANGE_TRADED = 25    # 场内基金总推荐数

# ============ 四维高阶选股配置 v7.3 ============
# 基于专业短线交易逻辑的四维度选股体系
# 维度一: 估值相对合理 (PEG/历史分位/市值空间)
# 维度二: 触底反弹拐点确认 (底背离/金针探底/倍量阳)
# 维度三: 主力资金真流入 (DDX/大单/龙虎榜)
# 维度四: 主升浪启动检测 (均线粘合/量能递增/筹码峰)
ADVANCED_DIM_WEIGHTS = {
    'base_weight': 0.4,          # 基础层权重(动量/流动性/活跃度/波动/规模)
    'advanced_weight': 0.6,      # 四维高阶层权重
    'valuation': 0.25,           # 估值维度权重
    'bottom_reversal': 0.25,     # 拐点维度权重
    'capital_flow': 0.25,        # 资金维度权重
    'main_rally': 0.25,          # 主升浪维度权重
}

# 四维共振阈值: 维度得分 >= 此值视为该维度触发
RESONANCE_THRESHOLD = 65

# 止损纪律
STOP_LOSS_RULES = {
    'break_ma5': '减仓50%',       # 破5日线减仓
    'break_ma10': '清仓',         # 破10日线清仓
    'top_divergence': '止盈',     # MACD顶背离止盈
    'volume_anomaly': '警惕',     # 放量滞涨警惕
}
