# 多因子选股量化框架

一个面向全 A 股的多因子选股量化交易框架,支持因子计算、因子合成、风险控制、历史回测与实时选股。

## 特性

- **20 个因子**:覆盖价值(EP/BP/SP/CFP)、成长(营收/利润/EPS增长)、动量(20/60/120日)、反转(5/20日)、质量(ROE/ROA/毛利率/资产负债率)、波动率、流动性
- **因子预处理**:MAD去极值 + 分位数截断 + Z-Score标准化 + 行业市值中性化
- **3 种合成方法**:等权、IC加权、截面回归
- **风险控制**:行业暴露限制、单股权重上限、换手率控制、个股止损、回撤预警
- **回测引擎**:按月/季/周调仓,组合收益与基准对比
- **绩效分析**:年化收益、最大回撤、夏普比率、索提诺比率、卡玛比率、Alpha/Beta、胜率、盈亏比
- **可视化**:净值曲线、回撤曲线、月度收益热力图
- **实时选股**:基于实时行情快速筛选
- **磁盘缓存**:避免重复下载,提升运行速度
- **离线模式**:内置数据生成器(GARCH波动率模型 + 行业Beta),无网络环境下也能完整演示

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行回测

```bash
# 完整回测(全A股,2020-2026)
python main.py --mode backtest

# 指定区间
python main.py --mode backtest --start 2022-01-01 --end 2026-07-25

# 指定股票池
python main.py --mode backtest --codes 000001,600519,000858
```

### 3. 实时选股

```bash
# 基于最新行情筛选
python main.py --mode screen

# 全流程(回测+选股)
python main.py --mode all
```

### 4. 离线模式(无网络环境)

```bash
# 使用内置数据生成器,生成模拟A股数据后运行回测
python -c "
from data_generator import generate_sample_panel, generate_sample_financial, generate_sample_industry
# 生成数据后会自动缓存,之后 main.py 可直接使用
"
```

## 项目结构

```
quant_factor_model/
├── main.py              # 主入口
├── config.py            # 全局配置(因子参数、回测参数、风控参数)
├── utils.py             # 工具函数(缓存、日志、日期)
├── data_loader.py       # 数据获取层(akshare + 磁盘缓存)
├── data_generator.py    # 离线数据生成器(GARCH模型,无网络时使用)
├── factors.py           # 因子计算层(20个因子)
├── preprocessing.py     # 因子预处理(去极值/标准化/中性化)
├── factor_combiner.py   # 因子合成(等权/IC/回归)
├── selector.py          # 选股与风控
├── backtester.py        # 回测引擎
├── performance.py       # 绩效分析与可视化
├── requirements.txt     # 依赖
├── data_cache/          # 数据缓存(自动生成)
└── output/              # 回测结果(自动生成)
```

## 配置说明

编辑 `config.py` 调整策略参数:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BACKTEST_START` | 2020-01-01 | 回测起始日期 |
| `BACKTEST_END` | 2026-07-25 | 回测结束日期 |
| `REBALANCE_FREQ` | M | 调仓频率(M月/Q季/W周) |
| `TOP_N` | 50 | 持仓股票数量 |
| `MAX_WEIGHT` | 0.05 | 单股最大权重 5% |
| `MAX_INDUSTRY_WEIGHT` | 0.30 | 单行业最大权重 30% |
| `MAX_TURNOVER` | 0.50 | 单次最大换手率 50% |
| `FACTOR_COMBINE_METHOD` | ic_weight | 因子合成方法 |
| `STOP_LOSS_THRESHOLD` | -0.10 | 个股止损线 -10% |
| `DRAWDOWN_ALERT` | 0.15 | 组合回撤预警 15% |

## 因子列表

| 类别 | 因子 | 方向 |
|------|------|------|
| 价值 | EP, BP, SP, CFP | 正向(越大越好) |
| 成长 | 营收增长, 利润增长, EPS增长 | 正向 |
| 动量 | 20日/60日/120日动量 | 正向 |
| 反转 | 5日/20日反转 | 反向 |
| 质量 | ROE, ROA, 毛利率 | 正向 |
| 质量 | 资产负债率 | 反向 |
| 波动率 | 20日/60日波动率 | 反向(低波动溢价) |
| 流动性 | 20日换手率 | 反向 |
| 流动性 | 20日成交额 | 正向 |

## 输出文件

运行后 `output/` 目录包含:

- `backtest_report.txt` — 绩效报告(文本)
- `equity_curve.png` — 净值曲线与回撤图
- `monthly_returns.png` — 月度收益热力图
- `ic_summary.csv` — 因子IC/IR分析
- `holdings_history.csv` — 历次调仓持仓
- `screen_result_*.csv` — 实时选股结果

## 风险提示

- 数据来自公开第三方(东方财富/新浪),约有 15 分钟延迟
- 回测结果不代表未来表现,存在过拟合风险
- 本框架仅供研究学习使用,不构成任何投资建议
- 投资有风险,入市需谨慎
