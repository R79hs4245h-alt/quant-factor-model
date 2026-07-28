"""
v7.0 统一因子注册表模块
======================
将回测因子(config.FACTOR_CONFIG, 27个)、扩展因子(factors_v7, 10个)、
推荐维度(config.WEIGHTS, 16个)统一注册到一张注册表中，提供:

  1. FactorRegistry   — 统一因子注册表,支持按类别/来源/映射查询
  2. FactorEvaluator  — 因子有效性评估器(IC/分组收益/IC衰减/批量评估)
  3. UNIFIED_WEIGHTS  — 融合推荐16维权重与回测因子权重的统一配置

导入依赖:
  - from config import FACTOR_CONFIG, WEIGHTS
  - from factors_v7 import get_extended_factor_config
  - from utils import get_logger
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple

from config import FACTOR_CONFIG, WEIGHTS
from factors_v7 import get_extended_factor_config
from utils import get_logger

logger = get_logger("factor_registry")


# ======================================================================
#  1. 推荐维度到回测因子(含扩展因子)的映射关系
# ======================================================================
CATEGORY_FACTOR_MAPPING: Dict[str, List[str]] = {
    "value":     ["ep", "bp", "sp", "cfp"],
    "growth":    ["revenue_growth", "profit_growth", "eps_growth"],
    "quality":   ["roe", "roa", "gross_margin", "debt_ratio",
                  "f_score", "altman_z", "up_ratio"],
    "momentum":  ["mom_20d", "mom_60d", "mom_120d",
                  "mom_3m", "mom_6m", "rps", "mom_accel", "high_low_ratio"],
    "reversal":  ["reversal_5d", "reversal_20d"],
    "volatility": ["vol_20d", "vol_60d", "atr_pct", "downside_dev"],
    "liquidity": ["turnover_20d", "amount_20d", "amihud"],
    "risk":      ["beta", "coskew", "downside_dev", "altman_z"],
    "earnings_quality": ["sloan_accrual", "sue", "earnings_persist"],
    "technical": ["vwap_dev", "mom_accel", "high_low_ratio"],
    # 以下推荐维度无直接回测因子对应,保留空列表或标记为非回测维度
    "fund_flow":           [],
    "sector":              [],
    "sentiment":           [],
    "fundamental":         [],   # 与 quality/value/growth 重叠,通过上述类别间接映射
    "timing":              [],
    "chip_distribution":   [],
    "trend_strength":      [],
    "volume_price":        [],
    "industry_prosperity": [],
}


# ======================================================================
#  2. FactorRegistry — 统一因子注册表
# ======================================================================
class FactorRegistry:
    """
    统一因子注册表 v7.0
    ==================
    将三大因子来源(回测因子、扩展因子、推荐维度)统一注册,
    每个因子记录 name / direction / category / source / description。

    使用方式:
        registry = FactorRegistry()
        registry.get_factors_by_category("value")
        registry.get_factors_by_source("backtest")
        registry.get_mapping()
    """

    def __init__(self):
        self._factors: Dict[str, dict] = {}
        self._register_all()
        logger.info(f"因子注册完成: 共注册 {len(self._factors)} 个因子")

    # ------------------------------------------------------------------
    #  内部注册方法
    # ------------------------------------------------------------------
    def _register_all(self):
        """依次注册回测因子、扩展因子、推荐维度"""
        self._register_backtest_factors()
        self._register_extended_factors()
        self._register_recommend_dimensions()

    def _register_backtest_factors(self):
        """注册 config.FACTOR_CONFIG 中的回测因子(source='backtest')"""
        for name, cfg in FACTOR_CONFIG.items():
            self._factors[name] = {
                "name":      name,
                "direction": cfg.get("direction", 1),
                "category":  cfg.get("category", "unknown"),
                "source":    "backtest",
                "description": cfg.get("desc", ""),
            }

    def _register_extended_factors(self):
        """注册 factors_v7 中 get_extended_factor_config 的扩展因子(source='extended')"""
        ext_config = get_extended_factor_config()
        for name, cfg in ext_config.items():
            # 若回测因子已同名,跳过(以回测因子为准)
            if name in self._factors:
                logger.debug(f"扩展因子 '{name}' 已在回测因子中存在,跳过重复注册")
                continue
            self._factors[name] = {
                "name":      name,
                "direction": cfg.get("direction", 1),
                "category":  cfg.get("category", "unknown"),
                "source":    "extended",
                "description": cfg.get("desc", ""),
            }

    def _register_recommend_dimensions(self):
        """
        注册 config.WEIGHTS 中的推荐维度(source='recommend')。
        推荐维度不是具体因子,而是因子类别/维度名称,用于实时推荐。
        """
        for dim_name, weight in WEIGHTS.items():
            # 推荐维度可能已作为回测类别名存在于 _factors 中(如 'value', 'momentum' 等)
            # 为避免覆盖,在名称前加前缀 "dim_" 存储
            reg_key = f"dim_{dim_name}"
            self._factors[reg_key] = {
                "name":      dim_name,
                "direction": 1,           # 推荐维度本身无方向,设为中性
                "category":  dim_name,
                "source":    "recommend",
                "description": f"推荐维度 '{dim_name}', 权重={weight:.0%}",
            }

    # ------------------------------------------------------------------
    #  公共查询方法
    # ------------------------------------------------------------------
    def get_factors_by_category(self, category: str) -> List[dict]:
        """
        按类别获取因子列表(仅 backtest 和 extended 来源)。

        :param category: 因子类别名称,如 'value', 'momentum', 'quality' 等
        :return: 匹配类别的因子信息字典列表
        """
        return [
            f for f in self._factors.values()
            if f["category"] == category and f["source"] in ("backtest", "extended")
        ]

    def get_factors_by_source(self, source: str) -> List[dict]:
        """
        按来源获取因子列表。

        :param source: 来源标识,可选 'backtest' / 'extended' / 'recommend'
        :return: 匹配来源的因子信息字典列表
        """
        return [
            f for f in self._factors.values()
            if f["source"] == source
        ]

    def get_factor_directions(self) -> Dict[str, int]:
        """
        获取所有回测/扩展因子的方向映射。

        :return: {因子名: direction} 字典
                 direction=1 表示越大越好, direction=-1 表示越小越好, 0 表示中性
        """
        return {
            f["name"]: f["direction"]
            for f in self._factors.values()
            if f["source"] in ("backtest", "extended")
        }

    def get_mapping(self) -> Dict[str, List[str]]:
        """
        返回推荐维度到回测因子(含扩展因子)的映射关系。

        :return: {维度名: [因子名列表]} 字典。
                 例如: 'value' -> ['ep', 'bp', 'sp', 'cfp']
                       'momentum' -> ['mom_20d', 'mom_60d', ...]
        """
        # 以注册表中的实际因子为准,过滤掉不在注册表中的因子名
        valid_names = set(self.get_factor_directions().keys())
        mapping = {}
        for dim, factors in CATEGORY_FACTOR_MAPPING.items():
            mapped = [f for f in factors if f in valid_names]
            mapping[dim] = mapped
        return mapping

    def get_all_factor_names(self) -> List[str]:
        """
        获取所有已注册因子名称(不含推荐维度)。

        :return: 因子名称列表
        """
        return [
            f["name"] for f in self._factors.values()
            if f["source"] in ("backtest", "extended")
        ]

    def get_categories(self) -> List[str]:
        """
        获取所有因子类别名称。

        :return: 类别名称列表(去重排序)
        """
        cats = set(
            f["category"] for f in self._factors.values()
            if f["source"] in ("backtest", "extended")
        )
        return sorted(cats)

    def __len__(self) -> int:
        return len(self._factors)

    def __repr__(self) -> str:
        n_bt = len(self.get_factors_by_source("backtest"))
        n_ext = len(self.get_factors_by_source("extended"))
        n_rec = len(self.get_factors_by_source("recommend"))
        return (f"FactorRegistry(backtest={n_bt}, extended={n_ext}, "
                f"recommend={n_rec}, total={len(self)})")


# ======================================================================
#  3. FactorEvaluator — 因子有效性评估器
# ======================================================================
class FactorEvaluator:
    """
    因子有效性评估器 v7.0
    ====================
    提供因子有效性分析的核心指标计算:
      - IC (Information Coefficient, Spearman秩相关)
      - IC_IR (IC的信息比率 = mean(IC) / std(IC))
      - Rank IC (因子排名与收益排名的相关系数)
      - 分组收益分析 (按因子值分n组,计算各组平均收益)
      - 多空收益 (最高组 - 最低组)
      - 覆盖率 (非NaN比例)
      - t统计量 (IC显著性的t检验)
      - IC衰减分析 (不同滞后期IC的变化趋势)

    使用方式:
        evaluator = FactorEvaluator()
        result = evaluator.evaluate_single(factor_values, forward_returns)
        results_df = evaluator.evaluate_all(factors_df, forward_returns_series)
    """

    def __init__(self):
        self.logger = get_logger("factor_evaluator")

    # ------------------------------------------------------------------
    #  单因子评估
    # ------------------------------------------------------------------
    def evaluate_single(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
    ) -> Dict:
        """
        对单个因子进行有效性评估。

        :param factor_values: 因子值序列 (index=股票代码)
        :param forward_returns: 对应的持有期收益率序列 (index=股票代码)
        :return: 评估结果字典,包含以下字段:
            - ic: Spearman秩相关系数
            - ic_abs: |IC|
            - ic_pvalue: IC的p值
            - ic_tstat: IC的t统计量
            - rank_ic: Rank IC (因子排名与收益排名的Pearson相关)
            - ic_ir: 信息比率 (需时序数据,此处返回NaN,需用ic_decay分析)
            - coverage: 覆盖率(有效数据占比)
            - group_returns: 各组平均收益字典 {组号: 平均收益}
            - long_short: 多空收益(最高组 - 最低组)
            - long_short_annualized: 多空收益年化
            - n_stocks: 有效股票数
        """
        # 对齐数据
        common_idx = factor_values.dropna().index.intersection(
            forward_returns.dropna().index
        )
        if len(common_idx) < 10:
            self.logger.warning(
                f"有效样本不足(N={len(common_idx)}),评估结果不可靠"
            )
            return {
                "ic": np.nan, "ic_abs": np.nan, "ic_pvalue": np.nan,
                "ic_tstat": np.nan, "rank_ic": np.nan, "ic_ir": np.nan,
                "coverage": len(common_idx) / max(len(factor_values), 1),
                "group_returns": {}, "long_short": np.nan,
                "long_short_annualized": np.nan, "n_stocks": len(common_idx),
            }

        fvals = factor_values.loc[common_idx]
        frets = forward_returns.loc[common_idx]

        # Spearman秩相关 (IC)
        ic, ic_pvalue = stats.spearmanr(fvals, frets)
        ic_abs = abs(ic)

        # t统计量: t = IC * sqrt(N-2) / sqrt(1 - IC^2)
        n = len(common_idx)
        if ic_abs < 1.0:
            ic_tstat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2)
        else:
            ic_tstat = np.inf if ic > 0 else -np.inf

        # Rank IC: 因子排名 与 收益排名 的 Pearson 相关
        fvals_rank = fvals.rank(pct=True)
        frets_rank = frets.rank(pct=True)
        rank_ic = fvals_rank.corr(frets_rank)

        # 分组收益
        group_result = self.group_return_analysis(fvals, frets, n_groups=5)
        group_returns = group_result.get("group_returns", {})
        long_short = group_result.get("long_short", np.nan)

        # 多空年化 (假设月度调仓, *12)
        long_short_annualized = long_short * 12 if not np.isnan(long_short) else np.nan

        coverage = len(common_idx) / max(len(factor_values), 1)

        return {
            "ic": ic,
            "ic_abs": ic_abs,
            "ic_pvalue": ic_pvalue,
            "ic_tstat": ic_tstat,
            "rank_ic": rank_ic,
            "ic_ir": np.nan,  # 单截面无法计算IC_IR,需时序
            "coverage": coverage,
            "group_returns": group_returns,
            "long_short": long_short,
            "long_short_annualized": long_short_annualized,
            "n_stocks": n,
        }

    # ------------------------------------------------------------------
    #  批量评估
    # ------------------------------------------------------------------
    def evaluate_all(
        self,
        factors_df: pd.DataFrame,
        forward_returns: pd.Series,
        sort_by: str = "ic_abs",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        对所有因子进行批量有效性评估,返回排序后的评估表。

        :param factors_df: 因子值 DataFrame (index=股票代码, columns=因子名)
        :param forward_returns: 持有期收益率序列 (index=股票代码)
        :param sort_by: 排序字段,默认 'ic_abs'(按IC绝对值降序)
        :param ascending: 是否升序排列,默认 False(降序)
        :return: 评估结果 DataFrame,每行一个因子,列包含所有评估指标
        """
        results = []
        for col in factors_df.columns:
            try:
                eval_result = self.evaluate_single(
                    factors_df[col], forward_returns
                )
                eval_result["factor"] = col
                results.append(eval_result)
            except Exception as e:
                self.logger.warning(f"因子 '{col}' 评估异常: {e}")
                results.append({
                    "factor": col, "ic": np.nan, "ic_abs": np.nan,
                    "ic_pvalue": np.nan, "ic_tstat": np.nan,
                    "rank_ic": np.nan, "ic_ir": np.nan, "coverage": np.nan,
                    "group_returns": {}, "long_short": np.nan,
                    "long_short_annualized": np.nan, "n_stocks": 0,
                })

        df_result = pd.DataFrame(results)
        df_result = df_result.set_index("factor")

        # 将 group_returns 展开为独立列
        if "group_returns" in df_result.columns:
            for i in range(5):
                col_name = f"group_{i+1}"
                df_result[col_name] = df_result["group_returns"].apply(
                    lambda x: x.get(i + 1, np.nan) if isinstance(x, dict) else np.nan
                )
            df_result = df_result.drop(columns=["group_returns"])

        # 排序
        if sort_by in df_result.columns:
            df_result = df_result.sort_values(
                by=sort_by, ascending=ascending
            )

        return df_result

    # ------------------------------------------------------------------
    #  分组收益分析
    # ------------------------------------------------------------------
    def group_return_analysis(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        n_groups: int = 5,
    ) -> Dict:
        """
        按因子值分组,计算各组平均收益和多空收益。

        将因子值按分位数分为 n_groups 组(组1=最低值, 组n_groups=最高值),
        计算每组股票的平均持有期收益。

        :param factor_values: 因子值序列 (index=股票代码)
        :param forward_returns: 持有期收益率序列 (index=股票代码)
        :param n_groups: 分组数量,默认5组
        :return: 字典,包含:
            - group_returns: {组号: 平均收益}
            - group_sizes: {组号: 股票数量}
            - long_short: 多空收益(最高组平均收益 - 最低组平均收益)
        """
        # 对齐数据
        common_idx = factor_values.dropna().index.intersection(
            forward_returns.dropna().index
        )
        if len(common_idx) < n_groups:
            return {
                "group_returns": {},
                "group_sizes": {},
                "long_short": np.nan,
            }

        fvals = factor_values.loc[common_idx]
        frets = forward_returns.loc[common_idx]

        # 按分位数分组
        try:
            groups = pd.qcut(fvals, q=n_groups, labels=False, duplicates="drop") + 1
        except ValueError:
            # 若分位数计算失败(如过多相同值),使用rank分组
            self.logger.debug("qcut分组失败,降级使用rank分组")
            groups = pd.cut(
                fvals.rank(method="first"),
                bins=n_groups,
                labels=False,
            ) + 1

        # 合并分组与收益
        combined = pd.DataFrame({"group": groups, "ret": frets})

        group_returns = {}
        group_sizes = {}
        for g in sorted(combined["group"].unique()):
            g_mask = combined["group"] == g
            group_returns[g] = combined.loc[g_mask, "ret"].mean()
            group_sizes[g] = g_mask.sum()

        # 多空收益: 最高组 - 最低组
        if len(group_returns) >= 2:
            max_group = max(group_returns.keys())
            min_group = min(group_returns.keys())
            long_short = group_returns[max_group] - group_returns[min_group]
        else:
            long_short = np.nan

        return {
            "group_returns": group_returns,
            "group_sizes": group_sizes,
            "long_short": long_short,
        }

    # ------------------------------------------------------------------
    #  IC衰减分析
    # ------------------------------------------------------------------
    def ic_decay_analysis(
        self,
        factor_values: pd.DataFrame,
        forward_returns_lagged: Dict[int, pd.Series],
        max_lag: int = 12,
    ) -> pd.Series:
        """
        计算因子在不同滞后期下的IC,判断因子衰减速度。

        IC衰减是评估因子预测能力持续性的重要指标:
          - IC衰减慢 -> 因子信号持续时间长,适合低频调仓
          - IC衰减快 -> 因子信号短暂,需要更频繁调仓

        :param factor_values: 因子值 DataFrame (index=日期, columns=股票代码)
                              需为多期面板数据
        :param forward_returns_lagged: {滞后期: 收益率Series} 的字典。
            滞后期为月数,如 {1: 月度收益, 3: 季度收益, 6: 半年收益}
            若为None,则自动计算(需要因子值面板数据)
        :param max_lag: 最大滞后期,默认12个月
        :return: pd.Series (index=滞后期, values=IC值)
        """
        ic_series = pd.Series(dtype=float, name="IC")

        if isinstance(factor_values, pd.Series):
            # 单期因子值,无法做IC衰减分析
            self.logger.warning(
                "IC衰减分析需要多期因子面板数据,传入的是单期数据,返回空结果"
            )
            return ic_series

        for lag in range(1, max_lag + 1):
            if lag in forward_returns_lagged:
                ret = forward_returns_lagged[lag]
                # 取因子值与对应滞后期收益的截面IC
                ics = []
                for date in factor_values.index:
                    fval = factor_values.loc[date]
                    if date in ret.index:
                        fret = ret.loc[date]
                        common = fval.dropna().index.intersection(fret.dropna().index)
                        if len(common) >= 10:
                            ic, _ = stats.spearmanr(
                                fval.loc[common], fret.loc[common]
                            )
                            ics.append(ic)

                if ics:
                    ic_series[lag] = np.mean(ics)
                else:
                    ic_series[lag] = np.nan
            else:
                ic_series[lag] = np.nan

        return ic_series


# ======================================================================
#  4. UNIFIED_WEIGHTS — 统一因子权重配置
# ======================================================================
def build_unified_weights() -> Dict[str, float]:
    """
    构建统一因子权重配置:将推荐16维权重按 category 分配到对应回测因子。

    分配逻辑:
      1. 对于有对应回测因子的推荐维度(如 value -> [ep, bp, sp, cfp]):
         维度权重 / 该类别下因子数 = 每个因子的权重
      2. 对于无对应回测因子的推荐维度(如 fund_flow, sentiment):
         权重保留在维度级别,不分配到具体因子(这些维度在实时推荐中使用)

    :return: {因子名: 权重} 字典(仅包含回测/扩展因子)
    """
    registry = FactorRegistry()
    mapping = registry.get_mapping()
    directions = registry.get_factor_directions()

    unified: Dict[str, float] = {}

    for dim_name, dim_weight in WEIGHTS.items():
        # 查找该维度映射到的因子列表
        mapped_factors = mapping.get(dim_name, [])

        if not mapped_factors:
            # 该维度无对应回测因子,跳过
            logger.debug(
                f"推荐维度 '{dim_name}' 无对应回测因子,权重保留在维度级别"
            )
            continue

        # 均匀分配维度权重到各因子
        n_factors = len(mapped_factors)
        per_factor_weight = dim_weight / n_factors

        for fname in mapped_factors:
            if fname in unified:
                # 同一因子可能被多个维度映射,取累加最大值
                unified[fname] = max(unified[fname], per_factor_weight)
            else:
                unified[fname] = per_factor_weight

    # 归一化:确保总权重为1.0(仅对有分配权重的因子归一化)
    total = sum(unified.values())
    if total > 0:
        unified = {k: v / total for k, v in unified.items()}

    logger.info(
        f"统一因子权重构建完成: 共 {len(unified)} 个因子获得权重分配"
    )
    return unified


# 模块级单例:统一因子权重
UNIFIED_WEIGHTS: Dict[str, float] = build_unified_weights()


# ======================================================================
#  5. 便捷函数
# ======================================================================
def get_registry() -> FactorRegistry:
    """获取全局因子注册表单例"""
    return _global_registry


def get_unified_weights() -> Dict[str, float]:
    """
    获取统一因子权重配置。

    :return: {因子名: 权重} 字典
    """
    return UNIFIED_WEIGHTS


# 模块加载时初始化全局注册表
_global_registry = FactorRegistry()


# ======================================================================
#  6. 模块自测
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("v7.0 统一因子注册表 - 模块自测")
    print("=" * 60)

    # 测试 FactorRegistry
    reg = get_registry()
    print(f"\n[FactorRegistry] {reg}")
    print(f"  总注册因子数: {len(reg)}")
    print(f"  回测因子数:   {len(reg.get_factors_by_source('backtest'))}")
    print(f"  扩展因子数:   {len(reg.get_factors_by_source('extended'))}")
    print(f"  推荐维度数:   {len(reg.get_factors_by_source('recommend'))}")
    print(f"  因子类别:     {reg.get_categories()}")
    print(f"  全部因子名:   {reg.get_all_factor_names()}")

    # 按类别查询
    for cat in ["value", "momentum", "quality", "risk"]:
        factors = reg.get_factors_by_category(cat)
        print(f"\n  [{cat}] 因子({len(factors)}个):")
        for f in factors:
            print(f"    - {f['name']:20s} dir={f['direction']:+d}  src={f['source']:10s}  {f['description']}")

    # 映射关系
    mapping = reg.get_mapping()
    print(f"\n[推荐维度 -> 因子映射]:")
    for dim, factors in mapping.items():
        if factors:
            print(f"  {dim:25s} -> {factors}")

    # 方向映射
    directions = reg.get_factor_directions()
    print(f"\n[因子方向] 共 {len(directions)} 个因子:")
    for name, d in directions.items():
        arrow = "↑" if d == 1 else ("↓" if d == -1 else "-")
        print(f"  {name:20s} {arrow} (direction={d:+d})")

    # 统一权重
    weights = get_unified_weights()
    print(f"\n[统一因子权重] 共 {len(weights)} 个因子:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} weight={w:.4f} ({w*100:.2f}%)")
    print(f"  总权重: {sum(weights.values()):.6f}")

    # 测试 FactorEvaluator
    print(f"\n{'=' * 60}")
    print("[FactorEvaluator] 单因子评估测试")
    print(f"{'=' * 60}")

    np.random.seed(42)
    n_stocks = 200
    codes = [f"SH{str(i).zfill(6)}" for i in range(n_stocks)]
    fake_factor = pd.Series(
        np.random.randn(n_stocks),
        index=codes,
        name="test_factor",
    )
    fake_returns = pd.Series(
        np.random.randn(n_stocks) * 0.02,
        index=codes,
        name="fwd_ret",
    )
    # 注入一定的单调关系
    fake_returns += fake_factor * 0.01

    evaluator = FactorEvaluator()
    result = evaluator.evaluate_single(fake_factor, fake_returns)
    print(f"\n  单因子评估结果:")
    for k, v in result.items():
        if k != "group_returns":
            print(f"    {k:30s} = {v}")

    # 分组收益
    group_res = evaluator.group_return_analysis(fake_factor, fake_returns)
    print(f"\n  分组收益分析:")
    for g, ret in group_res["group_returns"].items():
        size = group_res["group_sizes"].get(g, 0)
        print(f"    Group {g}: avg_ret={ret:.4f}, n={size}")
    print(f"    多空收益(Long-Short): {group_res['long_short']:.4f}")

    print(f"\n{'=' * 60}")
    print("模块自测完成")
    print(f"{'=' * 60}")
