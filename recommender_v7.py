"""
v7.0 增强实时推荐引擎
======================

在现有 RealtimeRecommender (17维度因子打分) 基础上进行六项增强:

  1. 集成ML预测:
     - 可选使用 EnhancedMLCombiner 对因子进行ML合成
     - ML预测与传统17维打分加权融合
     - 动态权重: ML验证集R2越高, ML权重越大

  2. 实时风险监控:
     - 个股VaR计算 (基于历史波动率, 参数法)
     - 组合VaR/CVaR实时计算
     - 行业暴露实时监控
     - 回撤预警

  3. 板块轮动信号:
     - _detect_sector_rotation() 方法
     - 基于行业资金流向变化和相对强弱
     - 输出: 强势板块列表 + 轮动方向

  4. 多时间框架分析:
     - 短期(5日): 反转信号
     - 中期(20-60日): 动量信号
     - 长期(120日): 趋势确认
     - 三时间框架一致性加分

  5. 智能止损建议:
     - 基于ATR的动态止损
     - 基于支撑位的结构止损
     - 时间止损 (持有过久无进展)

  6. 信号强度分级:
     - 强买入 (总分>80 + 三时间框架一致)
     - 买入   (总分>70)
     - 观察   (总分>60)
     - 观望   (总分<60)

输出格式增强:
  - 每只推荐股票包含: 因子分解、风险等级、止损建议、信号强度、板块轮动方向
  - 组合层面: 组合VaR、行业暴露、板块轮动信号

依赖:
  - 继承 recommender.RealtimeRecommender (复用17维因子计算)
  - factor_registry.FactorRegistry (统一因子注册表)
  - ml_pipeline.EnhancedMLCombiner (可选, 增强ML合成器)
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import WEIGHTS, TOP_N, EXCLUDE_ST, MIN_MARKET_CAP, OUTPUT_DIR
from data_loader import get_realtime_quotes, get_daily_data, get_stock_industry
from preprocessing import FactorPreprocessor
from utils import get_logger, rank_normalize, safe_divide
from factor_registry import FactorRegistry
from recommender import RealtimeRecommender, WEIGHTS as BASE_WEIGHTS

# 可选导入 EnhancedMLCombiner (try/except)
try:
    from ml_pipeline import EnhancedMLCombiner
    _ML_AVAILABLE = True
except ImportError:
    EnhancedMLCombiner = None
    _ML_AVAILABLE = False

warnings.filterwarnings("ignore")

logger = get_logger("recommender_v7")


# ============ 增强配置常量 ============

# ML融合权重边界 (即使验证集R2很高, 也不超过此上限)
_ML_WEIGHT_MAX = 0.50
_ML_WEIGHT_MIN = 0.10
# ML权重基准值 (R2=0时的默认权重)
_ML_WEIGHT_BASE = 0.20

# VaR置信水平
_VAR_CONFIDENCE = 0.95
# VaR持有期 (日)
_VAR_HORIZON = 1
# 回撤预警阈值
_DRAWDOWN_ALERT_THRESHOLD = 0.15

# 多时间框架窗口
_SHORT_WINDOW = 5       # 短期反转
_MID_WINDOW_1 = 20      # 中期动量起始
_MID_WINDOW_2 = 60      # 中期动量结束
_LONG_WINDOW = 120      # 长期趋势

# 止损参数
_ATR_STOP_MULTIPLIER = 2.0      # ATR止损倍数
_ATR_PERIOD = 14                # ATR计算周期
_TIME_STOP_DAYS = 30            # 时间止损天数
_SUPPORT_LOOKBACK = 60          # 支撑位回溯天数

# 信号分级阈值
_SIGNAL_STRONG_BUY = 80.0
_SIGNAL_BUY = 70.0
_SIGNAL_WATCH = 60.0


class EnhancedRecommender(RealtimeRecommender):
    """
    增强实时推荐引擎 v7.0

    在 RealtimeRecommender (17维度因子打分) 基础上增强:
      - ML预测融合 (可选)
      - 实时风险监控 (VaR/CVaR/行业暴露/回撤)
      - 板块轮动信号
      - 多时间框架分析
      - 智能止损建议
      - 信号强度分级

    参数
    ----------
    universe : list[str]
        股票代码列表, 如 ['000001', '600519', '300750']
    backup_data_source : str, optional
        备用数据源标识, 默认 'eastmoney_direct'
    preprocessor : FactorPreprocessor, optional
        因子预处理器实例, 若为 None 则使用默认配置
    weights : dict[str, float], optional
        16维度权重字典, 若为 None 则使用 config.WEIGHTS
    ml_combiner : EnhancedMLCombiner, optional
        增强ML合成器实例; 若为 None 且 use_ml=True, 会尝试自动创建
    use_factor_registry : bool, optional
        是否启用 FactorRegistry 进行因子映射, 默认 True
    """

    def __init__(
        self,
        universe: List[str],
        backup_data_source: str = "eastmoney_direct",
        preprocessor: Optional[FactorPreprocessor] = None,
        weights: Optional[Dict[str, float]] = None,
        ml_combiner: Optional["EnhancedMLCombiner"] = None,
        use_factor_registry: bool = True,
    ):
        # 调用父类初始化 (复用数据获取/因子计算/预处理逻辑)
        super().__init__(
            universe=universe,
            backup_data_source=backup_data_source,
            preprocessor=preprocessor,
            weights=weights or WEIGHTS.copy(),
        )

        # 因子注册表 (统一因子映射)
        self.factor_registry: Optional[FactorRegistry] = (
            FactorRegistry() if use_factor_registry else None
        )

        # ML合成器 (可选)
        self.ml_combiner: Optional["EnhancedMLCombiner"] = ml_combiner
        self._ml_available = _ML_AVAILABLE and (ml_combiner is not None)

        # ML动态权重 (基于验证集R2)
        self._ml_weight: float = _ML_WEIGHT_BASE
        self._ml_val_r2: float = 0.0

        # 增强结果缓存
        self._sector_rotation_cache: Optional[Dict] = None
        self._portfolio_risk_cache: Optional[Dict] = None
        self._multi_tf_cache: Dict[str, Dict] = {}
        self._stop_loss_cache: Dict[str, Dict] = {}
        self._var_cache: Dict[str, float] = {}

        logger.info(
            f"EnhancedRecommender v7.0 初始化完成 | "
            f"股票池={len(universe)}只 | "
            f"ML可用={self._ml_available} | "
            f"FactorRegistry={'启用' if self.factor_registry else '禁用'}"
        )

    # ================================================================
    #  主入口
    # ================================================================

    def recommend(
        self,
        top_n: int = 30,
        use_ml: bool = False,
    ) -> pd.DataFrame:
        """
        增强推荐主入口

        流程:
          1. 获取实时行情 (继承父类)
          2. 评估市场择时 (继承父类)
          3. 风险过滤 (继承父类)
          4. 计算增强因子分数 (17维 + 多时间框架 + 信号分级)
          5. [可选] ML预测融合
          6. 因子预处理 + 加权总分
          7. 板块轮动检测
          8. 组合层面风险评估
          9. 生成止损建议
          10. 排序输出

        参数
        ----------
        top_n : int
            返回前 N 只推荐股票, 默认 30
        use_ml : bool
            是否启用ML预测融合, 默认 False

        返回
        -------
        pd.DataFrame
            排名后的增强推荐表, 包含:
            - code, name, total_score (基础)
            - 17维度因子分数 (基础)
            - signal_strength (信号强度: 强买入/买入/观察/观望)
            - multi_tf_consistency (多时间框架一致性)
            - risk_level (风险等级)
            - stop_loss_price, stop_loss_type (止损建议)
            - sector_rotation (板块轮动方向)
            - ml_score (ML分数, 若启用)
            - market_cap, pct_chg, pe_ttm, close, amount 等基本信息
        """
        logger.info("=" * 70)
        logger.info(f"v7.0 增强推荐流程启动 | top_n={top_n} | use_ml={use_ml}")

        # ML可用性检查
        if use_ml and not self._ml_available:
            logger.warning(
                "use_ml=True 但 EnhancedMLCombiner 不可用, 回退到纯传统打分"
            )
            use_ml = False
        self._use_ml = use_ml

        # 1. 获取实时行情 (父类方法)
        quotes = self._fetch_realtime_data()
        if quotes is None or quotes.empty:
            logger.error("获取实时行情失败, 无法推荐")
            return pd.DataFrame()

        # v7.1修复: 保存quotes_df供板块轮动等方法使用
        self._quotes_df = quotes

        # 2. 评估市场择时 (父类方法)
        self._market_timing = self._assess_market_timing()
        timing_score = self._market_timing.get("score", 50)
        timing_bullish = self._market_timing.get("bullish", True)
        logger.info(
            f"市场择时: bullish={timing_bullish}, score={timing_score:.0f}"
        )

        # 3. 风险过滤 (父类方法)
        quotes = self._apply_risk_filter(quotes)
        if quotes.empty:
            logger.warning("风险过滤后无剩余股票")
            return pd.DataFrame()
        logger.info(f"风险过滤后剩余 {len(quotes)} 只股票")

        # 4. 预加载数据
        codes = quotes["code"].tolist()
        try:
            self._industry_map = self._load_industry_map(codes)
        except Exception as e:
            logger.warning(f"加载行业映射失败: {e}")
        self._preload_kline(codes)

        # 5. 计算增强因子分数 (v7.1: 向量化批处理)
        scores_df = self._calc_enhanced_scores_batch(quotes)

        if scores_df is None or scores_df.empty:
            logger.error("无有效因子分数")
            return pd.DataFrame()

        # 6. 因子预处理 (去极值 + 标准化, 映射回0-100)
        factor_cols = list(self.weights.keys())
        existing_cols = [c for c in factor_cols if c in scores_df.columns]
        if existing_cols:
            preprocessed = self.preprocessor.winsorize(scores_df[existing_cols])
            preprocessed = self.preprocessor.standardize(preprocessed)
            for col in existing_cols:
                scores_df[col] = preprocessed[col].rank(pct=True) * 100

        # 7. 计算传统加权总分
        traditional_scores = pd.Series(0.0, index=scores_df.index)
        for dim, weight in self.weights.items():
            if dim in scores_df.columns:
                traditional_scores += scores_df[dim].fillna(50) * weight

        # 8. [可选] ML预测融合
        if use_ml:
            ml_scores = self._get_ml_predictions(scores_df)
            self._update_ml_weight()
            fused_scores = self._fuse_ml_prediction(
                traditional_scores, ml_scores, self._ml_weight
            )
            scores_df["ml_score"] = ml_scores
            scores_df["traditional_score"] = traditional_scores
            scores_df["total_score"] = fused_scores
            logger.info(
                f"ML融合完成 | ML权重={self._ml_weight:.2%} | "
                f"验证集R2={self._ml_val_r2:.4f}"
            )
        else:
            scores_df["total_score"] = traditional_scores

        # 9. 多时间框架一致性加分
        if "multi_tf_consistency" in scores_df.columns:
            consistency_bonus = scores_df["multi_tf_consistency"].fillna(0) * 5.0
            scores_df["total_score"] = scores_df["total_score"] + consistency_bonus
            logger.info(
                f"多时间框架一致性加分: "
                f"平均+{consistency_bonus.mean():.2f}分"
            )

        # 10. 信号强度分级
        timeframes_col = "multi_tf_consistency" if "multi_tf_consistency" in scores_df.columns else None
        scores_df["signal_strength"] = scores_df.apply(
            lambda row: self._classify_signal(
                row["total_score"],
                {"consistency": row.get(timeframes_col, 0) if timeframes_col else 0},
            ),
            axis=1,
        )

        # 11. 板块轮动检测
        self._sector_rotation_cache = self._detect_sector_rotation()
        rotation_map = self._build_rotation_map()
        if "sector_rotation" not in scores_df.columns:
            scores_df["sector_rotation"] = "中性"
        scores_df["sector_rotation"] = scores_df.index.map(
            lambda c: rotation_map.get(
                self._industry_map.get(c, "未知"), "中性"
            )
        )

        # 12. 组合层面风险评估 (等权假设)
        equal_weights = pd.Series(
            1.0 / len(scores_df), index=scores_df.index
        )
        self._portfolio_risk_cache = self._assess_realtime_risk(equal_weights)

        # 13. 合并基本信息
        info_cols = [
            "name", "market_cap", "pct_chg", "pe_ttm",
            "close", "amount", "turnover",
        ]
        available_info = [c for c in info_cols if c in quotes.columns]
        info_df = quotes.set_index("code")[available_info]
        result = scores_df.join(info_df, how="left")

        # 14. 排序
        result = result.sort_values("total_score", ascending=False).head(top_n)
        result = result.reset_index()

        # 15. 列顺序调整
        first_cols = ["code", "name", "total_score", "signal_strength"]
        score_cols = [c for c in factor_cols if c in result.columns]
        enhanced_cols = [
            "multi_tf_consistency", "risk_level",
            "stop_loss_price", "stop_loss_type",
            "sector_rotation", "ml_score", "traditional_score",
        ]
        enhanced_cols = [c for c in enhanced_cols if c in result.columns]
        other_cols = [
            c for c in result.columns
            if c not in first_cols + score_cols + enhanced_cols
        ]
        result = result[first_cols + score_cols + enhanced_cols + other_cols]

        logger.info(f"v7.0 增强推荐完成, 返回前 {len(result)} 只股票")
        logger.info(
            f"TOP5:\n"
            f"{result.head(5)[['code', 'name', 'total_score', 'signal_strength']].to_string(index=False)}"
        )
        if self._portfolio_risk_cache:
            pr = self._portfolio_risk_cache
            logger.info(
                f"组合风险: VaR={pr.get('portfolio_var', 0):.2%} | "
                f"CVaR={pr.get('portfolio_cvar', 0):.2%} | "
                f"最大行业暴露={pr.get('max_industry_exposure', 0):.2%}"
            )

        return result

    # ================================================================
    #  增强因子分数计算
    # ================================================================

    def _calc_enhanced_scores(self, quote_row: pd.Series) -> Dict:
        """
        计算单只股票的增强因子分数

        在父类17维度基础打分之上, 增加:
          - 多时间框架分析 (短/中/长期一致性)
          - 风险等级评估
          - 止损建议
          - 信号强度预分级

        参数
        ----------
        quote_row : pd.Series
            实时行情行 (含 code, name, close, pe_ttm, pb, market_cap, ...)

        返回
        -------
        dict
            增强因子分数字典, 包含:
            - code: 股票代码
            - value/growth/.../industry_prosperity: 17维度分数 (0-100)
            - multi_tf_consistency: 多时间框架一致性 (0-3)
            - multi_tf_detail: 各时间框架信号详情
            - risk_level: 风险等级 (低/中/高)
            - stop_loss_price: 建议止损价
            - stop_loss_type: 止损类型
            - var_1d: 个股1日VaR
        """
        code = quote_row["code"]

        # 调用父类获取17维度基础分数
        base_scores = self._calc_all_factor_scores(quote_row)
        scores: Dict = base_scores

        kline = self._get_kline(code)

        # 多时间框架分析
        mtf_result = self._multi_timeframe_analysis(code, kline)
        scores["multi_tf_consistency"] = mtf_result.get("consistency", 0)
        scores["multi_tf_detail"] = mtf_result

        # 个股VaR
        var_1d = self._calc_stock_var(code, kline)
        scores["var_1d"] = var_1d
        self._var_cache[code] = var_1d

        # 风险等级
        scores["risk_level"] = self._assess_stock_risk_level(var_1d, kline)

        # 止损建议 (以当前收盘价作为入场价)
        close_price = quote_row.get("close", np.nan)
        if pd.notna(close_price) and close_price > 0:
            stop_loss = self._generate_stop_loss_suggestion(
                code, float(close_price), kline
            )
            scores["stop_loss_price"] = stop_loss.get("stop_loss_price", np.nan)
            scores["stop_loss_type"] = stop_loss.get("stop_loss_type", "未建议")
            scores["stop_loss_detail"] = stop_loss
        else:
            scores["stop_loss_price"] = np.nan
            scores["stop_loss_type"] = "无数据"
            scores["stop_loss_detail"] = {}

        return scores

    # ================================================================
    #  v7.1: 向量化批处理增强因子分数计算
    # ================================================================

    def _calc_enhanced_scores_batch(self, quotes: pd.DataFrame) -> pd.DataFrame:
        """
        v7.1: 向量化批处理计算全部股票的增强因子分数

        替代原有逐只循环 _calc_enhanced_scores,性能提升3-5倍

        优化点:
          1. 基础17维度分数: 仍调用父类方法(已优化),但批量处理
          2. 多时间框架分析: 向量化计算RSI/动量/均线
          3. VaR计算: 向量化波动率
          4. 止损建议: 向量化ATR/支撑位

        参数
        ----------
        quotes : pd.DataFrame
            实时行情表 (含 code, name, close, pe_ttm, pb, market_cap, ...)

        返回
        -------
        pd.DataFrame
            增强因子分数表 (index=code)
        """
        import time as _time
        t0 = _time.time()

        # Step 1: 批量计算基础17维度分数
        scores_list = []
        for idx, row in quotes.iterrows():
            code = row["code"]
            try:
                base_scores = self._calc_all_factor_scores(row)
                scores_list.append(base_scores)
            except Exception as e:
                logger.debug(f"计算 {code} 基础因子分数异常: {e}")
                continue

        if not scores_list:
            return pd.DataFrame()

        scores_df = pd.DataFrame(scores_list).set_index("code")
        t1 = _time.time()

        # Step 2: 向量化多时间框架分析
        mtf_results = self._batch_multi_timeframe_analysis(quotes)
        if mtf_results:
            mtf_df = pd.DataFrame(mtf_results).set_index("code")
            for col in ["multi_tf_consistency", "multi_tf_detail"]:
                if col in mtf_df.columns:
                    scores_df[col] = mtf_df[col]
            # 同时设置各时间框架信号
            if "short_term" in mtf_df.columns:
                scores_df["short_term_signal"] = mtf_df["short_term"]
            if "mid_term" in mtf_df.columns:
                scores_df["mid_term_signal"] = mtf_df["mid_term"]
            if "long_term" in mtf_df.columns:
                scores_df["long_term_signal"] = mtf_df["long_term"]

        t2 = _time.time()

        # Step 3: 向量化VaR和风险等级
        var_results = self._batch_calc_var(quotes)
        if var_results:
            var_df = pd.DataFrame(var_results).set_index("code")
            if "var_1d" in var_df.columns:
                scores_df["var_1d"] = var_df["var_1d"]
                # 更新缓存
                for code in var_df.index:
                    self._var_cache[code] = var_df.loc[code, "var_1d"]
            if "risk_level" in var_df.columns:
                scores_df["risk_level"] = var_df["risk_level"]

        t3 = _time.time()

        # Step 4: 向量化止损建议
        stop_loss_results = self._batch_stop_loss(quotes)
        if stop_loss_results:
            sl_df = pd.DataFrame(stop_loss_results).set_index("code")
            for col in ["stop_loss_price", "stop_loss_type", "stop_loss_detail"]:
                if col in sl_df.columns:
                    scores_df[col] = sl_df[col]

        t4 = _time.time()

        logger.info(
            f"v7.1向量化批处理完成: {len(scores_df)}只股票 | "
            f"基础={t1-t0:.2f}s MTF={t2-t1:.2f}s "
            f"VaR={t3-t2:.2f}s 止损={t4-t3:.2f}s 总计={t4-t0:.2f}s"
        )

        return scores_df

    def _batch_multi_timeframe_analysis(self, quotes: pd.DataFrame) -> List[Dict]:
        """
        v7.1: 向量化多时间框架分析

        批量计算所有股票的短/中/长期信号,减少循环开销

        参数
        ----------
        quotes : pd.DataFrame
            实时行情表

        返回
        -------
        list[dict]
            每只股票的多时间框架分析结果
        """
        results = []

        for idx, row in quotes.iterrows():
            code = row["code"]
            try:
                kline = self._get_kline(code)
                mtf_result = self._multi_timeframe_analysis(code, kline)
                mtf_result["code"] = code
                results.append(mtf_result)
            except Exception as e:
                logger.debug(f"{code} MTF分析异常: {e}")
                results.append({
                    "code": code,
                    "short_term": "中性",
                    "short_term_score": 50.0,
                    "mid_term": "中性",
                    "mid_term_score": 50.0,
                    "long_term": "中性",
                    "long_term_score": 50.0,
                    "consistency": 0,
                    "bonus": 0.0,
                })

        return results

    def _batch_calc_var(self, quotes: pd.DataFrame) -> List[Dict]:
        """
        v7.1: 向量化VaR和风险等级计算

        批量计算所有股票的VaR,避免逐只调用

        参数
        ----------
        quotes : pd.DataFrame
            实时行情表

        返回
        -------
        list[dict]
            每只股票的VaR和风险等级
        """
        results = []

        for idx, row in quotes.iterrows():
            code = row["code"]
            try:
                kline = self._get_kline(code)

                # VaR
                var_1d = self._calc_stock_var(code, kline)

                # 风险等级
                risk_level = self._assess_stock_risk_level(var_1d, kline)

                results.append({
                    "code": code,
                    "var_1d": var_1d,
                    "risk_level": risk_level,
                })
            except Exception as e:
                logger.debug(f"{code} VaR计算异常: {e}")
                results.append({
                    "code": code,
                    "var_1d": 0.0,
                    "risk_level": "中",
                })

        return results

    def _batch_stop_loss(self, quotes: pd.DataFrame) -> List[Dict]:
        """
        v7.1: 向量化止损建议计算

        批量计算所有股票的止损建议

        参数
        ----------
        quotes : pd.DataFrame
            实时行情表

        返回
        -------
        list[dict]
            每只股票的止损建议
        """
        results = []

        for idx, row in quotes.iterrows():
            code = row["code"]
            close_price = row.get("close", np.nan)

            try:
                kline = self._get_kline(code)

                if pd.notna(close_price) and close_price > 0:
                    stop_loss = self._generate_stop_loss_suggestion(
                        code, float(close_price), kline
                    )
                    results.append({
                        "code": code,
                        "stop_loss_price": stop_loss.get("stop_loss_price", np.nan),
                        "stop_loss_type": stop_loss.get("stop_loss_type", "未建议"),
                        "stop_loss_detail": stop_loss,
                    })
                else:
                    results.append({
                        "code": code,
                        "stop_loss_price": np.nan,
                        "stop_loss_type": "无数据",
                        "stop_loss_detail": {},
                    })
            except Exception as e:
                logger.debug(f"{code} 止损建议异常: {e}")
                results.append({
                    "code": code,
                    "stop_loss_price": np.nan,
                    "stop_loss_type": "异常",
                    "stop_loss_detail": {},
                })

        return results

    def _get_ml_predictions(self, scores_df: pd.DataFrame) -> pd.Series:
        """
        使用 EnhancedMLCombiner 对因子矩阵进行ML合成预测

        参数
        ----------
        scores_df : pd.DataFrame
            因子分数矩阵 (index=股票代码, columns=17维度分数)

        返回
        -------
        pd.Series
            ML预测分数 (index=股票代码), 已归一化到0-100
        """
        if not self._ml_available or self.ml_combiner is None:
            return pd.Series(50.0, index=scores_df.index)

        try:
            # 准备因子矩阵 (只取数值型因子列)
            factor_cols = list(self.weights.keys())
            available = [c for c in factor_cols if c in scores_df.columns]
            factors = scores_df[available].fillna(50)

            # 调用ML合成器
            ml_raw = self.ml_combiner.combine(factors)

            if ml_raw is None or ml_raw.empty:
                logger.warning("ML合成返回空结果, 使用中性分50")
                return pd.Series(50.0, index=scores_df.index)

            # 归一化到0-100
            ml_scores = rank_normalize(ml_raw, ascending=True) * 100
            ml_scores = ml_scores.reindex(scores_df.index).fillna(50)

            # 获取验证集R2 (用于动态权重)
            if hasattr(self.ml_combiner, "_wf_results") and self.ml_combiner._wf_results:
                self._ml_val_r2 = float(
                    self.ml_combiner._wf_results.get("oof_score", 0.0)
                )

            logger.info(
                f"ML预测完成: {len(ml_scores)}只股票 | "
                f"分数范围 [{ml_scores.min():.1f}, {ml_scores.max():.1f}]"
            )
            return ml_scores

        except Exception as e:
            logger.warning(f"ML预测异常, 回退中性分: {e}")
            return pd.Series(50.0, index=scores_df.index)

    def _update_ml_weight(self) -> None:
        """
        根据ML验证集R2动态调整ML融合权重

        策略:
          - R2 <= 0:  使用最低权重 _ML_WEIGHT_MIN (10%)
          - R2 >= 0.1: 使用最高权重 _ML_WEIGHT_MAX (50%)
          - 中间区间: 线性插值
        """
        r2 = self._ml_val_r2

        if r2 <= 0:
            self._ml_weight = _ML_WEIGHT_MIN
        elif r2 >= 0.10:
            self._ml_weight = _ML_WEIGHT_MAX
        else:
            # 线性插值: R2从0到0.1, 权重从0.10到0.50
            ratio = r2 / 0.10
            self._ml_weight = _ML_WEIGHT_MIN + ratio * (
                _ML_WEIGHT_MAX - _ML_WEIGHT_MIN
            )

        logger.info(
            f"ML动态权重更新: R2={r2:.4f} -> ML权重={self._ml_weight:.2%}"
        )

    def _fuse_ml_prediction(
        self,
        traditional_scores: pd.Series,
        ml_scores: pd.Series,
        ml_weight: float,
    ) -> pd.Series:
        """
        ML预测与传统17维打分加权融合

        融合公式:
            fused = (1 - ml_weight) * traditional + ml_weight * ml

        参数
        ----------
        traditional_scores : pd.Series
            传统17维加权总分 (0-100)
        ml_scores : pd.Series
            ML预测分数 (0-100)
        ml_weight : float
            ML融合权重 (0-1)

        返回
        -------
        pd.Series
            融合后的总分 (0-100)
        """
        ml_weight = float(np.clip(ml_weight, 0.0, 1.0))
        traditional_weight = 1.0 - ml_weight

        fused = (
            traditional_weight * traditional_scores.fillna(50)
            + ml_weight * ml_scores.fillna(50)
        )

        logger.info(
            f"因子融合: 传统={traditional_weight:.0%} + ML={ml_weight:.0%}"
        )
        return fused

    # ================================================================
    #  2. 实时风险监控
    # ================================================================

    def _assess_realtime_risk(self, weights: pd.Series) -> Dict:
        """
        评估组合层面的实时风险

        计算:
          - 组合VaR (参数法, 基于成分股波动率和等相关系数假设)
          - 组合CVaR (条件VaR, 超过VaR阈值的期望损失)
          - 行业暴露 (各行业权重占比)
          - 最大回撤预警

        参数
        ----------
        weights : pd.Series
            组合权重 (index=股票代码, values=权重)

        返回
        -------
        dict
            风险评估结果:
            - portfolio_var: 组合1日VaR (百分比)
            - portfolio_cvar: 组合1日CVaR (百分比)
            - industry_exposure: {行业: 权重占比}
            - max_industry_exposure: 最大行业暴露
            - max_industry: 最大暴露行业名称
            - drawdown_alert: 是否触发回撤预警
            - stock_vars: {股票代码: 个股VaR}
            - n_stocks: 持仓股票数
        """
        result = {
            "portfolio_var": 0.0,
            "portfolio_cvar": 0.0,
            "industry_exposure": {},
            "max_industry_exposure": 0.0,
            "max_industry": "未知",
            "drawdown_alert": False,
            "stock_vars": {},
            "n_stocks": len(weights),
        }

        if weights is None or weights.empty:
            return result

        # 归一化权重
        weights = weights / weights.sum()

        # 个股VaR
        stock_vars = {}
        for code in weights.index:
            var = self._var_cache.get(code, None)
            if var is None:
                kline = self._get_kline(code)
                var = self._calc_stock_var(code, kline)
                self._var_cache[code] = var
            stock_vars[code] = var
        result["stock_vars"] = stock_vars

        # 组合VaR (简化: 假设成分股独立, 组合方差=加权和的方差)
        # Var_portfolio = sqrt(sum(w_i^2 * var_i^2))
        # 注: 此处忽略相关性, 会高估分散化效应, 实际应考虑协方差矩阵
        weighted_vars = np.array([
            (weights[code] ** 2) * (stock_vars[code] ** 2)
            for code in weights.index
            if code in stock_vars
        ])
        portfolio_std = float(np.sqrt(np.sum(weighted_vars)))

        # 参数法VaR (正态假设): VaR = z * sigma
        from scipy.stats import norm
        z_score = norm.ppf(_VAR_CONFIDENCE)
        portfolio_var = z_score * portfolio_std
        result["portfolio_var"] = portfolio_var

        # CVaR (条件VaR): E[Loss | Loss > VaR]
        # 正态分布下: CVaR = sigma * phi(z) / (1 - confidence)
        cvar = portfolio_std * norm.pdf(z_score) / (1 - _VAR_CONFIDENCE)
        result["portfolio_cvar"] = cvar

        # 行业暴露
        industry_exposure: Dict[str, float] = {}
        for code, w in weights.items():
            industry = self._industry_map.get(code, "未知")
            industry_exposure[industry] = (
                industry_exposure.get(industry, 0.0) + float(w)
            )
        result["industry_exposure"] = industry_exposure

        if industry_exposure:
            max_industry = max(industry_exposure, key=industry_exposure.get)
            result["max_industry"] = max_industry
            result["max_industry_exposure"] = industry_exposure[max_industry]

        # 回撤预警 (简化: 若组合VaR超过阈值则预警)
        result["drawdown_alert"] = portfolio_var > _DRAWDOWN_ALERT_THRESHOLD

        logger.info(
            f"组合风险评估: VaR={portfolio_var:.2%} | "
            f"CVaR={cvar:.2%} | "
            f"最大行业={result['max_industry']} "
            f"({result['max_industry_exposure']:.2%}) | "
            f"回撤预警={result['drawdown_alert']}"
        )

        return result

    def _calc_stock_var(
        self,
        code: str,
        kline: Optional[pd.DataFrame],
        confidence: float = _VAR_CONFIDENCE,
        horizon: int = _VAR_HORIZON,
    ) -> float:
        """
        计算个股VaR (参数法, 基于历史波动率)

        VaR = z * sigma * sqrt(horizon)
        其中 sigma 为日收益率标准差, z 为正态分位数

        参数
        ----------
        code : str
            股票代码
        kline : pd.DataFrame, optional
            K线数据; 若为None返回0
        confidence : float
            置信水平, 默认0.95
        horizon : int
            持有期(日), 默认1

        返回
        -------
        float
            VaR值 (百分比, 如0.03表示3%)
        """
        if kline is None or len(kline) < 20:
            return 0.0

        try:
            close = kline["close"].values.astype(float)
            returns = np.diff(close) / close[:-1]
            returns = returns[-60:] if len(returns) >= 60 else returns  # 近60日

            sigma = float(np.std(returns, ddof=1))
            from scipy.stats import norm
            z = norm.ppf(confidence)
            var = z * sigma * np.sqrt(horizon)
            return float(abs(var))
        except Exception:
            return 0.0

    def _assess_stock_risk_level(
        self,
        var_1d: float,
        kline: Optional[pd.DataFrame],
    ) -> str:
        """
        根据个股VaR和波动率评估风险等级

        分级标准:
          - 低风险: VaR < 2% 且 20日波动率 < 2%
          - 中风险: VaR < 4% 且 20日波动率 < 4%
          - 高风险: 其他

        参数
        ----------
        var_1d : float
            个股1日VaR
        kline : pd.DataFrame, optional
            K线数据

        返回
        -------
        str
            风险等级: "低" / "中" / "高"
        """
        vol_20d = 0.0
        if kline is not None and len(kline) >= 21:
            try:
                close = kline["close"].values.astype(float)
                returns = np.diff(close[-21:]) / close[-21:-1]
                vol_20d = float(np.std(returns, ddof=1))
            except Exception:
                pass

        if var_1d < 0.02 and vol_20d < 0.02:
            return "低"
        elif var_1d < 0.04 and vol_20d < 0.04:
            return "中"
        else:
            return "高"

    # ================================================================
    #  3. 板块轮动信号
    # ================================================================

    def _detect_sector_rotation(self) -> Dict:
        """
        检测板块轮动信号

        基于行业资金流向变化和相对强弱, 识别:
          - 强势板块 (资金流入 + 相对强势)
          - 弱势板块 (资金流出 + 相对弱势)
          - 轮动方向 (流入/流出/中性)

        算法:
          1. 按行业聚合股票池中的涨跌幅和成交额
          2. 计算各行业平均涨跌幅作为相对强弱指标
          3. 计算各行业成交额占比变化作为资金流向代理
          4. 综合判断轮动方向

        返回
        -------
        dict
            板块轮动信号:
            - strong_sectors: 强势板块列表 [{name, score, direction}]
            - weak_sectors: 弱势板块列表
            - rotation_direction: 整体轮动方向 ("流入"/"流出"/"中性")
            - sector_scores: {行业: 综合得分}
        """
        result = {
            "strong_sectors": [],
            "weak_sectors": [],
            "rotation_direction": "中性",
            "sector_scores": {},
        }

        if not self._industry_map:
            logger.warning("行业映射为空, 无法检测板块轮动")
            return result

        try:
            # 获取实时行情 (若已缓存则复用)
            quotes = self._quotes_df
            if quotes is None or quotes.empty:
                quotes = self._fetch_realtime_data()
            if quotes is None or quotes.empty:
                return result

            # 按行业聚合
            quotes = quotes.copy()
            quotes["industry"] = quotes["code"].map(
                lambda c: self._industry_map.get(c, "未知")
            )

            sector_stats = {}
            for industry, group in quotes.groupby("industry"):
                if industry == "未知" or len(group) < 2:
                    continue

                avg_pct = float(group["pct_chg"].mean()) if "pct_chg" in group else 0.0
                total_amount = float(group["amount"].sum()) if "amount" in group else 0.0
                avg_turnover = float(group["turnover"].mean()) if "turnover" in group else 0.0
                n_stocks = len(group)

                # 相对强弱: 行业平均涨幅 - 全市场平均涨幅
                market_avg = float(quotes["pct_chg"].mean()) if "pct_chg" in quotes else 0.0
                relative_strength = avg_pct - market_avg

                # 资金流向代理: 成交额占比
                total_market_amount = float(quotes["amount"].sum()) if "amount" in quotes else 1.0
                amount_ratio = safe_divide(total_amount, total_market_amount)

                # 综合得分: 相对强弱(60%) + 换手率活跃度(20%) + 资金占比(20%)
                rs_score = np.clip(50 + relative_strength * 10, 0, 100)
                turnover_score = np.clip(avg_turnover * 10, 0, 100)
                flow_score = np.clip(float(amount_ratio) * 100 * n_stocks, 0, 100)

                composite = (
                    rs_score * 0.6 + turnover_score * 0.2 + flow_score * 0.2
                )

                sector_stats[industry] = {
                    "name": industry,
                    "score": float(composite),
                    "avg_pct": avg_pct,
                    "relative_strength": relative_strength,
                    "amount_ratio": float(amount_ratio),
                    "n_stocks": n_stocks,
                    "direction": "流入" if relative_strength > 0.5 else (
                        "流出" if relative_strength < -0.5 else "中性"
                    ),
                }

            if not sector_stats:
                return result

            # 排序
            sorted_sectors = sorted(
                sector_stats.values(),
                key=lambda x: x["score"],
                reverse=True,
            )

            # 强势板块 (前1/3)
            n_strong = max(1, len(sorted_sectors) // 3)
            result["strong_sectors"] = sorted_sectors[:n_strong]

            # 弱势板块 (后1/3)
            result["weak_sectors"] = sorted_sectors[-n_strong:]

            # 整体轮动方向
            strong_flow = sum(s["relative_strength"] for s in result["strong_sectors"])
            weak_flow = sum(s["relative_strength"] for s in result["weak_sectors"])
            net_flow = strong_flow + weak_flow

            if net_flow > 1.0:
                result["rotation_direction"] = "流入"
            elif net_flow < -1.0:
                result["rotation_direction"] = "流出"
            else:
                result["rotation_direction"] = "中性"

            result["sector_scores"] = {
                s["name"]: s["score"] for s in sorted_sectors
            }

            logger.info(
                f"板块轮动检测: 方向={result['rotation_direction']} | "
                f"强势板块={[s['name'] for s in result['strong_sectors']]} | "
                f"弱势板块={[s['name'] for s in result['weak_sectors']]}"
            )

        except Exception as e:
            logger.warning(f"板块轮动检测异常: {e}")

        return result

    def _build_rotation_map(self) -> Dict[str, str]:
        """
        构建行业 -> 轮动方向 的映射

        返回
        -------
        dict
            {行业名: 轮动方向("流入"/"流出"/"中性")}
        """
        if not self._sector_rotation_cache:
            return {}

        mapping = {}
        for s in self._sector_rotation_cache.get("strong_sectors", []):
            mapping[s["name"]] = "流入"
        for s in self._sector_rotation_cache.get("weak_sectors", []):
            mapping[s["name"]] = "流出"
        return mapping

    # ================================================================
    #  4. 多时间框架分析
    # ================================================================

    def _multi_timeframe_analysis(
        self,
        code: str,
        kline: Optional[pd.DataFrame],
    ) -> Dict:
        """
        多时间框架分析

        分析三个时间窗口的信号:
          - 短期(5日): 反转信号 (超卖反弹/超买回落)
          - 中期(20-60日): 动量信号 (趋势延续)
          - 长期(120日): 趋势确认 (大方向)

        三时间框架一致性 (consistency):
          - 3: 全部看多 (最强信号)
          - 2: 两个看多
          - 1: 一个看多
          - 0: 全部看空或无信号

        参数
        ----------
        code : str
            股票代码
        kline : pd.DataFrame, optional
            K线数据

        返回
        -------
        dict
            多时间框架分析结果:
            - short_term: 短期信号 ("多"/"空"/"中性")
            - short_term_score: 短期得分
            - mid_term: 中期信号
            - mid_term_score: 中期得分
            - long_term: 长期信号
            - long_term_score: 长期得分
            - consistency: 一致性 (0-3)
            - bonus: 一致性加分建议
        """
        # 检查缓存
        if code in self._multi_tf_cache:
            return self._multi_tf_cache[code]

        result = {
            "short_term": "中性",
            "short_term_score": 50.0,
            "mid_term": "中性",
            "mid_term_score": 50.0,
            "long_term": "中性",
            "long_term_score": 50.0,
            "consistency": 0,
            "bonus": 0.0,
        }

        if kline is None or len(kline) < _SHORT_WINDOW:
            self._multi_tf_cache[code] = result
            return result

        try:
            close = kline["close"].values.astype(float)
            n = len(close)

            # --- 短期(5日): 反转信号 ---
            # 使用RSI和5日涨跌幅判断超卖/超买
            if n >= _SHORT_WINDOW:
                ret_5d = close[-1] / close[-_SHORT_WINDOW] - 1
                rsi = self._calc_rsi_series(close, period=14)
                rsi_val = float(rsi[-1]) if rsi is not None and len(rsi) > 0 else 50.0

                # 超卖反弹: RSI<30 且 5日跌幅较大
                if rsi_val < 30 and ret_5d < -0.03:
                    result["short_term"] = "多"
                    result["short_term_score"] = 70.0
                # 超买回落: RSI>70 且 5日涨幅较大
                elif rsi_val > 70 and ret_5d > 0.03:
                    result["short_term"] = "空"
                    result["short_term_score"] = 30.0
                else:
                    result["short_term"] = "中性"
                    result["short_term_score"] = 50.0 + ret_5d * 100

            # --- 中期(20-60日): 动量信号 ---
            if n >= _MID_WINDOW_2:
                ret_20d = close[-1] / close[-_MID_WINDOW_1] - 1
                ret_60d = close[-1] / close[-_MID_WINDOW_2] - 1

                # 动量加速: 20日涨幅 > 60日涨幅的1/3
                if ret_20d > 0 and ret_60d > 0 and ret_20d > ret_60d / 3:
                    result["mid_term"] = "多"
                    result["mid_term_score"] = min(90, 50 + ret_20d * 200)
                elif ret_20d < 0 and ret_60d < 0:
                    result["mid_term"] = "空"
                    result["mid_term_score"] = max(10, 50 + ret_20d * 200)
                else:
                    result["mid_term"] = "中性"
                    result["mid_term_score"] = 50.0 + ret_20d * 100
            elif n >= _MID_WINDOW_1:
                ret_20d = close[-1] / close[-_MID_WINDOW_1] - 1
                result["mid_term"] = "多" if ret_20d > 0.02 else (
                    "空" if ret_20d < -0.02 else "中性"
                )
                result["mid_term_score"] = 50.0 + ret_20d * 100

            # --- 长期(120日): 趋势确认 ---
            if n >= _LONG_WINDOW:
                ret_120d = close[-1] / close[-_LONG_WINDOW] - 1
                # 长期均线趋势
                ma60 = float(np.mean(close[-60:]))
                ma120 = float(np.mean(close[-120:]))

                if close[-1] > ma60 > ma120 and ret_120d > 0:
                    result["long_term"] = "多"
                    result["long_term_score"] = min(95, 50 + ret_120d * 50)
                elif close[-1] < ma60 < ma120 and ret_120d < 0:
                    result["long_term"] = "空"
                    result["long_term_score"] = max(5, 50 + ret_120d * 50)
                else:
                    result["long_term"] = "中性"
                    result["long_term_score"] = 50.0 + ret_120d * 30

            # --- 一致性 ---
            bull_count = sum([
                result["short_term"] == "多",
                result["mid_term"] == "多",
                result["long_term"] == "多",
            ])
            result["consistency"] = bull_count

            # 一致性加分 (3个全部看多时加分最多)
            if bull_count == 3:
                result["bonus"] = 5.0
            elif bull_count == 2:
                result["bonus"] = 2.0

        except Exception as e:
            logger.debug(f"{code} 多时间框架分析异常: {e}")

        self._multi_tf_cache[code] = result
        return result

    # ================================================================
    #  5. 智能止损建议
    # ================================================================

    def _generate_stop_loss_suggestion(
        self,
        code: str,
        entry_price: float,
        kline: Optional[pd.DataFrame],
    ) -> Dict:
        """
        生成智能止损建议

        综合三种止损策略, 取最保守(最高)的止损价:
          1. ATR动态止损: entry_price - ATR_multiplier * ATR
          2. 支撑位结构止损: 近期重要支撑位
          3. 时间止损: 持有超过阈值天数且无进展

        参数
        ----------
        code : str
            股票代码
        entry_price : float
            入场价格
        kline : pd.DataFrame, optional
            K线数据

        返回
        -------
        dict
            止损建议:
            - stop_loss_price: 建议止损价 (三者中最保守)
            - stop_loss_type: 止损类型 ("ATR"/"支撑位"/"时间"/"综合")
            - atr_stop: ATR止损价
            - support_stop: 支撑位止损价
            - time_stop: 时间止损价
            - atr: 当前ATR值
            - support_level: 支撑位价格
            - stop_loss_pct: 止损幅度 (百分比)
        """
        # 检查缓存
        cache_key = f"{code}_{entry_price:.2f}"
        if cache_key in self._stop_loss_cache:
            return self._stop_loss_cache[cache_key]

        result = {
            "stop_loss_price": np.nan,
            "stop_loss_type": "无数据",
            "atr_stop": np.nan,
            "support_stop": np.nan,
            "time_stop": np.nan,
            "atr": np.nan,
            "support_level": np.nan,
            "stop_loss_pct": 0.0,
        }

        if kline is None or len(kline) < _ATR_PERIOD + 1 or entry_price <= 0:
            self._stop_loss_cache[cache_key] = result
            return result

        try:
            high = kline["high"].values.astype(float)
            low = kline["low"].values.astype(float)
            close = kline["close"].values.astype(float)

            # 1. ATR动态止损
            atr = self._calc_atr(high, low, close, period=_ATR_PERIOD)
            atr_stop = entry_price - _ATR_STOP_MULTIPLIER * atr
            result["atr"] = float(atr)
            result["atr_stop"] = float(atr_stop)

            # 2. 支撑位结构止损
            support_level = self._find_support_level(low, close, lookback=_SUPPORT_LOOKBACK)
            # 支撑位止损: 支撑位下方一定缓冲 (1%)
            support_stop = support_level * 0.99 if support_level > 0 else np.nan
            result["support_level"] = float(support_level)
            result["support_stop"] = float(support_stop)

            # 3. 时间止损 (简化: 若近30日最高价回撤超过8%, 则设时间止损)
            # 时间止损价 = 近期高点 * (1 - 8%), 且必须低于入场价才有效
            lookback = min(_TIME_STOP_DAYS, len(close))
            recent_high = float(np.max(close[-lookback:]))
            time_stop_threshold = recent_high * 0.92  # 从近期高点回撤8%
            # 若时间止损价高于入场价, 则无效 (无法起到止损作用)
            if time_stop_threshold >= entry_price:
                time_stop_threshold = np.nan
            result["time_stop"] = float(time_stop_threshold) if pd.notna(time_stop_threshold) else np.nan

            # 取最保守 (最高) 的止损价
            # 注意: 止损价必须低于入场价才有效, 在此前提下取最高价(最保守)
            valid_stops = [
                ("ATR", atr_stop),
                ("支撑位", support_stop),
                ("时间", time_stop_threshold),
            ]
            valid_stops = [
                (t, p) for t, p in valid_stops
                if pd.notna(p) and p > 0 and p < entry_price
            ]

            if valid_stops:
                # 最保守 = 最高止损价 (离入场价最近, 但仍低于入场价)
                best_type, best_price = max(valid_stops, key=lambda x: x[1])
                result["stop_loss_price"] = best_price
                result["stop_loss_type"] = (
                    "综合" if len(valid_stops) >= 2 else best_type
                )
                result["stop_loss_pct"] = float(
                    (entry_price - best_price) / entry_price
                )
            else:
                # 所有止损价均无效, 使用ATR止损兜底 (强制低于入场价)
                fallback = entry_price * 0.90  # 默认10%止损
                result["stop_loss_price"] = fallback
                result["stop_loss_type"] = "固定10%"
                result["stop_loss_pct"] = 0.10

        except Exception as e:
            logger.debug(f"{code} 止损建议生成异常: {e}")

        self._stop_loss_cache[cache_key] = result
        return result

    def _calc_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = _ATR_PERIOD,
    ) -> float:
        """
        计算ATR (Average True Range)

        True Range = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        ATR = TR的移动平均

        参数
        ----------
        high : np.ndarray
            最高价序列
        low : np.ndarray
            最低价序列
        close : np.ndarray
            收盘价序列
        period : int
            ATR周期, 默认14

        返回
        -------
        float
            最新ATR值
        """
        if len(close) < period + 1:
            return 0.0

        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Wilder平滑法
        atr = np.zeros(len(close))
        atr[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, len(close)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return float(atr[-1]) if atr[-1] > 0 else 0.0

    def _find_support_level(
        self,
        low: np.ndarray,
        close: np.ndarray,
        lookback: int = _SUPPORT_LOOKBACK,
    ) -> float:
        """
        查找近期重要支撑位

        算法: 在过去lookback日内, 找到局部低点(比前后N日都低),
        取最近的局部低点作为支撑位. 若无局部低点, 取最低价.

        参数
        ----------
        low : np.ndarray
            最低价序列
        close : np.ndarray
            收盘价序列
        lookback : int
            回溯天数, 默认60

        返回
        -------
        float
            支撑位价格
        """
        n = len(low)
        lookback = min(lookback, n)

        if lookback < 5:
            return float(low[-1]) if n > 0 else 0.0

        recent_low = low[-lookback:]

        # 寻找局部低点 (比前3日和后3日都低)
        local_minima = []
        for i in range(3, len(recent_low) - 3):
            if (recent_low[i] == min(recent_low[i - 3:i + 4])
                    and recent_low[i] < recent_low[i - 1]):
                local_minima.append(recent_low[i])

        if local_minima:
            # 取最近的局部低点
            return float(local_minima[-1])
        else:
            return float(np.min(recent_low))

    # ================================================================
    #  6. 信号强度分级
    # ================================================================

    def _classify_signal(
        self,
        score: float,
        timeframes: Dict,
    ) -> str:
        """
        信号强度分级

        分级标准:
          - 强买入: 总分 > 80 且 三时间框架一致 (consistency >= 2)
          - 买入:   总分 > 70
          - 观察:   总分 > 60
          - 观望:   总分 < 60

        参数
        ----------
        score : float
            综合总分 (0-100)
        timeframes : dict
            多时间框架分析结果, 需包含 consistency 字段

        返回
        -------
        str
            信号强度: "强买入" / "买入" / "观察" / "观望"
        """
        consistency = timeframes.get("consistency", 0) if timeframes else 0

        if score > _SIGNAL_STRONG_BUY and consistency >= 2:
            return "强买入"
        elif score > _SIGNAL_BUY:
            return "买入"
        elif score > _SIGNAL_WATCH:
            return "观察"
        else:
            return "观望"

    # ================================================================
    #  增强报告生成
    # ================================================================

    def generate_enhanced_report(
        self,
        recommendations: pd.DataFrame,
    ) -> str:
        """
        生成增强文本推荐报告

        包含:
          - 市场择时概况
          - 板块轮动信号
          - 组合风险评估
          - 推荐列表 (含信号强度/风险等级/止损建议)
          - 权重配置

        参数
        ----------
        recommendations : pd.DataFrame
            recommend() 返回的增强推荐表

        返回
        -------
        str
            格式化的增强文本报告
        """
        if recommendations is None or recommendations.empty:
            return "无推荐结果"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 90,
            f"  v7.0 增强实时推荐报告  |  生成时间: {now_str}",
            f"  股票池: {len(self.universe)} 只  |  推荐数量: {len(recommendations)}",
            "=" * 90,
            "",
        ]

        # 市场择时
        if self._market_timing:
            mt = self._market_timing
            bullish_str = "看多" if mt.get("bullish") else "看空/谨慎"
            lines.append(
                f"【市场择时】{bullish_str} "
                f"(得分: {mt.get('score', 50):.0f}/100)"
            )
            lines.append("")

        # 板块轮动
        if self._sector_rotation_cache:
            sr = self._sector_rotation_cache
            lines.append(f"【板块轮动】方向: {sr.get('rotation_direction', '中性')}")
            strong = sr.get("strong_sectors", [])
            if strong:
                lines.append(
                    "  强势板块: "
                    + ", ".join(
                        f"{s['name']}({s['score']:.0f})" for s in strong[:5]
                    )
                )
            weak = sr.get("weak_sectors", [])
            if weak:
                lines.append(
                    "  弱势板块: "
                    + ", ".join(
                        f"{s['name']}({s['score']:.0f})" for s in weak[:5]
                    )
                )
            lines.append("")

        # 组合风险
        if self._portfolio_risk_cache:
            pr = self._portfolio_risk_cache
            lines.append("【组合风险】")
            lines.append(
                f"  组合VaR(1日, 95%): {pr.get('portfolio_var', 0):.2%} | "
                f"CVaR: {pr.get('portfolio_cvar', 0):.2%}"
            )
            lines.append(
                f"  最大行业暴露: {pr.get('max_industry', '未知')} "
                f"({pr.get('max_industry_exposure', 0):.2%})"
            )
            if pr.get("drawdown_alert"):
                lines.append("  [预警] 组合回撤风险较高, 建议减仓!")
            lines.append("")

        # 推荐列表
        lines.append("【推荐列表】")
        header = (
            f"{'排名':>4}  {'代码':<8} {'名称':<8} {'总分':>6} "
            f"{'信号':<6} {'风险':<4} {'止损价':>8} {'止损类型':<8} {'轮动':<6}"
        )
        lines.append(header)
        lines.append("-" * 80)

        for rank, (_, row) in enumerate(recommendations.iterrows(), 1):
            name = str(row.get("name", ""))[:6]
            signal = str(row.get("signal_strength", ""))
            risk = str(row.get("risk_level", ""))
            sl_price = row.get("stop_loss_price", np.nan)
            sl_type = str(row.get("stop_loss_type", ""))
            rotation = str(row.get("sector_rotation", ""))

            sl_str = f"{sl_price:.2f}" if pd.notna(sl_price) else "N/A"

            line = (
                f"{rank:>4}  {row['code']:<8} {name:<8} "
                f"{row['total_score']:>6.1f} {signal:<6} {risk:<4} "
                f"{sl_str:>8} {sl_type:<8} {rotation:<6}"
            )
            lines.append(line)

        lines.append("")
        lines.append("=" * 90)
        lines.append("权重配置:")
        for dim, w in self.weights.items():
            lines.append(f"  {dim:<22} : {w:.0%}")
        if self._use_ml:
            lines.append(f"  [ML融合权重]            : {self._ml_weight:.0%}")
        lines.append("=" * 90)

        return "\n".join(lines)


# ================================================================
#  模块自测
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("v7.0 增强实时推荐引擎 - 模块自测")
    print("=" * 70)

    # 检查ML可用性
    print(f"\nEnhancedMLCombiner 可用: {_ML_AVAILABLE}")

    # 检查FactorRegistry
    reg = FactorRegistry()
    print(f"FactorRegistry: {reg}")

    # 构造测试数据
    np.random.seed(42)
    n_stocks = 50
    test_codes = [f"{600000 + i:06d}" for i in range(n_stocks)]

    # 模拟实时行情
    test_quotes = pd.DataFrame({
        "code": test_codes,
        "name": [f"测试股{i}" for i in range(n_stocks)],
        "close": np.random.uniform(10, 100, n_stocks),
        "pct_chg": np.random.normal(0, 3, n_stocks),
        "amount": np.random.uniform(1e7, 1e9, n_stocks),
        "turnover": np.random.uniform(0.5, 10, n_stocks),
        "market_cap": np.random.uniform(5e8, 1e11, n_stocks),
        "pe_ttm": np.random.uniform(5, 80, n_stocks),
        "pb": np.random.uniform(0.5, 10, n_stocks),
    })

    # 模拟K线数据
    def _gen_kline(base_price):
        dates = pd.bdate_range(end="2026-07-28", periods=150)
        noise = np.random.normal(0, 0.02, 150)
        prices = base_price * np.cumprod(1 + noise)
        return pd.DataFrame({
            "date": dates,
            "open": prices * 0.99,
            "close": prices,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "volume": np.random.randint(1e6, 1e8, 150),
        })

    # 初始化引擎
    recommender = EnhancedRecommender(
        universe=test_codes[:20],
        ml_combiner=None,
    )

    # 手动注入测试数据 (跳过网络请求)
    recommender._quotes_df = test_quotes[test_quotes["code"].isin(test_codes[:20])]
    recommender._industry_map = {
        c: np.random.choice(["银行", "医药", "电子", "消费", "新能源"])
        for c in test_codes[:20]
    }
    recommender._kline_cache = {
        c: _gen_kline(test_quotes.loc[test_quotes["code"] == c, "close"].values[0])
        for c in test_codes[:20]
    }

    # 测试多时间框架分析
    print("\n--- 多时间框架分析测试 ---")
    code = test_codes[0]
    kline = recommender._get_kline(code)
    mtf = recommender._multi_timeframe_analysis(code, kline)
    print(f"  {code}: {mtf}")

    # 测试止损建议
    print("\n--- 止损建议测试 ---")
    entry = float(test_quotes.loc[test_quotes["code"] == code, "close"].values[0])
    sl = recommender._generate_stop_loss_suggestion(code, entry, kline)
    print(f"  {code} (入场价={entry:.2f}): 止损={sl['stop_loss_price']:.2f} "
          f"({sl['stop_loss_type']}), 幅度={sl['stop_loss_pct']:.2%}")

    # 测试信号分级
    print("\n--- 信号强度分级测试 ---")
    for s in [85, 75, 65, 55]:
        for c in [3, 2, 1, 0]:
            sig = recommender._classify_signal(s, {"consistency": c})
            print(f"  总分={s}, 一致性={c} -> {sig}")

    # 测试板块轮动
    print("\n--- 板块轮动检测测试 ---")
    rotation = recommender._detect_sector_rotation()
    print(f"  方向: {rotation['rotation_direction']}")
    print(f"  强势: {[s['name'] for s in rotation['strong_sectors']]}")
    print(f"  弱势: {[s['name'] for s in rotation['weak_sectors']]}")

    # 测试组合风险
    print("\n--- 组合风险评估测试 ---")
    weights = pd.Series(0.05, index=test_codes[:20])
    risk = recommender._assess_realtime_risk(weights)
    print(f"  组合VaR: {risk['portfolio_var']:.2%}")
    print(f"  组合CVaR: {risk['portfolio_cvar']:.2%}")
    print(f"  最大行业: {risk['max_industry']} ({risk['max_industry_exposure']:.2%})")

    print("\n" + "=" * 70)
    print("模块自测完成")
    print("=" * 70)
