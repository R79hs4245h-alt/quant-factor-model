"""
自适应Alpha因子挖掘模块 v6.2
功能:
  1. 因子组合挖掘: 通过运算符(加减乘除、时序运算)自动生成新因子
  2. IC有效性筛选: 自动测试新因子IC,保留有效因子
  3. 因子衰减监控: 跟踪因子IC随时间的变化,识别衰减因子
  4. 因子正交化: 对高度相关的因子做正交化处理
  5. 因子重要性排序: 基于IC/IR/衰减率综合排序
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from config import FACTOR_CONFIG
from utils import get_logger

logger = get_logger("alpha_factor_mining")

# ============ 全局参数 ============
IC_ABS_THRESHOLD = 0.02          # IC绝对值阈值
IC_IR_THRESHOLD = 0.3             # IC_IR阈值
MAX_NEW_FACTORS = 20              # 最多保留新因子数量
ORTHOGONAL_CORR_THRESHOLD = 0.7  # 正交化相关系数阈值
DECAY_MIN_RECORDS = 6             # 判定衰减所需最少记录数


class AlphaFactorMiner:
    """
    自适应Alpha因子挖掘器

    通过算术运算和时序运算自动组合基础因子,
    基于IC/IR指标筛选有效因子并进行正交化处理。
    """

    # 支持的二元运算符 (用于两因子组合)
    BINARY_OPS = ["+", "-", "*", "/"]

    # 支持的一元运算符 (用于单因子变换)
    UNARY_OPS = ["rank", "ts_mean_5", "ts_mean_10", "ts_std_5", "ts_std_10",
                 "ts_rank_10", "delay_1", "delay_5", "delta_1", "delta_5"]

    def __init__(self, factor_config: Optional[Dict] = None):
        """
        初始化因子挖掘器

        :param factor_config: 因子配置字典, 默认使用 config.FACTOR_CONFIG
        """
        self.factor_config = factor_config or FACTOR_CONFIG
        self.mined_factors: OrderedDict[str, Dict] = OrderedDict()
        self._factor_values: Dict[str, pd.Series] = {}  # 缓存已计算因子值
        logger.info("AlphaFactorMiner 初始化完成, 基础因子数: %d", len(self.factor_config))

    def mine_factors(self, base_factors: pd.DataFrame,
                     forward_returns: pd.Series) -> pd.DataFrame:
        """
        挖掘新因子: 自动组合基础因子, 筛选有效新因子

        :param base_factors: 基础因子面板 DataFrame(index=code, columns=factor_names)
        :param forward_returns: 未来收益序列 Series(index=code), 与base_factors对齐
        :return: 新因子 DataFrame(index=code, columns=new_factor_names)
        """
        logger.info("开始因子挖掘, 基础因子维度: %s, 收益序列长度: %d",
                     base_factors.shape, len(forward_returns))

        # 第一步: 生成因子组合表达式
        expressions = self._generate_combinations(base_factors)
        logger.info("生成因子组合表达式: %d 条", len(expressions))

        # 第二步: 逐条计算并评估
        valid_factors: OrderedDict[str, pd.Series] = OrderedDict()
        for expr in expressions:
            if len(valid_factors) >= MAX_NEW_FACTORS:
                logger.info("已达到最大新因子数量 %d, 停止挖掘", MAX_NEW_FACTORS)
                break

            try:
                factor_values = self._compute_expression(base_factors, expr)
                if factor_values is None:
                    continue

                # 对齐收益序列
                common_idx = factor_values.index.intersection(forward_returns.index)
                if len(common_idx) < 30:
                    continue

                fv_aligned = factor_values.loc[common_idx]
                fr_aligned = forward_returns.loc[common_idx]

                # 评估因子有效性
                eval_result = self._evaluate_factor(fv_aligned, fr_aligned)
                if eval_result is None:
                    continue

                # 通过阈值检查
                if abs(eval_result["ic"]) >= IC_ABS_THRESHOLD and eval_result["ic_ir"] >= IC_IR_THRESHOLD:
                    # 正交化处理
                    existing_df = pd.DataFrame(valid_factors)
                    ortho_factor = self._orthogonalize(fv_aligned, existing_df)

                    # 正交化后再次验证IC
                    common_idx2 = ortho_factor.index.intersection(forward_returns.index)
                    if len(common_idx2) >= 30:
                        re_eval = self._evaluate_factor(
                            ortho_factor.loc[common_idx2],
                            forward_returns.loc[common_idx2],
                        )
                        if re_eval is not None and abs(re_eval["ic"]) >= IC_ABS_THRESHOLD * 0.8:
                            valid_factors[expr] = ortho_factor
                            eval_result["expression"] = expr
                            eval_result["orthogonalized_ic"] = re_eval["ic"]
                            self.mined_factors[expr] = eval_result
                            logger.info("  保留因子: %s | IC=%.4f IR=%.4f 覆盖=%.2f%%",
                                        expr, eval_result["ic"],
                                        eval_result["ic_ir"],
                                        eval_result["coverage"] * 100)
                            continue

                    # 正交化后退化, 保留原始因子
                    valid_factors[expr] = fv_aligned
                    eval_result["expression"] = expr
                    eval_result["orthogonalized_ic"] = None
                    self.mined_factors[expr] = eval_result
                    logger.info("  保留因子(原始): %s | IC=%.4f IR=%.4f",
                                expr, eval_result["ic"], eval_result["ic_ir"])

            except Exception as e:
                logger.debug("因子计算异常 [%s]: %s", expr, e)
                continue

        # 组装结果
        if valid_factors:
            result_df = pd.DataFrame(valid_factors)
            logger.info("因子挖掘完成, 有效新因子: %d / %d",
                        len(valid_factors), len(expressions))
        else:
            result_df = pd.DataFrame()
            logger.info("因子挖掘完成, 未发现有效新因子")

        return result_df

    def _generate_combinations(self, base_factors: pd.DataFrame) -> List[str]:
        """
        生成因子组合表达式列表

        策略:
          1. 二元运算: 选取因子方向相同的因子对, 应用四则运算
          2. 一元运算: 对每个基础因子应用时序/排名变换
          3. 限制总表达式数量以控制计算量

        :param base_factors: 基础因子 DataFrame
        :return: 表达式字符串列表
        """
        factor_names = list(base_factors.columns)
        expressions: List[str] = []

        # 按类别分组, 优先组合同类因子
        category_groups: Dict[str, List[str]] = {}
        for name in factor_names:
            if name in self.factor_config:
                cat = self.factor_config[name].get("category", "other")
            else:
                cat = "other"
            category_groups.setdefault(cat, []).append(name)

        # --- 二元运算: 同类因子两两组合 ---
        used_pairs = set()
        for cat, names in category_groups.items():
            if len(names) < 2:
                continue
            # 控制同类组合数量
            for i in range(min(len(names), 6)):
                for j in range(i + 1, min(len(names), 6)):
                    for op in self.BINARY_OPS:
                        expr = f"{names[i]}{op}{names[j]}"
                        reverse_expr = f"{names[j]}{op}{names[i]}"
                        # 避免加法和乘法的重复
                        if op in ("+", "*") and reverse_expr in used_pairs:
                            continue
                        used_pairs.add(expr)
                        expressions.append(expr)

        # --- 跨类别组合: 价值 * 动量, 质量 * 反转 等 ---
        priority_pairs = [
            ("value", "momentum"),
            ("quality", "momentum"),
            ("value", "reversal"),
            ("growth", "momentum"),
            ("liquidity", "momentum"),
            ("quality", "volatility"),
        ]
        for cat_a, cat_b in priority_pairs:
            if cat_a in category_groups and cat_b in category_groups:
                names_a = category_groups[cat_a][:3]
                names_b = category_groups[cat_b][:3]
                for na in names_a:
                    for nb in names_b:
                        for op in ["+", "-", "*", "/"]:
                            expressions.append(f"{na}{op}{nb}")

        # --- 一元运算: 对每个因子应用变换 ---
        for name in factor_names:
            for op in self.UNARY_OPS:
                expressions.append(f"{op}({name})")

        # --- 二元 + 一元混合: 变换后的因子做简单加减 ---
        selected_unary = []
        for name in factor_names[:5]:  # 限制范围
            for op in ["rank", "ts_mean_5", "ts_std_5"]:
                selected_unary.append(f"{op}({name})")

        for i, expr_a in enumerate(selected_unary):
            for j, expr_b in enumerate(selected_unary):
                if i < j:
                    expressions.append(f"{expr_a}+{expr_b}")
                    expressions.append(f"{expr_a}-{expr_b}")

        logger.info("组合表达式总数: %d (二元配对: %d, 一元变换: %d)",
                     len(expressions), len(used_pairs),
                     len(factor_names) * len(self.UNARY_OPS))

        return expressions

    def _compute_expression(self, base_factors: pd.DataFrame,
                            expression: str) -> Optional[pd.Series]:
        """
        根据表达式计算因子值

        :param base_factors: 基础因子 DataFrame
        :param expression: 因子表达式字符串
        :return: 计算得到的因子 Series, 或 None(计算失败)
        """
        try:
            # 处理一元运算: op(factor_name)
            for op in self.UNARY_OPS:
                prefix = f"{op}("
                if expression.startswith(prefix) and expression.endswith(")"):
                    factor_name = expression[len(prefix):-1]
                    if factor_name not in base_factors.columns:
                        return None
                    raw = base_factors[factor_name].astype(float)
                    return self._apply_unary_op(raw, op)

            # 处理二元运算: 解析 operator
            # 按优先级尝试解析, 优先匹配 + 和 - (因为它们可能出现在表达式中)
            parsed = self._parse_binary_expression(expression, base_factors.columns)
            if parsed is None:
                return None

            left_name, op, right_name = parsed
            left = base_factors[left_name].astype(float)
            right = base_factors[right_name].astype(float)

            if op == "+":
                result = left + right
            elif op == "-":
                result = left - right
            elif op == "*":
                result = left * right
            elif op == "/":
                result = np.where(
                    (right.abs() > 1e-10) & (~np.isnan(right)),
                    left / right,
                    np.nan,
                )
                result = pd.Series(result, index=base_factors.index, dtype=float)
            else:
                return None

            # 清理无穷值
            result = result.replace([np.inf, -np.inf], np.nan)
            return result

        except Exception as e:
            logger.debug("表达式计算失败 [%s]: %s", expression, e)
            return None

    def _parse_binary_expression(self, expression: str,
                                 available_factors: List[str]) -> Optional[Tuple[str, str, str]]:
        """
        解析二元表达式, 提取左因子、运算符、右因子

        :param expression: 表达式字符串
        :param available_factors: 可用因子名列表
        :return: (left, op, right) 或 None
        """
        # 尝试在所有可用因子中找到匹配
        # 按因子名长度降序排列, 优先匹配长名称避免歧义
        sorted_factors = sorted(available_factors, key=len, reverse=True)

        for op in self.BINARY_OPS:
            # 在表达式中定位运算符
            # 需要确保运算符不在一元表达式内部
            op_positions = []
            pos = 0
            while True:
                idx = expression.find(op, pos)
                if idx == -1:
                    break
                op_positions.append(idx)
                pos = idx + 1

            for op_idx in op_positions:
                left_part = expression[:op_idx]
                right_part = expression[op_idx + len(op):]

                # 跳过不合法的分割
                if not left_part or not right_part:
                    continue
                if left_part.endswith("(") or right_part.startswith("("):
                    continue

                if left_part in available_factors and right_part in available_factors:
                    return (left_part, op, right_part)

        return None

    def _apply_unary_op(self, series: pd.Series, op: str) -> pd.Series:
        """
        应用一元运算

        :param series: 输入因子值
        :param op: 运算符名称
        :return: 变换后的 Series
        """
        s = series.copy()

        if op == "rank":
            return s.rank(pct=True)

        elif op.startswith("ts_mean_"):
            window = int(op.split("_")[-1])
            return s.rolling(window, min_periods=window // 2).mean()

        elif op.startswith("ts_std_"):
            window = int(op.split("_")[-1])
            return s.rolling(window, min_periods=window // 2).std()

        elif op.startswith("ts_rank_"):
            window = int(op.split("_")[-1])
            return s.rolling(window, min_periods=window // 2).rank(pct=True).iloc[:, -1] \
                if hasattr(s, 'rolling') else s

        elif op.startswith("delay_"):
            periods = int(op.split("_")[-1])
            return s.shift(periods)

        elif op.startswith("delta_"):
            periods = int(op.split("_")[-1])
            return s.diff(periods)

        else:
            logger.debug("未知一元运算: %s", op)
            return s

    def _evaluate_factor(self, factor: pd.Series,
                         forward_returns: pd.Series) -> Optional[Dict]:
        """
        评估单因子有效性: 计算IC、IR、覆盖率

        :param factor: 因子值 Series
        :param forward_returns: 未来收益 Series
        :return: 评估结果字典, 或 None(评估失败)
        """
        # 对齐
        common_idx = factor.index.intersection(forward_returns.index)
        if len(common_idx) < 30:
            return None

        f = factor.loc[common_idx].astype(float)
        r = forward_returns.loc[common_idx].astype(float)

        # 去除NaN
        mask = f.notna() & r.notna() & (r.abs() < 1.0)  # 排除极端收益
        f_clean = f[mask]
        r_clean = r[mask]

        if len(f_clean) < 30:
            return None

        # 计算IC (Spearman秩相关)
        ic = f_clean.corr(r_clean, method="spearman")
        if pd.isna(ic):
            return None

        # 计算IC_IR (IC的绝对值 / IC的标准差代理)
        # 使用bootstrap估算IC的稳定性
        n_samples = min(len(f_clean), 200)
        if n_samples < 30:
            return None

        ic_samples = []
        for _ in range(100):
            sample_idx = np.random.choice(len(f_clean), size=n_samples, replace=True)
            ic_sample = pd.Series(f_clean.iloc[sample_idx].values).corr(
                pd.Series(r_clean.iloc[sample_idx].values), method="spearman"
            )
            if pd.notna(ic_sample):
                ic_samples.append(ic_sample)

        if not ic_samples:
            return None

        ic_std = np.std(ic_samples, ddof=1)
        ic_ir = abs(ic) / (ic_std + 1e-10)

        # 覆盖率
        coverage = len(f_clean) / len(common_idx)

        return {
            "ic": float(ic),
            "ic_std": float(ic_std),
            "ic_ir": float(ic_ir),
            "coverage": float(coverage),
            "n_stocks": int(len(f_clean)),
        }

    def _orthogonalize(self, factor: pd.Series,
                       existing_factors: pd.DataFrame) -> pd.Series:
        """
        Schmidt正交化: 将因子与已有因子正交化

        对因子依次减去在已有因子方向上的投影,
        降低冗余性, 提升因子独立信息含量。

        :param factor: 待正交化的因子 Series
        :param existing_factors: 已有因子 DataFrame
        :return: 正交化后的因子 Series
        """
        if existing_factors.empty:
            return factor

        result = factor.copy().astype(float).fillna(0)

        for col in existing_factors.columns:
            base = existing_factors[col].astype(float).fillna(0)
            common_idx = result.index.intersection(base.index)
            if len(common_idx) < 10:
                continue

            r = result.loc[common_idx]
            b = base.loc[common_idx]

            norm_sq = (b ** 2).sum()
            if norm_sq < 1e-10:
                continue

            # 检查相关性, 只对高相关因子做正交化
            corr = r.corr(b)
            if pd.isna(corr) or abs(corr) < ORTHOGONAL_CORR_THRESHOLD:
                continue

            # Schmidt投影: 减去在base方向上的分量
            projection = (r * b).sum() / norm_sq * b
            result.loc[common_idx] = r - projection.values
            logger.debug("  正交化: 减去 %s 方向投影 (corr=%.3f)", col, corr)

        # 恢复NaN位置
        nan_mask = factor.isna()
        result.loc[nan_mask] = np.nan

        return result

    def get_factor_report(self) -> pd.DataFrame:
        """
        生成因子报告: 按综合重要性排序

        综合得分 = 0.4 * |IC|_norm + 0.3 * IR_norm + 0.3 * (1 - 衰减率)
        (若无衰减信息则衰减项取默认值0.5)

        :return: 因子报告 DataFrame
        """
        if not self.mined_factors:
            logger.info("暂无已挖掘因子, 返回空报告")
            return pd.DataFrame()

        records = []
        for expr, info in self.mined_factors.items():
            record = {
                "expression": expr,
                "ic": info.get("ic", 0),
                "ic_ir": info.get("ic_ir", 0),
                "ic_std": info.get("ic_std", 0),
                "coverage": info.get("coverage", 0),
                "n_stocks": info.get("n_stocks", 0),
                "orthogonalized_ic": info.get("orthogonalized_ic"),
            }
            records.append(record)

        report = pd.DataFrame(records)

        # 归一化
        if len(report) > 1:
            ic_abs = report["ic"].abs()
            report["ic_norm"] = (ic_abs - ic_abs.min()) / (ic_abs.max() - ic_abs.min() + 1e-10)

            ir_vals = report["ic_ir"]
            report["ir_norm"] = (ir_vals - ir_vals.min()) / (ir_vals.max() - ir_vals.min() + 1e-10)

            # 无衰减信息时, 衰减项默认0.5
            report["decay_norm"] = 0.5

            # 综合得分
            report["score"] = (
                0.4 * report["ic_norm"]
                + 0.3 * report["ir_norm"]
                + 0.3 * report["decay_norm"]
            )
        else:
            report["score"] = 1.0

        report = report.sort_values("score", ascending=False).reset_index(drop=True)
        report.index = report.index + 1  # 排名从1开始
        report.index.name = "rank"

        logger.info("因子报告生成完成, 共 %d 个因子, Top1: %s (score=%.3f)",
                     len(report), report.iloc[0]["expression"],
                     report.iloc[0]["score"])
        return report


class FactorDecayTracker:
    """
    因子衰减追踪器

    记录因子IC随时间的变化趋势, 识别正在衰减的因子,
    支持设置衰减阈值来筛选需要关注的因子。
    """

    def __init__(self):
        """初始化衰减追踪器"""
        self._history: Dict[str, List[Dict]] = {}  # factor_name -> [{date, ic}, ...]
        logger.info("FactorDecayTracker 初始化完成")

    def track(self, factor_name: str, ic: float, date: str) -> None:
        """
        追踪因子在某日期的IC值

        :param factor_name: 因子名称
        :param ic: 当期IC值
        :param date: 日期字符串 (YYYY-MM-DD 或 YYYYMMDD)
        """
        if factor_name not in self._history:
            self._history[factor_name] = []

        self._history[factor_name].append({
            "date": date,
            "ic": float(ic),
        })

        # 保留最近 120 条记录
        if len(self._history[factor_name]) > 120:
            self._history[factor_name] = self._history[factor_name][-120:]

        logger.debug("追踪因子 %s IC: %.4f @ %s", factor_name, ic, date)

    def get_decay_status(self, factor_name: str) -> Dict:
        """
        获取因子的衰减状态

        通过对比近期IC均值与远期IC均值来衡量衰减程度:
          - decay_ratio = |近期IC均值| / |远期IC均值| (越大越不衰减)
          - decay_rate = 1 - decay_ratio (越大衰减越严重)
          - 衰减方向通过近期IC趋势斜率判断

        :param factor_name: 因子名称
        :return: 衰减状态字典
        """
        default_status = {
            "factor_name": factor_name,
            "decay_rate": 0.0,
            "status": "unknown",
            "recent_ic_mean": 0.0,
            "remote_ic_mean": 0.0,
            "trend_slope": 0.0,
            "n_records": 0,
        }

        if factor_name not in self._history:
            return default_status

        records = self._history[factor_name]
        n = len(records)

        if n < DECAY_MIN_RECORDS:
            default_status["n_records"] = n
            default_status["status"] = "insufficient_data"
            return default_status

        # 按时间排序
        sorted_records = sorted(records, key=lambda x: x["date"])

        # 前1/2为远期, 后1/2为近期
        mid = n // 2
        remote_ics = [r["ic"] for r in sorted_records[:mid]]
        recent_ics = [r["ic"] for r in sorted_records[mid:]]

        remote_mean = np.mean(remote_ics)
        recent_mean = np.mean(recent_ics)

        # 衰减率: 近期IC有效性相对于远期的比值
        if abs(remote_mean) > 1e-10:
            decay_ratio = abs(recent_mean) / abs(remote_mean)
        else:
            decay_ratio = 0.0

        decay_rate = max(0.0, 1.0 - decay_ratio)

        # 趋势斜率: 对IC序列做线性回归
        ic_array = np.array([r["ic"] for r in sorted_records])
        x = np.arange(len(ic_array), dtype=float)
        if len(x) > 1:
            slope = float(np.polyfit(x, ic_array, 1)[0])
        else:
            slope = 0.0

        # 判断状态
        if decay_rate >= 0.5:
            status = "severe_decay"
        elif decay_rate >= 0.3:
            status = "decaying"
        elif decay_rate >= 0.15:
            status = "mild_decay"
        else:
            status = "stable"

        return {
            "factor_name": factor_name,
            "decay_rate": round(decay_rate, 4),
            "status": status,
            "recent_ic_mean": round(recent_mean, 4),
            "remote_ic_mean": round(remote_mean, 4),
            "trend_slope": round(slope, 6),
            "n_records": n,
        }

    def get_decaying_factors(self, threshold: float = 0.3) -> List[str]:
        """
        获取衰减因子列表

        :param threshold: 衰减率阈值, 默认0.3(衰减率>=0.3视为衰减)
        :return: 衰减因子名称列表, 按衰减率降序排列
        """
        decaying = []
        for factor_name in self._history:
            status = self.get_decay_status(factor_name)
            if status["decay_rate"] >= threshold:
                decaying.append((factor_name, status["decay_rate"]))

        # 按衰减率降序排列
        decaying.sort(key=lambda x: x[1], reverse=True)

        result = [item[0] for item in decaying]
        logger.info("衰减因子检测 (阈值=%.2f): %d / %d 个因子衰减",
                     threshold, len(result), len(self._history))
        return result

    def get_all_status(self) -> pd.DataFrame:
        """
        获取所有因子的衰减状态汇总

        :return: DataFrame, 包含每个因子的衰减指标
        """
        if not self._history:
            return pd.DataFrame()

        records = []
        for factor_name in self._history:
            status = self.get_decay_status(factor_name)
            records.append(status)

        df = pd.DataFrame(records)
        df = df.sort_values("decay_rate", ascending=False).reset_index(drop=True)
        return df
