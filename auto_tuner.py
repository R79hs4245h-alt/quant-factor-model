"""
自动调参模块 v6.2 (v6.2新增模块)
功能:
  1. 网格搜索: 对关键参数进行网格搜索找最优组合
  2. 贝叶斯优化: 使用scikit-optimize进行高效参数优化
  3. 交叉验证: 时间序列交叉验证防过拟合
  4. 参数敏感性分析: 分析各参数对策略表现的影响
  5. 最优参数持久化: 保存和加载最优参数配置

依赖:
  - pandas / numpy (必需)
  - scikit-optimize (可选, 不可用时贝叶斯优化退化为随机搜索)

用法示例:
    from auto_tuner import (
        ParamSpec, GridSearchTuner, BayesianTuner,
        TimeSeriesCV, DEFAULT_PARAM_SPECS,
        save_best_params, load_best_params,
    )

    # 1. 定义评估函数: 输入参数dict, 输出含 metric 字段的指标dict
    def evaluate(params):
        # ... 运行回测 ...
        return {"sharpe_ratio": 1.23, "annual_return": 0.25, "max_drawdown": -0.15}

    # 2. 网格搜索
    tuner = GridSearchTuner(DEFAULT_PARAM_SPECS, metric="sharpe_ratio")
    best = tuner.search(evaluate, n_trials=100)
    print(best)
    print(tuner.get_results().head())

    # 3. 贝叶斯优化
    btuner = BayesianTuner(DEFAULT_PARAM_SPECS, metric="sharpe_ratio")
    best_b = btuner.search(evaluate, n_trials=50)

    # 4. 时间序列交叉验证
    cv = TimeSeriesCV(n_splits=5, min_train_periods=12)
    for train_idx, test_idx in cv.split(rebalance_dates):
        ...

    # 5. 持久化
    save_best_params(best, "best_params.json")
    best_loaded = load_best_params("best_params.json")
"""
import json
import random
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 尝试导入 scikit-optimize, 不可用时贝叶斯优化退化为随机搜索
try:
    from skopt import Optimizer as _SkoptOptimizer
    from skopt.space import Real as _SkoptReal
    from skopt.space import Integer as _SkoptInteger
    from skopt.space import Categorical as _SkoptCategorical
    _SKOPT_AVAILABLE = True
    _SKOPT_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - 环境依赖相关
    _SKOPT_AVAILABLE = False
    _SKOPT_IMPORT_ERROR = _e
    _SkoptOptimizer = None
    _SkoptReal = None
    _SkoptInteger = None
    _SkoptCategorical = None

from config import (
    TOP_N, MAX_WEIGHT, MAX_INDUSTRY_WEIGHT, MAX_TURNOVER,
    WINSORIZE_QUANTILE, FACTOR_COMBINE_METHOD,
)
from utils import get_logger

logger = get_logger("auto_tuner")


# ============ 参数规格定义 ============
@dataclass
class ParamSpec:
    """
    参数规格定义

    Attributes:
        name: 参数名(需与 evaluate_fn 接收的参数字典键一致)
        param_type: 参数类型, "int" / "float" / "choice"
        low: 连续/整数参数的下界(对 choice 类型不使用)
        high: 连续/整数参数的上界(对 choice 类型不使用)
        choices: 离散候选值列表(仅 param_type="choice" 时使用)
        step: 网格搜索时的步长(为 None 时:
              - int/float 网格搜索退化为在 [low, high] 均匀取若干点;
              - 贝叶斯优化/随机搜索忽略 step, 在连续区间内采样)
    """
    name: str
    param_type: str = "float"
    low: float = 0.0
    high: float = 1.0
    choices: Optional[List[Any]] = None
    step: Optional[float] = None

    def __post_init__(self):
        if self.param_type not in ("int", "float", "choice"):
            raise ValueError(
                f"ParamSpec.param_type 必须是 'int'/'float'/'choice', 得到 {self.param_type!r}"
            )
        if self.param_type == "choice":
            if not self.choices:
                raise ValueError(f"choice 类型参数 {self.name!r} 必须提供非空 choices")
        else:
            if self.low > self.high:
                raise ValueError(
                    f"参数 {self.name!r} 的 low({self.low}) 不能大于 high({self.high})"
                )


# ============ 默认参数规格 (基于 config 中的策略参数) ============
DEFAULT_PARAM_SPECS: List[ParamSpec] = [
    ParamSpec(name="TOP_N", param_type="choice",
              choices=[30, 50, 80, 100]),
    ParamSpec(name="MAX_WEIGHT", param_type="choice",
              choices=[0.03, 0.05, 0.08, 0.10]),
    ParamSpec(name="MAX_INDUSTRY_WEIGHT", param_type="choice",
              choices=[0.20, 0.25, 0.30, 0.35]),
    ParamSpec(name="MAX_TURNOVER", param_type="choice",
              choices=[0.3, 0.5, 0.7]),
    ParamSpec(name="WINSORIZE_QUANTILE", param_type="choice",
              choices=[0.01, 0.025, 0.05]),
    ParamSpec(name="FACTOR_COMBINE_METHOD", param_type="choice",
              choices=["equal_weight", "ic_weight", "ic_ir_weight"]),
]


def build_default_param_specs() -> List[ParamSpec]:
    """返回默认参数规格的深拷贝, 避免外部修改全局常量"""
    return [
        ParamSpec(
            name=s.name, param_type=s.param_type,
            low=s.low, high=s.high,
            choices=list(s.choices) if s.choices is not None else None,
            step=s.step,
        )
        for s in DEFAULT_PARAM_SPECS
    ]


# ============ 内部工具函数 ============
def _sample_param(spec: ParamSpec, rng: random.Random) -> Any:
    """从参数规格中随机采样一个值(用于随机搜索退化)"""
    if spec.param_type == "choice":
        return rng.choice(spec.choices)
    if spec.param_type == "int":
        return int(rng.randint(int(spec.low), int(spec.high)))
    # float
    return rng.uniform(spec.low, spec.high)


def _extract_score(result: Any, metric: str) -> float:
    """
    从 evaluate_fn 的返回值中提取目标分数, 越大越好。
    支持:
      - 数值 (int/float)
      - dict (取 metric 键, 缺失则尝试 'score'/'value')
    """
    if isinstance(result, dict):
        if metric in result:
            val = result[metric]
        elif "score" in result:
            val = result["score"]
        elif "value" in result:
            val = result["value"]
        else:
            raise KeyError(
                f"评估结果 dict 中找不到指标 {metric!r}, 也无 'score'/'value' 兜底键。"
                f"可用键: {list(result.keys())}"
            )
    else:
        val = result
    try:
        return float(val)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"无法将评估结果转换为 float: {result!r} (metric={metric!r})"
        ) from e


# ============ 网格搜索调参器 ============
class GridSearchTuner:
    """网格搜索调参器: 遍历参数网格寻找最优组合"""

    def __init__(self, param_specs: List[ParamSpec], metric: str = "sharpe_ratio"):
        if not param_specs:
            raise ValueError("param_specs 不能为空")
        self.param_specs = param_specs
        self.metric = metric
        self._results: List[Dict[str, Any]] = []
        self._best: Optional[Dict[str, Any]] = None

    def _generate_grid(self) -> List[Dict[str, Any]]:
        """
        生成参数网格。
        - choice: 直接使用 choices
        - int: 在 [low, high] 内按 step 取整; step 为 None 时均匀取 5 个点
        - float: 在 [low, high] 内按 step 取点; step 为 None 时均匀取 5 个点
        """
        axes: List[List[Any]] = []
        for spec in self.param_specs:
            if spec.param_type == "choice":
                axes.append(list(spec.choices))
            elif spec.param_type == "int":
                lo, hi = int(spec.low), int(spec.high)
                step = spec.step if spec.step is not None else max(1, (hi - lo) / 4)
                step = max(1, int(round(step)))
                vals = list(range(lo, hi + 1, step))
                if not vals:
                    vals = [lo]
                if vals[-1] != hi and hi <= int(spec.high):
                    vals.append(hi)
                axes.append(vals)
            else:  # float
                lo, hi = float(spec.low), float(spec.high)
                step = spec.step if spec.step is not None else (hi - lo) / 4
                if step <= 0:
                    step = (hi - lo) / 4 if hi > lo else 1.0
                vals = []
                v = lo
                while v <= hi + 1e-12:
                    vals.append(round(v, 8))
                    v += step
                if not vals:
                    vals = [lo]
                if vals[-1] < hi - 1e-12:
                    vals.append(round(hi, 8))
                axes.append(vals)

        grid = []
        for combo in itertools.product(*axes):
            grid.append({spec.name: val for spec, val in zip(self.param_specs, combo)})
        return grid

    def search(self, evaluate_fn: Callable[[Dict[str, Any]], Any],
               n_trials: int = 100) -> Dict[str, Any]:
        """
        执行网格搜索。

        Args:
            evaluate_fn: 评估函数, 接收参数 dict, 返回数值或含 metric 的 dict
            n_trials: 最多评估的参数组合数(截断网格)

        Returns:
            最优参数 dict (附带 '__score__' 与 '__metrics__' 字段)
        """
        grid = self._generate_grid()
        total = len(grid)
        if total > n_trials:
            logger.info(f"网格规模 {total} 超过 n_trials={n_trials}, 仅评估前 {n_trials} 组")
            grid = grid[:n_trials]
        logger.info(f"网格搜索: 共评估 {len(grid)} 组参数 (metric={self.metric})")

        self._results = []
        best_score = -np.inf
        best_params = None
        best_metrics = None

        for i, params in enumerate(grid, 1):
            try:
                result = evaluate_fn(params)
                score = _extract_score(result, self.metric)
                metrics = result if isinstance(result, dict) else {self.metric: score}
            except Exception as e:
                logger.warning(f"第 {i}/{len(grid)} 组评估失败 ({params}): {e}")
                score = -np.inf
                metrics = {"error": str(e), self.metric: -np.inf}

            record = {**params, "__score__": score, "__trial__": i}
            if isinstance(metrics, dict):
                record["__metrics__"] = metrics
                for k, v in metrics.items():
                    record.setdefault(f"metric_{k}", v)
            self._results.append(record)

            if score > best_score:
                best_score = score
                best_params = dict(params)
                best_metrics = metrics

        if best_params is None:
            logger.warning("网格搜索未得到任何有效结果, 返回空参数")
            self._best = {"params": {}, "score": -np.inf, "metrics": {}}
            return self._best

        logger.info(f"网格搜索完成, 最优 {self.metric}={best_score:.4f}, 参数={best_params}")
        self._best = {
            "params": best_params,
            "score": best_score,
            "metrics": best_metrics if best_metrics is not None else {},
        }
        return self._best

    def get_results(self) -> pd.DataFrame:
        """获取搜索结果 DataFrame, 按 __score__ 降序"""
        if not self._results:
            return pd.DataFrame()
        df = pd.DataFrame(self._results)
        df = df.sort_values("__score__", ascending=False).reset_index(drop=True)
        return df


# ============ 贝叶斯优化调参器 ============
class BayesianTuner:
    """
    贝叶斯优化调参器。
    优先使用 scikit-optimize (skopt) 的高斯过程优化;
    若 skopt 不可用, 自动退化为随机搜索。
    """

    def __init__(self, param_specs: List[ParamSpec], metric: str = "sharpe_ratio"):
        if not param_specs:
            raise ValueError("param_specs 不能为空")
        self.param_specs = param_specs
        self.metric = metric
        self._results: List[Dict[str, Any]] = []
        self._best: Optional[Dict[str, Any]] = None
        self._used_skopt: Optional[bool] = None  # 标记本次 search 是否真正用了 skopt

    def _to_skopt_space(self) -> List[Any]:
        """
        转换为 scikit-optimize 参数空间。
        返回 skopt.space 维度对象列表, 顺序与 param_specs 一致。
        若 skopt 不可用返回 None。
        """
        if not _SKOPT_AVAILABLE:
            return None
        dims = []
        for spec in self.param_specs:
            if spec.param_type == "choice":
                dims.append(_SkoptCategorical(categories=list(spec.choices)))
            elif spec.param_type == "int":
                dims.append(_SkoptInteger(low=int(spec.low), high=int(spec.high)))
            else:  # float
                dims.append(_SkoptReal(low=float(spec.low), high=float(spec.high)))
        return dims

    def _x_to_params(self, x: List[Any]) -> Dict[str, Any]:
        """将 skopt 采样点 x 转换为参数 dict"""
        params = {}
        for spec, val in zip(self.param_specs, x):
            if spec.param_type == "int":
                params[spec.name] = int(val)
            elif spec.param_type == "float":
                params[spec.name] = float(val)
            else:
                params[spec.name] = val
        return params

    def _random_search(self, evaluate_fn: Callable[[Dict[str, Any]], Any],
                       n_trials: int) -> Tuple[Dict[str, Any], Any, float]:
        """随机搜索退化实现, 返回 (best_params, best_metrics, best_score)"""
        rng = random.Random()
        best_score = -np.inf
        best_params = None
        best_metrics = None

        for i in range(1, n_trials + 1):
            params = {spec.name: _sample_param(spec, rng) for spec in self.param_specs}
            try:
                result = evaluate_fn(params)
                score = _extract_score(result, self.metric)
                metrics = result if isinstance(result, dict) else {self.metric: score}
            except Exception as e:
                logger.warning(f"随机搜索第 {i}/{n_trials} 组评估失败 ({params}): {e}")
                score = -np.inf
                metrics = {"error": str(e), self.metric: -np.inf}

            record = {**params, "__score__": score, "__trial__": i}
            if isinstance(metrics, dict):
                record["__metrics__"] = metrics
                for k, v in metrics.items():
                    record.setdefault(f"metric_{k}", v)
            self._results.append(record)

            if score > best_score:
                best_score = score
                best_params = dict(params)
                best_metrics = metrics

        return best_params, best_metrics, best_score

    def search(self, evaluate_fn: Callable[[Dict[str, Any]], Any],
               n_trials: int = 50) -> Dict[str, Any]:
        """
        执行贝叶斯优化(最大化 metric)。

        Args:
            evaluate_fn: 评估函数, 接收参数 dict, 返回数值或含 metric 的 dict
            n_trials: 优化迭代次数

        Returns:
            最优参数 dict (附带 '__score__' 与 '__metrics__' 字段)
        """
        self._results = []
        self._used_skopt = False

        # ---- 分支1: skopt 可用, 使用贝叶斯优化 ----
        if _SKOPT_AVAILABLE:
            try:
                space = self._to_skopt_space()
                n_initial = max(3, min(10, n_trials))
                optimizer = _SkoptOptimizer(
                    dimensions=space,
                    base_estimator="GP",
                    n_initial_points=n_initial,
                    random_state=42,
                )

                best_score = -np.inf
                best_params = None
                best_metrics = None

                for i in range(1, n_trials + 1):
                    x = optimizer.ask()
                    params = self._x_to_params(x)
                    try:
                        result = evaluate_fn(params)
                        score = _extract_score(result, self.metric)
                        metrics = result if isinstance(result, dict) else {self.metric: score}
                    except Exception as e:
                        logger.warning(f"贝叶斯优化第 {i}/{n_trials} 组评估失败 ({params}): {e}")
                        score = -np.inf
                        metrics = {"error": str(e), self.metric: -np.inf}

                    # skopt 最小化目标, 此处最大化 metric -> 取负
                    optimizer.tell(x, -score)

                    record = {**params, "__score__": score, "__trial__": i}
                    if isinstance(metrics, dict):
                        record["__metrics__"] = metrics
                        for k, v in metrics.items():
                            record.setdefault(f"metric_{k}", v)
                    self._results.append(record)

                    if score > best_score:
                        best_score = score
                        best_params = dict(params)
                        best_metrics = metrics

                self._used_skopt = True
                logger.info(
                    f"贝叶斯优化(scikit-optimize GP)完成, "
                    f"最优 {self.metric}={best_score:.4f}, 参数={best_params}"
                )
                if best_params is not None:
                    self._best = {
                        "params": best_params,
                        "score": best_score,
                        "metrics": best_metrics if best_metrics is not None else {},
                    }
                    return self._best
            except Exception as e:
                logger.warning(
                    f"scikit-optimize 贝叶斯优化执行失败({e}), 退化为随机搜索"
                )
                self._results = []

        # ---- 分支2: skopt 不可用或失败, 退化为随机搜索 ----
        if not _SKOPT_AVAILABLE:
            logger.warning(
                f"scikit-optimize 不可用({_SKOPT_IMPORT_ERROR}), 贝叶斯优化退化为随机搜索"
            )
        best_params, best_metrics, best_score = self._random_search(evaluate_fn, n_trials)
        self._used_skopt = False

        if best_params is None:
            logger.warning("随机搜索未得到任何有效结果, 返回空参数")
            self._best = {"params": {}, "score": -np.inf, "metrics": {}}
            return self._best

        logger.info(
            f"随机搜索完成, 最优 {self.metric}={best_score:.4f}, 参数={best_params}"
        )
        self._best = {
            "params": best_params,
            "score": best_score,
            "metrics": best_metrics if best_metrics is not None else {},
        }
        return self._best

    @property
    def used_skopt(self) -> Optional[bool]:
        """本次 search 是否真正使用了 skopt (None 表示尚未 search)"""
        return self._used_skopt

    def get_results(self) -> pd.DataFrame:
        """获取搜索结果 DataFrame, 按 __score__ 降序"""
        if not self._results:
            return pd.DataFrame()
        df = pd.DataFrame(self._results)
        df = df.sort_values("__score__", ascending=False).reset_index(drop=True)
        return df


# ============ 时间序列交叉验证 ============
class TimeSeriesCV:
    """
    时间序列交叉验证 (扩展窗口 / Expanding Window)。

    防止未来信息泄漏: 训练集始终在测试集之前, 且训练集随时间递增。
    """

    def __init__(self, n_splits: int = 5, min_train_periods: int = 12):
        if n_splits < 1:
            raise ValueError("n_splits 必须 >= 1")
        if min_train_periods < 1:
            raise ValueError("min_train_periods 必须 >= 1")
        self.n_splits = n_splits
        self.min_train_periods = min_train_periods

    def split(self, dates: List[str]) -> List[Tuple[List[int], List[int]]]:
        """
        生成时间序列交叉验证分割。

        Args:
            dates: 按时间升序排列的日期列表(字符串, 如调仓日)

        Returns:
            [(train_idx, test_idx), ...], idx 为 dates 的整数下标。
            train_idx 与 test_idx 不重叠, 且 train 全部早于 test。
        """
        n = len(dates)
        if n < self.min_train_periods + self.n_splits:
            raise ValueError(
                f"数据长度 {n} 不足以做 {self.n_splits} 折交叉验证 "
                f"(需 >= min_train_periods({self.min_train_periods}) + n_splits({self.n_splits}))"
            )

        # 可用于测试的样本数
        n_test_total = n - self.min_train_periods
        # 每折测试集大小 (向下取整, 保证不越界)
        test_size = max(1, n_test_total // self.n_splits)

        splits: List[Tuple[List[int], List[int]]] = []
        train_end = self.min_train_periods  # 训练集结束下标(不含)

        for k in range(self.n_splits):
            test_start = train_end
            test_end = test_start + test_size
            # 最后一折吸收所有剩余样本, 避免尾部数据浪费
            if k == self.n_splits - 1:
                test_end = n

            train_idx = list(range(0, test_start))
            test_idx = list(range(test_start, test_end))
            splits.append((train_idx, test_idx))
            train_end = test_end

        return splits

    def get_split_summary(self, dates: List[str]) -> pd.DataFrame:
        """返回各折训练/测试区间的日期范围摘要, 便于检查"""
        rows = []
        for k, (tr, te) in enumerate(self.split(dates), 1):
            rows.append({
                "fold": k,
                "n_train": len(tr),
                "n_test": len(te),
                "train_start": dates[tr[0]] if tr else None,
                "train_end": dates[tr[-1]] if tr else None,
                "test_start": dates[te[0]] if te else None,
                "test_end": dates[te[-1]] if te else None,
            })
        return pd.DataFrame(rows)


# ============ 参数敏感性分析 ============
def analyze_sensitivity(results: pd.DataFrame, metric: str = "sharpe_ratio",
                        param_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    参数敏感性分析: 统计每个参数取值下目标指标的平均表现与波动,
    用于判断哪些参数对策略影响较大。

    Args:
        results: 调参器 get_results() 返回的 DataFrame
        metric: 目标指标列名(在 results 中以 'metric_<metric>' 或 metric 本身存在)
        param_names: 需分析的参数列名列表, 为 None 时自动推断(排除内部列)

    Returns:
        DataFrame, 列: param, value, n_trials, metric_mean, metric_std,
                metric_min, metric_max, metric_range
    """
    if results is None or len(results) == 0:
        return pd.DataFrame()

    # 定位指标列
    metric_col = f"metric_{metric}" if f"metric_{metric}" in results.columns else metric
    if metric_col not in results.columns:
        # 网格/贝叶斯结果用 __score__ 兜底
        if "__score__" in results.columns:
            metric_col = "__score__"
        else:
            raise KeyError(f"结果 DataFrame 中找不到指标列 {metric!r}")

    internal_cols = {"__score__", "__trial__", "__metrics__"}
    if param_names is None:
        param_names = [c for c in results.columns
                       if c not in internal_cols and not c.startswith("metric_")]

    rows = []
    for pname in param_names:
        if pname not in results.columns:
            continue
        grp = results.groupby(pname)[metric_col]
        for val, sub in grp:
            arr = sub.astype(float)
            rows.append({
                "param": pname,
                "value": val,
                "n_trials": int(len(arr)),
                "metric_mean": float(arr.mean()),
                "metric_std": float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
                "metric_min": float(arr.min()),
                "metric_max": float(arr.max()),
                "metric_range": float(arr.max() - arr.min()),
            })
    return pd.DataFrame(rows).sort_values(["param", "metric_mean"], ascending=[True, False])


# ============ 最优参数持久化 ============
def save_best_params(params: Dict[str, Any], path: str) -> None:
    """
    保存最优参数到 JSON 文件。

    支持两种输入格式:
      1. search() 返回的 {"params": ..., "score": ..., "metrics": ...} 格式
      2. 直接的策略参数 dict (如 {"TOP_N": 50, "MAX_WEIGHT": 0.05})
    会自动识别并提取真正的策略参数,写入元信息。
    """
    # 识别输入格式
    if isinstance(params, dict) and "params" in params and isinstance(params.get("params"), dict):
        # search() 返回格式
        clean = dict(params["params"])
        score = params.get("score")
        metrics = params.get("metrics")
    else:
        # 直接参数 dict 或旧格式
        clean = {k: v for k, v in params.items()
                 if not (str(k).startswith("__") or str(k).startswith("metric_"))}
        score = params.get("__score__") if isinstance(params, dict) else None
        metrics = params.get("__metrics__") if isinstance(params, dict) else None

    payload = {
        "module": "auto_tuner",
        "version": "v6.2",
        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metric_score": score,
        "metrics": metrics if isinstance(metrics, dict) else {},
        "skopt_available": _SKOPT_AVAILABLE,
        "params": clean,
    }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info(f"最优参数已保存至 {out_path} (共 {len(clean)} 个参数)")


def load_best_params(path: str) -> Dict[str, Any]:
    """
    从 JSON 文件加载最优参数。

    Returns:
        参数 dict。若文件包含元信息包装, 则返回 payload 中 'params' 字段;
        否则直接返回整个 JSON 内容。
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"参数文件不存在: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "params" in payload and "module" in payload:
        logger.info(
            f"从 {in_path} 加载最优参数 "
            f"(模块={payload.get('module')}, 保存时间={payload.get('saved_at')})"
        )
        return payload["params"]
    logger.info(f"从 {in_path} 加载参数(无元信息包装)")
    return payload


def _json_default(obj: Any) -> Any:
    """JSON 序列化兜底: 处理 numpy 类型与 Timestamp"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"无法序列化对象: {type(obj)}")


# ============ 模块自检 ============
def _self_test() -> None:
    """轻量自检: 用一个简单的模拟评估函数验证各组件可运行"""
    logger.info("=== auto_tuner 自检开始 ===")
    logger.info(f"scikit-optimize 可用: {_SKOPT_AVAILABLE}")

    # 模拟评估函数: 鼓励 TOP_N=50, MAX_WEIGHT=0.05, FACTOR_COMBINE_METHOD=ic_weight
    def mock_evaluate(params):
        score = 0.0
        score -= abs(params.get("TOP_N", 50) - 50) * 0.01
        score -= abs(params.get("MAX_WEIGHT", 0.05) - 0.05) * 5
        score += 1.0 if params.get("FACTOR_COMBINE_METHOD") == "ic_weight" else 0.0
        return {
            "sharpe_ratio": score,
            "annual_return": score * 0.2,
            "max_drawdown": -0.1 - abs(score) * 0.01,
        }

    specs = build_default_param_specs()

    # 网格搜索
    gt = GridSearchTuner(specs, metric="sharpe_ratio")
    best_g = gt.search(mock_evaluate, n_trials=50)
    logger.info(f"网格搜索最优: {best_g}")

    # 贝叶斯优化(可能退化为随机搜索)
    bt = BayesianTuner(specs, metric="sharpe_ratio")
    best_b = bt.search(mock_evaluate, n_trials=20)
    logger.info(f"贝叶斯优化最优 (used_skopt={bt.used_skopt}): {best_b}")

    # 敏感性分析
    sens = analyze_sensitivity(gt.get_results(), metric="sharpe_ratio")
    logger.info(f"敏感性分析行数: {len(sens)}")

    # 时间序列CV
    dates = [f"2023-{m:02d}-28" for m in range(1, 13)] + \
            [f"2024-{m:02d}-28" for m in range(1, 13)]
    cv = TimeSeriesCV(n_splits=4, min_train_periods=8)
    splits = cv.split(dates)
    logger.info(f"时间序列CV折数: {len(splits)}")
    logger.info(f"CV摘要:\n{cv.get_split_summary(dates).to_string()}")

    # 持久化
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "_auto_tuner_test_params.json")
    save_best_params(best_g, tmp)
    loaded = load_best_params(tmp)
    assert isinstance(loaded, dict) and len(loaded) > 0, "持久化往返失败"
    logger.info(f"持久化往返成功, 加载参数: {loaded}")
    os.remove(tmp)

    logger.info("=== auto_tuner 自检通过 ===")


if __name__ == "__main__":
    _self_test()
