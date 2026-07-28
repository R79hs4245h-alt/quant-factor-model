"""
增强机器学习管线模块 v7.0
v7.0 新增功能:
  1. FeatureSelector - 特征选择器(VIF/IC/相关性三维筛选)
  2. WalkForwardAnalyzer - Walk-Forward分析器(滚动窗口训练测试,过拟合检测)
  3. ModelVersionManager - 模型版本管理器(保存/加载/比较/选优)
  4. EnhancedMLCombiner - 增强ML合成器(集成特征选择/Walk-Forward/多模型集成/自适应超参数)
依赖: numpy, pandas, sklearn; 可选: xgboost, lightgbm
"""
import os
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from config import FACTOR_CONFIG
from utils import get_logger
from ml_combiner import MLFactorCombiner

warnings.filterwarnings("ignore")

logger = get_logger("ml_pipeline")


# ============================================================
#  1. FeatureSelector - 特征选择器
# ============================================================
class FeatureSelector:
    """
    特征选择器 v7.0

    提供三种特征筛选方法,可单独使用或组合使用:
    - VIF筛选: 剔除多重共线性严重的因子
    - IC筛选: 保留与收益有稳定预测能力的因子
    - 相关性筛选: 剔除高度相关的冗余因子,保留IC更高的

    组合筛选顺序: VIF -> IC -> 相关性
    """

    def __init__(self, ic_history: Optional[Dict[str, List[float]]] = None):
        """
        初始化特征选择器

        :param ic_history: 因子IC历史记录, 格式为 {因子名: [IC值列表]}
                           用于IC筛选和相关性筛选中判断保留哪个因子
        """
        self.ic_history = ic_history or {}

    def select_by_vif(self, factors_df: pd.DataFrame, threshold: float = 10.0) -> List[str]:
        """
        使用方差膨胀因子(VIF)筛选因子, 剔除VIF超过阈值的因子

        VIF衡量一个因子被其他因子线性解释的程度, VIF越大说明多重共线性越严重.
        通常 VIF > 10 表示存在严重的多重共线性, 应考虑剔除.

        算法: 迭代式剔除 -- 每次计算所有因子的VIF, 剔除VIF最大的因子,
        然后重新计算, 直到所有因子VIF均低于阈值.

        :param factors_df: 因子矩阵 DataFrame, 行为截面样本, 列为因子
        :param threshold: VIF阈值, 默认10.0; 超过该值的因子将被剔除
        :return: 通过VIF筛选的因子名称列表
        """
        X = factors_df.dropna(axis=1, how="all").fillna(0)
        if X.shape[1] < 2:
            return list(X.columns)

        selected = list(X.columns)
        max_iter = len(selected)  # 防止无限循环

        for _ in range(max_iter):
            if len(selected) < 2:
                break

            X_sub = X[selected].values
            # 计算相关系数矩阵的逆(即通过OLS拟合的VIF)
            try:
                corr_matrix = np.corrcoef(X_sub, rowvar=False)
                if np.any(np.isnan(corr_matrix)) or np.any(np.isinf(corr_matrix)):
                    break
                # 添加微小扰动确保矩阵可逆
                corr_matrix += np.eye(corr_matrix.shape[0]) * 1e-8
                try:
                    inv_corr = np.linalg.inv(corr_matrix)
                except np.linalg.LinAlgError:
                    break
                vif_values = np.diag(inv_corr)

                if np.all(vif_values <= threshold):
                    break

                # 剔除VIF最大的因子
                worst_idx = int(np.argmax(vif_values))
                removed = selected.pop(worst_idx)
                logger.debug(f"VIF筛选剔除: {removed} (VIF={vif_values[worst_idx]:.2f})")

            except Exception as e:
                logger.warning(f"VIF计算异常: {e}")
                break

        logger.info(f"VIF筛选: {len(factors_df.columns)} -> {len(selected)} 个因子 "
                    f"(阈值={threshold})")
        return selected

    def select_by_ic(self, factors_df: pd.DataFrame,
                     forward_returns: pd.Series,
                     min_ic: float = 0.02) -> List[str]:
        """
        基于IC绝对值筛选因子, 保留IC绝对值超过阈值的因子

        IC (Information Coefficient) 衡量因子值与未来收益的秩相关系数.
        |IC|越高说明因子的预测能力越强.

        当 ic_history 不为空时, 优先使用历史IC均值; 否则使用当期截面IC.

        :param factors_df: 因子矩阵 DataFrame
        :param forward_returns: 前向收益 Series, 与 factors_df 的 index 对齐
        :param min_ic: IC绝对值最低阈值, 默认0.02; 低于该值的因子将被剔除
        :return: 通过IC筛选的因子名称列表
        """
        ic_scores: Dict[str, float] = {}

        # 优先使用历史IC均值
        for col in factors_df.columns:
            if col in self.ic_history and len(self.ic_history[col]) >= 3:
                ic_scores[col] = float(np.mean(self.ic_history[col][-12:]))

        # 如果没有足够的历史IC, 计算当期截面IC
        common_idx = factors_df.index.intersection(forward_returns.index)
        if len(common_idx) < 30 or len(ic_scores) < len(factors_df.columns):
            for col in factors_df.columns:
                if col not in ic_scores:
                    try:
                        ic = factors_df.loc[common_idx, col].corr(
                            forward_returns.loc[common_idx], method="spearman"
                        )
                        if not np.isnan(ic):
                            ic_scores[col] = float(ic)
                        else:
                            ic_scores[col] = 0.0
                    except Exception:
                        ic_scores[col] = 0.0

        selected = [col for col in factors_df.columns
                    if ic_scores.get(col, 0.0) is not None
                    and abs(ic_scores.get(col, 0.0)) >= min_ic]

        logger.info(f"IC筛选: {len(factors_df.columns)} -> {len(selected)} 个因子 "
                    f"(最低IC={min_ic})")
        return selected

    def select_by_correlation(self, factors_df: pd.DataFrame,
                              max_corr: float = 0.7) -> List[str]:
        """
        剔除高度相关的冗余因子, 保留IC更高的因子

        算法:
        1. 计算因子间的相关系数矩阵
        2. 按IC绝对值从高到低排序因子
        3. 依次遍历每个因子, 如果与已选因子中任一因子的相关性超过阈值,
           则剔除该因子(保留IC更高的)

        这样可以保证:
        - IC最强的因子总是被保留
        - 相关因子组中只保留预测能力最强的一个

        :param factors_df: 因子矩阵 DataFrame
        :param max_corr: 最大允许相关系数, 默认0.7; 超过则视为冗余
        :return: 去冗余后的因子名称列表
        """
        if factors_df.shape[1] < 2:
            return list(factors_df.columns)

        X = factors_df.fillna(0)
        corr_matrix = X.corr().abs()

        # 获取IC评分(绝对值), 用于决定保留哪个因子
        ic_scores: Dict[str, float] = {}
        for col in factors_df.columns:
            if col in self.ic_history and len(self.ic_history[col]) >= 3:
                ic_scores[col] = abs(float(np.mean(self.ic_history[col][-12:])))
            else:
                ic_scores[col] = 0.0

        # 按IC绝对值降序排列
        sorted_factors = sorted(factors_df.columns,
                                key=lambda c: ic_scores.get(c, 0.0),
                                reverse=True)

        selected: List[str] = []
        for factor in sorted_factors:
            # 检查与已选因子的相关性
            is_redundant = False
            for selected_factor in selected:
                if factor in corr_matrix.columns and selected_factor in corr_matrix.index:
                    corr_val = corr_matrix.loc[selected_factor, factor]
                    if not np.isnan(corr_val) and corr_val > max_corr:
                        is_redundant = True
                        break

            if not is_redundant:
                selected.append(factor)
            else:
                logger.debug(f"相关性筛选剔除: {factor} (与已选因子相关>{max_corr})")

        logger.info(f"相关性筛选: {len(factors_df.columns)} -> {len(selected)} 个因子 "
                    f"(最大相关={max_corr})")
        return selected

    def select_all(self, factors_df: pd.DataFrame,
                   forward_returns: pd.Series) -> List[str]:
        """
        组合三种特征筛选方法, 依次执行: VIF -> IC -> 相关性

        筛选流程:
        1. VIF筛选: 剔除多重共线性严重的因子
        2. IC筛选: 保留有预测能力的因子
        3. 相关性筛选: 在剩余因子中剔除冗余因子

        逐步缩小候选集, 最终得到高质量的特征子集.

        :param factors_df: 因子矩阵 DataFrame
        :param forward_returns: 前向收益 Series
        :return: 通过全部筛选的因子名称列表
        """
        logger.info(f"组合特征筛选开始: {len(factors_df.columns)} 个候选因子")

        # 第一步: VIF筛选
        selected = self.select_by_vif(factors_df, threshold=10.0)
        if not selected:
            logger.warning("VIF筛选后无因子剩余, 返回全部因子")
            return list(factors_df.columns)

        # 第二步: IC筛选
        selected_df = factors_df[selected]
        selected = self.select_by_ic(selected_df, forward_returns, min_ic=0.02)
        if not selected:
            logger.warning("IC筛选后无因子剩余, 返回VIF筛选结果")
            return self.select_by_vif(factors_df, threshold=10.0)

        # 第三步: 相关性筛选
        selected_df = factors_df[selected]
        selected = self.select_by_correlation(selected_df, max_corr=0.7)

        logger.info(f"组合特征筛选完成: {len(factors_df.columns)} -> {len(selected)} 个因子")
        return selected


# ============================================================
#  2. WalkForwardAnalyzer - Walk-Forward分析器
# ============================================================
class WalkForwardAnalyzer:
    """
    Walk-Forward分析器 v7.0

    通过滚动窗口训练测试评估模型的样本外泛化能力.
    核心思想: 模拟真实交易中的时间序列特性 -- 用历史训练, 预测未来.

    主要用途:
    - 量化过拟合程度 (train_ic vs test_ic 差异)
    - 评估样本外表现 (out-of-fold score)
    - 为模型选择提供可靠依据

    典型调用方式:
        factors_history: List[pd.DataFrame]  -- 每期截面因子数据
        returns_history: List[pd.Series]     -- 每期截面前向收益
    """

    def analyze(self,
                factors_history: List[pd.DataFrame],
                returns_history: List[pd.Series],
                train_window: int = 12,
                test_window: int = 3,
                model_type: str = "rf") -> Dict:
        """
        滚动窗口训练测试分析

        滚动窗口机制:
        - 将历史数据按时间排列为 T 期
        - 第1次: 用第 1~train_window 期训练, 用第 train_window+1~train_window+test_window 期测试
        - 第2次: 窗口向前滑动 test_window 期, 重复上述过程
        - 直到剩余数据不足一个完整的训练+测试窗口

        每次训练测试:
        1. 合并训练窗口内的截面数据为一个训练集
        2. 用训练集拟合模型
        3. 在测试集上计算预测IC (Spearman秩相关)
        4. 记录训练集IC和测试集IC

        :param factors_history: 历史截面因子数据列表, 按时间顺序排列
        :param returns_history: 历史截面前向收益列表, 按时间顺序排列
        :param train_window: 训练窗口大小(期数), 默认12
        :param test_window: 测试窗口大小(期数), 默认3
        :param model_type: 模型类型, 默认"rf"(随机森林)
        :return: 分析结果字典, 包含:
            - train_ics: 每段训练集IC列表
            - test_ics: 每段测试集IC列表
            - oof_score: 样本外平均IC (所有测试集IC的均值)
            - overfit_gap: 过拟合指标 (训练集平均IC - 测试集平均IC)
        """
        if len(factors_history) != len(returns_history):
            raise ValueError(
                f"因子历史({len(factors_history)})与收益历史({len(returns_history)})长度不一致"
            )

        total_periods = len(factors_history)
        min_required = train_window + test_window
        if total_periods < min_required:
            logger.warning(
                f"历史数据不足: 需要至少{min_required}期, 实际{total_periods}期"
            )
            return {
                "train_ics": [],
                "test_ics": [],
                "oof_score": 0.0,
                "overfit_gap": 0.0,
            }

        train_ics: List[float] = []
        test_ics: List[float] = []
        step = test_window  # 每次滑动test_window期

        start = 0
        fold_idx = 0
        while start + train_window + test_window <= total_periods:
            fold_idx += 1

            # 训练窗口
            train_factors = factors_history[start:start + train_window]
            train_returns = returns_history[start:start + train_window]
            # 测试窗口
            test_factors = factors_history[start + train_window:
                                           start + train_window + test_window]
            test_returns = returns_history[start + train_window:
                                          start + train_window + test_window]

            # 合并训练数据
            X_train_list, y_train_list = [], []
            for f_df, r_s in zip(train_factors, train_returns):
                common = f_df.index.intersection(r_s.index)
                if len(common) >= 20:
                    X_train_list.append(f_df.loc[common].fillna(0))
                    y_train_list.append(r_s.loc[common])

            if not X_train_list:
                start += step
                continue

            X_train = pd.concat(X_train_list)
            y_train = pd.concat(y_train_list)

            # 合并测试数据
            X_test_list, y_test_list = [], []
            for f_df, r_s in zip(test_factors, test_returns):
                common = f_df.index.intersection(r_s.index)
                if len(common) >= 20:
                    X_test_list.append(f_df.loc[common].fillna(0))
                    y_test_list.append(r_s.loc[common])

            if not X_test_list:
                start += step
                continue

            X_test = pd.concat(X_test_list)
            y_test = pd.concat(y_test_list)

            # 训练模型
            model = self._create_model(model_type)
            try:
                model.fit(X_train.values, y_train.values)

                # 训练集IC
                train_pred = model.predict(X_train.values)
                train_ic = float(np.nan_to_num(
                    pd.Series(train_pred).corr(pd.Series(y_train), method="spearman")
                ))
                train_ics.append(train_ic)

                # 测试集IC
                test_pred = model.predict(X_test.values)
                test_ic = float(np.nan_to_num(
                    pd.Series(test_pred).corr(pd.Series(y_test), method="spearman")
                ))
                test_ics.append(test_ic)

                logger.debug(
                    f"Walk-Forward Fold {fold_idx}: "
                    f"train_IC={train_ic:.4f}, test_IC={test_ic:.4f}"
                )
            except Exception as e:
                logger.warning(f"Walk-Forward Fold {fold_idx} 训练失败: {e}")

            start += step

        # 汇总结果
        train_ic_mean = float(np.mean(train_ics)) if train_ics else 0.0
        test_ic_mean = float(np.mean(test_ics)) if test_ics else 0.0
        oof_score = test_ic_mean
        overfit_gap = train_ic_mean - test_ic_mean

        result = {
            "train_ics": train_ics,
            "test_ics": test_ics,
            "oof_score": oof_score,
            "overfit_gap": overfit_gap,
        }

        logger.info(
            f"Walk-Forward分析完成: {fold_idx} folds, "
            f"train_IC_mean={train_ic_mean:.4f}, "
            f"test_IC_mean={test_ic_mean:.4f}, "
            f"OOF_score={oof_score:.4f}, "
            f"过拟合差距={overfit_gap:.4f}"
        )
        return result

    def detect_overfit(self, train_scores: List[float],
                        test_scores: List[float]) -> Dict:
        """
        检测过拟合: 比较训练集和测试集得分的差异

        过拟合判断标准:
        - 轻度过拟合: 过拟合差距 < 0.03
        - 中度过拟合: 0.03 <= 过拟合差距 < 0.06
        - 重度过拟合: 过拟合差距 >= 0.06

        同时考虑训练集得分和测试集得分的稳定性:
        - 如果训练集得分高且稳定, 测试集得分低且波动大 -> 严重过拟合
        - 如果两者都低 -> 模型欠拟合

        :param train_scores: 训练集得分列表
        :param test_scores: 测试集得分列表
        :return: 过拟合检测结果字典:
            - overfit_gap: 训练-测试平均得分差
            - overfit_level: 过拟合程度 ("无"/"轻度"/"中度"/"重度")
            - train_mean: 训练集平均得分
            - test_mean: 测试集平均得分
            - train_std: 训练集得分标准差
            - test_std: 测试集得分标准差
            - suggestion: 改进建议
        """
        if not train_scores or not test_scores:
            return {
                "overfit_gap": 0.0,
                "overfit_level": "未知",
                "train_mean": 0.0,
                "test_mean": 0.0,
                "train_std": 0.0,
                "test_std": 0.0,
                "suggestion": "训练或测试得分为空, 无法判断过拟合程度",
            }

        train_mean = float(np.mean(train_scores))
        test_mean = float(np.mean(test_scores))
        train_std = float(np.std(train_scores))
        test_std = float(np.std(test_scores))
        gap = train_mean - test_mean

        # 判断过拟合程度
        if gap < 0.03:
            level = "无"
        elif gap < 0.06:
            level = "轻度"
        elif gap < 0.10:
            level = "中度"
        else:
            level = "重度"

        # 生成建议
        if level == "无":
            suggestion = "模型泛化能力良好, 无明显过拟合"
        elif level == "轻度":
            suggestion = "存在轻微过拟合, 可考虑: 减少模型复杂度、增加正则化、增大训练样本"
        elif level == "中度":
            suggestion = ("中度过拟合, 建议: 降低max_depth、增大min_samples_leaf、"
                           "增加L1/L2正则化、减少特征数量")
        else:
            suggestion = ("严重过拟合! 模型在样本外几乎无预测能力. "
                           "建议: 大幅简化模型、使用更强的正则化、"
                           "检查特征是否有前瞻偏差")

        result = {
            "overfit_gap": gap,
            "overfit_level": level,
            "train_mean": train_mean,
            "test_mean": test_mean,
            "train_std": train_std,
            "test_std": test_std,
            "suggestion": suggestion,
        }

        logger.info(
            f"过拟合检测: {level} (差距={gap:.4f}, "
            f"train={train_mean:.4f}+/-{train_std:.4f}, "
            f"test={test_mean:.4f}+/-{test_std:.4f})"
        )
        return result

    @staticmethod
    def _create_model(model_type: str):
        """创建指定类型的ML模型"""
        if model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100, max_depth=5,
                min_samples_leaf=20, random_state=42, n_jobs=-1,
            )
        elif model_type == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, verbosity=0,
                )
            except ImportError:
                logger.warning("xgboost不可用, 退化为随机森林")
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=100, max_depth=5,
                    min_samples_leaf=20, random_state=42, n_jobs=-1,
                )
        elif model_type == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, verbose=-1,
                )
            except ImportError:
                logger.warning("lightgbm不可用, 退化为随机森林")
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=100, max_depth=5,
                    min_samples_leaf=20, random_state=42, n_jobs=-1,
                )
        else:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100, max_depth=5,
                min_samples_leaf=20, random_state=42, n_jobs=-1,
            )


# ============================================================
#  3. ModelVersionManager - 模型版本管理器
# ============================================================
class ModelVersionManager:
    """
    模型版本管理器 v7.0

    管理ML模型的保存、加载、比较和选优.
    每个模型版本以独立目录保存, 包含:
    - model.pkl: 序列化的模型对象
    - metadata.pkl: 模型元数据 (训练时间、特征列表、性能指标等)

    目录结构示例:
        models/
        ├── v20240101_120000/
        │   ├── model.pkl
        │   └── metadata.pkl
        └── v20240201_120000/
            ├── model.pkl
            └── metadata.pkl
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化模型版本管理器

        :param base_dir: 模型存储根目录; 默认为 quant_factor_model/models/
        """
        if base_dir is None:
            base_dir = str(Path(__file__).parent / "models")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"模型版本管理器初始化: 存储目录={self.base_dir}")

    def save_model(self, model: Any, metadata: Dict, path: Optional[str] = None) -> None:
        """
        保存模型和元数据到指定路径

        元数据中会自动补充以下字段(如果未提供):
        - save_time: 保存时间戳
        - save_datetime: 保存日期时间字符串

        :param model: 已训练的ML模型对象
        :param metadata: 模型元数据字典, 建议包含:
            - feature_cols: 特征列名列表
            - train_size: 训练集样本数
            - val_r2: 验证集R2
            - train_r2: 训练集R2
            - train_periods: 训练期数
            - model_type: 模型类型
            - feature_count: 特征数量
        :param path: 保存路径(目录); 如果为None则自动生成版本目录名
        """
        if path is None:
            version_name = datetime.now().strftime("v%Y%m%d_%H%M%S")
            path = str(self.base_dir / version_name)

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 自动补充元数据
        metadata["save_time"] = time.time()
        metadata["save_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 保存模型
        model_path = save_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # 保存元数据
        meta_path = save_dir / "metadata.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"模型已保存: {save_dir} "
                    f"(类型={metadata.get('model_type', 'unknown')}, "
                    f"特征数={metadata.get('feature_count', '?')}, "
                    f"val_R2={metadata.get('val_r2', '?')})")

    def load_model(self, path: str) -> Tuple[Any, Dict]:
        """
        从指定路径加载模型和元数据

        :param path: 模型版本目录路径
        :return: (model, metadata) 元组; model为模型对象, metadata为元数据字典
        :raises FileNotFoundError: 如果模型文件不存在
        """
        load_dir = Path(path)
        model_path = load_dir / "model.pkl"
        meta_path = load_dir / "metadata.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        metadata: Dict = {}
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)

        logger.info(f"模型已加载: {load_dir} "
                    f"(类型={metadata.get('model_type', 'unknown')}, "
                    f"保存时间={metadata.get('save_datetime', 'unknown')})")
        return model, metadata

    def compare_models(self, model_paths: List[str]) -> pd.DataFrame:
        """
        比较多个模型版本的性能指标

        读取每个模型版本的元数据, 整理为可比较的DataFrame.
        如果某些模型缺少某些指标, 填充为NaN.

        :param model_paths: 模型版本目录路径列表
        :return: 模型比较结果 DataFrame, 每行一个模型版本, 列为各项指标
        """
        if not model_paths:
            return pd.DataFrame()

        rows: List[Dict] = []
        for path in model_paths:
            try:
                _, metadata = self.load_model(path)
                row = {
                    "path": path,
                    "version": Path(path).name,
                }
                # 复制元数据中所有标量值
                for key, val in metadata.items():
                    if isinstance(val, (int, float, str, bool)):
                        row[key] = val
                rows.append(row)
            except Exception as e:
                logger.warning(f"加载模型失败 ({path}): {e}")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # 按保存时间降序排列
        if "save_time" in df.columns:
            df = df.sort_values("save_time", ascending=False).reset_index(drop=True)

        logger.info(f"模型比较: {len(df)} 个版本")
        return df

    def get_best_model(self, model_paths: List[str],
                       metric: str = "val_r2") -> Tuple[Any, Dict]:
        """
        根据指定指标选择最优模型

        遍历所有模型版本, 比较指定的性能指标,
        返回指标值最优的模型和元数据.

        :param model_paths: 模型版本目录路径列表
        :param metric: 用于比较的指标名称, 默认"val_r2"(验证集R2)
        :return: (best_model, best_metadata) 元组
        :raises ValueError: 如果没有可比较的模型或指标不存在
        """
        if not model_paths:
            raise ValueError("模型路径列表为空")

        best_model = None
        best_metadata: Dict = {}
        best_score = -np.inf

        for path in model_paths:
            try:
                model, metadata = self.load_model(path)
                score = metadata.get(metric, None)
                if score is None:
                    logger.warning(f"模型 {path} 缺少指标 {metric}, 跳过")
                    continue

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_metadata = metadata
            except Exception as e:
                logger.warning(f"加载模型失败 ({path}): {e}")

        if best_model is None:
            raise ValueError(
                f"没有找到包含指标 '{metric}' 的模型版本"
            )

        logger.info(
            f"最优模型: {best_metadata.get('version', 'unknown')} "
            f"({metric}={best_score:.4f}, "
            f"保存时间={best_metadata.get('save_datetime', 'unknown')})"
        )
        return best_model, best_metadata

    def list_models(self) -> List[str]:
        """
        列出所有已保存的模型版本路径

        :return: 模型版本目录路径列表, 按名称排序
        """
        if not self.base_dir.exists():
            return []

        model_dirs = []
        for d in sorted(self.base_dir.iterdir()):
            if d.is_dir() and (d / "model.pkl").exists():
                model_dirs.append(str(d))

        return model_dirs


# ============================================================
#  4. EnhancedMLCombiner - 增强ML合成器
# ============================================================
class EnhancedMLCombiner(MLFactorCombiner):
    """
    增强ML合成器 v7.0 (继承 MLFactorCombiner)

    在基础ML合成器之上集成:
    1. 特征选择: 每次训练前自动筛选特征, 减少噪声和共线性
    2. Walk-Forward分析: 训练后自动评估过拟合风险
    3. 模型版本管理: 自动保存模型版本, 便于回溯和比较
    4. 多模型集成: 同时训练RF/XGBoost/LightGBM, 加权平均集成
    5. 自适应超参数: 根据数据量和特征数自动调整模型复杂度

    使用示例:
        combiner = EnhancedMLCombiner(
            method="enhanced_ensemble",
            enable_feature_selection=True,
            enable_walk_forward=True,
            enable_version_control=True,
        )
        score = combiner.combine(factors_df, forward_returns)
    """

    def __init__(self,
                 method: str = "enhanced_ensemble",
                 model_type: str = "auto",
                 rolling_window: int = 12,
                 min_samples: int = 200,
                 enable_feature_selection: bool = True,
                 enable_walk_forward: bool = True,
                 enable_version_control: bool = True,
                 model_save_dir: Optional[str] = None):
        """
        初始化增强ML合成器

        :param method: 合成方法, 支持 "enhanced_ensemble" / "multi_model" /
                       "single_model"(退化为父类行为)
        :param model_type: 主模型类型, 默认"auto"(自动选择)
        :param rolling_window: 滚动窗口大小
        :param min_samples: 最小训练样本数
        :param enable_feature_selection: 是否启用自动特征选择, 默认True
        :param enable_walk_forward: 是否启用Walk-Forward分析, 默认True
        :param enable_version_control: 是否启用模型版本管理, 默认True
        :param model_save_dir: 模型保存目录; 默认为 quant_factor_model/models/
        """
        super().__init__(
            method=method,
            model_type=model_type,
            rolling_window=rolling_window,
            min_samples=min_samples,
        )

        self.enable_feature_selection = enable_feature_selection
        self.enable_walk_forward = enable_walk_forward
        self.enable_version_control = enable_version_control

        # 特征选择器
        self._feature_selector = FeatureSelector(ic_history=dict(self.ic_history))

        # Walk-Forward分析器
        self._wf_analyzer = WalkForwardAnalyzer()

        # 模型版本管理器
        self._version_manager = ModelVersionManager(base_dir=model_save_dir)

        # 多模型集成: 存储多个训练好的模型及其权重
        self._ensemble_models: Dict[str, Any] = {}  # {模型类型: 模型对象}
        self._ensemble_weights: Dict[str, float] = {}  # {模型类型: 权重}
        self._ensemble_feature_cols: Optional[List[str]] = None

        # Walk-Forward分析结果
        self._wf_results: Optional[Dict] = None

        # 特征选择结果
        self._selected_features: Optional[List[str]] = None

    def _adaptive_hyperparams(self, n_samples: int, n_features: int) -> Dict:
        """
        根据数据量和特征数自适应调整模型超参数

        自适应策略:
        - 小数据集(<1000样本): 使用较浅的模型, 强正则化, 防止过拟合
        - 中等数据集(1000~5000): 标准参数
        - 大数据集(>5000): 可以使用更深的模型, 捕获更复杂的模式

        同时考虑特征数:
        - 特征数 > 样本数/10: 增加正则化, 降低特征采样比例

        :param n_samples: 训练样本数
        :param n_features: 特征数量
        :return: 超参数字典, 可用于初始化各模型
        """
        params: Dict[str, Any] = {}

        if n_samples < 500:
            # 小数据集: 简单模型 + 强正则化
            params = {
                "n_estimators": 50,
                "max_depth": 3,
                "min_samples_leaf": max(30, n_samples // 50),
                "learning_rate": 0.03,
                "subsample": 0.7,
                "colsample_bytree": 0.6,
                "reg_alpha": 1.0,
                "reg_lambda": 1.0,
            }
        elif n_samples < 2000:
            # 中等数据集: 标准参数
            params = {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_leaf": 20,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
            }
        else:
            # 大数据集: 可以更复杂
            params = {
                "n_estimators": 200,
                "max_depth": 7,
                "min_samples_leaf": 10,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.05,
                "reg_lambda": 0.05,
            }

        # 特征数过多时增加正则化
        if n_features > 0 and n_samples / n_features < 10:
            params["reg_alpha"] = max(params.get("reg_alpha", 0.1) * 2, 0.5)
            params["reg_lambda"] = max(params.get("reg_lambda", 0.1) * 2, 0.5)
            params["colsample_bytree"] = max(params.get("colsample_bytree", 0.8) - 0.2, 0.4)

        logger.info(f"自适应超参数: 样本={n_samples}, 特征={n_features}, "
                    f"max_depth={params['max_depth']}, "
                    f"reg_alpha={params['reg_alpha']}")
        return params

    def _create_model_with_params(self, model_type: str, params: Dict) -> Any:
        """
        使用指定超参数创建模型

        :param model_type: 模型类型 ("rf" / "xgboost" / "lightgbm" / "linear")
        :param params: 超参数字典
        :return: 模型对象
        """
        if model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                min_samples_leaf=params.get("min_samples_leaf", 20),
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 5),
                    learning_rate=params.get("learning_rate", 0.05),
                    subsample=params.get("subsample", 0.8),
                    colsample_bytree=params.get("colsample_bytree", 0.8),
                    reg_alpha=params.get("reg_alpha", 0.1),
                    reg_lambda=params.get("reg_lambda", 0.1),
                    random_state=42,
                    verbosity=0,
                )
            except ImportError:
                logger.warning("xgboost不可用, 跳过")
                return None
        elif model_type == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMRegressor(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 5),
                    learning_rate=params.get("learning_rate", 0.05),
                    subsample=params.get("subsample", 0.8),
                    colsample_bytree=params.get("colsample_bytree", 0.8),
                    reg_alpha=params.get("reg_alpha", 0.1),
                    reg_lambda=params.get("reg_lambda", 0.1),
                    random_state=42,
                    verbose=-1,
                )
            except ImportError:
                logger.warning("lightgbm不可用, 跳过")
                return None
        elif model_type == "linear":
            from sklearn.linear_model import Ridge
            return Ridge(
                alpha=params.get("reg_alpha", 1.0),
                random_state=42,
            )
        else:
            return None

    def _feature_selection_step(self, factors: pd.DataFrame,
                                forward_returns: Optional[pd.Series]) -> pd.DataFrame:
        """
        执行特征选择步骤

        如果启用特征选择且有forward_returns, 执行组合筛选(VIF->IC->相关性).
        否则返回原始因子.

        v7.1修复: 每次调用前同步最新IC历史到FeatureSelector

        :param factors: 因子矩阵
        :param forward_returns: 前向收益(可选)
        :return: 筛选后的因子矩阵
        """
        if not self.enable_feature_selection or forward_returns is None:
            return factors

        # v7.1修复: 同步最新IC历史到特征选择器
        self._feature_selector.ic_history = dict(self.ic_history)

        try:
            selected = self._feature_selector.select_all(factors, forward_returns)
            self._selected_features = selected

            if len(selected) < len(factors.columns):
                logger.info(f"特征选择: {len(factors.columns)} -> {len(selected)} 个因子")
                return factors[selected]
        except Exception as e:
            logger.warning(f"特征选择失败, 使用全部因子: {e}")

        return factors

    def _train_multi_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           n_features: int) -> Dict[str, float]:
        """
        训练多个模型并计算集成权重

        同时训练RF、XGBoost、LightGBM(如果可用), 根据交叉验证R2
        分配集成权重. R2越高权重越大, 但最低权重不低于10%.

        :param X_train: 训练特征矩阵
        :param y_train: 训练目标向量
        :param n_features: 特征数量
        :return: 各模型权重字典 {模型类型: 权重}
        """
        params = self._adaptive_hyperparams(len(X_train), n_features)
        model_types = ["rf", "xgboost", "lightgbm"]
        model_scores: Dict[str, float] = {}
        trained_models: Dict[str, Any] = {}

        for mtype in model_types:
            model = self._create_model_with_params(mtype, params)
            if model is None:
                continue

            try:
                model.fit(X_train, y_train)
                trained_models[mtype] = model

                # 用验证集R2评估
                try:
                    from sklearn.model_selection import cross_val_score
                    cv_model = self._create_model_with_params(mtype, params)
                    scores = cross_val_score(cv_model, X_train, y_train,
                                             cv=3, scoring="r2")
                    model_scores[mtype] = float(np.mean(scores))
                except Exception:
                    # 交叉验证失败, 使用训练R2
                    model_scores[mtype] = float(model.score(X_train, y_train))

                logger.info(f"多模型训练: {mtype} "
                           f"(cv_R2={model_scores[mtype]:.4f})")
            except Exception as e:
                logger.warning(f"模型 {mtype} 训练失败: {e}")

        if not trained_models:
            logger.warning("所有多模型训练失败, 退化为单模型")
            return {}

        # 存储训练好的模型
        self._ensemble_models = trained_models

        # 计算权重: R2越大权重越高
        weights: Dict[str, float] = {}
        for mtype, score in model_scores.items():
            weights[mtype] = max(score, 0.01)  # 最低分保护

        # 归一化, 并设最低权重
        total = sum(weights.values())
        for mtype in weights:
            weights[mtype] = max(weights[mtype] / total, 0.1)

        # 二次归一化
        total = sum(weights.values())
        for mtype in weights:
            weights[mtype] /= total

        self._ensemble_weights = weights
        logger.info(f"多模型集成权重: {weights}")
        return weights

    def _predict_multi_model(self, X: np.ndarray) -> np.ndarray:
        """
        使用多模型集成预测

        各模型预测值的加权平均.

        :param X: 特征矩阵
        :return: 集成预测值
        """
        if not self._ensemble_models or not self._ensemble_weights:
            return np.zeros(len(X))

        predictions = np.zeros(len(X))
        for mtype, model in self._ensemble_models.items():
            weight = self._ensemble_weights.get(mtype, 0.0)
            try:
                pred = model.predict(X)
                predictions += pred * weight
            except Exception as e:
                logger.warning(f"模型 {mtype} 预测失败: {e}")

        return predictions

    def _walk_forward_step(self) -> Optional[Dict]:
        """
        执行Walk-Forward分析步骤

        需要足够的历史数据(至少 train_window + test_window 期).

        :return: Walk-Forward分析结果, 如果数据不足则返回None
        """
        if not self.enable_walk_forward:
            return None

        if len(self._history_factors) < 15:
            logger.debug("历史数据不足, 跳过Walk-Forward分析")
            return None

        try:
            result = self._wf_analyzer.analyze(
                factors_history=self._history_factors,
                returns_history=self._history_returns,
                train_window=self.rolling_window,
                test_window=3,
            )
            self._wf_results = result

            # 检测过拟合
            if result["train_ics"] and result["test_ics"]:
                overfit = self._wf_analyzer.detect_overfit(
                    result["train_ics"], result["test_ics"]
                )
                logger.info(f"过拟合检测结果: {overfit['overfit_level']} "
                           f"- {overfit['suggestion']}")

            return result
        except Exception as e:
            logger.warning(f"Walk-Forward分析失败: {e}")
            return None

    def _version_save_step(self, feature_cols: List[str],
                           n_samples: int, val_r2: float) -> None:
        """
        执行模型版本保存步骤

        保存当前最佳模型和元数据.

        :param feature_cols: 使用的特征列表
        :param n_samples: 训练样本数
        :param val_r2: 验证集R2
        """
        if not self.enable_version_control:
            return

        try:
            # 选择要保存的模型(优先保存集成中的主模型)
            if self._ensemble_models:
                primary_type = max(self._ensemble_weights,
                                   key=self._ensemble_weights.get)
                model_to_save = self._ensemble_models[primary_type]
            elif self._model is not None:
                model_to_save = self._model
            else:
                return

            metadata = {
                "model_type": self.model_type,
                "feature_cols": feature_cols,
                "feature_count": len(feature_cols),
                "train_size": n_samples,
                "val_r2": val_r2,
                "train_r2": float(model_to_save.score(
                    self._get_last_train_X(), self._get_last_train_y()
                )) if hasattr(model_to_save, "score") else 0.0,
                "train_periods": len(self._history_factors),
                "rolling_window": self.rolling_window,
                "selected_features": self._selected_features,
                "ensemble_weights": dict(self._ensemble_weights),
                "wf_overfit_gap": self._wf_results.get("overfit_gap", 0.0)
                if self._wf_results else None,
                "wf_oof_score": self._wf_results.get("oof_score", 0.0)
                if self._wf_results else None,
            }
            self._version_manager.save_model(model_to_save, metadata)
            logger.info("模型版本已自动保存")
        except Exception as e:
            logger.warning(f"模型版本保存失败: {e}")

    def _get_last_train_X(self) -> np.ndarray:
        """获取最近一次训练的特征矩阵"""
        if not self._history_factors:
            return np.array([]).reshape(0, 0)
        all_factors = [f for f in self._history_factors]
        return pd.concat(all_factors).fillna(0).values

    def _get_last_train_y(self) -> np.ndarray:
        """获取最近一次训练的目标向量"""
        if not self._history_returns:
            return np.array([])
        return pd.concat(self._history_returns).values

    def combine(self, factors: pd.DataFrame,
                forward_returns: Optional[pd.Series] = None) -> pd.Series:
        """
        增强ML因子合成主入口

        流程:
        1. 应用因子方向(继承自父类)
        2. 特征选择(如果启用)
        3. 根据方法选择合成策略:
           - enhanced_ensemble: 多模型集成
           - 其他: 退化为父类行为
        4. 用forward_returns更新IC和训练(继承自父类)
        5. Walk-Forward分析(如果启用且数据充足)
        6. 模型版本保存(如果启用)

        :param factors: 因子矩阵 DataFrame
        :param forward_returns: 前向收益 Series(可选, 仅用于事后更新)
        :return: 综合得分 Series
        """
        if factors.empty:
            return pd.Series(dtype=float)

        # 应用因子方向(继承自父类)
        adjusted = factors.copy()
        for col, direction in self.directions.items():
            if col in adjusted.columns:
                adjusted[col] = adjusted[col] * direction

        # 特征选择
        adjusted = self._feature_selection_step(adjusted, forward_returns)

        # 合成
        if self.method == "enhanced_ensemble":
            score = self._enhanced_ensemble_combine(adjusted)
        elif self.method == "multi_model":
            score = self._multi_model_combine(adjusted)
        else:
            # 退化为父类行为
            score = super().combine(adjusted, forward_returns)
            return score

        # 事后更新(继承自父类逻辑)
        if forward_returns is not None:
            self._update_ic(adjusted, forward_returns)
            self._store_history(adjusted, forward_returns)
            # v7.1修复: 先训练父类单模型, 再训练增强集成模型
            super()._train_for_next_period()
            self._train_enhanced_for_next(adjusted, forward_returns)

            # Walk-Forward分析
            self._walk_forward_step()

        logger.info(f"增强ML合成完成: {len(score)} 只股票得分")
        return score

    def _enhanced_ensemble_combine(self, factors: pd.DataFrame) -> pd.Series:
        """
        增强集成合成

        优先使用多模型集成预测, 如果模型未训练则退化为IC加权.

        :param factors: 因子矩阵
        :return: 综合得分 Series
        """
        if self._ensemble_models and self._ensemble_feature_cols is not None:
            try:
                feature_cols = [c for c in self._ensemble_feature_cols
                                if c in factors.columns]
                if len(feature_cols) > 0:
                    X = factors[feature_cols].fillna(0).values
                    predictions = self._predict_multi_model(X)
                    return pd.Series(predictions, index=factors.index)
            except Exception as e:
                logger.warning(f"增强集成预测失败: {e}, 退化为IC加权")

        return self._ic_weight_fallback(factors)

    def _multi_model_combine(self, factors: pd.DataFrame) -> pd.Series:
        """
        多模型集成合成(仅使用多模型, 不混合IC/等权)

        :param factors: 因子矩阵
        :return: 综合得分 Series
        """
        return self._enhanced_ensemble_combine(factors)

    def _train_enhanced_for_next(self, factors: pd.DataFrame,
                                 forward_returns: pd.Series) -> None:
        """
        增强训练: 训练供下一期使用的多模型集成

        v7.1修复: 父类单模型(self._model)已在combine()中通过super()._train_for_next_period()训练
        本方法仅训练多模型集成(self._ensemble_models)
        两者使用相同的训练数据但训练不同的模型,无冗余

        :param factors: 当期因子矩阵
        :param forward_returns: 当期前向收益
        """
        if len(self._history_factors) < 3:
            return

        # 合并历史数据
        all_factors = []
        all_returns = []
        for f, r in zip(self._history_factors, self._history_returns):
            common = f.index.intersection(r.index)
            if len(common) >= 30:
                all_factors.append(f.loc[common])
                all_returns.append(r.loc[common])

        if len(all_factors) < 3:
            return

        X_train = pd.concat(all_factors).fillna(0)
        y_train = pd.concat(all_returns)

        if len(X_train) < self.min_samples:
            return

        n_features = X_train.shape[1]

        # v7.1修复: 父类单模型已在combine()中通过super()._train_for_next_period()训练
        # 本方法仅训练多模型集成
        self._train_multi_model(X_train.values, y_train.values, n_features)

        if self._ensemble_models:
            self._ensemble_feature_cols = list(X_train.columns)

            # 计算验证集R2
            val_r2 = 0.0
            try:
                from sklearn.model_selection import cross_val_score
                # 用主模型类型做交叉验证
                primary_model = self._select_model()
                if primary_model is not None:
                    cv_scores = cross_val_score(
                        primary_model, X_train.values, y_train.values,
                        cv=3, scoring="r2"
                    )
                    val_r2 = float(np.mean(cv_scores))
            except Exception:
                pass

            # 版本保存
            self._version_save_step(list(X_train.columns), len(X_train), val_r2)

    def get_walk_forward_results(self) -> Optional[Dict]:
        """
        获取最近的Walk-Forward分析结果

        :return: Walk-Forward分析结果字典, 如果未执行过则返回None
        """
        return self._wf_results

    def get_selected_features(self) -> Optional[List[str]]:
        """
        获取最近一次特征选择的结果

        :return: 选中因子名称列表, 如果未执行过则返回None
        """
        return self._selected_features

    def get_ensemble_info(self) -> Dict:
        """
        获取多模型集成的详细信息

        :return: 集成信息字典, 包含模型类型、权重等
        """
        return {
            "models": list(self._ensemble_models.keys()),
            "weights": dict(self._ensemble_weights),
            "feature_cols": self._ensemble_feature_cols,
        }
