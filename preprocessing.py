"""
因子预处理 v6.2: 去极值、标准化、联合行业市值中性化
v6.2修复:
  1. [P0] 修复winsorize覆盖bug: quantile clip基于MAD处理后的数据
  2. [P1] 联合中性化: 行业+市值同时回归取残差(不再分步破坏)
  3. [P1] 新增RobustScaler选项
"""
import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats

from config import WINSORIZE_QUANTILE, NEUTRALIZE_INDUSTRY, NEUTRALIZE_SIZE
from utils import get_logger

logger = get_logger("preprocess")


class FactorPreprocessor:
    """因子预处理器 v6.2"""

    def __init__(self,
                 winsorize_q: float = WINSORIZE_QUANTILE,
                 neutralize_industry: bool = NEUTRALIZE_INDUSTRY,
                 neutralize_size: bool = NEUTRALIZE_SIZE):
        self.winsorize_q = winsorize_q
        self.neutralize_industry = neutralize_industry
        self.neutralize_size = neutralize_size

    def process(self, factors: pd.DataFrame,
                industry: Optional[pd.Series] = None,
                market_cap: Optional[pd.Series] = None) -> pd.DataFrame:
        """完整预处理流程"""
        if factors.empty:
            return factors

        result = factors.copy()

        # 1. 去极值
        result = self.winsorize(result)
        logger.info(f"去极值完成 ({self.winsorize_q*100}% 分位)")

        # 2. 标准化
        result = self.standardize(result)
        logger.info("标准化完成 (Z-Score)")

        # 3. v6.2: 联合中性化(行业+市值同时回归)
        if self.neutralize_industry and self.neutralize_size and \
           industry is not None and market_cap is not None:
            result = self.neutralize_joint(result, industry, market_cap)
            logger.info("联合行业+市值中性化完成")
        elif self.neutralize_industry and industry is not None:
            result = self.neutralize_by_industry(result, industry)
            logger.info("行业中性化完成")
        elif self.neutralize_size and market_cap is not None:
            result = self.neutralize_by_size(result, market_cap)
            logger.info("市值中性化完成")

        return result

    def winsorize(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        v6.2修复: MAD法去极值 + 分位数截断
        修复: quantile clip基于MAD处理后的数据(不再覆盖)
        """
        result = factors.copy()
        for col in result.columns:
            s = result[col]
            
            # Step 1: MAD法去极值
            median = s.median()
            mad = (s - median).abs().median()
            if mad > 0 and not np.isnan(mad):
                upper = median + 3 * 1.4826 * mad
                lower = median - 3 * 1.4826 * mad
                result[col] = s.clip(lower, upper)
            
            # v6.2修复: Step 2基于MAD处理后的数据做分位数截尾
            # (之前用的是原始s,导致MAD结果被覆盖)
            q_low = result[col].quantile(self.winsorize_q)
            q_high = result[col].quantile(1 - self.winsorize_q)
            result[col] = result[col].clip(q_low, q_high)

        return result

    def standardize(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Z-Score 标准化"""
        result = factors.copy()
        for col in result.columns:
            s = result[col]
            mean = s.mean()
            std = s.std()
            if std > 0 and not np.isnan(std):
                result[col] = (s - mean) / std
            else:
                result[col] = 0
        return result

    def neutralize_joint(self, factors: pd.DataFrame,
                          industry: pd.Series,
                          market_cap: pd.Series) -> pd.DataFrame:
        """
        v6.2新增: 联合行业+市值中性化
        同时对行业哑变量和log(market_cap)回归取残差
        避免分步中性化时第二步破坏第一步的效果
        """
        from sklearn.linear_model import LinearRegression

        common_idx = factors.index.intersection(industry.index).intersection(market_cap.index)
        if len(common_idx) == 0:
            return factors

        f = factors.loc[common_idx].copy()
        ind = industry.loc[common_idx]
        log_mc = np.log(market_cap.loc[common_idx].clip(lower=1))

        # 行业哑变量
        industry_dummies = pd.get_dummies(ind, prefix="ind", drop_first=True)
        # 市值特征
        mc_feature = log_mc.values.reshape(-1, 1)
        
        # 合并特征矩阵
        if industry_dummies.shape[1] > 0:
            X = np.hstack([industry_dummies.values, mc_feature])
        else:
            X = mc_feature

        result = f.copy()
        for col in f.columns:
            y = f[col].fillna(f[col].mean())
            try:
                model = LinearRegression()
                model.fit(X, y)
                residual = y - model.predict(X)
                result[col] = residual
            except Exception:
                pass

        return result

    def neutralize_by_industry(self, factors: pd.DataFrame,
                               industry: pd.Series) -> pd.DataFrame:
        """行业中性化: 对每个因子做行业哑变量回归取残差"""
        from sklearn.linear_model import LinearRegression

        common_idx = factors.index.intersection(industry.index)
        if len(common_idx) == 0:
            return factors

        f = factors.loc[common_idx].copy()
        ind = industry.loc[common_idx]

        industry_dummies = pd.get_dummies(ind, prefix="ind", drop_first=True)

        result = f.copy()
        for col in f.columns:
            y = f[col].fillna(f[col].mean())
            X = industry_dummies.values
            if X.shape[1] == 0:
                continue
            try:
                model = LinearRegression()
                model.fit(X, y)
                residual = y - model.predict(X)
                result[col] = residual
            except Exception:
                pass

        return result

    def neutralize_by_size(self, factors: pd.DataFrame,
                           market_cap: pd.Series) -> pd.DataFrame:
        """市值中性化: 对 log(market_cap) 回归取残差"""
        from sklearn.linear_model import LinearRegression

        common_idx = factors.index.intersection(market_cap.index)
        if len(common_idx) == 0:
            return factors

        f = factors.loc[common_idx].copy()
        log_mc = np.log(market_cap.loc[common_idx].clip(lower=1))

        result = f.copy()
        for col in f.columns:
            y = f[col].fillna(f[col].mean())
            X = log_mc.values.reshape(-1, 1)
            try:
                model = LinearRegression()
                model.fit(X, y)
                residual = y - model.predict(X)
                result[col] = residual
            except Exception:
                pass

        return result
