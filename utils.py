"""
工具函数: 缓存、日志、日期处理
"""
import os
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np

from config import CACHE_DIR, USE_CACHE, CACHE_EXPIRE_HOURS


# ============ 日志 ============
def get_logger(name: str = "quant") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


# ============ 缓存装饰器 ============
def disk_cache(key_prefix: str = ""):
    """磁盘缓存装饰器,基于函数参数生成缓存键"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not USE_CACHE:
                return func(*args, **kwargs)

            # 生成缓存键
            key_str = f"{key_prefix}_{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
            key_hash = hashlib.md5(key_str.encode()).hexdigest()[:16]
            cache_file = CACHE_DIR / f"{key_prefix}_{key_hash}.pkl"

            # 检查缓存
            if cache_file.exists():
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - mtime < timedelta(hours=CACHE_EXPIRE_HOURS):
                    try:
                        with open(cache_file, "rb") as f:
                            logger.debug(f"缓存命中: {cache_file.name}")
                            return pickle.load(f)
                    except Exception:
                        pass  # 缓存损坏,重新计算

            # 执行函数
            result = func(*args, **kwargs)

            # 写入缓存
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            except Exception as e:
                logger.warning(f"缓存写入失败: {e}")

            return result
        return wrapper
    return decorator


# ============ 日期工具 ============
def get_trade_dates(start: str, end: str) -> list:
    """获取交易日列表(使用 akshare)"""
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
        mask = (dates >= start.replace("-", "")) & (dates <= end.replace("-", ""))
        return dates[mask].tolist()
    except Exception as e:
        logger.warning(f"获取交易日失败,使用日历近似: {e}")
        return pd.bdate_range(start, end).strftime("%Y%m%d").tolist()


def get_month_end_dates(start: str, end: str) -> list:
    """获取每月最后一个交易日"""
    trade_dates = get_trade_dates(start, end)
    if not trade_dates:
        return []
    s = pd.Series(pd.to_datetime(trade_dates, format="%Y%m%d"))
    # 每月最后一个交易日
    month_ends = s.groupby(s.dt.to_period("M")).max()
    return month_ends.dt.strftime("%Y%m%d").tolist()


def get_quarter_end_dates(start: str, end: str) -> list:
    """获取每季最后一个交易日"""
    trade_dates = get_trade_dates(start, end)
    if not trade_dates:
        return []
    s = pd.Series(pd.to_datetime(trade_dates, format="%Y%m%d"))
    quarter_ends = s.groupby(s.dt.to_period("Q")).max()
    return quarter_ends.dt.strftime("%Y%m%d").tolist()


def get_rebalance_dates(start: str, end: str, freq: str = "M") -> list:
    """根据频率获取调仓日"""
    if freq == "M":
        return get_month_end_dates(start, end)
    elif freq == "Q":
        return get_quarter_end_dates(start, end)
    elif freq == "W":
        trade_dates = get_trade_dates(start, end)
        return trade_dates  # 每周都调仓(简化)
    else:
        return get_month_end_dates(start, end)


# ============ 数据处理工具 ============
def safe_divide(numerator, denominator):
    """安全除法,避免除零"""
    return np.where(
        (denominator != 0) & (~np.isnan(denominator)),
        numerator / denominator,
        np.nan,
    )


def rank_normalize(series: pd.Series, ascending: bool = True) -> pd.Series:
    """排名归一化到 [0, 1]"""
    ranked = series.rank(pct=True, ascending=ascending)
    return ranked


def zscore(series: pd.Series) -> pd.Series:
    """Z-Score 标准化"""
    return (series - series.mean()) / (series.std() + 1e-10)


def safe_read_pickle(path: Path) -> Optional[Any]:
    """安全读取 pickle"""
    try:
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        return None
    return None
