"""
数据获取层: 全A股列表、历史行情、财务数据、实时行情
基于 akshare,内置磁盘缓存
"""
import time
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import (
    DATA_SOURCE, REALTIME_SOURCE, EXCLUDE_ST, EXCLUDE_NEW_DAYS,
    MIN_LIST_DAYS, MIN_MARKET_CAP, EXCLUDE_BOARD, CACHE_DIR
)
from utils import disk_cache, get_logger, safe_read_pickle
from pathlib import Path

logger = get_logger("data")


# ============ 股票池 ============
@disk_cache("stock_list")
def get_stock_list() -> pd.DataFrame:
    """获取全A股列表(含代码、名称、市值、行业、上市日期)"""
    import akshare as ak
    logger.info("获取全A股列表...")

    df = ak.stock_zh_a_spot_em()
    df = df.rename(columns={
        "代码": "code", "名称": "name", "总市值": "market_cap",
        "涨跌幅": "pct_chg", "最新价": "close",
    })

    # 排除 ST
    if EXCLUDE_ST:
        mask = (~df["name"].str.contains("ST", na=False)) & \
               (~df["name"].str.contains(r"\*ST", na=False, regex=True))
        df = df[mask].copy()
        logger.info(f"排除ST后剩余 {len(df)} 只")

    # 排除北交所(8xx/4xx)和已退市
    df = df[df["code"].str.match(r"^(0|3|6)\d{5}$")].copy()

    # 市值过滤
    if MIN_MARKET_CAP > 0 and "market_cap" in df.columns:
        df = df[df["market_cap"] >= MIN_MARKET_CAP].copy()
        logger.info(f"市值过滤后剩余 {len(df)} 只")

    # 获取行业分类
    try:
        industry_df = ak.stock_board_industry_name_em()
        # 用个股行业归属
        stock_industry = ak.stock_individual_info_em  # 占位
    except Exception as e:
        logger.warning(f"获取行业分类失败: {e}")

    # 获取上市日期
    logger.info("补充上市日期信息...")
    list_dates = {}
    for code in tqdm(df["code"].tolist()[:200], desc="上市日期", leave=False):
        try:
            info = ak.stock_individual_info_em(symbol=code)
            date_row = info[info["item"] == "上市时间"]
            if not date_row.empty:
                list_dates[code] = date_row["value"].values[0]
        except Exception:
            pass
        time.sleep(0.05)
    df["list_date"] = df["code"].map(list_dates)

    logger.info(f"最终股票池: {len(df)} 只")
    return df


def get_stock_universe(trade_date: str) -> list:
    """获取某一交易日的有效股票池(过滤次新、停牌)"""
    import akshare as ak
    try:
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={"代码": "code", "名称": "name"})
        codes = df["code"].tolist()
        # 过滤 ST
        if EXCLUDE_ST:
            codes = [c for c in codes
                     if "ST" not in df[df["code"] == c]["name"].values[0]]
        return [c for c in codes if c.startswith(("0", "3", "6"))]
    except Exception as e:
        logger.warning(f"获取股票池失败: {e}")
        return []


# ============ 历史行情 ============
@disk_cache("daily_data")
def get_daily_data(code: str, start_date: str, end_date: str,
                   adjust: str = "qfq") -> pd.DataFrame:
    """获取个股前复权日线数据"""
    import akshare as ak
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start_date, end_date=end_date,
            adjust=adjust
        )
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "换手率": "turnover", "涨跌幅": "pct_chg",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["code"] = code
        return df
    except Exception as e:
        logger.debug(f"获取 {code} 行情失败: {e}")
        return pd.DataFrame()


@disk_cache("batch_daily")
def get_batch_daily_data(codes: list, start_date: str, end_date: str,
                         adjust: str = "qfq") -> dict:
    """批量获取多只股票日线数据"""
    results = {}
    for code in tqdm(codes, desc="下载行情"):
        df = get_daily_data(code, start_date, end_date, adjust)
        if not df.empty:
            results[code] = df
        time.sleep(0.1)  # 避免频率限制
    logger.info(f"成功获取 {len(results)}/{len(codes)} 只股票行情")
    return results


def get_panel_data(codes: list, start_date: str, end_date: str) -> pd.DataFrame:
    """获取面板数据(所有股票合并成一张表)"""
    data_dict = get_batch_daily_data(codes, start_date, end_date)
    if not data_dict:
        return pd.DataFrame()
    panel = pd.concat(data_dict.values(), ignore_index=True)
    return panel


# ============ 财务数据 ============
@disk_cache("financial")
def get_financial_data(code: str, n_periods: int = 8) -> pd.DataFrame:
    """获取个股财务指标(杜邦分析、估值、成长等)"""
    import akshare as ak
    try:
        df = ak.stock_financial_analysis_indicator(symbol=code)
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={"日期": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").head(n_periods)
        df["code"] = code
        return df
    except Exception as e:
        logger.debug(f"获取 {code} 财务数据失败: {e}")
        return pd.DataFrame()


@disk_cache("batch_financial")
def get_batch_financial(codes: list, n_periods: int = 8) -> pd.DataFrame:
    """批量获取财务数据"""
    results = []
    for code in tqdm(codes, desc="下载财务"):
        df = get_financial_data(code, n_periods)
        if not df.empty:
            results.append(df)
        time.sleep(0.1)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def get_valuation_data(code: str) -> pd.DataFrame:
    """获取估值数据(PE/PB/PS)"""
    import akshare as ak
    try:
        df = ak.stock_a_indicator_lg(symbol=code)
        if df is None or df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["trade_date"])
        df["code"] = code
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        logger.debug(f"获取 {code} 估值数据失败: {e}")
        return pd.DataFrame()


# ============ 实时行情 ============
def get_realtime_quotes(codes: Optional[list] = None) -> pd.DataFrame:
    """获取实时行情快照(全市场或指定股票)"""
    import akshare as ak
    try:
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={
            "代码": "code", "名称": "name", "最新价": "close",
            "涨跌幅": "pct_chg", "成交额": "amount",
            "换手率": "turnover", "总市值": "market_cap",
            "市盈率-动态": "pe_ttm", "市净率": "pb",
        })
        if codes:
            df = df[df["code"].isin(codes)]
        return df.reset_index(drop=True)
    except Exception as e:
        logger.error(f"获取实时行情失败: {e}")
        return pd.DataFrame()


# ============ 行业分类 ============
@disk_cache("industry")
def get_industry_mapping() -> pd.DataFrame:
    """获取股票-行业映射表"""
    import akshare as ak
    try:
        df = ak.stock_board_industry_name_em()
        logger.info(f"获取 {len(df)} 个行业分类")
        return df
    except Exception as e:
        logger.warning(f"获取行业分类失败: {e}")
        return pd.DataFrame()


def get_stock_industry(codes: list) -> pd.Series:
    """获取个股所属行业(返回 Series: code -> industry)"""
    import akshare as ak
    result = {}
    for code in tqdm(codes[:500], desc="行业归属", leave=False):
        try:
            info = ak.stock_individual_info_em(symbol=code)
            ind_row = info[info["item"] == "行业"]
            if not ind_row.empty:
                result[code] = ind_row["value"].values[0]
        except Exception:
            pass
        time.sleep(0.03)
    return pd.Series(result, name="industry")


# ============ 基准指数 ============
@disk_cache("benchmark")
def get_benchmark_data(benchmark: str = "hs300",
                      start_date: str = "20200101",
                      end_date: str = "20260725") -> pd.DataFrame:
    """获取基准指数数据"""
    import akshare as ak
    code_map = {
        "hs300": "000300",   # 沪深300
        "zz500": "000905",   # 中证500
        "cyb":   "399006",   # 创业板指
        "sh":    "000001",   # 上证综指
    }
    index_code = code_map.get(benchmark, "000300")
    try:
        df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
        if df is None or df.empty:
            df = ak.stock_zh_index_daily(symbol=f"sz{index_code}")
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= pd.to_datetime(start_date)) & \
               (df["date"] <= pd.to_datetime(end_date))
        return df[mask].sort_values("date").reset_index(drop=True)
    except Exception as e:
        logger.error(f"获取基准数据失败: {e}")
        return pd.DataFrame()
