"""
数据获取层 v6.0: 全A股列表、历史行情、财务数据、实时行情
基于 akshare,内置磁盘缓存 + 多源备份(腾讯/新浪)
v6.0新增: 资金流向、北向资金、龙虎榜、融资融券、板块热度
"""
import time
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import (
    DATA_SOURCE, REALTIME_SOURCE, EXCLUDE_ST, EXCLUDE_NEW_DAYS,
    MIN_LIST_DAYS, MIN_MARKET_CAP, EXCLUDE_BOARD, CACHE_DIR,
    BACKUP_SOURCE_ENABLED, DATA_FETCH_TIMEOUT, DATA_FETCH_RETRIES
)
from utils import disk_cache, get_logger, safe_read_pickle
from pathlib import Path

logger = get_logger("data")

# 尝试导入备份数据源
try:
    import backup_data_source as bds
    BACKUP_OK = True
except Exception:
    BACKUP_OK = False


def _safe_call(func, *args, retries=DATA_FETCH_RETRIES, delay=0.3, **kwargs):
    """安全调用,异常重试N次返回None"""
    for i in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception:
            if i < retries:
                time.sleep(delay)
    return None


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

    # v6.2修复: 获取上市日期(移除[:200]截断,改用批量接口)
    logger.info("补充上市日期信息...")
    list_dates = {}
    # 尝试使用批量接口获取上市时间
    try:
        info_df = _safe_call(ak.stock_info_a_code_name)
        if info_df is not None and not info_df.empty:
            # 部分版本akshare此接口含上市时间
            pass
    except Exception:
        pass
    
    # 逐只获取(限制并发,避免被限流)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def _fetch_list_date(code):
        try:
            info = ak.stock_individual_info_em(symbol=code)
            date_row = info[info["item"] == "上市时间"]
            if not date_row.empty:
                return code, str(date_row["value"].values[0])
        except Exception:
            pass
        return code, None

    # v6.2: 并发获取(最多20线程)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_fetch_list_date, code): code 
                   for code in df["code"].tolist()}
        for future in as_completed(futures):
            code, list_date = future.result()
            if list_date:
                list_dates[code] = list_date
    df["list_date"] = df["code"].map(list_dates)
    logger.info(f"获取上市日期: {len(list_dates)}/{len(df)} 只")

    logger.info(f"最终股票池: {len(df)} 只")
    return df


def get_stock_universe(trade_date: str) -> list:
    """获取某一交易日的有效股票池(过滤次新、停牌)"""
    import akshare as ak
    try:
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={"代码": "code", "名称": "name"})
        codes = df["code"].tolist()
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
    """获取个股前复权日线数据(多源: AKShare -> 腾讯备份)"""
    import akshare as ak

    # 1. AKShare主源
    df = _safe_call(
        ak.stock_zh_a_hist,
        symbol=code, period="daily",
        start_date=start_date, end_date=end_date, adjust=adjust
    )
    if df is not None and not df.empty:
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "换手率": "turnover", "涨跌幅": "pct_chg",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["code"] = code
        return df

    # 2. 腾讯备份源
    if BACKUP_OK and BACKUP_SOURCE_ENABLED:
        df = _safe_call(bds.get_tencent_kline, code, days=250, adjust=adjust)
        if df is not None and not df.empty:
            logger.debug(f"{code} 使用腾讯备份K线")
            return df

    logger.debug(f"获取 {code} 行情失败")
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
        time.sleep(0.1)
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
    df = _safe_call(ak.stock_financial_analysis_indicator, symbol=code)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"日期": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").head(n_periods)
    df["code"] = code
    return df


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
    df = _safe_call(ak.stock_a_indicator_lg, symbol=code)
    if df is None or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["trade_date"])
    df["code"] = code
    return df.sort_values("date").reset_index(drop=True)


# ============ 实时行情 ============
def get_realtime_quotes(codes: Optional[list] = None) -> pd.DataFrame:
    """获取实时行情快照(全市场或指定股票, 多源融合)"""
    import akshare as ak

    # 1. AKShare主源
    df = _safe_call(ak.stock_zh_a_spot_em)
    if df is not None and not df.empty:
        df = df.rename(columns={
            "代码": "code", "名称": "name", "最新价": "close",
            "涨跌幅": "pct_chg", "成交额": "amount",
            "换手率": "turnover", "总市值": "market_cap",
            "市盈率-动态": "pe_ttm", "市净率": "pb",
            "流通市值": "circ_mv", "量比": "volume_ratio",
            "振幅": "amplitude",
        })
        if codes:
            df = df[df["code"].isin(codes)]
        return df.reset_index(drop=True)

    # 2. 备份源逐只获取
    if BACKUP_OK and BACKUP_SOURCE_ENABLED and codes:
        logger.info("AKShare实时行情失败,使用备份源...")
        results = []
        for code in codes:
            q = _safe_call(bds.get_quote_multi_source, code)
            if q:
                results.append(q)
        if results:
            return pd.DataFrame(results)
    return pd.DataFrame()


# ============ 资金流向(v6.0新增) ============
def get_fund_flow(code: str) -> Optional[dict]:
    """获取个股资金流向"""
    import akshare as ak
    market = 'sh' if code.startswith('6') else 'sz'
    df = _safe_call(ak.stock_individual_fund_flow, stock=code, market=market)
    if df is None or df.empty:
        return None
    df = df.tail(10)
    latest = df.iloc[-1]
    return {
        'main_net': float(latest.get('主力净流入-净额', 0) or 0),
        'main_pct': float(latest.get('主力净流入-净占比', 0) or 0),
        'super_large_net': float(latest.get('超大单净流入-净额', 0) or 0),
        'large_net': float(latest.get('大单净流入-净额', 0) or 0),
        'medium_net': float(latest.get('中单净流入-净额', 0) or 0),
        'small_net': float(latest.get('小单净流入-净额', 0) or 0),
        'main_5d': float(df.tail(5)['主力净流入-净额'].sum()) if '主力净流入-净额' in df.columns else 0,
        'main_10d': float(df['主力净流入-净额'].sum()) if '主力净流入-净额' in df.columns else 0,
        'flow_status': '主力净流入' if float(latest.get('主力净流入-净额', 0) or 0) > 0 else '主力净流出'
    }


def get_fund_flow_ranking(top_n: int = 150) -> pd.DataFrame:
    """获取主力资金流向排名"""
    import akshare as ak
    df = _safe_call(ak.stock_individual_fund_flow_rank, indicator="今日")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"代码": "code", "名称": "name"})
    if "主力净流入-净额" in df.columns:
        df = df.sort_values("主力净流入-净额", ascending=False).head(top_n)
    return df.reset_index(drop=True)


# ============ 北向资金 ============
def get_northbound_flow() -> Optional[dict]:
    """获取北向资金数据(兼容新版接口)"""
    import akshare as ak

    # v6.2修复: 兼容多种北向资金接口
    df = _safe_call(getattr(ak, 'stock_hsgt_hist_em', None), symbol="北向资金")
    if df is None or df.empty:
        df = _safe_call(getattr(ak, 'stock_hsgt_fund_flow_summary_em', None))
    if df is None or df.empty:
        df = _safe_call(getattr(ak, 'stock_hsgt_north_net_flow_in_em', None), symbol="北上")
    if df is None or df.empty:
        return None
    latest = df.tail(5)
    return {
        'latest_net': float(latest.iloc[-1].get('当日成交净买额', 0) or 0),
        'latest_date': str(latest.iloc[-1].get('交易日', '') or ''),
        'trend': '净流入' if float(latest.iloc[-1].get('当日成交净买额', 0) or 0) > 0 else '净流出'
    }


# ============ 行业分类 ============
@disk_cache("industry")
def get_industry_mapping() -> pd.DataFrame:
    """获取股票-行业映射表"""
    import akshare as ak
    df = _safe_call(ak.stock_board_industry_name_em)
    if df is None or df.empty:
        return pd.DataFrame()
    logger.info(f"获取 {len(df)} 个行业分类")
    return df


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


def get_sector_heatmap() -> Optional[dict]:
    """获取行业板块涨跌热力图"""
    import akshare as ak
    df = _safe_call(ak.stock_board_industry_name_em)
    if df is None or df.empty:
        return None
    top5 = df.nlargest(5, '涨跌幅')[['板块名称', '涨跌幅']].to_dict('records')
    bottom5 = df.nsmallest(5, '涨跌幅')[['板块名称', '涨跌幅']].to_dict('records')
    return {
        'top5': [{'name': str(r['板块名称']), 'pct': float(r['涨跌幅'])} for r in top5],
        'bottom5': [{'name': str(r['板块名称']), 'pct': float(r['涨跌幅'])} for r in bottom5],
        'total': len(df)
    }


def get_market_overview() -> Optional[dict]:
    """获取大盘总览(上证/深证/创业板)"""
    import akshare as ak
    indices = {'上证指数': 'sh000001', '深证成指': 'sz399001', '创业板指': 'sz399006'}
    result = {}
    df = _safe_call(ak.stock_zh_index_spot_em)
    if df is None or df.empty:
        return None
    for name, code in indices.items():
        row = df[df['代码'] == code]
        if not row.empty:
            r = row.iloc[0]
            result[name] = {
                'price': float(r.get('最新价', 0) or 0),
                'change_pct': float(r.get('涨跌幅', 0) or 0),
                'amount': float(r.get('成交额', 0) or 0)
            }
    return result


# ============ 基准指数 ============
@disk_cache("benchmark")
def get_benchmark_data(benchmark: str = "hs300",
                      start_date: str = "20200101",
                      end_date: str = "20260728") -> pd.DataFrame:
    """获取基准指数数据"""
    import akshare as ak
    code_map = {
        "hs300": "000300",   # 沪深300
        "zz500": "000905",   # 中证500
        "cyb":   "399006",   # 创业板指
        "sh":    "000001",   # 上证综指
    }
    index_code = code_map.get(benchmark, "000300")
    df = _safe_call(ak.stock_zh_index_daily, symbol=f"sh{index_code}")
    if df is None or df.empty:
        df = _safe_call(ak.stock_zh_index_daily, symbol=f"sz{index_code}")
    if df is None or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.to_datetime(start_date)) & \
           (df["date"] <= pd.to_datetime(end_date))
    return df[mask].sort_values("date").reset_index(drop=True)


def get_index_daily(symbol: str, days: int = 120) -> pd.DataFrame:
    """获取指数日线数据(用于大盘择时)"""
    import akshare as ak
    end = datetime.now().strftime('%Y%m%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    df = _safe_call(ak.stock_zh_index_daily, symbol=symbol)
    if df is None or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
    return df[mask].sort_values("date").reset_index(drop=True)
