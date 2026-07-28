"""
v7.0 扩展因子模块: 高级Alpha因子
在 v6.0 基础因子体系(价值/成长/动量/反转/质量/波动率/流动性)之上,
新增10个高级Alpha因子,涵盖基本面质量、盈余质量、技术形态与风险度量:

  1. Piotroski F-Score       — 基本面综合质量评分(0-9)
  2. Altman Z-Score           — 破产风险评估
  3. Sloan应计质量因子         — 盈余质量(高应计=低质量, direction=-1)
  4. 标准化未预期盈余(SUE)    — 盈余惊喜
  5. 盈余持续性               — EPS自回归系数beta
  6. 价格动量加速度           — Mom_60d - Mom_120d
  7. 52周高低比               — 当前价在52周价格区间中的位置
  8. 成交量加权偏离度(VWAP_Dev) — 价格偏离VWAP的程度
  9. 下行风险因子             — 仅考虑负收益的年化波动, direction=-1
  10. 协偏度(Co-skewness)     — 个股在市场下跌时的尾部风险

所有因子按截面计算,返回 DataFrame(index=code, columns=[factor_names])
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from factors import FactorCalculator
from config import FACTOR_CONFIG
from utils import get_logger, safe_divide

logger = get_logger("factors_v7")


class ExtendedFactorCalculator(FactorCalculator):
    """扩展因子计算器 v7.0 — 继承 v6.0 全部因子,追加10个高级Alpha因子"""

    _FINANCIAL_COLUMN_ALIASES = {
        "总资产报酬率(%%)": ["总资产报酬率", "roa", "ROA"],
        "资产负债率(%%)": ["资产负债率", "debt_ratio", "debt_to_assets"],
        "销售毛利率(%%)": ["销售毛利率", "gross_margin", "gross_profit_margin"],
        "总资产周转率": ["asset_turnover", "total_asset_turnover"],
        "总资产": ["total_assets", "资产总计"],
        "流动负债": ["current_liabilities", "流动负债合计"],
        "流动资产": ["current_assets", "流动资产合计"],
        "留存收益": ["retained_earnings", "盈余公积"],
        "营业利润": ["operating_profit", "ebit"],
        "总负债": ["total_liabilities", "负债合计"],
        "流动比率": ["current_ratio"],
        "总股本": ["total_shares", "share_capital"],
    }

    def _normalize_financial_columns(self, financial):
        if financial.empty:
            return financial
        fin = financial.copy()
        rename_map = {}
        for canonical, aliases in self._FINANCIAL_COLUMN_ALIASES.items():
            if canonical in fin.columns:
                continue
            for alias in aliases:
                if alias in fin.columns:
                    rename_map[alias] = canonical
                    break
        if rename_map:
            fin = fin.rename(columns=rename_map)
        return fin

    def compute_all_factors(self, panel, financial, trade_date, benchmark_data=None):
        factors = super().compute_all_factors(panel, financial, trade_date, benchmark_data)
        if factors.empty:
            return factors
        td = pd.to_datetime(trade_date, format="%%Y%%m%%d")
        cross_data = panel[panel["date"] <= td].copy()
        financial = self._normalize_financial_columns(financial)
        # v7.0 高级Alpha因子
        try:
            f_piotroski = self._compute_piotroski_f_score(financial, td)
            factors = factors.join(f_piotroski, how="left")
        except Exception as e:
            logger.debug(f"Piotroski F-Score 计算异常: {e}")
        try:
            f_altman = self._compute_altman_z_score(financial, td, cross_data)
            factors = factors.join(f_altman, how="left")
        except Exception as e:
            logger.debug(f"Altman Z-Score 计算异常: {e}")
        try:
            f_accrual = self._compute_sloan_accrual(financial, td)
            factors = factors.join(f_accrual, how="left")
        except Exception as e:
            logger.debug(f"Sloan应计质量因子 计算异常: {e}")
        try:
            f_sue = self._compute_sue(financial, td)
            factors = factors.join(f_sue, how="left")
        except Exception as e:
            logger.debug(f"SUE 计算异常: {e}")
        try:
            f_earnings_persist = self._compute_earnings_persistence(financial, td)
            factors = factors.join(f_earnings_persist, how="left")
        except Exception as e:
            logger.debug(f"盈余持续性 计算异常: {e}")
        try:
            f_mom_accel = self._compute_momentum_acceleration(cross_data, td)
            factors = factors.join(f_mom_accel, how="left")
        except Exception as e:
            logger.debug(f"价格动量加速度 计算异常: {e}")
        try:
            f_high_low = self._compute_high_low_ratio(cross_data, td)
            factors = factors.join(f_high_low, how="left")
        except Exception as e:
            logger.debug(f"52周高低比 计算异常: {e}")
        try:
            f_vwap_dev = self._compute_vwap_deviation(cross_data, td)
            factors = factors.join(f_vwap_dev, how="left")
        except Exception as e:
            logger.debug(f"VWAP偏离度 计算异常: {e}")
        try:
            f_downside = self._compute_downside_risk(cross_data, td)
            factors = factors.join(f_downside, how="left")
        except Exception as e:
            logger.debug(f"下行风险因子 计算异常: {e}")
        try:
            f_coskew = self._compute_co_skewness(cross_data, td, benchmark_data)
            factors = factors.join(f_coskew, how="left")
        except Exception as e:
            logger.debug(f"协偏度 计算异常: {e}")
        logger.info(f"截面 {trade_date}: v7.0 扩展因子追加完成, 当前共 {factors.shape[1]} 个因子")
        return factors

    def _compute_piotroski_f_score(self, financial, td):
        factors = pd.DataFrame()
        if financial.empty:
            return factors
        try:
            fin = financial[financial["date"] <= td].copy()
            fin = fin.sort_values(["code", "date"])
            results = []
            for code, g in fin.groupby("code"):
                if len(g) < 2:
                    continue
                latest = g.iloc[-1]
                prev = g.iloc[-2]
                score = 0
                roa_now = latest.get("总资产报酬率(%%)", np.nan)
                if pd.notna(roa_now) and roa_now > 0:
                    score += 1
                ocf_now = latest.get("经营现金流", np.nan)
                if pd.notna(ocf_now) and ocf_now > 0:
                    score += 1
                roa_prev = prev.get("总资产报酬率(%%)", np.nan)
                if pd.notna(roa_now) and pd.notna(roa_prev) and roa_now > roa_prev:
                    score += 1
                net_profit = latest.get("净利润", np.nan)
                if pd.notna(ocf_now) and pd.notna(net_profit):
                    if net_profit > 0 and ocf_now > net_profit:
                        score += 1
                    elif net_profit < 0 and ocf_now > net_profit:
                        score += 1
                debt_now = latest.get("资产负债率(%%)", np.nan)
                debt_prev = prev.get("资产负债率(%%)", np.nan)
                if pd.notna(debt_now) and pd.notna(debt_prev) and debt_now < debt_prev:
                    score += 1
                curr_ratio_now = latest.get("流动比率", np.nan)
                curr_ratio_prev = prev.get("流动比率", np.nan)
                if pd.notna(curr_ratio_now) and pd.notna(curr_ratio_prev) and curr_ratio_now > curr_ratio_prev:
                    score += 1
                shares_now = latest.get("总股本", np.nan)
                shares_prev = prev.get("总股本", np.nan)
                if pd.notna(shares_now) and pd.notna(shares_prev) and shares_now <= shares_prev:
                    score += 1
                gm_now = latest.get("销售毛利率(%%)", np.nan)
                gm_prev = prev.get("销售毛利率(%%)", np.nan)
                if pd.notna(gm_now) and pd.notna(gm_prev) and gm_now > gm_prev:
                    score += 1
                turnover_now = latest.get("总资产周转率", np.nan)
                turnover_prev = prev.get("总资产周转率", np.nan)
                if pd.notna(turnover_now) and pd.notna(turnover_prev) and turnover_now > turnover_prev:
                    score += 1
                results.append({"code": code, "f_score": score})
            if results:
                factors = pd.DataFrame(results).set_index("code")
        except Exception as e:
            logger.debug(f"Piotroski F-Score 计算异常: {e}")
        return factors

    def _compute_altman_z_score(self, financial, td, cross_data):
        factors = pd.DataFrame()
        if financial.empty:
            return factors
        try:
            fin = financial[financial["date"] <= td].copy()
            latest = fin.sort_values("date").groupby("code").last()
            price_latest = cross_data.sort_values("date").groupby("code").last()
            market_cap = price_latest.get("market_cap", price_latest["close"] * 1e8)
            codes = latest.index.intersection(price_latest.index)
            z_scores = []
            for code in codes:
                row_fin = latest.loc[code]
                mc = market_cap.reindex([code]).values[0]
                total_assets = row_fin.get("总资产", np.nan)
                current_liab = row_fin.get("流动负债", np.nan)
                current_assets = row_fin.get("流动资产", np.nan)
                retained_earnings = row_fin.get("留存收益", np.nan)
                ebit = row_fin.get("营业利润", np.nan)
                total_liab = row_fin.get("总负债", np.nan)
                revenue = row_fin.get("营业收入", np.nan)
                if any(pd.isna(v) or v == 0 for v in [total_assets, total_liab]):
                    z_scores.append({"code": code, "altman_z": np.nan})
                    continue
                working_capital = (current_assets if pd.notna(current_assets) else 0) - (current_liab if pd.notna(current_liab) else 0)
                x1 = safe_divide(np.array([working_capital]), np.array([total_assets]))[0]
                x2 = safe_divide(np.array([retained_earnings if pd.notna(retained_earnings) else np.nan]), np.array([total_assets]))[0]
                x3 = safe_divide(np.array([ebit if pd.notna(ebit) else np.nan]), np.array([total_assets]))[0]
                x4 = safe_divide(np.array([mc]), np.array([total_liab]))[0]
                x5 = safe_divide(np.array([revenue if pd.notna(revenue) else np.nan]), np.array([total_assets]))[0]
                x_vals = [x1, x2, x3, x4, x5]
                if any(pd.isna(v) for v in x_vals):
                    z_scores.append({"code": code, "altman_z": np.nan})
                else:
                    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
                    z_scores.append({"code": code, "altman_z": z})
            if z_scores:
                factors = pd.DataFrame(z_scores).set_index("code")
        except Exception as e:
            logger.debug(f"Altman Z-Score 计算异常: {e}")
        return factors

    def _compute_sloan_accrual(self, financial, td):
        factors = pd.DataFrame()
        if financial.empty:
            return factors
        try:
            fin = financial[financial["date"] <= td].copy()
            latest = fin.sort_values("date").groupby("code").last()
            if all(c in latest.columns for c in ["净利润", "经营现金流", "总资产"]):
                net_income = latest["净利润"]
                ocf = latest["经营现金流"]
                total_assets = latest["总资产"]
                accrual = safe_divide((net_income - ocf).values, total_assets.values)
                factors = pd.DataFrame({"sloan_accrual": accrual}, index=latest.index)
        except Exception as e:
            logger.debug(f"Sloan应计质量因子 计算异常: {e}")
        return factors

    def _compute_sue(self, financial, td):
        factors = pd.DataFrame()
        if financial.empty:
            return factors
        try:
            fin = financial[financial["date"] <= td].copy()
            fin = fin.sort_values(["code", "date"])
            results = []
            for code, g in fin.groupby("code"):
                if "基本每股收益" not in g.columns or len(g) < 5:
                    continue
                eps_series = g["基本每股收益"].dropna().values
                if len(eps_series) < 5:
                    continue
                actual_eps = eps_series[-1]
                expected_eps = np.mean(eps_series[-5:-1])
                eps_std = np.std(eps_series[-5:-1], ddof=1)
                if eps_std > 1e-10:
                    sue = (actual_eps - expected_eps) / eps_std
                else:
                    sue = np.nan
                results.append({"code": code, "sue": sue})
            if results:
                factors = pd.DataFrame(results).set_index("code")
        except Exception as e:
            logger.debug(f"SUE 计算异常: {e}")
        return factors

    def _compute_earnings_persistence(self, financial, td):
        factors = pd.DataFrame()
        if financial.empty:
            return factors
        try:
            fin = financial[financial["date"] <= td].copy()
            fin = fin.sort_values(["code", "date"])
            results = []
            for code, g in fin.groupby("code"):
                if "基本每股收益" not in g.columns:
                    continue
                eps_series = g["基本每股收益"].dropna().values
                if len(eps_series) < 8:
                    continue
                eps_hist = eps_series[-8:]
                y = eps_hist[1:]
                x = eps_hist[:-1]
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                beta_num = np.sum((x - x_mean) * (y - y_mean))
                beta_den = np.sum((x - x_mean) ** 2)
                if beta_den > 1e-10:
                    beta = beta_num / beta_den
                else:
                    beta = np.nan
                results.append({"code": code, "earnings_persist": beta})
            if results:
                factors = pd.DataFrame(results).set_index("code")
        except Exception as e:
            logger.debug(f"盈余持续性 计算异常: {e}")
        return factors

    def _compute_momentum_acceleration(self, data, td):
        factors = pd.DataFrame()
        try:
            results = []
            for code, g in data.groupby("code"):
                g = g.sort_values("date")
                close = g["close"].values
                if len(close) < 121:
                    continue
                mom_60d = self._momentum(close, 60)
                mom_120d = self._momentum(close, 120)
                if pd.notna(mom_60d) and pd.notna(mom_120d):
                    results.append({"code": code, "mom_accel": mom_60d - mom_120d})
            if results:
                factors = pd.DataFrame(results).set_index("code")
        except Exception as e:
            logger.debug(f"价格动量加速度 计算异常: {e}")
        return factors

    def _compute_high_low_ratio(self, data, td):
        factors = pd.DataFrame()
        try:
            results = []
            for code, g in data.groupby("code"):
                g = g.sort_values("date")
                if len(g) < 252:
                    recent = g
                else:
                    recent = g.tail(252)
                if len(recent) < 20:
                    continue
                high_52w = recent["high"].max()
                low_52w = recent["low"].min()
                current = recent["close"].iloc[-1]
                ratio = safe_divide(np.array([current - low_52w]), np.array([high_52w - low_52w]))[0]
                results.append({"code": code, "high_low_ratio": ratio})
            if results:
                factors = pd.DataFrame(results).set_index("code")
        except Exception as e:
            logger.debug(f"52周高低比 计算异常: {e}")
        return factors

    def _compute_vwap_deviation(self, data, td):
        factors = pd.DataFrame()
        try:
            results = []
            for code, g in data.groupby("code"):
                g = g.sort_values("date")
                if len(g) < 20:
                    continue
                recent = g.tail(20)
                if "volume" not in recent.columns:
                    continue
                close = recent["close"].astype(float).values
                volume = recent["volume"].astype(float).values
                total_volume = volume.sum()
                if total_volume > 0:
                    vwap = np.sum(close * volume) / total_volume
                    last_close = close[-1]
                    vwap_dev = safe_divide(np.array([last_close - vwap]), np.array([vwap]))[0]
                else:
                    vwap_dev = np.nan
                results.append({"code": code, "vwap_dev": vwap_dev})
            if results:
                factors = pd.DataFrame(results).set_index("code")
        except Exception as e:
            logger.debug(f"VWAP偏离度 计算异常: {e}")
        return factors

    def _compute_downside_risk(self, data, td):
        factors = pd.DataFrame()
        try:
            results = []
            for code, g in data.groupby("code"):
                g = g.sort_values("date")
                pct = g["pct_chg"].dropna().values / 100.0
                if len(pct) < 20:
                    continue
                neg_returns = pct[pct < 0]
                if len(neg_returns) < 5:
                    results.append({"code": code, "downside_dev": np.nan})
                    continue
                downside_std = float(np.std(neg_returns, ddof=1)) * np.sqrt(252)
                results.append({"code": code, "downside_dev": downside_std})
            if results:
                factors = pd.DataFrame(results).set_index("code")
        except Exception as e:
            logger.debug(f"下行风险因子 计算异常: {e}")
        return factors

    def _compute_co_skewness(self, data, td, benchmark_data=None):
        factors = pd.DataFrame()
        if benchmark_data is None or benchmark_data.empty:
            return factors
        try:
            bench = benchmark_data.sort_values("date").copy()
            bench["ret"] = bench["close"].pct_change()
            bench_ret = bench[["date", "ret"]].dropna()
            bench_ret = bench_ret.rename(columns={"date": "date", "ret": "bench_ret"})
            data_sorted = data.sort_values(["code", "date"]).copy()
            data_sorted["ret"] = data_sorted.groupby("code")["close"].pct_change()
            merged = data_sorted.merge(bench_ret, on="date", how="inner")
            merged = merged.dropna(subset=["ret", "bench_ret"])
            merged["bench_ret_sq"] = merged["bench_ret"] ** 2
            results = []
            for code, g in merged.groupby("code"):
                if len(g) < 60:
                    continue
                ri = g["ret"].values
                rm = g["bench_ret"].values
                excess_stock = ri
                excess_market = rm
                market_sq = excess_market ** 2
                product = excess_stock * market_sq
                e_product = np.mean(product)
                std_ri = np.std(ri, ddof=1)
                var_rm = np.var(rm, ddof=1)
                if std_ri > 1e-10 and var_rm > 1e-10:
                    coskew = e_product / (std_ri * var_rm)
                else:
                    coskew = np.nan
                results.append({"code": code, "coskew": coskew})
            if results:
                factors = pd.DataFrame(results).set_index("code")
        except Exception as e:
            logger.debug(f"协偏度 计算异常: {e}")
        return factors


def get_extended_factor_config():
    return {
        "f_score": {"direction": 1, "category": "quality", "desc": "Piotroski F-Score"},
        "altman_z": {"direction": 1, "category": "quality", "desc": "Altman Z-Score"},
        "sloan_accrual": {"direction": -1, "category": "earnings_quality", "desc": "Sloan应计质量"},
        "sue": {"direction": 1, "category": "earnings_quality", "desc": "标准化未预期盈余"},
        "earnings_persist": {"direction": 1, "category": "earnings_quality", "desc": "盈余持续性"},
        "mom_accel": {"direction": 1, "category": "momentum", "desc": "价格动量加速度"},
        "high_low_ratio": {"direction": 1, "category": "momentum", "desc": "52周高低比"},
        "vwap_dev": {"direction": 1, "category": "technical", "desc": "VWAP偏离度"},
        "downside_dev": {"direction": -1, "category": "risk", "desc": "下行风险因子"},
        "coskew": {"direction": 1, "category": "risk", "desc": "协偏度"},
    }
