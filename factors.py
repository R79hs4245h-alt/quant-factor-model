"""
因子计算层: 价值/成长/动量/反转/质量/波动率/流动性
所有因子按截面计算,返回 DataFrame(index=code, columns=[factor_names])
"""
import pandas as pd
import numpy as np
from typing import Dict

from config import FACTOR_CONFIG
from utils import get_logger, safe_divide

logger = get_logger("factors")


class FactorCalculator:
    """多因子计算器"""

    def __init__(self):
        self.factor_config = FACTOR_CONFIG

    def compute_all_factors(self, panel: pd.DataFrame,
                            financial: pd.DataFrame,
                            trade_date: str) -> pd.DataFrame:
        """
        在某一交易日截面计算所有因子
        :param panel: 全部股票历史行情面板 (含 date, code, close, volume, amount, turnover, pct_chg)
        :param financial: 财务数据
        :param trade_date: 截面日期 YYYYMMDD
        :return: DataFrame(index=code, columns=factor_names)
        """
        td = pd.to_datetime(trade_date, format="%Y%m%d")
        # 截取截面之前的数据
        cross_data = panel[panel["date"] <= td].copy()
        if cross_data.empty:
            return pd.DataFrame()

        factors = pd.DataFrame(index=cross_data["code"].unique())

        # 价值因子
        val_factors = self._compute_value_factors(cross_data, financial, td)
        factors = factors.join(val_factors, how="left")

        # 成长因子
        growth_factors = self._compute_growth_factors(financial, td)
        factors = factors.join(growth_factors, how="left")

        # 动量因子
        mom_factors = self._compute_momentum_factors(cross_data, td)
        factors = factors.join(mom_factors, how="left")

        # 反转因子
        rev_factors = self._compute_reversal_factors(cross_data, td)
        factors = factors.join(rev_factors, how="left")

        # 质量因子
        quality_factors = self._compute_quality_factors(financial, td)
        factors = factors.join(quality_factors, how="left")

        # 波动率因子
        vol_factors = self._compute_volatility_factors(cross_data, td)
        factors = factors.join(vol_factors, how="left")

        # 流动性因子
        liq_factors = self._compute_liquidity_factors(cross_data, td)
        factors = factors.join(liq_factors, how="left")

        logger.info(f"截面 {trade_date}: 计算完成 {factors.shape[1]} 个因子, "
                    f"覆盖 {len(factors)} 只股票")
        return factors

    # ============ 价值因子 ============
    def _compute_value_factors(self, data: pd.DataFrame,
                               financial: pd.DataFrame,
                               td) -> pd.DataFrame:
        """EP, BP, SP, CFP"""
        # 取截面日最新价
        latest = data.sort_values("date").groupby("code").last()
        close = latest["close"]
        market_cap = latest.get("market_cap", close * 1e8)  # 近似

        factors = pd.DataFrame(index=latest.index)

        try:
            if not financial.empty:
                fin_latest = financial[financial["date"] <= td]
                fin_latest = fin_latest.sort_values("date").groupby("code").last()

                # 净利润 / 市值 = EP
                if "净利润" in fin_latest.columns:
                    factors["ep"] = safe_divide(
                        fin_latest["净利润"].reindex(latest.index),
                        market_cap
                    )
                # 净资产 / 市值 = BP
                if "净资产" in fin_latest.columns:
                    factors["bp"] = safe_divide(
                        fin_latest["净资产"].reindex(latest.index),
                        market_cap
                    )
                # 营业收入 / 市值 = SP
                if "营业收入" in fin_latest.columns:
                    factors["sp"] = safe_divide(
                        fin_latest["营业收入"].reindex(latest.index),
                        market_cap
                    )
                # 经营现金流 / 市值 = CFP
                if "经营现金流" in fin_latest.columns:
                    factors["cfp"] = safe_divide(
                        fin_latest["经营现金流"].reindex(latest.index),
                        market_cap
                    )
        except Exception as e:
            logger.debug(f"价值因子计算异常: {e}")

        return factors

    # ============ 成长因子 ============
    def _compute_growth_factors(self, financial: pd.DataFrame,
                                td) -> pd.DataFrame:
        """营收增长、净利润增长、EPS增长"""
        factors = pd.DataFrame()
        if financial.empty:
            return factors

        try:
            fin = financial[financial["date"] <= td].copy()
            fin = fin.sort_values(["code", "date"])

            for code, g in fin.groupby("code"):
                if len(g) < 2:
                    continue
                latest = g.iloc[-1]
                prev = g.iloc[-2]

                if code not in factors.index:
                    factors.loc[code, "revenue_growth"] = safe_divide(
                    latest.get("营业收入", np.nan) - prev.get("营业收入", np.nan),
                        abs(prev.get("营业收入", np.nan))
                    )[0] if pd.notna(latest.get("营业收入")) else np.nan

                    factors.loc[code, "profit_growth"] = safe_divide(
                    latest.get("净利润", np.nan) - prev.get("净利润", np.nan),
                        abs(prev.get("净利润", np.nan))
                    )[0] if pd.notna(latest.get("净利润")) else np.nan

                    if "基本每股收益" in g.columns:
                        factors.loc[code, "eps_growth"] = safe_divide(
                        latest.get("基本每股收益", np.nan) - prev.get("基本每股收益", np.nan),
                            abs(prev.get("基本每股收益", np.nan))
                        )[0] if pd.notna(latest.get("基本每股收益")) else np.nan
        except Exception as e:
            logger.debug(f"成长因子计算异常: {e}")

        return factors

    # ============ 动量因子 ============
    def _compute_momentum_factors(self, data: pd.DataFrame,
                                  td) -> pd.DataFrame:
        """20/60/120 日动量"""
        factors = pd.DataFrame()
        try:
            for code, g in data.groupby("code"):
                g = g.sort_values("date")
                if len(g) < 5:
                    continue
                close = g["close"].values
                factors.loc[code, "mom_20d"] = self._momentum(close, 20)
                factors.loc[code, "mom_60d"] = self._momentum(close, 60)
                factors.loc[code, "mom_120d"] = self._momentum(close, 120)
        except Exception as e:
            logger.debug(f"动量因子异常: {e}")
        return factors

    @staticmethod
    def _momentum(prices: np.ndarray, window: int) -> float:
        if len(prices) < window + 1:
            return np.nan
        return prices[-1] / prices[-window - 1] - 1

    # ============ 反转因子 ============
    def _compute_reversal_factors(self, data: pd.DataFrame,
                                  td) -> pd.DataFrame:
        """5/20 日反转(近期收益取负)"""
        factors = pd.DataFrame()
        try:
            for code, g in data.groupby("code"):
                g = g.sort_values("date")
                if len(g) < 5:
                    continue
                pct = g["pct_chg"].values
                factors.loc[code, "reversal_5d"] = -np.nansum(pct[-5:]) if len(pct) >= 5 else np.nan
                factors.loc[code, "reversal_20d"] = -np.nansum(pct[-20:]) if len(pct) >= 20 else np.nan
        except Exception as e:
            logger.debug(f"反转因子异常: {e}")
        return factors

    # ============ 质量因子 ============
    def _compute_quality_factors(self, financial: pd.DataFrame,
                                 td) -> pd.DataFrame:
        """ROE, ROA, 毛利率, 资产负债率"""
        factors = pd.DataFrame()
        if financial.empty:
            return factors
        try:
            fin = financial[financial["date"] <= td].copy()
            latest = fin.sort_values("date").groupby("code").last()

            col_map = {
                "净资产收益率(%)": "roe",
                "总资产报酬率(%)": "roa",
                "销售毛利率(%)": "gross_margin",
                "资产负债率(%)": "debt_ratio",
            }
            for src, dst in col_map.items():
                if src in latest.columns:
                    factors[dst] = latest[src]
        except Exception as e:
            logger.debug(f"质量因子异常: {e}")
        return factors

    # ============ 波动率因子 ============
    def _compute_volatility_factors(self, data: pd.DataFrame,
                                    td) -> pd.DataFrame:
        """20/60 日收益波动率"""
        factors = pd.DataFrame()
        try:
            for code, g in data.groupby("code"):
                g = g.sort_values("date")
                pct = g["pct_chg"].dropna().values
                if len(pct) >= 20:
                    factors.loc[code, "vol_20d"] = np.std(pct[-20:], ddof=1)
                if len(pct) >= 60:
                    factors.loc[code, "vol_60d"] = np.std(pct[-60:], ddof=1)
        except Exception as e:
            logger.debug(f"波动率因子异常: {e}")
        return factors

    # ============ 流动性因子 ============
    def _compute_liquidity_factors(self, data: pd.DataFrame,
                                   td) -> pd.DataFrame:
        """20日平均换手率、20日平均成交额"""
        factors = pd.DataFrame()
        try:
            for code, g in data.groupby("code"):
                g = g.sort_values("date")
                if len(g) < 20:
                    continue
                recent = g.tail(20)
                factors.loc[code, "turnover_20d"] = recent["turnover"].mean() if "turnover" in recent else np.nan
                factors.loc[code, "amount_20d"] = recent["amount"].mean() if "amount" in recent else np.nan
        except Exception as e:
            logger.debug(f"流动性因子异常: {e}")
        return factors


def get_factor_directions() -> Dict[str, int]:
    """获取因子方向(1=正向, -1=反向)"""
    return {name: cfg["direction"] for name, cfg in FACTOR_CONFIG.items()}
