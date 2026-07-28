"""
场内基金(ETF+LOF)推荐引擎 v7.2
================================

专门针对场内基金(ETF+LOF)的推荐系统:
  - 仅推荐可在交易所实时买卖的ETF和LOF基金
  - 不再推荐任何场外基金
  - 基于多维度因子打分: 动量、流动性、跟踪误差、折溢价率、规模、波动率
  - 主题轮动分析: 识别当前强势赛道
  - 风险控制: 流动性筛选、规模筛选、集中度控制

核心逻辑:
  1. 获取全市场ETF+LOF实时行情
  2. 按流动性(成交额)和规模筛选
  3. 多维度因子打分(动量/流动性/波动率/规模/折溢价)
  4. 主题分类与轮动分析
  5. 输出推荐列表 + 主题配置建议 + 风险提示

依赖:
  - config.py 中的场内基金配置
  - data_loader.py 获取行情数据
"""

import warnings
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd

from config import (
    ETF_MIN_AMOUNT, LOF_MIN_AMOUNT,
    CORE_BROAD_ETF, THEME_ETF, CORE_LOF,
    ETF_THEME_MAP,
    ETF_RECOMMEND_TOP, LOF_RECOMMEND_TOP,
    TOTAL_EXCHANGE_TRADED,
    OUTPUT_DIR,
    ADVANCED_DIM_WEIGHTS,
)
from utils import get_logger

try:
    from advanced_factors import FourDimensionEngine
    _ADVANCED_AVAILABLE = True
except ImportError:
    _ADVANCED_AVAILABLE = False

warnings.filterwarnings("ignore")
logger = get_logger("etf_recommender")


class ExchangeTradedFundRecommender:
    """
    场内基金推荐引擎 v7.3

    仅推荐场内基金(ETF+LOF),核心流程:
      1. 数据获取: ETF实时行情 + LOF实时行情
      2. 流动性筛选: 成交额/规模过滤
      3. 多因子打分: 动量/流动性/波动率/规模/折溢价 (基础层)
      4. 四维高阶分析: 估值/拐点/资金/主升浪 (v7.3增强层)
      5. 主题分类: 科技/消费/医药/金融/宽基等
      6. 主题轮动: 近期强势主题识别
      7. 组合建议: 跨主题配置 + 风险提示 + 共振信号

    v7.3新增:
      - 集成四维选股引擎(advanced_factors.FourDimensionEngine)
      - 估值维度: PEG/历史分位/市值空间
      - 拐点维度: 金针探底/底背离/倍量阳
      - 资金维度: DDX代理/大单/压盘吸筹
      - 主升浪维度: 均线粘合/量能递增/空中加油/筹码峰
      - 四维共振检测: 强买入信号需多维共振确认
    """

    def __init__(self):
        self.etf_data: pd.DataFrame = pd.DataFrame()
        self.lof_data: pd.DataFrame = pd.DataFrame()
        self.all_funds: pd.DataFrame = pd.DataFrame()
        self.scores: pd.DataFrame = pd.DataFrame()
        self.recommendations: List[Dict] = []
        self.theme_analysis: Dict[str, Any] = {}
        self.advanced_results: Dict[str, Dict] = {}  # code -> 四维分析结果
        self.kline_cache: Dict[str, pd.DataFrame] = {}  # code -> OHLC DataFrame
        # v7.4: 使用ETF模式初始化四维引擎
        if _ADVANCED_AVAILABLE:
            try:
                self.four_dim_engine = FourDimensionEngine(asset_type='etf')
            except TypeError:
                # 兼容v7.3旧版
                self.four_dim_engine = FourDimensionEngine()
        else:
            self.four_dim_engine = None

    # ==================== 数据获取 ====================

    def fetch_etf_realtime(self) -> pd.DataFrame:
        """获取ETF实时行情"""
        logger.info("获取ETF实时行情...")
        try:
            import akshare as ak
            df = ak.fund_etf_spot_em()
            if df is not None and len(df) > 0:
                logger.info(f"获取到 {len(df)} 只ETF")
                self.etf_data = df
                return df
        except Exception as e:
            logger.warning(f"akshare获取ETF失败: {e}")
        return pd.DataFrame()

    def fetch_lof_realtime(self) -> pd.DataFrame:
        """获取LOF实时行情"""
        logger.info("获取LOF实时行情...")
        try:
            import akshare as ak
            df = ak.fund_lof_spot_em()
            if df is not None and len(df) > 0:
                logger.info(f"获取到 {len(df)} 只LOF")
                self.lof_data = df
                return df
        except Exception as e:
            logger.warning(f"akshare获取LOF失败: {e}")
        return pd.DataFrame()

    def load_from_browser_data(self, etf_json: str, lof_json: str) -> None:
        """
        从浏览器抓取的数据加载(JSON字符串)

        当akshare网络不可用时,使用浏览器抓取的数据
        """
        logger.info("从浏览器数据加载场内基金行情...")

        if etf_json:
            etf_list = json.loads(etf_json) if isinstance(etf_json, str) else etf_json
            if etf_list:
                self.etf_data = pd.DataFrame(etf_list)
                logger.info(f"加载ETF: {len(self.etf_data)} 只")

        if lof_json:
            lof_list = json.loads(lof_json) if isinstance(lof_json, str) else lof_json
            if lof_list:
                self.lof_data = pd.DataFrame(lof_list)
                logger.info(f"加载LOF: {len(self.lof_data)} 只")

    # ==================== 数据预处理 ====================

    def _normalize_etf_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化ETF数据列名"""
        if df.empty:
            return df

        rename_map = {
            '代码': 'code', '名称': 'name', '最新价': 'price',
            '涨跌幅': 'change_pct', '涨跌额': 'change_amt',
            '成交量': 'volume', '成交额': 'amount',
            '开盘价': 'open', '最高价': 'high', '最低价': 'low',
            '昨收': 'prev_close', '换手率': 'turnover',
        }
        available = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=available)

        # 先解析成交额和成交量字符串(保留原始中文格式: "8.61亿"等)
        # 必须在 pd.to_numeric 之前处理,否则中文字符串会变成NaN
        if 'amount' in df.columns:
            df['amount_yuan'] = df['amount'].apply(self._parse_amount_str)
        else:
            df['amount_yuan'] = 0.0

        if 'volume' in df.columns:
            df['volume_num'] = df['volume'].apply(self._parse_amount_str)
        else:
            df['volume_num'] = 0.0

        # 数值化其他列(amount和volume保持原始字符串用于显示)
        for col in ['price', 'change_pct', 'change_amt', 'turnover',
                     'open', 'high', 'low', 'prev_close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['fund_type'] = 'ETF'
        return df

    def _normalize_lof_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化LOF数据"""
        df = self._normalize_etf_data(df)
        if not df.empty:
            df['fund_type'] = 'LOF'
        return df

    @staticmethod
    def _parse_amount_str(val) -> float:
        """解析金额字符串: '8.61亿' -> 861000000, '3.64万' -> 36400"""
        if pd.isna(val) or val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        if not s:
            return 0.0
        try:
            if '亿' in s:
                return float(s.replace('亿', '')) * 1e8
            elif '万' in s:
                return float(s.replace('万', '')) * 1e4
            else:
                return float(s)
        except (ValueError, TypeError):
            return 0.0

    def merge_all_funds(self) -> pd.DataFrame:
        """合并ETF和LOF数据"""
        etf_df = self._normalize_etf_data(self.etf_data.copy())
        lof_df = self._normalize_lof_data(self.lof_data.copy())

        frames = [df for df in [etf_df, lof_df] if not df.empty]
        if not frames:
            self.all_funds = pd.DataFrame()
            return self.all_funds

        self.all_funds = pd.concat(frames, ignore_index=True)
        logger.info(f"合并后场内基金: {len(self.all_funds)} 只")
        return self.all_funds

    # ==================== 流动性筛选 ====================

    def filter_by_liquidity(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        按流动性筛选:
          - ETF: 日成交额 >= ETF_MIN_AMOUNT (500万)
          - LOF: 日成交额 >= LOF_MIN_AMOUNT (100万)
        """
        if df is None:
            df = self.all_funds

        if df.empty:
            return df

        mask = (
            ((df['fund_type'] == 'ETF') & (df['amount_yuan'] >= ETF_MIN_AMOUNT)) |
            ((df['fund_type'] == 'LOF') & (df['amount_yuan'] >= LOF_MIN_AMOUNT))
        )
        filtered = df[mask].copy()
        logger.info(f"流动性筛选后: {len(filtered)} 只 (ETF≥{ETF_MIN_AMOUNT/1e4:.0f}万, LOF≥{LOF_MIN_AMOUNT/1e4:.0f}万)")
        return filtered

    # ==================== 多因子打分 ====================

    def calculate_scores(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        多维度因子打分 v7.3

        基础层因子(权重由 ADVANCED_DIM_WEIGHTS['base_weight'] 控制):
          1. momentum: 今日涨跌幅
          2. liquidity: 成交额排名
          3. activity: 换手率
          4. volatility_adj: 日内振幅(越低越好)
          5. scale: 核心池加分

        高阶四维层(v7.3新增,权重由 ADVANCED_DIM_WEIGHTS 控制):
          6. valuation: 估值相对合理(PEG/历史分位/市值空间)
          7. bottom_reversal: 触底反弹拐点确认(底背离/金针探底/倍量阳)
          8. capital_flow: 主力资金真流入(DDX代理/大单/龙虎榜)
          9. main_rally: 主升浪启动检测(均线粘合/量能递增/筹码峰)

        当四维引擎可用时,综合得分 = 基础层×base_weight + 四维×advanced_weight
        当四维引擎不可用时,回退到纯基础层打分(v7.2兼容)
        """
        if df is None:
            df = self.all_funds

        if df.empty:
            self.scores = df
            return df

        scored = df.copy()

        # === 基础层打分 ===

        # 1. 动量得分 (今日涨跌幅)
        if 'change_pct' in scored.columns:
            scored['mom_score'] = scored['change_pct'].rank(pct=True)
        else:
            scored['mom_score'] = 0.5

        # 2. 流动性得分 (成交额)
        scored['liq_score'] = scored['amount_yuan'].rank(pct=True)

        # 3. 活跃度得分 (换手率)
        if 'turnover' in scored.columns and scored['turnover'].notna().any():
            scored['act_score'] = scored['turnover'].rank(pct=True)
        else:
            scored['act_score'] = 0.5

        # 4. 波动率调整 (日内振幅,越低越好)
        if all(c in scored.columns for c in ['high', 'low', 'prev_close']):
            scored['intraday_range'] = (
                (scored['high'] - scored['low']) / scored['prev_close'].replace(0, np.nan)
            ).fillna(0)
            scored['vol_score'] = 1 - scored['intraday_range'].rank(pct=True)
        else:
            scored['vol_score'] = 0.5

        # 5. 规模/核心池加分
        all_core_codes = set(CORE_BROAD_ETF.keys()) | set(THEME_ETF.keys()) | set(CORE_LOF.keys())
        scored['is_core'] = scored['code'].isin(all_core_codes).astype(int)
        scored['scale_score'] = scored['is_core'] * 0.8 + scored['amount_yuan'].rank(pct=True) * 0.2

        # 基础层综合得分
        scored['base_score'] = (
            scored['mom_score'] * 0.30 +
            scored['liq_score'] * 0.25 +
            scored['act_score'] * 0.15 +
            scored['vol_score'] * 0.15 +
            scored['scale_score'] * 0.15
        )

        # === 高阶四维层打分 ===

        if self.four_dim_engine is not None:
            # v7.4: 预取Top候选的K线数据 (仅对前100只做四维分析以平衡性能)
            top_for_analysis = scored.head(100)
            self._fetch_klines_batch(top_for_analysis)

            adv_scores = []
            for _, row in scored.iterrows():
                adv_result = self._calc_advanced_score_for_row(row)
                adv_scores.append(adv_result)
                self.advanced_results[row.get('code', '')] = adv_result

            scored['advanced_score'] = [a['total_score'] / 100.0 for a in adv_scores]
            scored['resonance'] = [a['resonance'] for a in adv_scores]
            scored['resonance_level'] = [a['resonance_level'] for a in adv_scores]
            scored['advanced_signal'] = [a['signal'] for a in adv_scores]

            # v7.4: 右侧确认信号
            scored['right_side_action'] = [
                a.get('right_side', {}).get('final_action', '') for a in adv_scores
            ]

            # 混合: 基础层×base_weight + 四维×advanced_weight
            base_w = ADVANCED_DIM_WEIGHTS.get('base_weight', 0.4)
            adv_w = ADVANCED_DIM_WEIGHTS.get('advanced_weight', 0.6)
            scored['total_score'] = scored['base_score'] * base_w + scored['advanced_score'] * adv_w
        else:
            # 四维引擎不可用,回退到纯基础层
            scored['total_score'] = scored['base_score']

        # 百分制
        scored['total_score_100'] = (scored['total_score'] * 100).round(2)

        self.scores = scored.sort_values('total_score', ascending=False)
        logger.info(f"打分完成: {len(self.scores)} 只基金 (四维引擎: {'启用' if self.four_dim_engine else '未启用'})")
        return self.scores

    def _fetch_klines_batch(self, df: pd.DataFrame):
        """v7.4: 批量获取K线数据"""
        codes = df['code'].tolist() if 'code' in df.columns else []
        fetched = 0
        for code in codes:
            if code in self.kline_cache:
                continue
            ohlc = self._fetch_single_kline(code)
            if ohlc is not None and len(ohlc) > 0:
                self.kline_cache[code] = ohlc
                fetched += 1
        if fetched > 0:
            logger.info(f"K线预取: {fetched}/{len(codes)} 只获取成功")

    def _fetch_single_kline(self, code: str) -> pd.DataFrame:
        """v7.4: 获取单只ETF/LOF的K线数据 (腾讯API)"""
        import requests

        # 判断沪深前缀
        if code.startswith('5') or code.startswith('6') or code.startswith('9'):
            prefix = 'sh'
        else:
            prefix = 'sz'

        url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
        params = {
            'param': f'{prefix}{code},day,2025-01-01,2026-12-31,640,qfq',
        }

        proxies = {
            'http': os.environ.get('HTTPS_PROXY', 'http://127.0.0.1:18080'),
            'https': os.environ.get('HTTPS_PROXY', 'http://127.0.0.1:18080'),
        }

        try:
            resp = requests.get(url, params=params, timeout=8, proxies=proxies)
            data = resp.json()
            key = f'{prefix}{code}'
            if data and data.get('data') and data['data'].get(key):
                stock_data = data['data'][key]
                klines = stock_data.get('qfqday') or stock_data.get('day') or []
                if klines:
                    df = pd.DataFrame(klines, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                    df['date'] = pd.to_datetime(df['date'])
                    for c in ['open', 'close', 'high', 'low', 'volume']:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                    return df
        except Exception:
            pass
        return pd.DataFrame()

    def _calc_advanced_score_for_row(self, row: pd.Series) -> Dict:
        """
        v7.4: 对单只ETF/LOF执行四维高阶分析

        升级点:
          1. 优先使用真实K线数据 (从kline_cache获取)
          2. ETF估值适配: 折溢价率 + 价格历史分位 + 规模变化
          3. 资金维度: 主力净流入 + 换手率代理
          4. 右侧确认三步框架
        """
        if self.four_dim_engine is None:
            return {
                'total_score': 50.0,
                'signal': '观望',
                'resonance': 0,
                'resonance_level': '无信号',
                'resonance_dims': [],
                'action': '四维引擎未启用',
                'stop_loss_rule': '',
                'details': '引擎未启用',
                'right_side': {},
            }

        code = row.get('code', '')

        # v7.4: 优先使用真实K线
        ohlc = self.kline_cache.get(code)
        if ohlc is None or len(ohlc) == 0:
            # 降级: 构造单日代理K线
            if all(c in row.index for c in ['open', 'high', 'low']):
                if pd.notna(row.get('open')) and pd.notna(row.get('high')):
                    ohlc = pd.DataFrame([{
                        'open': float(row.get('open', 0) or 0),
                        'high': float(row.get('high', 0) or 0),
                        'low': float(row.get('low', 0) or 0),
                        'close': float(row.get('price', 0) or 0),
                        'volume': float(row.get('volume_num', 0) or 0),
                    }])

        # 当前价格
        current_price = float(row.get('price', 0)) if pd.notna(row.get('price', 0)) else None

        # 折溢价率
        discount_rate = None
        if 'discount_rate' in row.index and pd.notna(row.get('discount_rate')):
            discount_rate = float(row['discount_rate'])
        elif current_price and pd.notna(row.get('prev_close', 0)) and float(row.get('prev_close', 0)) > 0:
            discount_rate = (current_price - float(row['prev_close'])) / float(row['prev_close'])

        # 主力净流入 (万元)
        main_inflow = None
        if 'main_inflow' in row.index and pd.notna(row.get('main_inflow')):
            main_inflow = float(row['main_inflow'])
        elif pd.notna(row.get('amount_yuan', 0)):
            # 代理: 成交额的10%作为主力净流入估计
            main_inflow = float(row['amount_yuan']) * 0.1 / 1e4

        # 换手率
        turnover_rate = float(row.get('turnover', 0)) if pd.notna(row.get('turnover', 0)) else None

        # 执行四维分析 (v7.4 ETF模式)
        try:
            result = self.four_dim_engine.analyze(
                ohlc=ohlc,
                current_price=current_price,
                discount_rate=discount_rate,
                main_inflow=main_inflow,
                turnover_rate=turnover_rate,
            )
        except TypeError:
            # 兼容v7.3: 不支持新参数
            result = self.four_dim_engine.analyze(
                ohlc=ohlc,
                large_order_net=main_inflow,
            )

        return result

    # ==================== 主题分类 ====================

    def classify_by_theme(self, df: pd.DataFrame = None) -> Dict[str, List[Dict]]:
        """按主题分类基金"""
        if df is None:
            df = self.scores

        if df.empty:
            return {}

        themes = defaultdict(list)
        classified = set()

        # 使用配置的主题映射
        for theme_name, codes in ETF_THEME_MAP.items():
            for code in codes:
                rows = df[df['code'] == code]
                if len(rows) > 0:
                    row = rows.iloc[0].to_dict()
                    row['theme'] = theme_name
                    themes[theme_name].append(row)
                    classified.add(code)

        # 未分类的按名称关键词归入
        keyword_map = {
            "科技AI": ["科技", "AI", "人工智能", "信息", "软件", "计算机", "通信"],
            "半导体芯片": ["半导体", "芯片", "集成电路"],
            "消费医药": ["消费", "医药", "医疗", "生物", "食品", "酒"],
            "新能源": ["新能源", "光伏", "锂电", "碳", "电池", "储能"],
            "金融地产": ["金融", "银行", "证券", "券商", "地产"],
            "宽基指数": ["沪深300", "中证500", "中证1000", "创业板", "科创", "上证50", "深100"],
            "港股相关": ["港股", "恒生", "中概", "H股"],
            "红利策略": ["红利", "国企"],
        }

        for _, row in df.iterrows():
            code = row.get('code', '')
            if code in classified:
                continue
            name = str(row.get('name', ''))
            for theme_name, keywords in keyword_map.items():
                if any(k in name for k in keywords):
                    d = row.to_dict()
                    d['theme'] = theme_name
                    themes[theme_name].append(d)
                    classified.add(code)
                    break

        # 剩余未分类的归入"其他"
        for _, row in df.iterrows():
            code = row.get('code', '')
            if code not in classified:
                d = row.to_dict()
                d['theme'] = '其他'
                themes['其他'].append(d)

        # 每个主题内按得分排序
        for theme in themes:
            themes[theme].sort(key=lambda x: x.get('total_score_100', 0), reverse=True)

        return dict(themes)

    # ==================== 主题轮动分析 ====================

    def analyze_theme_rotation(self, themes: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        分析主题轮动:
          - 计算各主题平均涨跌幅
          - 识别强势/弱势主题
          - 输出配置建议
        """
        theme_stats = []

        for theme_name, funds in themes.items():
            if not funds or theme_name == '其他':
                continue

            changes = [f.get('change_pct', 0) or 0 for f in funds]
            amounts = [f.get('amount_yuan', 0) or 0 for f in funds]

            avg_change = np.mean(changes) if changes else 0
            total_amount = sum(amounts)
            max_change = max(changes) if changes else 0
            min_change = min(changes) if changes else 0

            theme_stats.append({
                'theme': theme_name,
                'fund_count': len(funds),
                'avg_change_pct': round(avg_change, 2),
                'max_change_pct': round(max_change, 2),
                'min_change_pct': round(min_change, 2),
                'total_amount_yuan': total_amount,
                'total_amount_str': self._format_amount(total_amount),
            })

        theme_stats.sort(key=lambda x: x['avg_change_pct'], reverse=True)

        # 强势/弱势主题
        strong = [t for t in theme_stats if t['avg_change_pct'] > 1.0]
        neutral = [t for t in theme_stats if -1.0 <= t['avg_change_pct'] <= 1.0]
        weak = [t for t in theme_stats if t['avg_change_pct'] < -1.0]

        self.theme_analysis = {
            'theme_stats': theme_stats,
            'strong_themes': strong,
            'neutral_themes': neutral,
            'weak_themes': weak,
            'top_theme': theme_stats[0]['theme'] if theme_stats else 'N/A',
        }

        return self.theme_analysis

    @staticmethod
    def _format_amount(amount: float) -> str:
        """格式化金额显示"""
        if amount >= 1e8:
            return f"{amount/1e8:.2f}亿"
        elif amount >= 1e4:
            return f"{amount/1e4:.0f}万"
        else:
            return f"{amount:.0f}"

    # ==================== 生成推荐 ====================

    def generate_recommendations(self) -> List[Dict]:
        """
        生成最终推荐列表

        策略:
          1. 从打分排名中取Top基金
          2. 确保主题分散(每主题最多3只)
          3. ETF和LOF比例约 60:40
          4. 加入核心宽基ETF作为底仓
        """
        if self.scores.empty:
            return []

        # 取流动性筛选后的高分基金
        filtered = self.filter_by_liquidity(self.scores)
        if filtered.empty:
            filtered = self.scores

        themes = self.classify_by_theme(filtered)
        rotation = self.analyze_theme_rotation(themes)

        recommendations = []
        theme_count = defaultdict(int)
        max_per_theme = 3
        etf_count = 0
        lof_count = 0

        # 先从强势主题中选
        for theme_info in rotation.get('strong_themes', []):
            theme_name = theme_info['theme']
            for fund in themes.get(theme_name, []):
                if theme_count[theme_name] >= max_per_theme:
                    break
                if etf_count >= ETF_RECOMMEND_TOP and fund.get('fund_type') == 'ETF':
                    continue
                if lof_count >= LOF_RECOMMEND_TOP and fund.get('fund_type') == 'LOF':
                    continue

                rec = self._format_recommendation(fund, theme_name, rotation)
                recommendations.append(rec)
                theme_count[theme_name] += 1
                if fund.get('fund_type') == 'ETF':
                    etf_count += 1
                else:
                    lof_count += 1

                if len(recommendations) >= TOTAL_EXCHANGE_TRADED:
                    break
            if len(recommendations) >= TOTAL_EXCHANGE_TRADED:
                break

        # 如果数量不够，从中性主题补充
        if len(recommendations) < TOTAL_EXCHANGE_TRADED:
            for theme_info in rotation.get('neutral_themes', []):
                theme_name = theme_info['theme']
                for fund in themes.get(theme_name, []):
                    if theme_count[theme_name] >= max_per_theme:
                        break
                    if len(recommendations) >= TOTAL_EXCHANGE_TRADED:
                        break

                    code = fund.get('code', '')
                    if any(r['code'] == code for r in recommendations):
                        continue

                    rec = self._format_recommendation(fund, theme_name, rotation)
                    recommendations.append(rec)
                    theme_count[theme_name] += 1

        # 确保包含核心宽基ETF
        core_added = set()
        for rec in recommendations:
            core_added.add(rec['code'])

        for code, name in CORE_BROAD_ETF.items():
            if code not in core_added and len(recommendations) < TOTAL_EXCHANGE_TRADED + 5:
                # 从打分数据中找
                rows = self.scores[self.scores['code'] == code]
                if len(rows) > 0:
                    fund = rows.iloc[0].to_dict()
                    rec = self._format_recommendation(fund, '宽基指数', rotation)
                    rec['note'] = '核心宽基底仓'
                    recommendations.append(rec)
                    core_added.add(code)

        self.recommendations = recommendations
        logger.info(f"生成推荐: {len(recommendations)} 只场内基金 "
                    f"(ETF={etf_count}, LOF={lof_count})")
        return recommendations

    def _format_recommendation(self, fund: Dict, theme: str, rotation: Dict) -> Dict:
        """格式化推荐项 (v7.4: 增加右侧确认三步框架)"""
        change_pct = fund.get('change_pct', 0) or 0
        code = fund.get('code', '')

        # 基础信号强度
        score = fund.get('total_score_100', 0)
        base_signal = '观望'
        if score >= 80 and change_pct > 2:
            base_signal = '强买入'
        elif score >= 70:
            base_signal = '买入'
        elif score >= 60:
            base_signal = '观察'

        # 四维共振信号 (v7.3)
        adv_result = self.advanced_results.get(code, {})
        resonance = adv_result.get('resonance', 0)
        resonance_level = adv_result.get('resonance_level', '')
        adv_signal = adv_result.get('signal', '')
        adv_action = adv_result.get('action', '')
        adv_details = adv_result.get('details', '')
        stop_loss = adv_result.get('stop_loss_rule', '')

        # v7.4: 右侧确认三步框架
        right_side = adv_result.get('right_side', {})
        rs_action = right_side.get('final_action', '')
        rs_detail = right_side.get('action_detail', '')
        rs_entry = right_side.get('entry_timing', '')
        rs_exit = right_side.get('exit_rule', '')
        rs_step1 = right_side.get('step1_screen', False)
        rs_step2 = right_side.get('step2_confirm', False)
        rs_step3 = right_side.get('step3_order', False)

        # 信号融合: 右侧确认 > 四维共振 > 基础信号
        if rs_action == '果断介入':
            signal = '强买入'
        elif rs_action == '轻仓试探':
            signal = '买入'
        elif adv_signal in ('强买入',):
            signal = '强买入'
        elif adv_signal == '买入' and base_signal in ('强买入', '买入'):
            signal = '强买入'
        elif adv_signal == '买入':
            signal = '买入'
        elif base_signal in ('强买入', '买入'):
            signal = base_signal
        elif adv_signal == '观察' or rs_action == '加入备选池':
            signal = '观察'
        else:
            signal = base_signal

        # 风险等级
        amount = fund.get('amount_yuan', 0) or 0
        if amount >= 1e8:
            risk = '低'
        elif amount >= 5e7:
            risk = '中低'
        elif amount >= 1e7:
            risk = '中'
        else:
            risk = '高'

        # 是否有K线数据
        has_kline = code in self.kline_cache and len(self.kline_cache[code]) > 0
        kline_days = len(self.kline_cache.get(code, pd.DataFrame())) if has_kline else 0

        return {
            'code': code,
            'name': fund.get('name', ''),
            'fund_type': fund.get('fund_type', 'ETF'),
            'theme': theme,
            'price': float(fund.get('price', 0) or 0),
            'change_pct': round(change_pct, 2),
            'amount_str': fund.get('amount', '') if pd.notna(fund.get('amount', '')) else self._format_amount(amount),
            'amount_yuan': amount,
            'score': float(score) if pd.notna(score) else 0,
            'signal': signal,
            'risk_level': risk,
            'is_core': int(fund.get('is_core', 0)) if pd.notna(fund.get('is_core', 0)) else 0,
            'note': '',
            'has_kline': has_kline,
            'kline_days': kline_days,
            # v7.3 四维共振字段
            'resonance': resonance,
            'resonance_level': resonance_level,
            'resonance_dims': adv_result.get('resonance_dims', []),
            'advanced_signal': adv_signal,
            'advanced_action': adv_action,
            'advanced_details': adv_details,
            'stop_loss_rule': stop_loss,
            'val_score': adv_result.get('valuation', {}).get('score', 0),
            'br_score': adv_result.get('bottom_reversal', {}).get('score', 0),
            'cf_score': adv_result.get('capital_flow', {}).get('score', 0),
            'mr_score': adv_result.get('main_rally', {}).get('score', 0),
            # v7.4 估值详情
            'val_price_percentile': adv_result.get('valuation', {}).get('price_percentile'),
            'val_discount_rate': adv_result.get('valuation', {}).get('discount_rate'),
            'val_details': adv_result.get('valuation', {}).get('details', ''),
            # v7.4 拐点详情
            'br_hammer': adv_result.get('bottom_reversal', {}).get('hammer', {}).get('detected', False),
            'br_divergence': adv_result.get('bottom_reversal', {}).get('divergence', {}).get('detected', False),
            'br_vol_breakout': adv_result.get('bottom_reversal', {}).get('volume_breakout', {}).get('detected', False),
            # v7.4 资金详情
            'cf_ddx_consecutive': adv_result.get('capital_flow', {}).get('ddx_consecutive', 0),
            'cf_ddx_ok': adv_result.get('capital_flow', {}).get('ddx_ok', False),
            'cf_large_order_ok': adv_result.get('capital_flow', {}).get('large_order_ok', False),
            'cf_accumulating': adv_result.get('capital_flow', {}).get('is_accumulating', False),
            # v7.4 主升浪详情
            'mr_ma_convergence': adv_result.get('main_rally', {}).get('ma_convergence', {}).get('detected', False),
            'mr_bull_alignment': adv_result.get('main_rally', {}).get('ma_convergence', {}).get('bull_alignment', False),
            'mr_vol_gradient': adv_result.get('main_rally', {}).get('volume_gradient', {}).get('detected', False),
            'mr_air_refuel': adv_result.get('main_rally', {}).get('air_refueling', {}).get('detected', False),
            'mr_chip_profit_ratio': adv_result.get('main_rally', {}).get('chip_distribution', {}).get('profit_ratio', 0),
            'mr_chip_single_peak': adv_result.get('main_rally', {}).get('chip_distribution', {}).get('is_single_peak', False),
            # v7.4 右侧确认三步
            'rs_step1': rs_step1,
            'rs_step2': rs_step2,
            'rs_step3': rs_step3,
            'rs_action': rs_action,
            'rs_detail': rs_detail,
            'rs_entry': rs_entry,
            'rs_exit': rs_exit,
        }

    # ==================== 输出 ====================

    def to_dict(self) -> Dict:
        """输出完整推荐结果 (v7.4: 含四维分析+右侧确认)"""
        return {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'fund_type': 'exchange_traded',
            'model_version': 'v7.4',
            'total_etf': len(self.etf_data),
            'total_lof': len(self.lof_data),
            'total_funds': len(self.all_funds),
            'total_filtered': len(self.filter_by_liquidity()),
            'kline_count': len(self.kline_cache),
            'recommendations': self.recommendations,
            'theme_analysis': self.theme_analysis,
            'advanced_analysis': {
                'enabled': self.four_dim_engine is not None,
                'asset_type': 'etf',
                'dimensions': ['估值相对合理(ETF适配)', '触底反弹拐点', '主力资金流入', '主升浪启动'] if self.four_dim_engine else [],
                'resonance_threshold': 65,
                'right_side_framework': {
                    'step1_screen': '估值分位低+底背离 → 备选池',
                    'step2_confirm': '倍量阳/金针探底+DDX确认 → 触发',
                    'step3_order': '均线发散+筹码密集 → 下单',
                },
                'results': {code: {
                    'total_score': r.get('total_score', 0),
                    'signal': r.get('signal', ''),
                    'resonance': r.get('resonance', 0),
                    'resonance_level': r.get('resonance_level', ''),
                    'resonance_dims': r.get('resonance_dims', []),
                    'action': r.get('action', ''),
                    'details': r.get('details', ''),
                    'valuation': r.get('valuation', {}).get('score', 0),
                    'bottom_reversal': r.get('bottom_reversal', {}).get('score', 0),
                    'capital_flow': r.get('capital_flow', {}).get('score', 0),
                    'main_rally': r.get('main_rally', {}).get('score', 0),
                    'right_side': r.get('right_side', {}),
                } for code, r in self.advanced_results.items()},
            },
            'config': {
                'etf_min_amount': ETF_MIN_AMOUNT,
                'lof_min_amount': LOF_MIN_AMOUNT,
                'etf_recommend_top': ETF_RECOMMEND_TOP,
                'lof_recommend_top': LOF_RECOMMEND_TOP,
                'total_recommend': TOTAL_EXCHANGE_TRADED,
                'advanced_weights': ADVANCED_DIM_WEIGHTS,
            },
        }

    def save_report(self, filepath: str = None) -> str:
        """保存推荐结果为JSON"""
        if filepath is None:
            filepath = str(OUTPUT_DIR / f"etf_recommend_{datetime.now().strftime('%Y%m%d')}.json")

        result = self.to_dict()

        def default_serializer(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return str(obj)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=default_serializer)

        logger.info(f"报告已保存: {filepath}")
        return filepath

    # ==================== 一键运行 ====================

    def run(self, etf_browser_data: str = None, lof_browser_data: str = None) -> Dict:
        """
        一键执行完整推荐流程

        参数:
          etf_browser_data: 浏览器抓取的ETF数据(JSON字符串), akshare不可用时使用
          lof_browser_data: 浏览器抓取的LOF数据(JSON字符串)

        返回:
          完整推荐结果字典
        """
        logger.info("=" * 60)
        logger.info("场内基金推荐引擎 v7.3 启动 (四维高阶选股)")
        logger.info("=" * 60)

        # 1. 数据获取
        if etf_browser_data or lof_browser_data:
            self.load_from_browser_data(etf_browser_data or '', lof_browser_data or '')
        else:
            self.fetch_etf_realtime()
            self.fetch_lof_realtime()

        # 2. 合并数据
        self.merge_all_funds()

        if self.all_funds.empty:
            logger.error("未获取到场内基金数据!")
            return {'error': '未获取到场内基金数据', 'recommendations': []}

        # 3. 打分
        self.calculate_scores()

        # 4. 生成推荐
        self.generate_recommendations()

        # 5. 保存报告
        self.save_report()

        logger.info("=" * 60)
        logger.info(f"推荐完成: {len(self.recommendations)} 只场内基金")
        logger.info("=" * 60)

        return self.to_dict()
