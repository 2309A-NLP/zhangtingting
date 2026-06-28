"""数据库服务 — 元数据提取与连接管理"""

from __future__ import annotations

import sqlite3
from typing import Optional

from config import settings
from src.core.models import ColumnInfo, TableSchema


class DatabaseService:
    """数据库服务：元数据提取、连接池管理"""

    # 实际表名（中文）→ 业务描述
    TABLE_DESCRIPTIONS = {
        "基金基本信息": "基金基本信息，包含基金代码、名称、类型、管理人、成立日期等",
        "基金股票持仓明细": "基金股票持仓明细，包含每只基金持有的股票代码、名称、数量、市值、占比",
        "基金债券持仓明细": "基金债券持仓明细，包含每只基金持有的债券名称、类型、市值、占比",
        "基金可转债持仓明细": "基金可转债持仓明细，包含可转债名称、对应股票代码、市值、占比",
        "基金日行情表": "基金日行情表，包含单位净值、复权净值、累计净值、资产净值",
        "A股票日行情表": "A股日行情表，包含昨收盘、今开盘、最高价、最低价、收盘价、成交量、成交额",
        "港股票日行情表": "港股日行情表，包含昨收盘、今开盘、最高价、最低价、收盘价、成交量、成交额",
        "A股公司行业划分表": "A股公司行业划分表（中信行业分类），包含股票代码、一级行业、二级行业",
        "基金规模变动表": "基金规模变动表，包含期初份额、申购份额、赎回份额、期末份额",
        "基金份额持有人结构": "基金份额持有人结构，包含机构持有份额、个人持有份额及比例",
    }

    # 字段业务描述映射
    FIELD_DESCRIPTIONS = {
        "基金基本信息": {
            "基金代码": "基金唯一标识代码，格式如000001.OF",
            "基金全称": "基金完整名称",
            "基金简称": "基金简称",
            "管理人": "基金管理公司名称",
            "托管人": "基金托管银行",
            "基金类型": "基金类型：股票型/债券型/混合型等",
            "成立日期": "基金成立日期，格式YYYYMMDD",
            "到期日期": "基金到期日期，格式YYYYMMDD",
            "管理费率": "基金管理人收取的管理费比例",
            "托管费率": "基金托管人收取的托管费比例",
        },
        "基金股票持仓明细": {
            "基金代码": "基金唯一标识代码",
            "基金简称": "基金简称",
            "持仓日期": "报告期持仓日期，格式YYYYMMDD",
            "股票代码": "持仓股票代码，格式如600519.SH",
            "股票名称": "持仓股票名称",
            "数量": "持有股票数量（股）",
            "市值": "持仓市值（元）",
            "市值占基金资产净值比": "持仓市值占基金资产净值比例",
            "第N大重仓股": "持仓排名，1表示第一大重仓股",
            "所在证券市场": "股票所在市场：沪市/深市",
            "所属国家(地区)": "股票所属国家/地区",
            "报告类型": "报告类型：季报/半年报/年报",
        },
        "基金债券持仓明细": {
            "基金代码": "基金唯一标识代码",
            "基金简称": "基金简称",
            "持仓日期": "报告期持仓日期，格式YYYYMMDD",
            "债券类型": "债券类型：国债/企业债/金融债等",
            "债券名称": "债券名称",
            "持债数量": "持有债券数量",
            "持债市值": "持仓市值（元）",
            "持债市值占基金资产净值比": "持仓市值占基金资产净值比例",
            "第N大重仓股": "持仓排名",
            "所在证券市场": "债券所在市场",
            "所属国家(地区)": "债券所属国家/地区",
            "报告类型": "报告类型",
        },
        "基金可转债持仓明细": {
            "基金代码": "基金唯一标识代码",
            "基金简称": "基金简称",
            "持仓日期": "报告期持仓日期，格式YYYYMMDD",
            "对应股票代码": "可转债对应的正股股票代码",
            "债券名称": "可转债名称",
            "数量": "持有数量",
            "市值": "持仓市值（元）",
            "市值占基金资产净值比": "持仓市值占基金资产净值比例",
            "第N大重仓股": "持仓排名",
            "所在证券市场": "可转债所在市场",
            "所属国家(地区)": "可转债所属国家/地区",
            "报告类型": "报告类型",
        },
        "基金日行情表": {
            "基金代码": "基金唯一标识代码",
            "交易日期": "交易日期，格式YYYYMMDD",
            "单位净值": "基金单位净值（元）",
            "复权单位净值": "复权后的单位净值",
            "累计单位净值": "累计单位净值",
            "资产净值": "基金资产净值（元）",
        },
        "A股票日行情表": {
            "股票代码": "股票代码，格式如600519.SH",
            "交易日": "交易日期，格式YYYYMMDD",
            "昨收盘(元)": "前一交易日收盘价（元）",
            "今开盘(元)": "当日开盘价（元）",
            "最高价(元)": "当日最高价（元）",
            "最低价(元)": "当日最低价（元）",
            "收盘价(元)": "当日收盘价（元）",
            "成交量(股)": "成交量（股）",
            "成交金额(元)": "成交金额（元）",
        },
        "港股票日行情表": {
            "股票代码": "港股股票代码",
            "交易日": "交易日期，格式YYYYMMDD",
            "昨收盘(元)": "前一交易日收盘价",
            "今开盘(元)": "当日开盘价",
            "最高价(元)": "当日最高价",
            "最低价(元)": "当日最低价",
            "收盘价(元)": "当日收盘价",
            "成交量(股)": "成交量",
            "成交金额(元)": "成交金额",
        },
        "A股公司行业划分表": {
            "股票代码": "股票代码",
            "交易日期": "交易日期",
            "行业划分标准": "行业划分标准：中信/申万等",
            "一级行业名称": "一级行业名称，如：综合金融、银行、非银金融等",
            "二级行业名称": "二级行业名称",
        },
        "基金规模变动表": {
            "基金代码": "基金唯一标识代码",
            "基金简称": "基金简称",
            "公告日期": "公告日期",
            "截止日期": "报告期截止日期，格式YYYYMMDD",
            "报告期期初基金总份额": "期初份额",
            "报告期基金总申购份额": "本期申购份额",
            "报告期基金总赎回份额": "本期赎回份额",
            "报告期期末基金总份额": "期末份额",
            "定期报告所属年度": "报告所属年度",
            "报告类型": "报告类型：季报/半年报/年报",
        },
        "基金份额持有人结构": {
            "基金代码": "基金唯一标识代码",
            "基金简称": "基金简称",
            "公告日期": "公告日期",
            "截止日期": "报告期截止日期",
            "机构投资者持有的基金份额": "机构持有份额",
            "机构投资者持有的基金份额占总份额比例": "机构持有比例",
            "个人投资者持有的基金份额": "个人持有份额",
            "个人投资者持有的基金份额占总份额比例": "个人持有比例",
            "定期报告所属年度": "报告所属年度",
            "报告类型": "报告类型",
        },
    }

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or str(settings.DB_PATH_RESOLVED)
        self._schemas: dict[str, TableSchema] | None = None

    def get_all_table_schemas(self) -> list[TableSchema]:
        """获取所有表的 Schema"""
        if self._schemas is not None:
            return list(self._schemas.values())

        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        table_names = [row[0] for row in cursor.fetchall()]

        schemas: dict[str, TableSchema] = {}
        for name in table_names:
            cursor.execute(f'PRAGMA table_info("{name}");')
            columns = []
            table_field_descs = self.FIELD_DESCRIPTIONS.get(name, {})
            for col in cursor.fetchall():
                col_id, col_name, col_type, not_null, default_val, is_pk = col
                columns.append(ColumnInfo(
                    name=col_name,
                    data_type=col_type or "TEXT",
                    description=table_field_descs.get(col_name, ""),
                    is_primary_key=bool(is_pk),
                ))

            # 获取外键
            cursor.execute(f'PRAGMA foreign_key_list("{name}");')
            fk_info = cursor.fetchall()
            for fk in fk_info:
                _, _, fk_table, fk_from, fk_to, *_ = fk
                for col in columns:
                    if col.name == fk_from:
                        col.is_foreign_key = True
                        col.fk_ref_table = fk_table
                        col.fk_ref_column = fk_to

            schemas[name] = TableSchema(
                name=name,
                description=self.TABLE_DESCRIPTIONS.get(name, ""),
                columns=columns,
            )

        conn.close()
        self._schemas = schemas
        return list(schemas.values())

    def get_table_schema(self, table_name: str) -> Optional[TableSchema]:
        """获取单张表 Schema"""
        schemas = self.get_all_table_schemas()
        for s in schemas:
            if s.name == table_name:
                return s
        return None

    @property
    def db_path(self) -> str:
        return self._db_path
