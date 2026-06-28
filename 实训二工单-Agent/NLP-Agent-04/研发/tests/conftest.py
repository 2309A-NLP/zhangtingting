"""测试配置"""

from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture
def sample_question() -> str:
    return "景顺长城中短债债券C基金在20210331的季报里，前三大持仓占比的债券名称是什么？"


@pytest.fixture
def sample_sql() -> str:
    return """SELECT b.bond_name, b.proportion 
FROM fund_bond_holdings b 
JOIN fund_basic_info f ON b.fund_code = f.fund_code 
WHERE f.fund_name LIKE '%景顺长城中短债债券C%' 
AND b.report_date = '2021-03-31' 
ORDER BY b.proportion DESC LIMIT 3;"""
