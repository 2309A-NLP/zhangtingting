"""
给博金杯比赛数据库添加索引，大幅加速 JOIN 查询
"""
import sqlite3
import time

db_path = "data/raw/bs_challenge_financial_14b_dataset/dataset/博金杯比赛数据.db"
print(f"连接数据库: {db_path}")
conn = sqlite3.connect(db_path)
c = conn.cursor()

indexes = [
    "CREATE INDEX IF NOT EXISTS idx_a_daily_stock ON A股票日行情表(股票代码);",
    "CREATE INDEX IF NOT EXISTS idx_a_daily_date ON A股票日行情表(交易日);",
    "CREATE INDEX IF NOT EXISTS idx_industry_stock ON A股公司行业划分表(股票代码);",
    "CREATE INDEX IF NOT EXISTS idx_fund_stock_code ON 基金股票持仓明细(基金代码);",
    "CREATE INDEX IF NOT EXISTS idx_fund_bond_code ON 基金债券持仓明细(基金代码);",
    "CREATE INDEX IF NOT EXISTS idx_fund_daily_code ON 基金日行情表(基金代码);",
    "CREATE INDEX IF NOT EXISTS idx_fund_daily_date ON 基金日行情表(交易日期);",
    "CREATE INDEX IF NOT EXISTS idx_fund_size_code ON 基金规模变动表(基金代码);",
]

for sql in indexes:
    name = sql.split("ON ")[1].split("(")[0]
    print(f"  创建索引: {name}...", end=" ")
    start = time.time()
    try:
        c.execute(sql)
        print(f"完成 ({time.time()-start:.1f}s)")
    except Exception as e:
        print(f"失败: {e}")

conn.commit()
conn.close()
print("\n所有索引创建完成！")
print("重启服务后 JOIN 查询速度将大幅提升。")
