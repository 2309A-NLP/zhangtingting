CREATE TABLE IF NOT EXISTS money_notes (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    date     TEXT    NOT NULL,
    member   TEXT    NOT NULL,
    type     TEXT    NOT NULL CHECK (type IN ('支出', '收入')),
    category TEXT    NOT NULL,
    amount   REAL    NOT NULL CHECK (amount > 0),
    note     TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_member_date ON money_notes(member, date);

-- CREATE INDEX              -- 创建索引
-- IF NOT EXISTS             -- 如果不存在才创建（避免重复创建报错）
-- idx_member_date           -- 索引的名称
-- ON money_notes            -- 在 money_notes 表上
-- (member, date);           -- 包含 member 和 date 两列