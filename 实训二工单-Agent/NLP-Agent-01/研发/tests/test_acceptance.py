# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
# 来源：北京八维信息集团 · 八维文化与产业研究院

from __future__ import annotations

import importlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class AcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "acceptance.db")
        os.environ["DB_PATH"] = self.db_path

        import config

        importlib.reload(config)
        import db.init
        import db.operations
        import tools.db_tools
        import agent.guardrails
        import agent.run

        importlib.reload(db.init)
        importlib.reload(db.operations)
        importlib.reload(tools.db_tools)
        importlib.reload(agent.guardrails)
        importlib.reload(agent.run)

        self.guardrails = agent.guardrails.GuardrailAgent()
        db.init.init_db()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _rows(self) -> list[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT date, member, type, category, amount, note FROM money_notes ORDER BY id"
            ).fetchall()

    def test_case_1_add_expense(self) -> None:
        reply = self.guardrails.reply("今天女儿买了双登山鞋499元")
        self.assertIn("确认无误吗", reply)
        final = self.guardrails.reply("确认")
        self.assertIn("已记录", final)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "女儿")
        self.assertEqual(rows[0][2], "支出")
        self.assertEqual(rows[0][4], 499.0)

    def test_case_2_add_income(self) -> None:
        reply = self.guardrails.reply("7月5日妈妈收到报销1000元")
        self.assertIn("确认无误吗", reply)
        final = self.guardrails.reply("确认")
        self.assertIn("已记录", final)
        rows = self._rows()
        self.assertEqual(rows[0][1], "妈妈")
        self.assertEqual(rows[0][2], "收入")
        self.assertEqual(rows[0][4], 1000.0)

    def test_case_3_query_month_details(self) -> None:
        self.guardrails.reply("今天女儿买了双登山鞋499元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("看下这个月家里花钱明细")
        self.assertTrue("您查看的时间段" in reply or "本月家庭总支出" in reply)

    def test_case_4_query_member_total(self) -> None:
        self.guardrails.reply("今天女儿买了双登山鞋499元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("这个月女儿花了多少钱？")
        self.assertIn("女儿本月总支出", reply)

    def test_case_5_delete_after_confirmation(self) -> None:
        self.guardrails.reply("今天女儿报旅游团费1200元")
        self.guardrails.reply("确认")
        prompt = self.guardrails.reply("删除女儿报旅游团的费用")
        self.assertIn("确认删除这条记录吗", prompt)
        done = self.guardrails.reply("确认")
        self.assertIn("已删除", done)
        self.assertEqual(self._rows(), [])

    def test_case_6_non_accounting_message(self) -> None:
        reply = self.guardrails.reply("你好")
        self.assertIn("欢迎使用咱们小家专属记账本", reply)

    def test_case_7_missing_fields_followup(self) -> None:
        reply = self.guardrails.reply("今天买东西")
        self.assertIn("需要确认一下", reply)
        self.assertIn("谁", reply)
        self.assertIn("多少钱", reply)

    def test_case_8_precise_keyword_query(self) -> None:
        self.guardrails.reply("今天爸爸买了三体50元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("我哪天买的三体？")
        self.assertTrue("三体" in reply or "相关记录" in reply)

    def test_case_9_category_total(self) -> None:
        self.guardrails.reply("今天爸爸买书50元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("我这个月买书花了多少钱")
        self.assertTrue("共计" in reply or "花费" in reply)

    def test_case_10_verify_after_insert(self) -> None:
        self.guardrails.reply("今天妈妈买菜88元")
        self.guardrails.reply("确认")
        rows = self._rows()
        self.assertEqual(len(rows), 1)

    def test_case_11_member_alias_normalization(self) -> None:
        reply = self.guardrails.reply("今天老妈买菜66元")
        self.assertIn("确认无误吗", reply)
        self.guardrails.reply("确认")
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "妈妈")

    def test_case_12_reject_insert_should_not_write(self) -> None:
        reply = self.guardrails.reply("今天爸爸买水果30元")
        self.assertIn("确认无误吗", reply)
        cancel = self.guardrails.reply("先不要")
        self.assertIn("先不写入", cancel)
        self.assertEqual(self._rows(), [])

    def test_case_13_reject_delete_should_keep_record(self) -> None:
        self.guardrails.reply("今天女儿报旅游团费1200元")
        self.guardrails.reply("确认")
        prompt = self.guardrails.reply("删除女儿报旅游团的费用")
        self.assertIn("确认删除这条记录吗", prompt)
        cancel = self.guardrails.reply("取消")
        self.assertIn("先不删", cancel)
        self.assertEqual(len(self._rows()), 1)

    def test_case_14_multi_record_insert(self) -> None:
        reply = self.guardrails.reply("今天妈妈买菜88元，爸爸买书50元")
        self.assertIn("一共识别到以下账目", reply)
        done = self.guardrails.reply("确认")
        self.assertIn("已全部记录", done)
        rows = self._rows()
        self.assertEqual(len(rows), 2)

    def test_case_15_yesterday_expense(self) -> None:
        reply = self.guardrails.reply("昨天爸爸买菜45元")
        self.assertIn("确认无误吗", reply)
        self.guardrails.reply("确认")
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "爸爸")
        self.assertEqual(rows[0][4], 45.0)

    def test_case_16_last_month_query(self) -> None:
        self.guardrails.reply("今天妈妈买菜88元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("看下上个月家里花钱明细")
        self.assertTrue("暂无记录" in reply or "您查看的时间段" in reply)

    def test_case_17_multi_turn_fill_slots_success(self) -> None:
        reply = self.guardrails.reply("今天买东西")
        self.assertIn("需要确认一下", reply)
        reply = self.guardrails.reply("妈妈的")
        self.assertIn("多少钱", reply)
        reply = self.guardrails.reply("88元买菜")
        self.assertIn("确认无误吗", reply)
        done = self.guardrails.reply("确认")
        self.assertIn("已记录", done)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "妈妈")
        self.assertEqual(rows[0][4], 88.0)

    def test_case_18_fill_slots_then_cancel(self) -> None:
        reply = self.guardrails.reply("今天买东西")
        self.assertIn("需要确认一下", reply)
        cancel = self.guardrails.reply("取消")
        self.assertTrue("先不写入" in cancel or "重新告诉我" in cancel)
        self.assertEqual(self._rows(), [])

    def test_case_19_delete_last_record_by_reference(self) -> None:
        self.guardrails.reply("今天妈妈买菜88元")
        self.guardrails.reply("确认")
        prompt = self.guardrails.reply("把刚才那笔删掉")
        self.assertIn("确认删除这条记录吗", prompt)
        done = self.guardrails.reply("确认")
        self.assertIn("已删除", done)
        self.assertEqual(self._rows(), [])

    def test_case_20_delete_pending_by_reference_reject(self) -> None:
        self.guardrails.reply("今天爸爸买水果30元")
        self.guardrails.reply("确认")
        prompt = self.guardrails.reply("不要那条了")
        self.assertIn("确认删除这条记录吗", prompt)
        cancel = self.guardrails.reply("不确认")
        self.assertIn("先不删", cancel)
        self.assertEqual(len(self._rows()), 1)

    def test_case_21_query_last_record_by_reference(self) -> None:
        self.guardrails.reply("今天爸爸买书50元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("那笔多少钱")
        self.assertIn("50", reply)

    def test_case_22_query_last_record_category_by_reference(self) -> None:
        self.guardrails.reply("今天妈妈买菜88元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("那条是什么")
        self.assertTrue("买菜" in reply or "88" in reply)

    def test_case_23_query_last_record_member_by_reference(self) -> None:
        self.guardrails.reply("今天女儿买文具35元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("那是谁买的")
        self.assertIn("女儿", reply)

    def test_case_24_query_first_record_amount_in_recent_batch(self) -> None:
        self.guardrails.reply("今天妈妈买菜88元，爸爸买书50元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("第一笔多少钱")
        self.assertIn("88", reply)

    def test_case_25_query_second_record_member_in_recent_batch(self) -> None:
        self.guardrails.reply("今天妈妈买菜88元，爸爸买书50元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("第二笔是谁的")
        self.assertIn("爸爸", reply)

    def test_case_26_query_last_record_in_recent_batch(self) -> None:
        self.guardrails.reply("今天妈妈买菜88元，爸爸买书50元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("最后一条是什么")
        self.assertTrue("买书" in reply or "50" in reply)

    def test_case_27_delete_first_record_in_recent_batch(self) -> None:
        self.guardrails.reply("今天妈妈买菜88元，爸爸买书50元")
        self.guardrails.reply("确认")
        prompt = self.guardrails.reply("把第一笔删掉")
        self.assertIn("确认删除这条记录吗", prompt)
        done = self.guardrails.reply("确认")
        self.assertIn("已删除", done)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "爸爸")

    def test_case_28_last_friday_date_parse(self) -> None:
        reply = self.guardrails.reply("上周五妈妈买菜30元")
        self.assertIn("确认无误吗", reply)
        self.guardrails.reply("确认")
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "妈妈")
        self.assertEqual(rows[0][4], 30.0)

    def test_case_29_spouse_alias_normalization(self) -> None:
        reply = self.guardrails.reply("今天老婆买菜66元")
        self.assertIn("确认无误吗", reply)
        self.guardrails.reply("确认")
        rows = self._rows()
        self.assertEqual(rows[0][1], "妈妈")

    def test_case_30_ambiguous_pronoun_should_ask_member(self) -> None:
        reply = self.guardrails.reply("今天他买书50元")
        self.assertIn("谁的消费", reply)
        self.assertEqual(self._rows(), [])

    def test_case_31_estimated_amount_should_require_confirmation(self) -> None:
        reply = self.guardrails.reply("今天妈妈大概买菜88元")
        self.assertTrue("大概" in reply or "估计" in reply or "确认无误吗" in reply)
        self.assertEqual(self._rows(), [])
        done = self.guardrails.reply("确认")
        self.assertIn("已记录", done)

    def test_case_32_duplicate_record_should_warn_before_insert(self) -> None:
        self.guardrails.reply("今天妈妈买菜88元")
        self.guardrails.reply("确认")
        reply = self.guardrails.reply("今天妈妈买菜88元")
        self.assertIn("已经存过", reply)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        done = self.guardrails.reply("确认")
        self.assertIn("已记录", done)
        self.assertEqual(len(self._rows()), 2)


if __name__ == "__main__":
    unittest.main()
