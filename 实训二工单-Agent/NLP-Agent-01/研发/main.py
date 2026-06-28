# 工单编号：人工智能NLP-Agent数字人项目-记账本任务
# 来源：北京八维信息集团 · 八维文化与产业研究院

from __future__ import annotations

import logging
import sys

from agent.run import run
from config import LOG_LEVEL
from db.init import init_db

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def print_welcome() -> None:
    print("=" * 50)
    print("您好，欢迎使用咱们小家专属记账本！")
    print("请按照\"x年x月x日，谁做什么事收入/支出多少钱\"的格式来输入。")
    print("请告诉我你的账目需求吧~")
    print("输入 '退出' 或 'quit' 结束对话")
    print("=" * 50)


def main() -> None:
    init_db()
    print_welcome()
    print()

    while True:
        try:
            user_input = input("您：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见，下次见~")
            break

        if not user_input:
            continue

        if user_input.lower() in ("退出", "quit", "exit", "q"):
            print("再见，下次见~")
            break

        logger.info("[USER] %s", user_input)
        try:
            response = run(user_input)
            print(f"小账：{response}")
            logger.info("[AGENT] %s", response)
        except Exception as exc:
            logger.error("[ERROR] %s", exc)
            print(f"小账：抱歉，遇到了一点问题：{exc}，请稍后重试~")


if __name__ == "__main__":
    main()
