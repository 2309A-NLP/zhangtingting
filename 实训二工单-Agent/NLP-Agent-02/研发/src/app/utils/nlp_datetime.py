import re
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class ParsedDateTime:
    schedule_date: str | None
    schedule_time: str | None


class ChineseDateTimeParser:
    def parse(self, user_input: str) -> ParsedDateTime:
        parsed_date = self._parse_date(user_input)
        parsed_time = self._parse_time(user_input)
        return ParsedDateTime(schedule_date=parsed_date, schedule_time=parsed_time)

    def _parse_date(self, user_input: str) -> str | None:
        today = date.today()
        if "明天" in user_input:
            return (today + timedelta(days=1)).isoformat()
        if "后天" in user_input:
            return (today + timedelta(days=2)).isoformat()
        if "今天" in user_input:
            return today.isoformat()

        if "下周" in user_input:
            weekday_map = {
                "一": 0,
                "二": 1,
                "三": 2,
                "四": 3,
                "五": 4,
                "六": 5,
                "日": 6,
                "天": 6,
            }
            for token, weekday in weekday_map.items():
                if f"下周{token}" in user_input or f"下星期{token}" in user_input:
                    days_until_next_week = 7 - today.weekday()
                    return (today + timedelta(days=days_until_next_week + weekday)).isoformat()

        weekday_map = {
            "周一": 0,
            "星期一": 0,
            "周二": 1,
            "星期二": 1,
            "周三": 2,
            "星期三": 2,
            "周四": 3,
            "星期四": 3,
            "周五": 4,
            "星期五": 4,
            "周六": 5,
            "星期六": 5,
            "周日": 6,
            "周天": 6,
            "星期日": 6,
            "星期天": 6,
        }
        for keyword, weekday in weekday_map.items():
            if keyword in user_input:
                delta_days = (weekday - today.weekday() + 7) % 7
                if delta_days == 0:
                    delta_days = 7
                return (today + timedelta(days=delta_days)).isoformat()

        return None

    def parse_interval_days(self, user_input: str) -> int | None:
        match = re.search(r"每隔([一二三四五六七八九十\d]+)天", user_input)
        if match:
            raw_value = match.group(1)
            if raw_value.isdigit():
                value = int(raw_value)
            else:
                chinese_numbers = {
                    "一": 1,
                    "二": 2,
                    "三": 3,
                    "四": 4,
                    "五": 5,
                    "六": 6,
                    "七": 7,
                    "八": 8,
                    "九": 9,
                    "十": 10,
                }
                value = chinese_numbers.get(raw_value, 0)
            return value if value > 0 else None
        return None

    def _parse_time(self, user_input: str) -> str | None:
        meridiem_match = re.search(
            r"(上午|早上|清晨|中午|下午|晚上|傍晚)\s*(\d{1,2})(?:点|时)(?:([0-5]?\d))?(?:分)?",
            user_input,
        )
        if meridiem_match:
            meridiem = meridiem_match.group(1)
            hour = int(meridiem_match.group(2))
            minute = int(meridiem_match.group(3) or 0)
            if meridiem in {"下午", "晚上", "傍晚"} and hour < 12:
                hour += 12
            if meridiem == "中午" and hour < 12:
                hour = 12 if hour == 0 else hour
            return f"{hour:02d}:{minute:02d}:00"

        if "下午" in user_input and "5点" in user_input:
            return "17:00:00"
        if "早上" in user_input and "9点" in user_input:
            return "09:00:00"
        if "下午5点" in user_input or "17:00" in user_input:
            return "17:00:00"
        if "早上9点" in user_input or "09:00" in user_input:
            return "09:00:00"

        match = re.search(r"(\d{1,2}):(\d{2})", user_input)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            return f"{hour:02d}:{minute:02d}:00"

        return None
