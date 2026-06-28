EXAMPLES = [
    "明天下午5点提醒我开会",
    "我今天的日程有哪些？",
]

TOOL_CALL_EXAMPLES = [
    {
        "tool_name": "schedule_create",
        "arguments": {
            "content": "开会",
            "schedule_date": "2025-01-15",
            "schedule_time": "17:00:00",
            "cycle_rule": "once",
            "cycle_value": None,
            "source_text": "明天下午5点提醒我开会",
        },
    },
    {
        "tool_name": "schedule_list",
        "arguments": {
            "date_value": "2025-01-15",
            "status": "active",
        },
    },
    {
        "tool_name": "schedule_delete",
        "arguments": {
            "schedule_id": 1,
        },
    },
]
