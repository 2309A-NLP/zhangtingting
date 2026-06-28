SCHEDULE_CREATE_TOOL_SCHEMA = {
    "name": "schedule_create",
    "description": "新增日程",
    "input_schema": {
        "type": "object",
            "properties": {
                "content": {"type": "string"},
                "schedule_date": {"type": ["string", "null"]},
                "schedule_time": {"type": "string"},
                "cycle_rule": {
                    "type": "string",
                    "description": "once/daily/weekday/weekly_custom/interval_days",
                },
                "cycle_value": {"type": ["string", "null"]},
                "source_text": {"type": ["string", "null"]},
            },
        "required": ["content", "schedule_time", "cycle_rule"],
    },
}

SCHEDULE_LIST_TOOL_SCHEMA = {
    "name": "schedule_list",
    "description": "查询日程列表",
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": ["string", "null"]},
            "start_date": {"type": ["string", "null"]},
            "end_date": {"type": ["string", "null"]},
            "status": {"type": ["string", "null"]},
        },
        "required": [],
    },
}

SCHEDULE_GET_TOOL_SCHEMA = {
    "name": "schedule_get",
    "description": "查询单条日程",
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
        },
        "required": ["id"],
    },
}

SCHEDULE_UPDATE_TOOL_SCHEMA = {
    "name": "schedule_update",
    "description": "更新日程",
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "content": {"type": ["string", "null"]},
            "schedule_date": {"type": ["string", "null"]},
            "schedule_time": {"type": ["string", "null"]},
            "cycle_rule": {"type": ["string", "null"]},
            "cycle_value": {"type": ["string", "null"]},
            "source_text": {"type": ["string", "null"]},
            "status": {"type": ["string", "null"]},
        },
        "required": ["id"],
    },
}

SCHEDULE_DELETE_TOOL_SCHEMA = {
    "name": "schedule_delete",
    "description": "删除或取消日程",
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
        },
        "required": ["id"],
    },
}
