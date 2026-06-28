# Schedule Reminder Agent

Enterprise-grade FastAPI scaffold for a schedule reminder agent, built to be both production-minded and educational.

## Current Status

The project is now at a fairly complete backend-demo stage:

- schedule CRUD and summary
- agent chat, confirmation, session history
- LLM fallback parser, planner, reply enhancement, and audit
- scheduler / worker / Redis runtime queue
- reminder retry, alerting, admin operations
- metrics, readiness, dashboard, and demo pages
- local + Docker Compose multi-role startup

Useful docs for the current stage:

- `docs/39-部署与联调手册.md`
- `docs/40-当前完成度与路线图.md`

## Tech Stack

- FastAPI
- Pydantic v2
- SQLAlchemy 2.0 Async
- Alembic
- APScheduler
- structlog
- pytest / ruff / mypy

## Quick Start

1. Install dependencies

```powershell
python -m pip install -e .[dev]
```

2. Prepare environment

```powershell
Copy-Item .env.example .env
```

3. Run database migrations

```powershell
$env:PYTHONPATH="src"
python -m alembic upgrade head
```

The app does not auto-create tables by default. This is intentional: schema changes should go through Alembic migrations so local/dev behavior stays aligned with production-style workflow.

4. Start the app

```powershell
.\scripts\dev.ps1 run -Port 8010
```

If you also want to run the scheduler as an independent process, open another terminal:

```powershell
.\scripts\dev.ps1 run-scheduler -Port 8010
```

If you also want to run the reminder worker as an independent process, open a third terminal:

```powershell
.\scripts\dev.ps1 run-worker -Port 8010
```

5. Open and verify

- API health: `http://127.0.0.1:8010/api/v1/health`
- API readiness: `http://127.0.0.1:8010/api/v1/health/ready`
- Demo hub: `http://127.0.0.1:8010/api/v1/demo`
- Chat demo: `http://127.0.0.1:8010/api/v1/demo/chat`
- Agent history demo: `http://127.0.0.1:8010/api/v1/demo/agent-history`
- OpenAPI docs: `http://127.0.0.1:8010/docs`

## Docker Compose

1. Prepare Docker environment

```powershell
Copy-Item .env.docker.example .env.docker
```

2. Start services

```powershell
docker compose up --build
```

or use the helper:

```powershell
.\scripts\dev.ps1 compose-up
```

3. Verify

- API health: `http://127.0.0.1:8000/api/v1/health`
- API readiness: `http://127.0.0.1:8000/api/v1/health/ready`
- Demo hub: `http://127.0.0.1:8000/api/v1/demo`
- Chat demo: `http://127.0.0.1:8000/api/v1/demo/chat`
- Agent history demo: `http://127.0.0.1:8000/api/v1/demo/agent-history`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

Quick Docker smoke check:

```powershell
.\scripts\dev.ps1 smoke-docker
```

This smoke check now also verifies the delivery queue summary endpoint, so you can confirm the scheduler/worker/Redis pipeline is visible from the admin side.

## Suggested Manual Acceptance Flow

1. Send `明天下午5点提醒我开会`
2. Confirm the returned `confirm` state
3. Send confirmation
4. Query schedules
5. Delete a target schedule
6. Query schedule summary overview

Example schedule list query with pagination and filters:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/schedule?date=2025-01-16&status=active&limit=10&offset=0"
```

Example schedule list query with keyword search and sorting:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/schedule?keyword=report&sort_by=schedule_date&sort_order=asc&limit=10&offset=0"
```

Example schedule list query with multi-status and time-range filters:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/schedule?statuses=active&statuses=cancelled&schedule_time_start=09:00:00&schedule_time_end=18:00:00&sort_by=schedule_time&sort_order=asc&limit=10&offset=0"
```

Example schedule summary query for dashboard-style overview:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/schedule/summary" | ConvertTo-Json -Depth 10
```

Example filtered summary query:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/schedule/summary?keyword=report&status=active" | ConvertTo-Json -Depth 10
```

Response shape:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "total": 1,
    "limit": 10,
    "offset": 0,
    "items": [
      {
        "id": 1,
        "content": "team sync",
        "schedule_date": "2025-01-16",
        "schedule_time": "17:00:00",
        "cycle_rule": "once",
        "cycle_value": null,
        "source_text": "tomorrow at 5pm remind me about team sync",
        "status": "active",
        "next_trigger_at": "2025-01-16T17:00:00",
        "created_at": "2025-01-15T10:00:00",
        "updated_at": "2025-01-15T10:00:00"
      }
    ]
  }
}
```

## Test

```powershell
python -m pytest -q
```

## Dev Script

Common local commands:

```powershell
.\scripts\dev.ps1 migrate
.\scripts\dev.ps1 lint
.\scripts\dev.ps1 typecheck
.\scripts\dev.ps1 test
.\scripts\dev.ps1 check
.\scripts\dev.ps1 precommit
.\scripts\dev.ps1 hooks
.\scripts\dev.ps1 run -Port 8010
.\scripts\dev.ps1 run-ai-debug -Port 8010
.\scripts\dev.ps1 run-scheduler -Port 8010
.\scripts\dev.ps1 run-worker -Port 8010
.\scripts\dev.ps1 compose-up
.\scripts\dev.ps1 compose-logs
.\scripts\dev.ps1 compose-down
.\scripts\dev.ps1 smoke-local
.\scripts\dev.ps1 smoke-docker
```

`hooks` installs Git hooks for `pre-commit` and `pre-push`. `precommit` runs the same checks manually across the whole project.

`run` starts the API role. `run-scheduler` starts the scheduler role. `run-worker` starts the reminder delivery worker role.

The scheduler role now also supports a DB-backed lease lock so multiple scheduler instances do not process the same job at the same time.

Relevant environment variables:

```powershell
SCHEDULER_LOCK_ENABLED=true
SCHEDULER_LOCK_OWNER=local-scheduler-1
SCHEDULER_LOCK_TTL_SECONDS=120
```

## Optional AI Fallback

The project supports an optional LLM fallback parser for `agent/chat`.

Set these variables in `.env` when you want to enable it:

```powershell
LLM_ENABLED=true
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=30
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=400
LLM_DEBUG_LOGGING=false
```

Behavior:

- Rule parser runs first
- LLM parser is used as fallback when the rule parser cannot confidently handle the input
- If LLM is disabled or unavailable, the original rule-based flow still works

For local AI debugging, you can also run:

```powershell
.\scripts\dev.ps1 run-ai-debug -Port 8010
```

This enables verbose LLM diagnostics in the application logs.

## Agent Session History

The project now stores each `agent/chat` interaction as conversation history, so you can inspect the full agent trail instead of only the latest session snapshot.

Query one session's full history:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/agent/sessions/session-demo-1/history" | ConvertTo-Json -Depth 10
```

Query paginated history list:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/agent/sessions/history?session_id=session-demo-1&limit=20&offset=0" | ConvertTo-Json -Depth 10
```

History APIs now return decoded fields such as `context`, `tool_arguments`, `missing_fields`, `suggested_inputs`, and structured `execution_result`, instead of exposing internal `*_json` storage fields.

## Optional LLM Reply Enhancement

The project can also let the LLM participate in user-facing reply generation after the backend has already made a deterministic execution decision.

Enable it in `.env`:

```powershell
LLM_REPLY_ENABLED=true
```

This mode does not change tool execution logic. It only rewrites the final reply text to sound more natural.

When enabled, it can also improve `confirm` and `clarify` messages, so the agent feels more conversational while still keeping execution deterministic.

For `clarify` states, the backend can also return `suggested_inputs`, which are short example follow-up messages that help the user complete missing information. These suggestions are also persisted into conversation history for later debugging and replay.

## Admin APIs

The project also exposes a small admin-style API group under `/api/v1/admin/...`.

These endpoints are intended for backend operations, audit, and dashboard scenarios such as:

- admin schedule list and summary
- reminder delivery log queries
- conversation session queries
- conversation history queries
- LLM audit log queries
- aggregated admin dashboard overview

Example endpoints:

- `GET /api/v1/admin/dashboard/overview`
- `GET /api/v1/admin/schedule`
- `GET /api/v1/admin/schedule/summary`
- `GET /api/v1/admin/schedule/reminder-logs`
- `GET /api/v1/admin/schedule/delivery-tasks`
- `GET /api/v1/admin/schedule/delivery-queue`
- `GET /api/v1/admin/agent/sessions`
- `GET /api/v1/admin/agent/sessions/history`
- `GET /api/v1/admin/agent/llm-audit`

### Admin Token Behavior

The admin API uses the `ADMIN_ACCESS_TOKEN` environment variable.

```powershell
ADMIN_ACCESS_TOKEN=secret-admin-token
```

Behavior:

- if `ADMIN_ACCESS_TOKEN` is empty, admin routes are open in local/dev mode for convenience
- if `ADMIN_ACCESS_TOKEN` is set, requests must send `X-Admin-Token`

PowerShell example with admin token:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8010/api/v1/admin/dashboard/overview" `
  -Headers @{ "X-Admin-Token" = "secret-admin-token" } |
  ConvertTo-Json -Depth 10
```

PowerShell example without token after enabling admin auth:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/dashboard/overview"
```

The second command should return `401 Unauthorized` after you set `ADMIN_ACCESS_TOKEN` and restart the app.

### Admin Verification Examples

Query admin dashboard overview:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/dashboard/overview" | ConvertTo-Json -Depth 10
```

Query admin schedule summary:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/schedule/summary?status=active" | ConvertTo-Json -Depth 10
```

Query admin reminder logs:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/schedule/reminder-logs?limit=10&offset=0" | ConvertTo-Json -Depth 10
```

Query admin worker delivery tasks:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/schedule/delivery-tasks?limit=10&offset=0" | ConvertTo-Json -Depth 10
```

Query admin worker delivery queue summary:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/schedule/delivery-queue" | ConvertTo-Json -Depth 10
```

Query scheduler lease state:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/scheduler/leases/scan_due_reminders" | ConvertTo-Json -Depth 10
```

Query admin session list:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/agent/sessions?limit=10&offset=0" | ConvertTo-Json -Depth 10
```

Query admin conversation history list:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/agent/sessions/history?limit=20&offset=0" | ConvertTo-Json -Depth 10
```

Query admin LLM audit logs:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/agent/llm-audit?limit=20&offset=0" | ConvertTo-Json -Depth 10
```

### Admin CSV Export

The admin API also supports CSV export endpoints. These are useful for operations review, offline analysis, and simple reporting.

Use `Invoke-WebRequest` for export endpoints, because CSV download returns raw file content instead of JSON.

Export schedules:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8010/api/v1/admin/schedule/export?status=active" `
  -OutFile ".\admin_schedules.csv"
```

Export reminder logs:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8010/api/v1/admin/schedule/reminder-logs/export" `
  -OutFile ".\admin_reminder_logs.csv"
```

Export delivery tasks:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8010/api/v1/admin/schedule/delivery-tasks/export" `
  -OutFile ".\admin_delivery_tasks.csv"
```

Export agent conversation history:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8010/api/v1/admin/agent/sessions/history/export?session_id=session-demo-1" `
  -OutFile ".\admin_agent_history.csv"
```

Export LLM audit logs:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8010/api/v1/admin/agent/llm-audit/export?success=true" `
  -OutFile ".\admin_llm_audit.csv"
```

If `ADMIN_ACCESS_TOKEN` is enabled, add the header:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8010/api/v1/admin/schedule/export" `
  -Headers @{ "X-Admin-Token" = "secret-admin-token" } `
  -OutFile ".\admin_schedules.csv"
```

## Reminder Queue and Worker

The reminder pipeline is now split into two backend roles:

- scheduler: scans due schedules and writes delivery tasks into the database queue
- worker: claims queued tasks, sends reminders, retries failures, and records alerts

This is closer to a real production backend than a single in-process timer loop, because API traffic, scheduling, and delivery execution can scale independently.

Useful endpoints for this phase:

- `GET /api/v1/metrics`
- `GET /api/v1/admin/schedule/reminder-reliability`
- `GET /api/v1/admin/schedule/delivery-tasks`
- `GET /api/v1/admin/schedule/delivery-queue`
- `POST /api/v1/admin/schedule/delivery-tasks/{task_id}/retry`
- `POST /api/v1/admin/schedule/delivery-tasks/{task_id}/unlock`

The worker now also includes stale-task recovery. If a delivery task stays in `processing` longer than `WORKER_LOCK_TIMEOUT_SECONDS`, the worker can automatically requeue it on the next polling cycle.

If Redis is enabled, it becomes the runtime queue for delivery task IDs. The database still remains the source of truth, but Redis is used for dispatch and worker throughput.

In Docker Compose, Redis is started as a dedicated container and `api`, `scheduler`, and `worker` all wait for it to become healthy before startup continues.

Relevant worker settings:

```powershell
WORKER_OWNER=local-worker
WORKER_POLL_INTERVAL_SECONDS=10
WORKER_BATCH_SIZE=20
WORKER_LOCK_TIMEOUT_SECONDS=300
REDIS_ENABLED=true
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_QUEUE_KEY=schedule:delivery:queue
```

Manual recovery examples:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/schedule/delivery-tasks/12/unlock" -Method Post | ConvertTo-Json -Depth 10
```

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/admin/schedule/delivery-tasks/12/retry" -Method Post | ConvertTo-Json -Depth 10
```

The metrics endpoint now also exposes `delivery_task_stale_processing_total`.

It also exposes:

- `redis_queue_enabled`
- `redis_queue_backlog_total`

Readiness check example:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/v1/health/ready" | ConvertTo-Json -Depth 10
```

## Multi-Turn Agent Confirmation

The agent now supports a richer confirmation-stage flow instead of only `confirm -> yes -> execute`.

Supported follow-up behaviors while a confirmation is pending:

- direct confirm, such as `确认` or `yes`
- direct cancel, such as `取消` or `cancel`
- LLM-assisted confirmation follow-up parsing for revision-style messages

This means the LLM is no longer limited to first-turn fallback parsing and reply polishing. It now also participates in pending-confirmation interpretation, which is closer to a real multi-turn agent orchestration path.

## LLM Planner

The agent also has a planner layer before fallback parsing.

- rule parser runs first
- complex inputs can be handled by the LLM planner
- parser audit now records `parse`, `repair`, `pending_confirmation`, and `plan`

So the LLM now participates in orchestration, not just fallback and wording.
