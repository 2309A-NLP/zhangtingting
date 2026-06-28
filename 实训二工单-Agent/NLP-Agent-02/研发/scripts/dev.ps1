param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "migrate", "lint", "typecheck", "test", "check", "precommit", "hooks", "run", "run-ai-debug", "run-scheduler", "run-worker", "compose-up", "compose-down", "compose-logs", "smoke-local", "smoke-docker")]
    [string]$Task = "check",
    [int]$Port = 8010
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

function Get-DotEnvValue {
    param(
        [string]$Key
    )

    if (-not (Test-Path ".env")) {
        return $null
    }

    $match = Get-Content ".env" | Where-Object { $_ -match "^$Key=(.*)$" } | Select-Object -First 1
    if (-not $match) {
        return $null
    }

    return ($match -replace "^$Key=", "").Trim()
}

function Get-AdminHeaders {
    $token = $env:ADMIN_ACCESS_TOKEN
    if (-not $token) {
        $token = Get-DotEnvValue -Key "ADMIN_ACCESS_TOKEN"
    }
    if ($token) {
        return @{ "X-Admin-Token" = $token }
    }
    return @{}
}

# 命令分发器
switch ($Task) {
    "install" {
        # 安装项目依赖
        # .：从 pyproject.toml 中读取[project]下的依赖进行安装
        # [dev]：从 pyproject.toml 中读取 [project.optional-dependencies] 下的 dev 组
        python -m pip install -e ".[dev]"
        # -m 告诉 Python："把这个文件当作模块来运行"
    }
    "migrate" {
        # 数据库迁移
        python -m alembic upgrade head
    }
    "lint" {
        # 代码风格检查 ruff：极快的 Python linter（替代 Flake8、isort 等）  检查代码风格问题。
        python -m ruff check src tests
    }
    "typecheck" {
        # 类型检查 mypy：Python 静态类型检查器   检查类型注解是否正确
        python -m mypy src
    }
    "test" {
        # 运行测试 pytest：Python 测试框架  -q：quiet 模式，减少输出信息  快速运行测试套件
        python -m pytest -q
    }
    "check" {
        # 组合任务  作用：CI/CD 中常用的"一键全检"。
        # CI 保证代码质量（自动检查），CD 保证快速交付（自动部署）。你的 "check" 任务就是 CI 流水线中的核心检查步骤。
        # 记住：CI = 自动检查，CD = 自动发布

        # CI = 持续集成 (Continuous Integration)
        # 核心思想：开发人员频繁地把代码合并到主干，每次合并都自动运行测试和检查。
        # CD = 持续交付/部署 (Continuous Delivery/Deployment)
        # 两种理解：
        # 概念	含义	区别
        # 持续交付	代码随时准备好可以部署到生产环境，但需要手动确认才真正上线	安全、可控
        # 持续部署	代码通过所有检查后自动部署到生产环境，无需人工干预	完全自动化
        # 工具	类型	特点
        # GitHub Actions	CI/CD	GitHub 原生集成，免费
        # GitLab CI	CI/CD	GitLab 自带，一体化
        # Jenkins	CI/CD	老牌，开源，高度可定制
        # CircleCI	CI/CD	云端，配置简单
        # Travis CI	CI	开源项目免费
        # ArgoCD	CD	Kubernetes 原生持续部署
        python -m ruff check src tests
        python -m mypy src
        python -m pytest -q
    }
    "precommit" {
        # 运行 pre-commit 钩子  一个统一管理代码检查的小框架
        # pre_commit：Git pre-commit 钩子管理工具
        # run --all-files：对所有文件运行钩子（不只是暂存的文件）
        # 作用：手动运行所有 pre-commit 检查（格式化、lint 等）。
        python -m pre_commit run --all-files
    }
    "hooks" {
        # 安装 Git 钩子
        # pre_commit install：安装 pre-commit 钩子（在 git commit 时触发）
        # pre_commit install --hook-type pre-push：安装 pre-push 钩子（在 git push 时触发）
        # 作用：配置 Git 钩子，确保代码质量在提交前被检查。
        python -m pre_commit install
        python -m pre_commit install --hook-type pre-push
    }
    "run" {
        # 启动 API 服务
        # $env:APP_ROLE = "api"：设置环境变量，告诉应用当前角色是 API 服务
        # uvicorn：ASGI 服务器，运行 FastAPI 应用
        # app.main:app：从 app/main.py 导入 app 对象
        # --reload：开发模式，代码变更自动重启
        # --port $Port：端口号由变量 $Port 指定（如 8000）
        $env:APP_ROLE = "api"
        python -m uvicorn app.main:app --reload --port $Port
    }
    "run-ai-debug" {
        # AI 调试模式
        # 多设置了 $env:LLM_DEBUG_LOGGING = "true"
        # 启用大语言模型（LLM）相关的详细调试     
        # 作用：调试 AI 相关功能时使用。
        $env:APP_ROLE = "api"
        $env:LLM_DEBUG_LOGGING = "true"
        python -m uvicorn app.main:app --reload --port $Port
    }
    "run-scheduler" {
        # 启动调度器
        # 启动任务调度服务（如定时任务、Celery beat 等）
        # Celery 是一个分布式任务队列
        $env:APP_ROLE = "scheduler"
        python -m uvicorn app.main:scheduler_app --port ($Port + 1)
    }
    "run-worker" {
        # 启动工作进程
        # 作用：启动后台任务工作进程（如 Celery worker）
        $env:APP_ROLE = "worker"
        python -m uvicorn app.main:worker_app --port ($Port + 2)
    }
    "compose-up" {
        # 启动容器
        # --build：重新构建镜像
        # -d：detach 模式，后台运行
        docker compose up --build -d
    }
    "compose-down" {
        # 停止容器
        docker compose down
    }
    "compose-logs" {
        # 查看日志
        docker compose logs --tail=120
    }
    "smoke-local" {
        # 本地冒烟测试  测试本地服务的关键端点是否正常工作。
        # 冒烟测试是软件测试中的一种快速、基础的功能验证，用来检查系统是否"活"着，核心功能是否正常。
        # Invoke-RestMethod  发送 HTTP 请求并自动解析 JSON 响应
        # ConvertTo-Json -Depth 10：将响应美化成可读的 JSON  格式化输出 ，不是解析！
        # -Depth 10 作用：指定 JSON 序列化的最大嵌套深度  大多数 API 响应不会超过 10 层嵌套   默认深度是 2
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v1/health" | ConvertTo-Json -Depth 10
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v1/health/ready" | ConvertTo-Json -Depth 10
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v1/metrics"
        # 查看调度队列中的配送任务
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/v1/admin/schedule/delivery-queue" -Headers (Get-AdminHeaders) | ConvertTo-Json -Depth 10
    }
    "smoke-docker" {
        # Docker 容器冒烟测试  端口固定为 8000（Docker 映射端口）
        # 多测试了一个端点：/admin/scheduler/leases/scan_due_reminders
        Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" | ConvertTo-Json -Depth 10
        Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health/ready" | ConvertTo-Json -Depth 10
        Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/admin/schedule/delivery-queue" -Headers (Get-AdminHeaders) | ConvertTo-Json -Depth 10
        # 触发调度器扫描到期的提醒任务
        Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/admin/scheduler/leases/scan_due_reminders" -Headers (Get-AdminHeaders) | ConvertTo-Json -Depth 10
    }
}
