from fastapi import APIRouter
#  FastAPI 路由的总入口文件  "路由注册表" 的设计模式，把所有路由集中管理
from app.api.demo import router as demo_router
from app.api.v1.routes.admin import router as admin_router
from app.api.v1.routes.agent import router as agent_router
from app.api.v1.routes.dashboard import router as dashboard_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.metrics import router as metrics_router
from app.api.v1.routes.schedule import router as schedule_router
from app.api.v1.routes.scheduler_audit import router as scheduler_audit_router

api_router = APIRouter()
api_router.include_router(demo_router, tags=["demo"])
api_router.include_router(admin_router)
api_router.include_router(scheduler_audit_router)
api_router.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(health_router, tags=["health"])
api_router.include_router(metrics_router, tags=["metrics"])
api_router.include_router(agent_router, prefix="/agent", tags=["agent"])
api_router.include_router(schedule_router, prefix="/schedule", tags=["schedule"])
'''
api_router.py
├── 导入所有子路由
├── 创建总路由器 (APIRouter)
└── 注册所有子路由（带不同前缀和标签）
    ├── /demo          → demo_router
    ├── /admin         → admin_router
    ├── /scheduler-audit → scheduler_audit_router
    ├── /dashboard     → dashboard_router
    ├── /health        → health_router
    ├── /metrics       → metrics_router
    ├── /agent         → agent_router
    └── /schedule      → schedule_router
'''