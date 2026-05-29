"""Application services."""
# services/ 文件夹是 API 层和数据库层之间的"业务逻辑层"（Service Layer）。
'''
services 文件夹的作用？	业务逻辑层，连接 API 和数据库
为什么需要？	解耦、复用、可测试、集中管理

API 层 (api/*.py)
    │
    ├─ 只负责：接收请求、参数验证、返回响应
    │
    ▼
Service 层 (services/*.py)  ← 你在这里
    │
    ├─ 负责：业务逻辑、权限校验、数据组装、事务协调
    │
    ▼
数据库层 (db/*.py) / 第三方服务
    │
    └─ 负责：实际的 CRUD 操作
'''