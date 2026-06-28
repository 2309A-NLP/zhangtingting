'''
1. admin.py 负责什么
[admin.py](D:/Desktop/NLP-Agent-02/src/app/api/v1/routes/admin.py) 主要是给“后台管理/排障/审计”用的。
它面向的不是普通用户，而是：
开发者
运维
管理后台
调试页面
所以它负责：
后台汇总数据
日程统计
提醒日志/告警日志/投递任务查看
agent 会话历史查看
LLM 审计日志查看
管理员访问审计
CSV 导出
人工重试/解锁任务
一句话：
admin = 后台管理面
2. schedule.py 负责什么
[schedule.py](D:/Desktop/NLP-Agent-02/src/app/api/v1/routes/schedule.py) 负责“日程这个核心业务对象”的标准业务接口。
它更像传统 REST API，主要做：
创建日程
列表查询
单条查询
更新日程
删除日程
日程摘要统计
提醒日志查询
告警日志查询
可靠性统计
它面向的是：
前端业务页面
普通调用方
其他系统直接操作日程
一句话：
schedule = 日程业务主接口
3. agent.py 负责什么
[agent.py](D:/Desktop/NLP-Agent-02/src/app/api/v1/routes/agent.py) 负责“自然语言智能体交互”。
它不是直接操作数据库资源，而是处理：
用户说一句自然语言
系统理解意图
进入确认态/回复态
调用工具执行
记录会话和历史
它主要做：
POST /agent/chat
POST /agent/execute
会话列表
会话历史
LLM 审计查询
它面向的是：
聊天式交互
agent 调用入口
智能体前端页面
一句话：
agent = 自然语言交互入口
4. 其他路由文件分别负责什么
[health.py](D:/Desktop/NLP-Agent-02/src/app/api/v1/routes/health.py)
负责健康检查。
用途：
服务活着没
数据库通没通
Redis 通没通
readiness 检查
一句话：
health = 系统体检接口
[metrics.py](D:/Desktop/NLP-Agent-02/src/app/api/v1/routes/metrics.py)
负责暴露 Prometheus 指标。
用途：
给 Prometheus 抓取监控数据
一句话：
metrics = 监控指标出口
[dashboard.py](D:/Desktop/NLP-Agent-02/src/app/api/v1/routes/dashboard.py)
负责仪表盘数据汇总。
用途：
给前端 dashboard 页面提供总览数据
一句话：
dashboard = 前台/中台总览数据接口
[scheduler_audit.py](D:/Desktop/NLP-Agent-02/src/app/api/v1/routes/scheduler_audit.py)
负责调度器审计和锁状态查询。
用途：
看定时任务执行记录
看 scheduler lease/lock 状态
辅助排查定时任务是否重复跑、是否卡住
一句话：
scheduler_audit = 定时任务运维观察接口
[demo.py](D:/Desktop/NLP-Agent-02/src/app/api/demo.py)
负责演示页面。
用途：
提供浏览器里的 demo / 调试页面
方便你本地直接点点看，不用全靠 Postman/PowerShell
一句话：
demo = 演示和调试页面入口
最简单的整体理解
你可以这样记：
schedule：管“日程”这个业务对象
agent：管“自然语言对话和智能体流程”
admin：管“后台管理、排障、审计、导出”
health：管“服务健不健康”
metrics：管“监控数据”
dashboard：管“总览面板”
scheduler_audit：管“定时任务执行观察”
demo：管“本地演示页面”
'''