from __future__ import annotations
'''
知识库任务队列系统
这是一个异步任务队列系统，用于处理知识库文档的后台摄入（ingest）任务。将耗时操作从请求链路中剥离，异步处理。

1. 应用启动
   └── start() → 创建 4 个 worker

2. 用户上传文件
   └── enqueue() 
       ├── Redis 记录状态 "queued"
       └── 放入内存队列

3. Worker 处理
   └── _worker_loop()
       ├── 取任务
       ├── ingest_document()（解析、向量化）
       ├── 更新 Redis 状态
       └── 清理临时文件

4. 用户查询状态
   └── GET /status/{task_id}
       └── 从 Redis 获取状态
'''
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.redis_client import get_redis, ingest_status_key
from app.knowledge.ingest import KnowledgeIngestService
from app.knowledge.models import RawDocument

logger = get_logger(__name__)

# 作用：封装一个知识库摄入任务的所有信息。
@dataclass(slots=True)
class KnowledgeIngestTask:
    task_id: str                          # 任务ID
    user_id: str                          # 用户ID
    role_id: str                          # 角色ID
    role_category: str                    # 角色分类
    mode: str                             # 增量/全量
    raw_document: RawDocument             # 原始文档
    collection_name: str | None = None    # 向量集合名
    replace_doc_id: str | None = None


class KnowledgeTaskQueue:
    def __init__(self) -> None:
        '''
        变量	           类型	           作用
        _queue	    asyncio.Queue	内存队列，存放待处理任务
        _workers	list[Task]	    工作协程列表
        _running	bool	        队列是否正在运行
        '''
        self.settings = get_settings()
        # 作用：创建一个线程安全（协程安全）的异步队列。
        # 类比：像一个任务管道，一端往里放任务，另一端取任务执行。
        # 特性：
        # await queue.put(task)：如果队列满了就等待（默认无上限，不会满）
        # await queue.get()：如果队列空了就等待
        # 天然适合生产者和消费者解耦
        self._queue: asyncio.Queue[KnowledgeIngestTask] = asyncio.Queue()
        # 作用：存放所有正在运行的异步任务（每个任务通常是一个无限循环的消费者）。
        # 类比：像工厂里的一群工人，每个工人不停从队列取任务处理。
        # 为什么用列表：便于启动、停止、监控所有 worker。
        # Task 后面的 [Any] 表示：这个 Task 执行完后返回的值可以是任意类型（不限制）
        # asyncio.Task 就是把一个异步函数包装成可以独立运行、可以随时取消、可以查询状态的“任务对象”。
        self._workers: list[asyncio.Task[Any]] = []
        # 作用：控制 worker 是否应继续运行。
        self._running = False

    # 启动队列
    async def start(self) -> None:
        if self._running:
            # 防止重复启动
            return
        # 从这行之后，其他方法（比如 stop()、_worker_loop()）就知道系统已经开始运行了。
        self._running = True
        # 防止配置写错了，比如设成 0 或者负数  至少保证有 1 个 worker 在干活
        worker_count = max(1, self.settings.knowledge_task_queue_workers)
        self._workers = [
            # asyncio.create_task(...)：把 _worker_loop 这个异步函数包装成一个 Task，让它立刻在后台开始运行
            # name=f"knowledge-worker-{index}"：给这个 Task 起个名字，方便调试（比如在日志里能看到是哪个 worker）
            # 每个 worker 就是一个"无限循环"，只要 _running 是 True，它就不断从队列取任务、执行任务。
            asyncio.create_task(self._worker_loop(index), name=f"knowledge-worker-{index}")
            for index in range(worker_count)
        ]
        logger.info("knowledge_task_queue_started", worker_count=worker_count)

    # 停止队列
    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for worker in self._workers:
            # 给这个 Task 发送一个取消信号  然后立即返回，不会等待 worker 真正退出。  所以不会堵塞
            # 不会立即强制停止 Task，而是让 Task 在遇到下一个 await 时抛出 CancelledError 异常
            worker.cancel()
        if self._workers:
            # if self._workers:	如果有 worker 存在（列表不为空）
            # *self._workers	把列表解开成多个参数（[t1, t2, t3] → t1, t2, t3）
            # asyncio.gather(...)	并发等待多个 Task 完成
            # return_exceptions=True	如果有异常（比如 CancelledError），不要抛出，而是作为结果返回
            '''
            为什么需要 asyncio.gather 这一行？
            cancel() 只是发送信号，不是等待完成
            方法结束了，但 worker 可能还在后台运行（正在清理或处理善后）
            可能导致资源泄漏或状态混乱
            gather方法会真正等待所有 worker 完全退出
            确保返回时，整个系统已经处于“干净”的停止状态
                        
            为什么要 return_exceptions=True？
            因为 worker 被取消时，会抛出 CancelledError 异常
            如果不加这个参数，gather 会在遇到第一个异常时就抛出，其他 worker 可能还没处理完
            加上这个参数，gather 会等待所有 worker 都退出，然后把异常作为普通结果收集起来
            效果：这一行会阻塞，直到所有 worker 的 Task 真正结束。
            '''
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("knowledge_task_queue_stopped")

    # 把一个知识注入任务放入队列，并在 Redis 中记录这个任务的状态
    '''
    为什么需要 Redis？
    1. 状态查询
    前端可能需要知道任务处理到哪一步了
    2. 跨服务共享
    如果以后有多个服务实例（分布式部署），asyncio.Queue 只在单个进程内有效，但 Redis 是共享的，所有服务都能读到状态。
    '''
    '''
    可优化：
    1.任务存redis，关机不丢
    2.使用框架Celery
    特性	   手写Redis队列	Celery
    分布式	✅ 能实现	✅ 内置
    任务重试	❌ 自己写	✅ 装饰器配置
    定时任务	❌ 自己写	✅ built-in
    监控界面	❌ 自己写	✅ Flower
    结果存储	❌ 自己写	✅ 内置
    代码量	多	        少
    '''
    async def enqueue(self, task: KnowledgeIngestTask) -> None:
        redis = get_redis()
        key = ingest_status_key(task.user_id, task.role_id, task.task_id)
        await redis.hset(
            key,
            mapping={
                "task_id": task.task_id,
                "doc_id": task.raw_document.file_id,
                "user_id": task.user_id,
                "role_id": task.role_id,
                "status": "queued",
                "mode": task.mode,
                "source_uri": task.raw_document.source_uri,
            },
        )
        await redis.expire(key, self.settings.redis_ingest_status_ttl_seconds)
        await self._queue.put(task)
        logger.info("knowledge_task_enqueued", task_id=task.task_id, user_id=task.user_id, role_id=task.role_id)

    async def _worker_loop(self, worker_index: int) -> None:
        # 从队列中持续获取文档处理任务，执行知识库注入（向量化+存储），并确保资源被正确清理，单个任务失败不会导致整个工作器崩溃。
        logger.info("knowledge_worker_started", worker_index=worker_index)
        while True:
            # 持续从队列中获取任务
            # await：如果队列为空，暂停等待直到有新任务
            # 每个工作器独立运行这个循环
            task = await self._queue.get()
            try:
                # 创建知识库注入服务实例
                ingest_service = KnowledgeIngestService()
                # 执行文档注入
                await ingest_service.ingest_document(
                    task.raw_document,
                    role_category=task.role_category,
                    mode=task.mode,
                    collection_name=task.collection_name,
                    replace_doc_id=task.replace_doc_id,
                )
            except asyncio.CancelledError:
                # 异步操作被取消时抛出的异常  外部主动调用  eg：用户不想等了，点击了"取消上传"按钮
                # 正常流程的一部分，通常用于优雅关闭
                # 应该重新抛出：取消的任务通常需要让上层知道被取消了，让上层处理（优雅关闭）
                # 如果不重新抛出（即捕获后不 raise），会发生什么？
                    # 工作器不会停止，而是继续运行
                    # 外部调用了 cancel()，但任务依然存活
                    # 造成资源泄漏：连接、文件句柄、队列引用都没有释放
                # 既能做清理，又能让上层知道取消发生。
                raise
            except Exception as exc:
                logger.exception("knowledge_task_failed", task_id=task.task_id, error=str(exc))
            finally:
                if self.settings.knowledge_cleanup_local_after_ingest:
                    await asyncio.to_thread(_cleanup_local_file, task.raw_document.local_path)
                # 通知队列该任务已完成（用于 queue.join()）
                # 每个 get() 后必须配对 task_done()，否则 join() 会永远等待
                # 用 finally 确保 task_done() 一定执行，即使处理出错
                # task_done() 只减少计数，不返回值
                # join() 阻塞直到所有任务完成 等待计数归零
                self._queue.task_done()

# 清理本地文件
def _cleanup_local_file(local_path: str) -> None:
    try:
        # unlink() 删除文件
        Path(local_path).unlink(missing_ok=True)
    except Exception:
        logger.warning("knowledge_local_cleanup_failed", local_path=local_path)

# 单例模式：整个应用只有一个任务队列实例。 + 生命周期管理
# 这种设计确保全局只有一个队列实例，避免任务分散、资源浪费和状态不一致，同时提供统一的启动/停止机制管理队列生命周期。
_knowledge_task_queue: KnowledgeTaskQueue | None = None


def get_knowledge_task_queue() -> KnowledgeTaskQueue:
    global _knowledge_task_queue
    if _knowledge_task_queue is None:
        _knowledge_task_queue = KnowledgeTaskQueue()
    return _knowledge_task_queue


async def start_knowledge_task_queue() -> None:
    await get_knowledge_task_queue().start()


async def stop_knowledge_task_queue() -> None:
    global _knowledge_task_queue
    if _knowledge_task_queue is not None:
        await _knowledge_task_queue.stop()
        _knowledge_task_queue = None
