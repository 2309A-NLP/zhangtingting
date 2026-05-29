"""Knowledge pipeline package."""
'''
我的项目是单进程（单进程只能同时使用 1 个 CPU 核来跑 Python 代码，但底层 C 代码可以用其他核）。
项目主要包含 2 种任务：一是 CPU 密集型（比如：数据向量化），二是 I/O 密集型（比如：HTTP 请求、数据库读写、文件读取）。
Python 有 GIL（全局解释器锁），导致同一个时刻只有一个线程在执行 Python 字节码。
但因为我的向量化是调用的底层 C/GPU 库（不是纯 Python 写的），所以执行向量化时会主动释放 GIL。
此时 Python 解释器（可以理解为主控制权持有者）是空闲的，还能去执行其他任务（比如处理新请求）。
这样就实现了并发效果：看起来像是"多线程同时在工作"，但实际上同时只有一份 Python 代码在跑。

协程 = 一个线程里的任务分时复用  简单说就是协程就是线程里的小任务
你用 asyncio.to_thread 是把重活扔给别的工人（线程池），自己这个工人的协程清单还能继续往下走
'''

'''
api/knowledge.py (上传接口)
    │
    ├─ 创建任务，放入队列
    │
    ▼
task_queue.py (队列)
    │
    ├─ 后台消费者取出任务
    │
    ▼
ingest.py (总控制器)
    │
    ├─ 调用 loader → cleaner → chunker → embedder
    │
    ▼
Milvus + MySQL
'''