from __future__ import annotations
'''
这是一个基于 Redis 的漏桶算法限流器，用于限制用户对特定角色的请求频率。
'''
import time
# Redis：类型注解用
from redis.asyncio import Redis

from app.core.config import get_settings
# rate_limit_key：生成限流器的 Redis key
from app.db.redis_client import get_redis, rate_limit_key

# 自定义异常，当用户超过限流阈值时抛出。
class RateLimitExceededError(RuntimeError):
    """Raised when a user exceeds the configured rate limit."""

# Redis漏桶限流器
class RedisLeakyBucketRateLimiter:
    def __init__(self, redis_client: Redis | None = None) -> None:
        self.settings = get_settings()
        self.redis = redis_client or get_redis()

    async def check(self, *, user_id: str, role_id: str) -> None:
        key = rate_limit_key(user_id, role_id)
        now = time.time()
        # 这个配置允许用户在 1 秒内突发 20 个请求，但长期平均不能超过 60 个/分钟（即 1 个/秒）。超过限制的请求会被拒绝，直到桶里的水漏出足够的空间。
        # 同一个用户同一个角色发请求，ttl会刷新，但请求数量会增加直到上限，除非用户不发请求，ttl的值才会减少
        # 限流计数器过期时间（秒），默认1分钟 / 键的过期时间（秒）
        ttl = self.settings.redis_rate_limit_ttl_seconds
        # 限流：突发流量允许的请求数 / 桶容量
        capacity = float(self.settings.rate_limit_burst)
        # rate_limit_requests_per_minute 每分钟允许的最大请求数
        # 漏水速率  计算漏水速率 = 每分钟请求数 ÷ 时间窗口（秒），让配置变化时速率自动适配。
        leak_rate = float(self.settings.rate_limit_requests_per_minute) / max(ttl, 1)
        # leak_rate = float(self.settings.rate_limit_requests_per_minute) / ttl
        '''
        请求进来，当前时间 now
            ↓
        取出上次记录：last_time（上次请求时间）, water（当时水量）
            ↓
        计算这段时间漏掉了多少：
            leaked = (now - last_time) × leak_rate
            ↓
        当前实际水量 = max(0, water - leaked)   ← 水不能漏成负数
            ↓
        当前水量 + 1（本次请求要"加水"）> capacity ?
            ├── 是 → ❌ 拒绝（桶满，突发超限）
            └── 否 → ✅ 允许
                      更新记录：water = 新水量, last_time = now
                      
        代码潜在问题
        leak_rate = float(requests_per_minute) / max(ttl, 1)
        问题：ttl 职责不单一
        职责	    说明
        职责1	Redis key 过期时间（数据清理）
        职责2	速率计算的除数（时间窗口）
        风险
        改 ttl=30 → leak_rate=2 req/s，限流变宽松
        改 ttl=1 → leak_rate=60 req/s，限流失效
        建议：解耦参数
        window = 60          # 统计窗口（固定）
        ttl = 70             # Redis过期（略大于window，防边界误差）
        leak_rate = requests_per_minute / window
        '''

        # Lua 脚本在 Redis 中执行，是原子性的。整个检查+更新操作不会被其他命令打断。
        script = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local leak_rate = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'water', 'ts')
local water = tonumber(data[1]) or 0
local ts = tonumber(data[2]) or now

local leaked = math.max(0, (now - ts) * leak_rate)
water = math.max(0, water - leaked)

if water + 1 > capacity then
  redis.call('EXPIRE', key, ttl)
  redis.call('HMSET', key, 'water', water, 'ts', now)
  return 0
end

water = water + 1
redis.call('HMSET', key, 'water', water, 'ts', now)
redis.call('EXPIRE', key, ttl)
return 1
"""
        allowed = await self.redis.eval(script, 1, key, now, capacity, leak_rate, ttl)
        if int(allowed) != 1:
            raise RateLimitExceededError("Too many requests for this user/role.")
'''
Lua 语言中，数组（表）的第一个元素索引是 1，不是 0。

local 的作用：声明局部变量
为什么要用 local？
原因1：避免命名冲突
原因2：性能更好
Lua 中访问局部变量比全局变量快。因为全局变量存在一个全局表中，需要查表；局部变量存在栈中，直接访问。
原因3：更安全
不会不小心修改了其他地方的同名变量。

tonumber()  Lua 中的转换函数，把字符串转成数字。

call 是 Redis Lua 脚本中执行 Redis 命令的函数。
在 Lua 脚本里，你不能直接写 Redis 命令，必须用 redis.call() 或 redis.pcall() 来调用。
redis.call(command, ...)
redis.call()	出错时，脚本停止执行，Redis 返回错误
redis.pcall()	出错时，脚本继续执行，返回 Lua 表包含错误信息

HMGET  在哈希表中取多个字段

注意：拒绝分支存的是原来的水量（没加 1），但更新了 ts（时间戳）到当前。
为什么？
时间在流逝，水在漏
虽然这次请求被拒绝了，但桶里的水还在继续漏
如果不更新 ts，下一次请求计算漏水量时会用旧的时间戳，导致漏水量计算错误
拒绝分支不加水但更新时间，允许分支加水并更新时间，返回值不同。

eval 是 Redis 提供的一个命令，用于在服务器端执行 Lua 脚本。
result = await self.redis.eval(script, numkeys, *args)
    script	Lua 脚本代码（字符串）
    numkeys	后面有多少个参数是 Redis 键名
    *args	键名 + 其他参数
关键特性：
原子性：Lua 脚本执行期间，其他命令不会插入
一次网络往返：所有逻辑在 Redis 内部完成
'''