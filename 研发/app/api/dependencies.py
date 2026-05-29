from __future__ import annotations
'''
这是整个 API 层的"依赖工厂"，提供各种依赖注入函数，供路由函数使用。
dependencies.py 是 API 层的"工具库"：
提供数据库连接（MySQL、Redis、Milvus）
提供用户认证（JWT token + 开发模式 Header）
提供服务实例（RoleService、AuthService）
提供权限校验（require_user_match）
'''
# Python 3.9+ 的类型注解增强，可以把额外信息附加到类型上
from typing import Annotated
# Header 是 FastAPI 的工具，用于从 HTTP 请求头中提取指定的字段。
from fastapi import Depends, Header, HTTPException
# HTTPBearer：FastAPI 的安全工具，用于解析 Bearer Token（JWT）
# HTTPAuthorizationCredentials：Bearer Token 解析后的结果对象
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import decode_access_token
from app.db.milvus_client import get_milvus_client
from app.db.mysql_client import get_mysql_session
from app.db.redis_client import get_redis_client
from app.services.auth_service import AuthService
from app.services.role_service import RoleService

# HTTPBearer：FastAPI 提供的安全工具，自动从请求头中提取 Authorization: Bearer <token>
# auto_error=False：
# True（默认）：没有 token 时自动返回 401 错误
# False：没有 token 时返回 None，让代码自己处理
bearer_scheme = HTTPBearer(auto_error=False)

# 返回一个异步生成器，负责创建和关闭连接
# FastAPI 会自动管理生命周期：请求开始 → yield session → 请求结束 → 关闭连接
# 生成器的好处：FastAPI 自动管理生命周期。请求结束后，自动执行 async with 块后面的清理代码。
async def get_db_session() -> AsyncSession:
    async for session in get_mysql_session():
        yield session

# Redis 依赖 : 异步生成器，管理 Redis 连接生命周期。
async def get_redis_dependency() -> Redis:
    async for client in get_redis_client():
        yield client

# 获取当前用户 ID（核心认证逻辑）
async def get_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> str | None:
    '''
    Annotated 语法：给类型注解附加额外信息
    # 等价于
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)
    x_user_id: str | None = Header(alias="X-User-Id")
    好处：让类型和依赖写在一起，能和默认值更好地区分，代码更清晰，可复用
    '''
    settings = get_settings()

    if credentials is not None:
        try:
            # 解码 JWT token
            '''
            HTTPAuthorizationCredentials 对象的内部结构：
            class HTTPAuthorizationCredentials:
                scheme: str      # 比如 "Bearer"
                credentials: str # 实际的 token 字符串，如 "eyJhbGciOiJIUzI1NiIs..."
            '''
            payload = decode_access_token(credentials.credentials)
        except Exception as exc:
            # from exc 的作用：异常链（Exception Chaining）from exc  把"原始错误"挂在"新错误"后面，让你知道问题是怎么一步步发生的。
            # 调试时，保留完整的错误上下文
            '''
            # 没有 from exc
                try:
                    x = 1 / 0
                except Exception as exc:
                    raise ValueError("计算错误")
                # 输出：ValueError: 计算错误
                # 原始 ZeroDivisionError 丢失了
            # 有 from exc
                try:
                    x = 1 / 0
                except Exception as exc:
                    raise ValueError("计算错误") from exc
                # 输出：ValueError: 计算错误
                # 还会显示：During handling of the above exception, another exception occurred
                # 同时显示 ZeroDivisionError: division by zero
            '''
            raise HTTPException(status_code=401, detail="Invalid or expired access token.") from exc
        subject = str(payload.get("sub", "")).strip()
        # JWT（JSON Web Token）定义了一组标准字段（Claim）：
        '''
        字段	全称	                 含义
        sub	Subject	        主题——通常是用户 ID
        iss	Issuer	        签发者——谁发的 token
        exp	Expiration Time	过期时间
        iat	Issued At	    签发时间
        aud	Audience	    接收方
        
        为什么叫 sub 不叫 user_id？
            因为 JWT 是行业标准，不同系统可能有不同的标识符（用户 ID、邮箱、手机号）。标准字段 sub 提供了一个通用的名称，让所有系统都能理解。
        '''
        if not subject:
            raise HTTPException(status_code=401, detail="Invalid access token subject.")
        return subject

    # auth_enable_dev_header：开发环境配置，允许通过请求头模拟用户身份
    # 方便前端开发调试，不用每次都带 token
    if settings.auth_enable_dev_header and x_user_id:
        return x_user_id

    return None

'''
客户端（你）                    服务器
    │                              │
    │  1. 登录（用户名+密码）        │
    │ ───────────────────────────→ │
    │                              │
    │  2. 验证通过，签发 token      │
    │ ←─────────────────────────── │
    │     "eyJhbGciOiJ..."         │
    │                              │
    │  3. 存储 token                │
    │                              │
    │  4. 发请求时，带上 token       │
    │     Authorization: Bearer eyJ... │
    │ ───────────────────────────→ │
    │                              │
    │                     5. 解析 token
    │                        得到 user_id = "张三"
    │                        这是 current_user_id
    │                              │
    │  6. 请求体里还有 user_id       │
    │     {"user_id": "张三"}       │
    │                              │
    │                     7. 比较两个 ID
    
步骤	                客户端给了什么	  服务器做了什么
给 token	        给了一个加密字符串	  解密后得到 user_id
给 body_user_id	    直接给了 "张三"	  直接读取
区别：
客户端给 token，但不是直接给 user_id。服务器自己从 token 里解密出 user_id。

为什么 token 可信，body_user_id 不可信？
因为 token 是加密的，客户端无法篡改。
'''
def require_user_match(body_user_id: str, current_user_id: str | None) -> None:
    if current_user_id and current_user_id != body_user_id:
        raise HTTPException(status_code=403, detail="User mismatch.")
    '''
    payload.user_id	用户想操作的"目标用户 ID"	请求体中的 JSON 字段
    current_user_id	当前登录的"实际用户 ID"	JWT token 解析出的身份
    
    变量	                  是谁	               来自哪里	              什么时候产生的
    payload.user_id	  请求体里写的 ID	      HTTP 请求的 JSON body	    客户端发送请求时
    current_user_id	  JWT 里解析出的 ID	  Authorization 头的 token   用户登录时（服务器签发）
    '''


def get_role_service() -> RoleService:
    return RoleService()


def get_auth_service() -> AuthService:
    return AuthService()


def get_milvus_dependency():
    return get_milvus_client()
