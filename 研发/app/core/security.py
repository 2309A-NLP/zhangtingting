from __future__ import annotations
# UTC	UTC 时区（避免时区混乱）
# datetime	处理 token 过期时间
# timedelta	计算未来时间（如 30 分钟后）
# bcrypt	密码加密库
# jwt	JWT token 编解码库
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.core.config import get_settings

UTC = timezone.utc

# 密码加密
def hash_password(password: str) -> str:
    """
    使用bcrypt算法对密码进行哈希加密

    Args:
        password: 明文密码
    Returns:
        加密后的密码哈希字符串
    Note:
        使用bcrypt.gensalt(rounds=12)生成盐值，rounds=12表示计算轮次
        轮次越高越安全但计算越慢，12是推荐的安全值
    """
    # bcrypt 只能处理 bytes，不能处理 str
    password_bytes = password.encode("utf-8")
    # salt = bcrypt.gensalt(rounds=12) # 随机生成盐值，如 b"$2b$12$abcdef..."
    # bcrypt.hashpw(password_bytes, salt)  # 加密
    # .decode("utf-8") 字节 → 字符串（方便存数据库）
    '''
    特点
    1. 同样的密码，每次加密结果不同（因为盐值随机）
    2. 包含盐值，不需要单独存储
    3. rounds=12 表示计算 2^12=4096 轮，安全且够快
    '''
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt(rounds=12)).decode("utf-8")

# 密码验证
def verify_password(password: str, password_hash: str) -> bool:
    """
    验证明文密码是否与哈希值匹配
    Args:
        password: 待验证的明文密码
        password_hash: 存储的密码哈希值
    Returns:
        密码是否匹配
    Note:
        bcrypt.checkpw会自动从hash中提取盐值进行验证
        捕获ValueError异常处理无效的hash格式
    """
    try:
        '''
        bcrypt.checkpw 会自动：
        1. 从 stored_hash 中提取盐值
        2. 用盐值对 user_input 进行加密
        3. 比较结果是否与 stored_hash 相同
        '''
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        '''
        为什么捕获 ValueError？
        如果 stored_hash 格式无效（比如数据损坏）
        bcrypt.checkpw 会抛出 ValueError
        此时返回 False（密码不匹配），而不是让程序崩溃
        '''
        return False

# 创建 JWT token
def create_access_token(*, subject: str, extra_claims: dict[str, Any] | None = None) -> str:
    """
    创建JWT访问令牌
    Args:
        subject: 令牌主题，通常是用户ID
        extra_claims: 额外的声明信息，如角色、权限等
    Returns:
        JWT令牌字符串
    Note:
        令牌包含标准声明：
        - sub(Subject): 令牌主题
        - iat(Issued At): 签发时间戳
        - exp(Expiration): 过期时间戳
        - type: 令牌类型标识
    """
    settings = get_settings()
    # 计算过期时长
    # timedelta 是 Python 的时间差对象
    # timedelta(minutes=30) 意思是："我想表达 30 分钟 这个时间长度"。
    expires_delta = timedelta(minutes=settings.app_access_token_expire_minutes)
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": subject,
        # iat 和 exp 字段必须使用 NumericDate 格式，也就是 Unix 时间戳（从 1970-01-01 到现在的秒数）。
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "type": "access",
    }
    if extra_claims:
        # dict.update()：合并字典 会覆盖相同的键
        payload.update(extra_claims)
    # 把 payload 转成 JSON 字符串
    # 用密钥和算法签名（防止篡改）
    # 把三部分拼成一个 token
    return jwt.encode(payload, settings.app_secret_key, algorithm=settings.auth_jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """
    解码并验证JWT访问令牌
    Args:
        token: JWT令牌字符串
    Returns:
        解码后的令牌payload字典
    Note:
        使用配置的密钥和算法进行验证
        自动检查令牌签名和过期时间
    """
    '''
    一、JWT 是什么？
    JWT（JSON Web Token）是一个加密的字符串，里面包含着用户信息。
    JWT 的组成（三部分，用点分隔）
    
    eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyXzEyMyIsImV4cCI6MTczNTY4OTYwMH0.signature
    │                                    │                              │
    │                                    │                              │
    │         Header（头部）              Payload（载荷）                 Signature（签名）
    │    {"alg":"HS256",                {"sub":"user_123",              加密签名
    │     "typ":"JWT"}                   "exp":1735689600}
    
    部分	         内容	                作用
    Header	    加密算法、类型	    告诉服务器怎么解密
    Payload	    用户ID、过期时间等	    真正要传递的数据
    Signature	签名	                 防止篡改
    
    二、代码逐行解析
    def decode_access_token(token: str) -> dict[str, Any]:
    输入：加密的 JWT 字符串（如 "eyJhbGciOiJIUzI1NiIs..."）
    输出：解密后的字典（如 {"sub": "user_123", "exp": 1735689600}）
    settings = get_settings()
    获取配置，里面包含：
    app_secret_key：加密密钥（只有服务器知道，用来验证签名）
    auth_jwt_algorithm：加密算法（如 "HS256"）
    return jwt.decode(token, settings.app_secret_key, algorithms=[settings.auth_jwt_algorithm])
    jwt.decode() 做了三件事：
    验证签名：确保 token 没有被篡改
    检查过期时间：如果 token 过期了，抛出异常
    解密返回：成功则返回 payload 字典
    '''
    settings = get_settings()
    return jwt.decode(token, settings.app_secret_key, algorithms=[settings.auth_jwt_algorithm])
