from __future__ import annotations
'''
AuthService 是一个用户认证服务，提供注册（检查重名+加密存密码）、
登录（验证密码+生成JWT）、查询用户的功能，所有数据库操作用原始 SQL，安全且高效。
'''
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password

logger = get_logger(__name__)


@dataclass(slots=True)
class AuthenticatedUser:
    user_id: str
    username: str
    email: str | None


class AuthService:
    async def register_user(
        self,
        db_session: AsyncSession,
        *,
        username: str,
        password: str,
        email: str | None = None,
    ) -> AuthenticatedUser:
        exists_stmt = text("SELECT id FROM users WHERE username = :username LIMIT 1")
        exists = await db_session.execute(exists_stmt, {"username": username})
        if exists.scalar() is not None:
            raise ValueError("Username already exists.")

        user_id = uuid.uuid4().hex
        password_hash = hash_password(password)
        insert_stmt = text(
            """
            INSERT INTO users (id, username, password_hash, email, created_at, updated_at)
            VALUES (:id, :username, :password_hash, :email, NOW(), NOW())
            """
        )
        await db_session.execute(
            insert_stmt,
            {
                "id": user_id,
                "username": username,
                "password_hash": password_hash,
                "email": email,
            },
        )
        await db_session.commit()
        logger.info("user_registered", user_id=user_id, username=username)
        return AuthenticatedUser(user_id=user_id, username=username, email=email)

    # 验证用户
    async def authenticate_user(
        self,
        db_session: AsyncSession,
        *,
        username: str,
        password: str,
    ) -> tuple[AuthenticatedUser, str]:
        stmt = text(
            """
            SELECT id, username, password_hash, email
            FROM users
            WHERE username = :username
            LIMIT 1
            """
        )
        result = await db_session.execute(stmt, {"username": username})
        row = result.mappings().first()
        if row is None:
            raise ValueError("Invalid username or password.")

        password_hash = str(row["password_hash"])
        is_valid = verify_password(password, password_hash)
        if not is_valid:
            logger.warning(
                "user_authentication_failed",
                username=username,
                reason="invalid_password_or_hash",
            )
            raise ValueError("Invalid username or password.")

        user = AuthenticatedUser(
            user_id=str(row["id"]),
            username=str(row["username"]),
            email=row["email"],
        )
        token = create_access_token(subject=user.user_id, extra_claims={"username": user.username})
        return user, token

    async def get_user_by_id(self, db_session: AsyncSession, *, user_id: str) -> AuthenticatedUser | None:
        stmt = text(
            """
            SELECT id, username, email
            FROM users
            WHERE id = :user_id
            LIMIT 1
            """
        )
        result = await db_session.execute(stmt, {"user_id": user_id})
        row = result.mappings().first()
        if row is None:
            return None
        return AuthenticatedUser(
            user_id=str(row["id"]),
            username=str(row["username"]),
            email=row["email"],
        )
