"""认证核心模块

提供 JWT 令牌签发/验证、密码哈希、FastAPI 依赖注入等认证基础设施。
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from app.config import config

# Bearer Token 提取
security = HTTPBearer()


def hash_password(password: str) -> str:
    """
    对明文密码进行 bcrypt 哈希

    Args:
        password: 明文密码

    Returns:
        str: 哈希后的密码字符串
    """
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证明文密码是否与哈希值匹配

    Args:
        plain_password: 明文密码
        hashed_password: 哈希后的密码

    Returns:
        bool: 匹配返回 True
    """
    password_bytes = plain_password.encode("utf-8")
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def create_access_token(user_id: int, username: str) -> str:
    """
    签发 Access Token（JWT）

    Args:
        user_id: 用户 ID
        username: 用户名

    Returns:
        str: JWT 字符串
    """
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=config.access_token_expire_minutes
    )
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    return jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)


def create_refresh_token() -> str:
    """
    生成 Refresh Token（UUID）

    Returns:
        str: UUID 字符串作为 refresh token
    """
    import uuid

    return str(uuid.uuid4())


def verify_access_token(token: str) -> Optional[dict]:
    """
    验证并解析 Access Token

    Args:
        token: JWT 字符串

    Returns:
        dict: 解码后的 payload，验证失败返回 None
    """
    try:
        payload = jwt.decode(
            token, config.jwt_secret_key, algorithms=[config.jwt_algorithm]
        )

        # 验证 token 类型
        if payload.get("type") != "access":
            logger.warning(f"Token 类型不匹配: {payload.get('type')}")
            return None

        # 验证必要字段
        if "sub" not in payload or "username" not in payload:
            logger.warning("Token 缺少必要字段")
            return None

        return payload

    except jwt.ExpiredSignatureError:
        logger.debug("Access Token 已过期")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"无效的 Access Token: {e}")
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> "User":
    """
    FastAPI 依赖注入：从 Authorization header 中提取并验证 JWT，返回当前用户

    Args:
        credentials: HTTP Bearer 凭证

    Returns:
        User: 当前登录的用户对象

    Raises:
        HTTPException: 认证失败时抛出 401
    """
    from app.services.auth_service import auth_service

    token = credentials.credentials
    payload = verify_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = int(payload["sub"])
    user = await auth_service.get_user_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用",
        )

    return user


# 类型提示用（避免循环导入）
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.user import User  # noqa: F401
