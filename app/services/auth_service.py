"""认证服务

负责用户账号的 CRUD 操作和认证逻辑。
"""

from datetime import timedelta
from typing import Optional

from loguru import logger
from sqlalchemy import select, func

from app.config import config
from app.core.auth import hash_password, verify_password
from app.core.mysql_client import mysql_manager
from app.core.redis_client import redis_manager
from app.models.user import User


class AuthService:
    """认证服务"""

    async def init_db(self) -> None:
        """初始化数据库表并创建初始管理员"""
        from app.models.user_profile import Base

        engine = await mysql_manager.get_engine()
        async with engine.begin() as conn:
            # 创建 users 表（如果不存在）
            await conn.run_sync(Base.metadata.create_all)

        logger.info("用户认证表初始化完成")

        # 创建初始管理员（如果不存在）
        await self._ensure_initial_admin()

    async def _ensure_initial_admin(self) -> None:
        """确保初始管理员存在"""
        admin_username = config.initial_admin_username
        admin_password = config.initial_admin_password

        if not admin_password:
            logger.warning("未配置 INITIAL_ADMIN_PASSWORD，跳过创建初始管理员")
            return

        async with mysql_manager.get_session() as session:
            result = await session.execute(
                select(User).where(User.username == admin_username)
            )
            existing_admin = result.scalar_one_or_none()

            if existing_admin is None:
                # 创建初始管理员
                admin_user = User(
                    username=admin_username,
                    hashed_password=hash_password(admin_password),
                    display_name="系统管理员",
                    is_active=True,
                    is_admin=True,
                )
                session.add(admin_user)
                logger.info(f"已创建初始管理员: {admin_username}")
            else:
                logger.debug(f"初始管理员已存在: {admin_username}")

    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        验证用户登录

        Args:
            username: 用户名
            password: 明文密码

        Returns:
            User: 验证成功返回用户对象，失败返回 None
        """
        async with mysql_manager.get_session() as session:
            result = await session.execute(
                select(User).where(User.username == username)
            )
            user = result.scalar_one_or_none()

            if user is None:
                logger.info(f"登录失败: 用户不存在 - {username}")
                return None

            if not user.is_active:
                logger.info(f"登录失败: 用户已禁用 - {username}")
                return None

            if not verify_password(password, user.hashed_password):
                logger.info(f"登录失败: 密码错误 - {username}")
                return None

            logger.info(f"用户登录成功: {username}")
            return user

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        根据 ID 获取用户

        Args:
            user_id: 用户 ID

        Returns:
            User: 用户对象或 None
        """
        async with mysql_manager.get_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        根据用户名获取用户

        Args:
            username: 用户名

        Returns:
            User: 用户对象或 None
        """
        async with mysql_manager.get_session() as session:
            result = await session.execute(
                select(User).where(User.username == username)
            )
            return result.scalar_one_or_none()

    async def create_user(
        self,
        username: str,
        password: str,
        display_name: str = "",
        is_admin: bool = False,
    ) -> User:
        """
        创建新用户

        Args:
            username: 用户名
            password: 明文密码
            display_name: 显示名称
            is_admin: 是否管理员

        Returns:
            User: 创建的用户对象

        Raises:
            ValueError: 用户名已存在
        """
        # 检查用户名是否已存在
        existing = await self.get_user_by_username(username)
        if existing is not None:
            raise ValueError(f"用户名 '{username}' 已存在")

        async with mysql_manager.get_session() as session:
            user = User(
                username=username,
                hashed_password=hash_password(password),
                display_name=display_name or username,
                is_active=True,
                is_admin=is_admin,
            )
            session.add(user)
            await session.flush()

            logger.info(f"创建新用户: {username} (admin={is_admin})")
            return user

    async def update_user(
        self,
        user_id: int,
        display_name: Optional[str] = None,
        is_active: Optional[bool] = None,
        is_admin: Optional[bool] = None,
        password: Optional[str] = None,
    ) -> Optional[User]:
        """
        更新用户信息

        Args:
            user_id: 用户 ID
            display_name: 显示名称
            is_active: 是否启用
            is_admin: 是否管理员
            password: 新密码

        Returns:
            User: 更新后的用户对象或 None
        """
        async with mysql_manager.get_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()

            if user is None:
                return None

            if display_name is not None:
                user.display_name = display_name
            if is_active is not None:
                user.is_active = is_active
            if is_admin is not None:
                user.is_admin = is_admin
            if password is not None:
                user.hashed_password = hash_password(password)

            await session.flush()
            logger.info(f"更新用户: {user.username} (id={user_id})")
            return user

    async def delete_user(self, user_id: int) -> bool:
        """
        删除用户

        Args:
            user_id: 用户 ID

        Returns:
            bool: 是否删除成功
        """
        async with mysql_manager.get_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()

            if user is None:
                return False

            username = user.username
            await session.delete(user)
            await session.flush()
            logger.info(f"删除用户: {username} (id={user_id})")
            return True

    async def list_users(
        self, offset: int = 0, limit: int = 50
    ) -> tuple[list[User], int]:
        """
        获取用户列表

        Args:
            offset: 偏移量
            limit: 每页数量

        Returns:
            tuple: (用户列表, 总数)
        """
        async with mysql_manager.get_session() as session:
            # 获取总数
            count_result = await session.execute(
                select(func.count()).select_from(User)
            )
            total = count_result.scalar() or 0

            # 获取用户列表
            result = await session.execute(
                select(User).order_by(User.id).offset(offset).limit(limit)
            )
            users = list(result.scalars().all())

            return users, total

    # ========== Refresh Token 管理 ==========

    async def store_refresh_token(
        self, refresh_token: str, user_id: int
    ) -> None:
        """
        存储 Refresh Token 到 Redis

        Args:
            refresh_token: Refresh Token（UUID）
            user_id: 用户 ID
        """
        redis_client = await redis_manager.get_client()
        key = f"refresh_token:{refresh_token}"
        ttl = timedelta(days=config.refresh_token_expire_days)

        await redis_client.set(key, str(user_id), ex=ttl)
        logger.debug(f"存储 refresh_token: user_id={user_id}, ttl={ttl}")

    async def verify_refresh_token(self, refresh_token: str) -> Optional[int]:
        """
        验证 Refresh Token

        Args:
            refresh_token: Refresh Token

        Returns:
            int: 用户 ID，验证失败返回 None
        """
        redis_client = await redis_manager.get_client()
        key = f"refresh_token:{refresh_token}"
        user_id_str = await redis_client.get(key)

        if user_id_str is None:
            logger.info("Refresh Token 无效或已过期")
            return None

        return int(user_id_str)

    async def revoke_refresh_token(self, refresh_token: str) -> bool:
        """
        吊销 Refresh Token（登出）

        Args:
            refresh_token: Refresh Token

        Returns:
            bool: 是否成功吊销
        """
        redis_client = await redis_manager.get_client()
        key = f"refresh_token:{refresh_token}"
        deleted = await redis_client.delete(key)

        if deleted:
            logger.info(f"已吊销 refresh_token: {refresh_token[:8]}...")
        return deleted > 0


# 全局单例
auth_service = AuthService()
