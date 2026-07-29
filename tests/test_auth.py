"""认证模块单元测试

覆盖：密码哈希/验证、JWT 签发/解析、认证服务逻辑。
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import jwt

from app.config import config
from app.core.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_access_token,
    get_current_user,
)


# ========== 密码哈希测试 ==========


class TestPasswordHashing:
    """密码哈希/验证测试"""

    def test_hash_password_returns_string(self):
        """hash_password 应返回字符串"""
        result = hash_password("test_password")
        assert isinstance(result, str)
        assert result != "test_password"

    def test_hash_password_different_each_time(self):
        """相同密码应产生不同哈希（自动加盐）"""
        h1 = hash_password("same_password")
        h2 = hash_password("same_password")
        assert h1 != h2

    def test_verify_password_correct(self):
        """正确密码应验证通过"""
        hashed = hash_password("my_secret")
        assert verify_password("my_secret", hashed) is True

    def test_verify_password_incorrect(self):
        """错误密码应验证失败"""
        hashed = hash_password("my_secret")
        assert verify_password("wrong_password", hashed) is False

    def test_verify_password_empty(self):
        """空密码应验证失败"""
        hashed = hash_password("my_secret")
        assert verify_password("", hashed) is False


# ========== JWT 令牌测试 ==========


class TestJWTTokens:
    """JWT 令牌签发/验证测试"""

    def test_create_access_token_returns_string(self):
        """create_access_token 应返回 JWT 字符串"""
        token = create_access_token(user_id=1, username="testuser")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_contains_correct_payload(self):
        """JWT payload 应包含正确的 sub、username、type 字段"""
        token = create_access_token(user_id=42, username="alice")
        payload = jwt.decode(
            token, config.jwt_secret_key, algorithms=[config.jwt_algorithm]
        )
        assert payload["sub"] == "42"
        assert payload["username"] == "alice"
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload

    def test_verify_access_token_valid(self):
        """有效 token 应成功解析"""
        token = create_access_token(user_id=1, username="testuser")
        payload = verify_access_token(token)
        assert payload is not None
        assert payload["sub"] == "1"
        assert payload["username"] == "testuser"

    def test_verify_access_token_expired(self):
        """过期 token 应返回 None"""
        # 手动创建一个已过期的 token
        expire = datetime.now(timezone.utc) - timedelta(minutes=1)
        payload = {
            "sub": "1",
            "username": "testuser",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access",
        }
        token = jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)
        result = verify_access_token(token)
        assert result is None

    def test_verify_access_token_invalid_signature(self):
        """无效签名的 token 应返回 None"""
        payload = {
            "sub": "1",
            "username": "testuser",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
            "type": "access",
        }
        token = jwt.encode(payload, "wrong-secret", algorithm="HS256")
        result = verify_access_token(token)
        assert result is None

    def test_verify_access_token_wrong_type(self):
        """非 access 类型的 token 应返回 None"""
        expire = datetime.now(timezone.utc) + timedelta(minutes=30)
        payload = {
            "sub": "1",
            "username": "testuser",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",  # 错误的类型
        }
        token = jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)
        result = verify_access_token(token)
        assert result is None

    def test_verify_access_token_missing_fields(self):
        """缺少必要字段的 token 应返回 None"""
        expire = datetime.now(timezone.utc) + timedelta(minutes=30)
        payload = {
            "sub": "1",
            # 缺少 username
            "exp": expire,
            "type": "access",
        }
        token = jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)
        result = verify_access_token(token)
        assert result is None

    def test_create_refresh_token_is_uuid(self):
        """refresh_token 应为 UUID 格式字符串"""
        token = create_refresh_token()
        assert isinstance(token, str)
        # UUID 格式：8-4-4-4-12
        parts = token.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[4]) == 12


# ========== 认证服务测试（Mock 数据库） ==========


class TestAuthService:
    """认证服务逻辑测试（Mock 数据库和 Redis）"""

    @pytest.fixture
    def mock_user(self):
        """模拟用户对象"""
        user = MagicMock()
        user.id = 1
        user.username = "testuser"
        user.hashed_password = hash_password("password123")
        user.display_name = "测试用户"
        user.is_active = True
        user.is_admin = False
        user.created_at = datetime.now()
        return user

    @pytest.fixture
    def mock_admin_user(self, mock_user):
        """模拟管理员对象"""
        mock_user.is_admin = True
        mock_user.username = "admin"
        return mock_user

    @pytest.mark.asyncio
    async def test_authenticate_success(self, mock_user):
        """正确凭据应认证成功"""
        from app.services.auth_service import AuthService

        service = AuthService()

        with patch("app.services.auth_service.mysql_manager") as mock_mysql:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_user
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_mysql.get_session.return_value = mock_session

            result = await service.authenticate("testuser", "password123")
            assert result is not None
            assert result.username == "testuser"

    @pytest.mark.asyncio
    async def test_authenticate_wrong_password(self, mock_user):
        """错误密码应返回 None"""
        from app.services.auth_service import AuthService

        service = AuthService()

        with patch("app.services.auth_service.mysql_manager") as mock_mysql:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_user
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_mysql.get_session.return_value = mock_session

            result = await service.authenticate("testuser", "wrong_password")
            assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self):
        """用户不存在应返回 None"""
        from app.services.auth_service import AuthService

        service = AuthService()

        with patch("app.services.auth_service.mysql_manager") as mock_mysql:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_mysql.get_session.return_value = mock_session

            result = await service.authenticate("nonexistent", "password")
            assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_disabled_user(self, mock_user):
        """已禁用用户应返回 None"""
        from app.services.auth_service import AuthService

        mock_user.is_active = False
        service = AuthService()

        with patch("app.services.auth_service.mysql_manager") as mock_mysql:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_user
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_mysql.get_session.return_value = mock_session

            result = await service.authenticate("testuser", "password123")
            assert result is None

    @pytest.mark.asyncio
    async def test_store_and_verify_refresh_token(self):
        """refresh_token 存储和验证应正常工作"""
        from app.services.auth_service import AuthService

        service = AuthService()

        with patch("app.services.auth_service.redis_manager") as mock_redis:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value="1")
            mock_client.set = AsyncMock()
            mock_redis.get_client = AsyncMock(return_value=mock_client)

            # 存储
            await service.store_refresh_token("test-uuid", user_id=1)
            mock_client.set.assert_called_once()

            # 验证
            result = await service.verify_refresh_token("test-uuid")
            assert result == 1

    @pytest.mark.asyncio
    async def test_verify_invalid_refresh_token(self):
        """无效的 refresh_token 应返回 None"""
        from app.services.auth_service import AuthService

        service = AuthService()

        with patch("app.services.auth_service.redis_manager") as mock_redis:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=None)
            mock_redis.get_client = AsyncMock(return_value=mock_client)

            result = await service.verify_refresh_token("invalid-uuid")
            assert result is None

    @pytest.mark.asyncio
    async def test_revoke_refresh_token(self):
        """吊销 refresh_token 应删除 Redis 中的记录"""
        from app.services.auth_service import AuthService

        service = AuthService()

        with patch("app.services.auth_service.redis_manager") as mock_redis:
            mock_client = AsyncMock()
            mock_client.delete = AsyncMock(return_value=1)
            mock_redis.get_client = AsyncMock(return_value=mock_client)

            result = await service.revoke_refresh_token("test-uuid")
            assert result is True
            mock_client.delete.assert_called_once_with("refresh_token:test-uuid")


# ========== get_current_user 依赖测试 ==========


class TestGetCurrentUser:
    """get_current_user 依赖注入测试"""

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self):
        """无效 token 应抛出 401"""
        from fastapi import HTTPException

        with patch("app.core.auth.verify_access_token", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                mock_creds = MagicMock()
                mock_creds.credentials = "invalid-token"
                await get_current_user(mock_creds)
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_user_not_found_raises_401(self):
        """用户不存在应抛出 401"""
        from fastapi import HTTPException
        import app.core.auth as auth_module

        payload = {"sub": "999", "username": "ghost"}
        mock_auth = MagicMock()
        mock_auth.get_user_by_id = AsyncMock(return_value=None)

        with patch.object(auth_module, "verify_access_token", return_value=payload):
            with patch("app.services.auth_service.auth_service", mock_auth):
                with pytest.raises(HTTPException) as exc_info:
                    mock_creds = MagicMock()
                    mock_creds.credentials = "valid-token"
                    await get_current_user(mock_creds)
                assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_disabled_user_raises_403(self):
        """已禁用用户应抛出 403"""
        from fastapi import HTTPException
        import app.core.auth as auth_module

        mock_user = MagicMock()
        mock_user.is_active = False
        payload = {"sub": "1", "username": "disabled"}

        mock_auth = MagicMock()
        mock_auth.get_user_by_id = AsyncMock(return_value=mock_user)

        with patch.object(auth_module, "verify_access_token", return_value=payload):
            with patch("app.services.auth_service.auth_service", mock_auth):
                with pytest.raises(HTTPException) as exc_info:
                    mock_creds = MagicMock()
                    mock_creds.credentials = "valid-token"
                    await get_current_user(mock_creds)
                assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_valid_token_returns_user(self):
        """有效 token 应返回用户对象"""
        import app.core.auth as auth_module

        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.username = "testuser"
        mock_user.is_active = True
        payload = {"sub": "1", "username": "testuser"}

        mock_auth = MagicMock()
        mock_auth.get_user_by_id = AsyncMock(return_value=mock_user)

        with patch.object(auth_module, "verify_access_token", return_value=payload):
            with patch("app.services.auth_service.auth_service", mock_auth):
                mock_creds = MagicMock()
                mock_creds.credentials = "valid-token"
                result = await get_current_user(mock_creds)
                assert result is mock_user
                assert result.username == "testuser"
