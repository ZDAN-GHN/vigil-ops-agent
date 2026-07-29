"""认证接口

提供用户登录、登出、令牌刷新、用户管理等接口。
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.core.auth import create_access_token, create_refresh_token, get_current_user
from app.models.auth_schema import (
    LoginRequest,
    LoginResponse,
    RefreshRequest,
    TokenResponse,
    UserCreateRequest,
    UserInfoResponse,
    UserListResponse,
    UserUpdateRequest,
)
from app.models.user import User
from app.services.auth_service import auth_service

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """用户登录

    使用用户名和密码登录，返回 access_token 和 refresh_token。

    Args:
        request: 登录请求

    Returns:
        LoginResponse: 包含令牌和用户信息

    Raises:
        HTTPException: 登录失败时抛出 401
    """
    user = await auth_service.authenticate(request.username, request.password)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 签发令牌
    access_token = create_access_token(user.id, user.username)
    refresh_token = create_refresh_token()

    # 存储 refresh_token 到 Redis
    await auth_service.store_refresh_token(refresh_token, user.id)

    logger.info(f"用户登录成功: {user.username}")

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=UserInfoResponse.model_validate(user),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest):
    """刷新 Access Token

    使用 refresh_token 获取新的 access_token。

    Args:
        request: 刷新请求

    Returns:
        TokenResponse: 新的 access_token

    Raises:
        HTTPException: 刷新失败时抛出 401
    """
    user_id = await auth_service.verify_refresh_token(request.refresh_token)

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的刷新令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 获取用户信息
    user = await auth_service.get_user_by_id(user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或已禁用",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 签发新的 access_token
    new_access_token = create_access_token(user.id, user.username)

    return TokenResponse(
        access_token=new_access_token,
        token_type="bearer",
    )


@router.post("/logout")
async def logout(request: RefreshRequest):
    """用户登出

    吊销 refresh_token，access_token 将在自然过期后失效。

    Args:
        request: 登出请求（包含 refresh_token）

    Returns:
        dict: 登出结果
    """
    success = await auth_service.revoke_refresh_token(request.refresh_token)

    if success:
        logger.info("用户登出成功")
        return {"message": "登出成功"}
    else:
        logger.warning("登出失败: refresh_token 无效")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的刷新令牌",
        )


@router.get("/me", response_model=UserInfoResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """获取当前登录用户信息

    Args:
        current_user: 当前用户（通过 JWT 自动注入）

    Returns:
        UserInfoResponse: 用户信息
    """
    return UserInfoResponse.model_validate(current_user)


# ========== 管理员接口 ==========


@router.post("/users", response_model=UserInfoResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
        request: UserCreateRequest,
        current_user: User = Depends(get_current_user),
):
    """创建新用户（仅管理员）

    Args:
        request: 创建用户请求
        current_user: 当前用户（需为管理员）

    Returns:
        UserInfoResponse: 创建的用户信息

    Raises:
        HTTPException: 权限不足或用户名已存在
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="仅管理员可创建用户",
        )

    try:
        user = await auth_service.create_user(
            username=request.username,
            password=request.password,
            display_name=request.display_name,
            is_admin=request.is_admin,
        )
        return UserInfoResponse.model_validate(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/users", response_model=UserListResponse)
async def list_users(
        offset: int = 0,
        limit: int = 50,
        current_user: User = Depends(get_current_user),
):
    """获取用户列表（仅管理员）

    Args:
        offset: 偏移量
        limit: 每页数量
        current_user: 当前用户（需为管理员）

    Returns:
        UserListResponse: 用户列表

    Raises:
        HTTPException: 权限不足
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="仅管理员可查看用户列表",
        )

    users, total = await auth_service.list_users(offset=offset, limit=limit)

    return UserListResponse(
        users=[UserInfoResponse.model_validate(u) for u in users],
        total=total,
    )


@router.put("/users/{user_id}", response_model=UserInfoResponse)
async def update_user(
        user_id: int,
        request: UserUpdateRequest,
        current_user: User = Depends(get_current_user),
):
    """更新用户信息（仅管理员）

    Args:
        user_id: 用户 ID
        request: 更新请求
        current_user: 当前用户（需为管理员）

    Returns:
        UserInfoResponse: 更新后的用户信息

    Raises:
        HTTPException: 权限不足或用户不存在
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="仅管理员可更新用户",
        )

    user = await auth_service.update_user(
        user_id=user_id,
        display_name=request.display_name,
        is_active=request.is_active,
        is_admin=request.is_admin,
        password=request.password,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )

    return UserInfoResponse.model_validate(user)


@router.delete("/users/{user_id}")
async def delete_user(
        user_id: int,
        current_user: User = Depends(get_current_user),
):
    """删除用户（仅管理员）

    Args:
        user_id: 用户 ID
        current_user: 当前用户（需为管理员）

    Returns:
        dict: 删除结果

    Raises:
        HTTPException: 权限不足或用户不存在
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="仅管理员可删除用户",
        )

    # 禁止删除自己
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能删除自己",
        )

    success = await auth_service.delete_user(user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )

    return {"message": "用户已删除"}
