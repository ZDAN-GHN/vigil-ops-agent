"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.config import config
from loguru import logger
from app.api import chat, health, file, aiops
from app.core.milvus_client import milvus_manager
from app.core.redis_client import redis_manager
from app.core.mysql_client import mysql_manager
from app.services.scheduler_service import scheduled_aiops_service
from app.services.long_term_memory_service import long_term_memory_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")

    # 连接 Milvus
    logger.info("🔌 正在连接 Milvus...")
    milvus_manager.connect()
    logger.info("✅ Milvus 连接成功")

    # 连接 Redis（用于短期记忆 - 会话 checkpoint）
    logger.info("🔌 正在连接 Redis...")
    await redis_manager.connect()
    logger.info("✅ Redis 连接成功")

    # 连接 MySQL（用于长期记忆 - 用户画像）
    # logger.info("🔌 正在连接 MySQL...")
    # await mysql_manager.connect()
    # logger.info("✅ MySQL 连接成功")

    # 初始化长期记忆数据库表
    # logger.info("📊 正在初始化长期记忆数据库表...")
    # await long_term_memory_service.init_db()
    # logger.info("✅ 长期记忆数据库表初始化完成")

    # 启动定时 AIOps 任务
    if config.enable_scheduled_aiops:
        await scheduled_aiops_service.start()

    logger.info("=" * 60)

    yield

    # 关闭时执行
    await scheduled_aiops_service.stop()
    logger.info("🔌 正在关闭连接...")
    milvus_manager.close()
    await redis_manager.close()
    await mysql_manager.close()
    logger.info(f"👋 {config.app_name} 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
