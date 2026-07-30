"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（config.py 在 app/ 下，所以向上取一级）
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "OnCallAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9999

    # DashScope 配置
    dashscope_api_key: str = "sk:dashscope_api_key"  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max" # 对话模型
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）
    dashscope_biz_space_api_base: str = "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1" # 业务空间的 api_base_url
    dashscope_rerank_model: str = "qwen3-rerank" # rerank 重排模型

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # 定时 AIOps 配置
    enable_scheduled_aiops: bool = False
    scheduled_aiops_interval_seconds: int = 300
    scheduled_aiops_webhook_url: str = ""
    scheduled_aiops_session_id: str = "_scheduled_aiops"

    # Redis 配置（用于短期记忆 - 会话 checkpoint）
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""  # 空字符串表示无密码
    redis_checkpoint_ttl: int = 604800  # checkpoint 过期时间，默认 7 天（秒）

    # 对话历史持久化配置（MySQL 备份 + Redis fallback）
    conversation_history_enabled: bool = True  # 是否启用对话历史 MySQL 持久化
    conversation_history_redis_ttl: int = 604800  # 从 MySQL 恢复到 Redis 时的 TTL，默认 7 天（秒）

    # 消息队列配置（异步持久化对话历史）
    mq_queue_key: str = "mq:conversation_history"  # Redis List key
    mq_batch_size: int = 10  # 每批消费的消息数量
    mq_retry_count: int = 3  # 消费失败重试次数
    mq_fallback_log_file: str = "logs/mq_fallback.jsonl"  # 兜底日志文件路径

    # MySQL 配置（用于长期记忆 - 用户画像）
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_db: str = "oncall_agent"
    mysql_user: str = "root"
    mysql_password: str = ""

    # JWT 认证配置
    jwt_secret_key: str = "change-me-in-production-use-a-random-string"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # 初始管理员配置（首次启动时自动创建）
    initial_admin_username: str = "admin"
    initial_admin_password: str = "admin123"

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()

if __name__ == '__main__':
    print(Settings().model_config)
