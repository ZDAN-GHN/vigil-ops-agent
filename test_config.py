import os
import sys

sys.modules['app'] = type(sys)('app')
sys.modules['app.utils'] = type(sys)('app.utils')

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    dashscope_api_key: str = "sk-default"
    dashscope_api_base: str = "https://default.example.com"

config = Settings()
os.environ["DASHSCOPE_API_KEY"] = config.dashscope_api_key
os.environ["DASHSCOPE_API_BASE"] = config.dashscope_api_base

print("config.dashscope_api_key:", config.dashscope_api_key)
print("config.dashscope_api_base:", config.dashscope_api_base)
print("os.environ DASHSCOPE_API_KEY:", os.environ.get("DASHSCOPE_API_KEY"))
print("os.environ DASHSCOPE_API_BASE:", os.environ.get("DASHSCOPE_API_BASE"))
