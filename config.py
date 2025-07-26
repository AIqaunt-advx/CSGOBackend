from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 应用基本信息
    APP_NAME: str = "DaoGO Backend"
    APP_DESCRIPTION: str = "CSGO Market Data Analysis Backend API"
    APP_VERSION: str = "1.0.0"

    # 服务器配置
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # API配置
    API_PREFIX: str = "/api/v1"

    # CORS配置
    ALLOWED_HOSTS: List[str] = ["*"]

    # 外部API配置
    BUFF_API_BASE_URL: str = "https://api.buff.163.com"
    STEAM_API_BASE_URL: str = "https://steamcommunity.com/market"
    YYYP_API_BASE_URL: str = "https://api.youpin898.com"

    # 请求配置
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3

    # 数据库配置 (预留)
    DATABASE_URL: str = ""

    # Redis配置 (预留)
    REDIS_URL: str = ""

    # 安全配置
    SECRET_KEY: str = "your-secret-key-here"

    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建全局设置实例
settings = Settings()
