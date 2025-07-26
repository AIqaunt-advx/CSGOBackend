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

    # MongoDB数据库配置
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "csgo_market"
    MONGODB_COLLECTION_MARKET_DATA: str = "market_data"
    MONGODB_COLLECTION_TREND_DATA: str = "trend_data"
    MONGODB_CONNECTION_TIMEOUT: int = 10000
    MONGODB_MAX_POOL_SIZE: int = 10

    # LLM服务器配置
    LLM_API_BASE_URL: str = "http://localhost:8001"
    LLM_API_KEY: str = "your-llm-api-key"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.7
    LLM_REQUEST_TIMEOUT: int = 60

    # 爬虫配置
    CRAWLER_INTERVAL: int = 3600  # 爬取间隔（秒）
    CRAWLER_DATA_RETENTION_HOURS: int = 168  # 数据保留时间（小时，7天）
    CRAWLER_BATCH_SIZE: int = 100  # 批处理大小
    CRAWLER_USER_AGENT: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    CRAWLER_DELAY_BETWEEN_REQUESTS: float = 1.0  # 请求间隔（秒）

    # 数据分析配置
    ANALYSIS_WINDOW_HOURS: int = 7  # 分析时间窗口（小时）
    PREDICTION_CONFIDENCE_THRESHOLD: float = 0.8  # 预测置信度阈值

    # Redis配置 (预留)
    REDIS_URL: str = ""

    # 安全配置
    SECRET_KEY: str = "your-secret-key-here"

    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建全局设置实例
settings = Settings()
