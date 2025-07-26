import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from router import api_router
from modules.database import db_manager
from modules.crawler import crawler_manager

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("Starting CSGO Backend application...")

    # 连接数据库
    success = await db_manager.connect()
    if not success:
        logger.error("Failed to connect to database")
        raise Exception("Database connection failed")

    logger.info("Database connected successfully")

    # 可选：启动时自动开始爬虫任务
    if settings.DEBUG:
        logger.info("Starting crawler in background...")
        asyncio.create_task(crawler_manager.start_scheduled_crawling())

    yield

    # 关闭时执行
    logger.info("Shutting down CSGO Backend application...")

    # 停止爬虫
    crawler_manager.stop_crawling()

    # 断开数据库连接
    await db_manager.disconnect()

    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含所有API路由
app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    """根路径，返回应用状态"""
    return {
        "status": "ok",
        "message": f"Welcome to the {settings.APP_NAME}!",
        "version": settings.APP_VERSION,
        "crawler_running": crawler_manager.is_running
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        # 检查数据库连接
        stats = await db_manager.get_statistics()

        return {
            "status": "healthy",
            "database": "connected",
            "crawler": "running" if crawler_manager.is_running else "stopped",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
