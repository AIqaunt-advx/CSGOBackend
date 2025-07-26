"""
爬虫管理API路由
提供爬虫控制和状态监控接口
"""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from modules.crawler import crawler_manager
from modules.database import db_manager

logger = logging.getLogger(__name__)

crawler_router = APIRouter()


class CrawlerStatusResponse(BaseModel):
    """爬虫状态响应模型"""
    is_running: bool
    last_crawl_time: str = None
    total_items_crawled: int = 0
    error_count: int = 0


@crawler_router.post("/start")
async def start_crawler(background_tasks: BackgroundTasks):
    """启动爬虫定时任务"""
    try:
        if crawler_manager.is_running:
            return {"success": False, "message": "爬虫已经在运行中"}

        # 在后台启动爬虫任务
        background_tasks.add_task(crawler_manager.start_scheduled_crawling)

        return {
            "success": True,
            "message": "爬虫定时任务已启动"
        }

    except Exception as e:
        logger.error(f"启动爬虫失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动爬虫失败: {str(e)}")


@crawler_router.post("/stop")
async def stop_crawler():
    """停止爬虫任务"""
    try:
        crawler_manager.stop_crawling()
        return {
            "success": True,
            "message": "爬虫任务已停止"
        }

    except Exception as e:
        logger.error(f"停止爬虫失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止爬虫失败: {str(e)}")


@crawler_router.post("/run-once")
async def run_crawler_once():
    """手动执行一次爬取"""
    try:
        await crawler_manager.run_crawl_cycle()
        return {
            "success": True,
            "message": "手动爬取完成"
        }

    except Exception as e:
        logger.error(f"手动爬取失败: {e}")
        raise HTTPException(status_code=500, detail=f"手动爬取失败: {str(e)}")


@crawler_router.get("/status")
async def get_crawler_status():
    """获取爬虫状态"""
    try:
        # 获取数据库统计信息
        stats = await db_manager.get_statistics()

        return {
            "success": True,
            "status": {
                "is_running": crawler_manager.is_running,
                "database_stats": stats
            }
        }

    except Exception as e:
        logger.error(f"获取爬虫状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@crawler_router.post("/cleanup")
async def cleanup_old_data(hours: int = None):
    """清理旧数据"""
    try:
        await db_manager.cleanup_old_data(hours)
        return {
            "success": True,
            "message": f"已清理 {hours or '默认时间'} 小时前的旧数据"
        }

    except Exception as e:
        logger.error(f"清理数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理数据失败: {str(e)}")
