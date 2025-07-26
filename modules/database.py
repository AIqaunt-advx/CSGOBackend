"""
数据库模块
负责MongoDB数据库的连接和操作
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import PyMongoError

from config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器"""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.market_data_collection: Optional[AsyncIOMotorCollection] = None

    async def connect(self) -> bool:
        """连接到数据库"""
        try:
            # 创建MongoDB客户端
            self.client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=settings.MONGODB_CONNECTION_TIMEOUT,
                maxPoolSize=settings.MONGODB_MAX_POOL_SIZE
            )

            # 测试连接
            await self.client.admin.command('ping')

            # 获取数据库和集合
            self.database = self.client[settings.MONGODB_DATABASE]
            self.market_data_collection = self.database[settings.MONGODB_COLLECTION_MARKET_DATA]

            # 创建索引
            await self.market_data_collection.create_index([("timestamp", -1)])
            await self.market_data_collection.create_index([("item_name", 1)])
            await self.market_data_collection.create_index([("price", 1)])

            logger.info("Database connected and indexes created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    async def disconnect(self):
        """断开数据库连接"""
        if self.client:
            self.client.close()
            logger.info("Database connection closed")

    async def batch_insert_market_data(self, data_list: List[Dict[str, Any]]) -> bool:
        """批量插入市场数据"""
        try:
            if not data_list:
                return True

            # 添加创建时间
            for data in data_list:
                data['created_at'] = datetime.utcnow()

            result = await self.market_data_collection.insert_many(data_list)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} market data records")
            return True

        except PyMongoError as e:
            logger.error(f"Failed to batch insert market data: {e}")
            return False

    async def get_market_data(self, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
        """获取市场数据"""
        try:
            cursor = self.market_data_collection.find().sort("timestamp", -1).skip(skip).limit(limit)
            return await cursor.to_list(length=limit)

        except PyMongoError as e:
            logger.error(f"Failed to get market data: {e}")
            return []

    async def cleanup_old_data(self, hours: int = None):
        """清理旧数据"""
        try:
            if hours is None:
                hours = settings.CRAWLER_DATA_RETENTION_HOURS

            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            # 删除旧的市场数据
            market_result = await self.market_data_collection.delete_many({
                "timestamp": {"$lt": cutoff_time}
            })

            logger.info(f"Cleaned up {market_result.deleted_count} market records")

        except PyMongoError as e:
            logger.error(f"Failed to cleanup old data: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            market_count = await self.market_data_collection.count_documents({})

            # 获取最新数据时间
            latest_market = await self.market_data_collection.find_one(
                {}, sort=[("timestamp", -1)]
            )

            return {
                "market_data_count": market_count,
                "latest_market_data": latest_market.get("timestamp") if latest_market else None
            }

        except PyMongoError as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


# 创建全局数据库管理器实例
db_manager = DatabaseManager()
