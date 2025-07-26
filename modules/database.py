"""
MongoDB数据库模块
负责管理MongoDB连接和数据操作
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import PyMongoError

from config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """MongoDB数据库管理器"""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.market_data_collection: Optional[AsyncIOMotorCollection] = None
        self.trend_data_collection: Optional[AsyncIOMotorCollection] = None

    async def connect(self) -> bool:
        """连接到MongoDB数据库"""
        try:
            self.client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=settings.MONGODB_CONNECTION_TIMEOUT,
                maxPoolSize=settings.MONGODB_MAX_POOL_SIZE
            )

            # 测试连接
            await self.client.admin.command('ping')

            self.database = self.client[settings.MONGODB_DATABASE]
            self.market_data_collection = self.database[settings.MONGODB_COLLECTION_MARKET_DATA]
            self.trend_data_collection = self.database[settings.MONGODB_COLLECTION_TREND_DATA]

            # 创建索引
            await self._create_indexes()

            logger.info("Successfully connected to MongoDB")
            return True

        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False

    async def disconnect(self):
        """断开数据库连接"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    async def _create_indexes(self):
        """创建数据库索引"""
        try:
            # 为市场数据集合创建索引
            await self.market_data_collection.create_index([("item_name", 1), ("timestamp", -1)])
            await self.market_data_collection.create_index([("timestamp", -1)])

            # 为趋势数据集合创建索引
            await self.trend_data_collection.create_index([("item_name", 1), ("timestamp", -1)])
            await self.trend_data_collection.create_index([("timestamp", -1)])

            logger.info("Database indexes created successfully")

        except PyMongoError as e:
            logger.error(f"Failed to create indexes: {e}")

    async def insert_market_data(self, data: Dict[str, Any]) -> bool:
        """插入市场数据"""
        try:
            data['created_at'] = datetime.utcnow()
            result = await self.market_data_collection.insert_one(data)
            return result.inserted_id is not None

        except PyMongoError as e:
            logger.error(f"Failed to insert market data: {e}")
            return False

    async def insert_trend_data(self, data: Dict[str, Any]) -> bool:
        """插入趋势数据"""
        try:
            data['created_at'] = datetime.utcnow()
            result = await self.trend_data_collection.insert_one(data)
            return result.inserted_id is not None

        except PyMongoError as e:
            logger.error(f"Failed to insert trend data: {e}")
            return False

    async def batch_insert_market_data(self, data_list: List[Dict[str, Any]]) -> bool:
        """批量插入市场数据"""
        try:
            if not data_list:
                return True

            # 添加创建时间
            for data in data_list:
                data['created_at'] = datetime.utcnow()

            result = await self.market_data_collection.insert_many(data_list)
            return len(result.inserted_ids) == len(data_list)

        except PyMongoError as e:
            logger.error(f"Failed to batch insert market data: {e}")
            return False

    async def get_recent_data(self, item_name: str = None, hours: int = 7) -> List[Dict[str, Any]]:
        """获取最近指定小时的数据"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            query = {"timestamp": {"$gte": cutoff_time}}
            if item_name:
                query["item_name"] = item_name

            cursor = self.market_data_collection.find(query).sort("timestamp", -1)
            return await cursor.to_list(length=None)

        except PyMongoError as e:
            logger.error(f"Failed to get recent data: {e}")
            return []

    async def get_trend_data(self, item_name: str, hours: int = 7) -> List[Dict[str, Any]]:
        """获取指定物品的趋势数据"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            query = {
                "item_name": item_name,
                "timestamp": {"$gte": cutoff_time}
            }

            cursor = self.trend_data_collection.find(query).sort("timestamp", 1)
            return await cursor.to_list(length=None)

        except PyMongoError as e:
            logger.error(f"Failed to get trend data: {e}")
            return []

    async def cleanup_old_data(self, hours: int = None):
        """清理旧数据"""
        try:
            hours = hours or settings.CRAWLER_DATA_RETENTION_HOURS
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            # 删除旧的市场数据
            market_result = await self.market_data_collection.delete_many({
                "timestamp": {"$lt": cutoff_time}
            })

            # 删除旧的趋势数据
            trend_result = await self.trend_data_collection.delete_many({
                "timestamp": {"$lt": cutoff_time}
            })

            logger.info(
                f"Cleaned up {market_result.deleted_count} market records and {trend_result.deleted_count} trend records")

        except PyMongoError as e:
            logger.error(f"Failed to cleanup old data: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            market_count = await self.market_data_collection.count_documents({})
            trend_count = await self.trend_data_collection.count_documents({})

            # 获取最新数据时间
            latest_market = await self.market_data_collection.find_one(
                {}, sort=[("timestamp", -1)]
            )
            latest_trend = await self.trend_data_collection.find_one(
                {}, sort=[("timestamp", -1)]
            )

            return {
                "market_data_count": market_count,
                "trend_data_count": trend_count,
                "latest_market_data": latest_market.get("timestamp") if latest_market else None,
                "latest_trend_data": latest_trend.get("timestamp") if latest_trend else None
            }

        except PyMongoError as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


# 创建全局数据库管理器实例
db_manager = DatabaseManager()
