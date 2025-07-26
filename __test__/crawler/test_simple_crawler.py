#!/usr/bin/env python3
"""简化的爬虫测试"""

import logging
import os
import sys
from datetime import datetime

from pymongo import MongoClient

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCrawler:
    """简化的爬虫类"""

    def __init__(self):
        self.client = None
        self.db = None
        self.records_collection = None

    def connect_db(self):
        """连接数据库"""
        try:
            self.client = MongoClient(settings.MONGODB_URL)
            self.db = self.client[settings.MONGODB_DATABASE]
            self.records_collection = self.db[f"{settings.MONGODB_COLLECTION_MARKET_DATA}_records"]
            logger.info("数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False

    def close_db(self):
        """关闭数据库连接"""
        if self.client:
            self.client.close()
            logger.info("数据库连接已关闭")

    def save_sample_data(self):
        """保存样本数据"""
        sample_data = [
            {
                "timestamp": int(datetime.now().timestamp()),
                "price": 100.0,
                "onSaleQuantity": 50,
                "seekPrice": 95.0,
                "seekQuantity": 10,
                "transactionAmount": 1000.0,
                "transcationNum": 10,
                "surviveNum": 5,
                "file_id": "test_crawler",
                "item_id": "test_item_1",
                "item_name": "测试物品1"
            },
            {
                "timestamp": int(datetime.now().timestamp()) + 60,
                "price": 102.0,
                "onSaleQuantity": 45,
                "seekPrice": 97.0,
                "seekQuantity": 12,
                "transactionAmount": 1200.0,
                "transcationNum": 12,
                "surviveNum": 6,
                "file_id": "test_crawler",
                "item_id": "test_item_2",
                "item_name": "测试物品2"
            }
        ]

        try:
            result = self.records_collection.insert_many(sample_data)
            logger.info(f"成功保存 {len(result.inserted_ids)} 条样本数据")
            return True
        except Exception as e:
            logger.error(f"保存样本数据失败: {e}")
            return False

    def test_crawler(self):
        """测试爬虫功能"""
        logger.info("开始测试爬虫功能")

        # 连接数据库
        if not self.connect_db():
            return False

        try:
            # 保存样本数据
            success = self.save_sample_data()

            if success:
                logger.info("✅ 爬虫测试成功")

                # 验证数据
                count = self.records_collection.count_documents({"file_id": "test_crawler"})
                logger.info(f"数据库中测试数据条数: {count}")

                return True
            else:
                logger.error("❌ 爬虫测试失败")
                return False

        finally:
            self.close_db()


def main():
    """主函数"""
    print("🧪 简化爬虫测试")
    print("=" * 30)

    crawler = SimpleCrawler()
    success = crawler.test_crawler()

    if success:
        print("🎉 测试通过！")

        # 使用LLM工具验证数据
        print("\n📊 验证保存的数据:")
        from tools.llm_test_cli import QuickDataRetriever
        retriever = QuickDataRetriever()

        try:
            data = retriever.get_data(method="latest", limit=3)
            if data:
                for i, record in enumerate(data, 1):
                    time_str = datetime.fromtimestamp(record['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"  {i}. 时间: {time_str}, 价格: {record['price']}, 物品: {record.get('item_name', 'N/A')}")
        finally:
            retriever.close()
    else:
        print("💥 测试失败！")


if __name__ == "__main__":
    main()
