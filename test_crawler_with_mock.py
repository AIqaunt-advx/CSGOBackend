#!/usr/bin/env python3
"""使用模拟数据测试爬虫"""

import asyncio
import logging
import time
from datetime import datetime

from modules.crawler import CSGOCrawler

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockCrawler(CSGOCrawler):
    """模拟数据的爬虫"""

    async def crawl_market_items(self):
        """模拟爬取市场物品"""
        logger.info("模拟爬取市场物品数据...")

        # 模拟延迟
        await asyncio.sleep(2)

        # 返回空列表模拟API失败
        return []

    async def crawl_item_details(self, items):
        """模拟爬取物品详情"""
        logger.info(f"模拟爬取 {len(items)} 个物品的详情...")

        # 模拟延迟
        await asyncio.sleep(1)

        # 生成模拟数据
        mock_details = []
        for i in range(3):
            detail = {
                "timestamp": int(time.time()) + i * 60,
                "price": 100.0 + i * 5,
                "onSaleQuantity": 50 - i,
                "seekPrice": 95.0 + i * 2,
                "seekQuantity": 10 + i,
                "transactionAmount": 1000.0 + i * 100,
                "transcationNum": 10 + i,
                "surviveNum": 5 + i,
                "item_id": f"mock_item_{i}",
                "item_name": f"模拟物品_{i}"
            }
            mock_details.append(detail)

        return mock_details


async def test_mock_crawler():
    """测试模拟爬虫"""
    print("🚀 测试模拟爬虫...")

    crawler = MockCrawler()

    # 运行一次爬取
    success = await crawler.run_single_crawl()

    if success:
        print("✅ 模拟爬虫测试成功")

        # 验证数据
        from tools.llm_test_cli import QuickDataRetriever
        retriever = QuickDataRetriever()

        try:
            data = retriever.get_data(method="latest", limit=5)
            if data:
                print("📊 最新保存的数据:")
                for i, record in enumerate(data, 1):
                    time_str = datetime.fromtimestamp(record['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    item_name = record.get('item_name', 'N/A')
                    print(f"  {i}. 时间: {time_str}, 价格: {record['price']}, 物品: {item_name}")
        finally:
            retriever.close()
    else:
        print("❌ 模拟爬虫测试失败")


async def test_mock_scheduled_crawler():
    """测试定时模拟爬虫"""
    print("🚀 测试定时模拟爬虫（运行30秒）...")

    crawler = MockCrawler()

    # 启动定时爬取
    crawl_task = asyncio.create_task(crawler.start_scheduled_crawling())

    # 运行30秒
    await asyncio.sleep(30)

    # 停止爬虫
    crawler.stop_crawling()

    # 等待任务完成
    try:
        await asyncio.wait_for(crawl_task, timeout=5)
    except asyncio.TimeoutError:
        crawl_task.cancel()

    print("✅ 定时爬虫测试完成")


async def main():
    """主函数"""
    print("🧪 爬虫模拟测试")
    print("=" * 40)

    choice = input("选择测试类型:\n1. 单次爬取测试\n2. 定时爬取测试\n请输入选择 (1-2, 默认1): ").strip()

    if choice == "2":
        await test_mock_scheduled_crawler()
    else:
        await test_mock_crawler()


if __name__ == "__main__":
    asyncio.run(main())
