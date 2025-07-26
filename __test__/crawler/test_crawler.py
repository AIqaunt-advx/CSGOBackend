#!/usr/bin/env python3
"""测试爬虫功能"""

import asyncio
import logging
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.crawler import csgo_crawler

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_single_crawl():
    """测试单次爬取"""
    print("🚀 开始测试爬虫单次爬取...")

    success = await csgo_crawler.run_single_crawl()

    if success:
        print("✅ 爬虫测试成功！")

        # 验证数据是否保存到数据库
        from tools.llm_test_cli import QuickDataRetriever
        retriever = QuickDataRetriever()

        try:
            data = retriever.get_data(method="latest", limit=5)
            if data:
                print(f"📊 数据库中最新的5条记录:")
                for i, record in enumerate(data, 1):
                    from datetime import datetime
                    time_str = datetime.fromtimestamp(record['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"  {i}. 时间: {time_str}, 价格: {record['price']}, 数量: {record['onSaleQuantity']}")
            else:
                print("⚠️ 数据库中没有找到数据")
        finally:
            retriever.close()
    else:
        print("❌ 爬虫测试失败")


async def test_scheduled_crawl():
    """测试定时爬取（运行几分钟后停止）"""
    print("🚀 开始测试定时爬取（将运行2分钟）...")

    # 启动定时爬取任务
    crawl_task = asyncio.create_task(csgo_crawler.start_scheduled_crawling())

    # 等待2分钟
    await asyncio.sleep(120)

    # 停止爬取
    csgo_crawler.stop_crawling()

    # 等待任务完成
    try:
        await asyncio.wait_for(crawl_task, timeout=10)
    except asyncio.TimeoutError:
        crawl_task.cancel()

    print("✅ 定时爬取测试完成")


async def main():
    """主测试函数"""
    print("🧪 CSGO爬虫测试套件")
    print("=" * 50)

    choice = input("选择测试类型:\n1. 单次爬取测试\n2. 定时爬取测试\n请输入选择 (1-2, 默认1): ").strip()

    if choice == "2":
        await test_scheduled_crawl()
    else:
        await test_single_crawl()


if __name__ == "__main__":
    asyncio.run(main())
