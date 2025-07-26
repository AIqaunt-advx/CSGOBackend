#!/usr/bin/env python3
"""测试爬虫线程"""

import asyncio
import threading
import time
import logging
from modules.crawler import csgo_crawler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_crawler_thread():
    """测试爬虫线程"""
    print("🚀 测试爬虫线程...")
    
    # 启动爬虫
    print("启动爬虫...")
    
    def run_crawler():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(csgo_crawler.start_scheduled_crawling())
        except Exception as e:
            print(f"爬虫线程异常: {e}")
            import traceback
            traceback.print_exc()
    
    # 启动线程
    crawler_thread = threading.Thread(target=run_crawler, daemon=True)
    crawler_thread.start()
    
    # 等待一段时间
    for i in range(30):
        print(f"等待中... {i+1}/30 秒")
        time.sleep(1)
        
        if not crawler_thread.is_alive():
            print("❌ 爬虫线程已停止")
            break
    else:
        print("✅ 爬虫线程运行正常")
    
    # 停止爬虫
    print("停止爬虫...")
    csgo_crawler.stop_crawling()
    
    # 等待线程结束
    crawler_thread.join(timeout=5)
    
    print("测试完成")

if __name__ == "__main__":
    test_crawler_thread()