"""
爬虫模块
负责从各个数据源爬取CSGO市场数据
"""

import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import json
from config import settings
from modules.database import db_manager

logger = logging.getLogger(__name__)


@dataclass
class MarketItem:
    """市场物品数据模型"""
    item_name: str
    price: float
    volume: int
    timestamp: datetime
    source: str
    additional_data: Dict[str, Any] = None


class BaseCrawler:
    """爬虫基类"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            'User-Agent': settings.CRAWLER_USER_AGENT,
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=self.headers
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """发起HTTP请求"""
        for attempt in range(settings.MAX_RETRIES):
            try:
                await asyncio.sleep(settings.CRAWLER_DELAY_BETWEEN_REQUESTS)

                async with self.session.get(url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(
                            f"Request failed with status {response.status}: {url}")

            except Exception as e:
                logger.error(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < settings.MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避

        return None


class BuffCrawler(BaseCrawler):
    """BUFF平台爬虫"""

    def __init__(self):
        super().__init__()
        self.base_url = settings.BUFF_API_BASE_URL

    async def crawl_market_data(self) -> List[MarketItem]:
        """爬取BUFF市场数据"""
        items = []
        try:
            # 这里需要根据实际的BUFF API接口来实现
            # 示例实现
            url = f"{self.base_url}/api/market/goods"
            params = {
                'game': 'csgo',
                'page_num': 1,
                'page_size': settings.CRAWLER_BATCH_SIZE
            }

            data = await self._make_request(url, params=params)
            if data and 'data' in data:
                for item_data in data['data'].get('items', []):
                    item = MarketItem(
                        item_name=item_data.get('name', ''),
                        price=float(item_data.get('sell_min_price', 0)),
                        volume=int(item_data.get('sell_num', 0)),
                        timestamp=datetime.utcnow(),
                        source='buff',
                        additional_data=item_data
                    )
                    items.append(item)

            logger.info(f"Crawled {len(items)} items from BUFF")

        except Exception as e:
            logger.error(f"Error crawling BUFF data: {e}")

        return items


class SteamCrawler(BaseCrawler):
    """Steam平台爬虫"""

    def __init__(self):
        super().__init__()
        self.base_url = settings.STEAM_API_BASE_URL

    async def crawl_market_data(self) -> List[MarketItem]:
        """爬取Steam市场数据"""
        items = []
        try:
            # Steam市场API实现
            # 注意：Steam API可能需要特殊处理和认证
            logger.info("Starting Steam market data crawl")

            # 这里是示例实现，需要根据实际Steam API调整
            popular_items = [
                "AK-47 | Redline",
                "AWP | Dragon Lore",
                "M4A4 | Howl",
                # 更多热门物品...
            ]

            for item_name in popular_items:
                url = f"{self.base_url}/priceoverview/"
                params = {
                    'appid': 730,  # CSGO app ID
                    'currency': 1,  # USD
                    'market_hash_name': item_name
                }

                data = await self._make_request(url, params=params)
                if data and data.get('success'):
                    item = MarketItem(
                        item_name=item_name,
                        price=float(
                            data.get('median_price', '0').replace('$', '')),
                        volume=int(data.get('volume', 0)),
                        timestamp=datetime.utcnow(),
                        source='steam',
                        additional_data=data
                    )
                    items.append(item)

            logger.info(f"Crawled {len(items)} items from Steam")

        except Exception as e:
            logger.error(f"Error crawling Steam data: {e}")

        return items


class YYYPCrawler(BaseCrawler):
    """悠悠有品平台爬虫"""

    def __init__(self):
        super().__init__()
        self.base_url = settings.YYYP_API_BASE_URL

    async def crawl_market_data(self) -> List[MarketItem]:
        """爬取悠悠有品市场数据"""
        items = []
        try:
            # 悠悠有品API实现
            logger.info("Starting YYYP market data crawl")

            # 示例实现
            url = f"{self.base_url}/api/homepage/es/template/get_es_homepage_index"

            data = await self._make_request(url)
            if data:
                # 解析数据并转换为MarketItem对象
                # 这里需要根据实际API响应格式来实现
                pass

            logger.info(f"Crawled {len(items)} items from YYYP")

        except Exception as e:
            logger.error(f"Error crawling YYYP data: {e}")

        return items


class CrawlerManager:
    """爬虫管理器"""

    def __init__(self):
        self.crawlers = {
            'buff': BuffCrawler(),
            'steam': SteamCrawler(),
            'yyyp': YYYPCrawler()
        }
        self.is_running = False

    async def crawl_all_sources(self) -> List[MarketItem]:
        """从所有数据源爬取数据"""
        all_items = []

        for name, crawler in self.crawlers.items():
            try:
                async with crawler:
                    items = await crawler.crawl_market_data()
                    all_items.extend(items)
                    logger.info(
                        f"Successfully crawled {len(items)} items from {name}")

            except Exception as e:
                logger.error(f"Failed to crawl from {name}: {e}")

        return all_items

    async def save_crawled_data(self, items: List[MarketItem]) -> bool:
        """保存爬取的数据到数据库"""
        try:
            if not items:
                logger.info("No data to save")
                return True

            # 转换为数据库格式
            db_items = []
            for item in items:
                db_item = {
                    'item_name': item.item_name,
                    'price': item.price,
                    'volume': item.volume,
                    'timestamp': item.timestamp,
                    'source': item.source,
                    'additional_data': item.additional_data or {}
                }
                db_items.append(db_item)

            # 批量插入数据库
            success = await db_manager.batch_insert_market_data(db_items)
            if success:
                logger.info(
                    f"Successfully saved {len(db_items)} items to database")
            else:
                logger.error("Failed to save data to database")

            return success

        except Exception as e:
            logger.error(f"Error saving crawled data: {e}")
            return False

    async def run_crawl_cycle(self):
        """运行一次完整的爬取周期"""
        logger.info("Starting crawl cycle")
        start_time = time.time()

        try:
            # 爬取数据
            items = await self.crawl_all_sources()

            # 保存数据
            if items:
                await self.save_crawled_data(items)

            # 清理旧数据
            await db_manager.cleanup_old_data()

            elapsed_time = time.time() - start_time
            logger.info(f"Crawl cycle completed in {elapsed_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error in crawl cycle: {e}")

    async def start_scheduled_crawling(self):
        """启动定时爬取任务"""
        self.is_running = True
        logger.info(
            f"Starting scheduled crawling with interval {settings.CRAWLER_INTERVAL} seconds")

        while self.is_running:
            try:
                await self.run_crawl_cycle()
                await asyncio.sleep(settings.CRAWLER_INTERVAL)

            except asyncio.CancelledError:
                logger.info("Crawling task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in scheduled crawling: {e}")
                await asyncio.sleep(60)  # 等待1分钟后重试

    def stop_crawling(self):
        """停止爬取任务"""
        self.is_running = False
        logger.info("Stopping crawling tasks")


# 创建全局爬虫管理器实例
crawler_manager = CrawlerManager()
