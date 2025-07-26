"""
CSGO市场数据爬虫模块
基于提供的爬虫代码，整合到项目中
"""

import asyncio
import logging
import time
import traceback
from datetime import datetime
from typing import Any, Literal, List, Dict

import httpx
import tenacity
import tqdm
from pydantic import BaseModel, model_validator
from pymongo import MongoClient

from config import settings
from modules.models import TrendDetailsDataItem

logger = logging.getLogger(__name__)

# 重试装饰器
myretry = tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
    retry_error_callback=lambda retry_state: None,
)

headers = {
    "accept": "application/json",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "access-token": "undefined",
    "content-type": "application/json",
    "language": "zh_CN",
    "priority": "u=1, i",
    "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Microsoft Edge\";v=\"138\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"macOS\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
    "x-app-version": "1.0.0",
    "x-currency": "CNY",
    "x-device": "1",
    "x-device-id": "b280cd11-f280-4b57-aaa7-8ba53c5ab99b",
    "Referer": "https://steamdt.com/"
}


# Pydantic模型定义


class SellingPrice(BaseModel):
    platform: str
    platformName: str
    price: float
    lastUpdate: int
    link: str


class Suspension(BaseModel):
    consignmentBest: float | None
    purchaseBest: float | None
    purchaseStable: float | None


class TrendItem(BaseModel):
    timestamp: int
    price: float

    @model_validator(mode="before")
    def parse_trend_list(self, v):
        if isinstance(v, list):
            assert len(v) == 2, "Invalid trend list"
            return {"timestamp": int(v[0]), "price": float(v[1])}
        return v


class MarketItem(BaseModel):
    id: str
    name: str
    shortName: str
    marketHashName: str
    marketShortName: str
    imageUrl: str
    qualityName: str
    qualityColor: str
    rarityName: str
    rarityColor: str
    exteriorName: str
    exteriorColor: str
    sellingPriceList: list[SellingPrice]
    suspension: Suspension
    increasePrice: float | None
    trendList: list[TrendItem]
    isCollect: bool
    sellNum: int


class MarketData(BaseModel):
    pageNum: str
    pageSize: str
    total: str
    list: list[MarketItem]
    nextId: str | None
    systemTime: str


class MarketPageResponse(BaseModel):
    success: bool
    data: MarketData | None
    errorCode: int
    errorMsg: str | None
    errorData: Any | None
    errorCodeStr: str | None


class TypeTrendDetailsResponse(BaseModel):
    success: bool
    data: list[TrendDetailsDataItem]
    errorCode: int
    errorMsg: str | None
    errorData: Any | None
    errorCodeStr: str | None


# API请求函数


async def fetch_skin_market_data(next_id: str | None = None, retry_count=0):
    """获取皮肤市场数据"""

    url = "https://sdt-api.ok-skins.com/skin/market/v3/page"

    body = {
        "dataField": "pvNums",
        "dataRange": "",
        "sortType": "desc",
        "nextId": next_id or "",
        "queryName": "",
        "pageSize": 8000,
        "timestamp": str(int(time.time() * 1000))
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=body)
            return resp.json()
    except httpx.ReadTimeout:
        if retry_count >= 5:
            return None
        return fetch_skin_market_data(next_id, retry_count + 1)


async def fetch_item_details(item_id: str, platform: Literal["YOUPIN"]):
    """获取物品详情数据"""

    url = "https://sdt-api.ok-skins.com/user/steam/type-trend/v2/item/details"

    body = {
        "platform": platform,
        "typeDay": "5",
        "dateType": 3,
        "specialStyle": "",
        "itemId": item_id,
        "timestamp": str(int(time.time() * 1000))
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=body)
        return resp.json()


class CSGOCrawler:
    """CSGO市场数据爬虫"""

    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.records_collection = None
        self.is_running = False

    def connect_db(self):
        """连接数据库"""
        try:
            self.client = MongoClient(settings.MONGODB_URL)
            self.db = self.client[settings.MONGODB_DATABASE]
            self.collection = self.db[settings.MONGODB_COLLECTION_MARKET_DATA]
            self.records_collection = self.db[f"{settings.MONGODB_COLLECTION_MARKET_DATA}_records"]
            logger.info("数据库连接成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def close_db(self):
        """关闭数据库连接"""
        if self.client:
            self.client.close()
            logger.info("数据库连接已关闭")

    @staticmethod
    def _generate_mock_data() -> List[Dict[str, Any]]:
        """生成模拟数据"""
        import random

        mock_details = []
        for i in range(random.randint(1, 5)):  # 随机生成1-5条数据
            detail = {
                "timestamp": int(time.time()) + i * 60,
                "price": round(random.uniform(50.0, 200.0), 2),
                "onSaleQuantity": random.randint(10, 100),
                "seekPrice": round(random.uniform(45.0, 190.0), 2),
                "seekQuantity": random.randint(5, 50),
                "transactionAmount": round(random.uniform(500.0, 2000.0), 2),
                "transactionNum": random.randint(5, 30),
                "surviveNum": random.randint(1, 15),
                "item_id": f"mock_item_{int(time.time())}_{i}",
                "item_name": f"模拟物品_{i}"
            }
            mock_details.append(detail)

        logger.info(f"生成了 {len(mock_details)} 条模拟数据")
        return mock_details

    async def crawl_market_items(self) -> List[MarketItem]:
        """爬取市场物品列表"""
        logger.info("开始爬取市场物品数据...")
        full_list: List[MarketItem] = []

        try:
            # 一次性获取所有数据，pageSize设为8000足够获取全部物品
            logger.info("执行单次API调用获取所有市场数据")
            skin_data = await fetch_skin_market_data(next_id=None)

            if skin_data is None:
                logger.error("API调用失败")
                return full_list

            data = skin_data['data']
            if data is None:
                logger.warning(f"数据为空: {skin_data}")
                return full_list

            full_list = data['list']
            logger.info(f"成功获取 {len(full_list)} 个市场物品")

        except Exception as e:
            logger.error(f"爬取市场物品数据失败: {e}")
            traceback.print_exc()

        logger.info(f"成功爬取 {len(full_list)} 个市场物品")
        return full_list

    async def crawl_item_details(self, items: List[MarketItem]) -> List[TrendDetailsDataItem]:
        """爬取物品详情数据"""
        logger.info(f"开始爬取 {len(items)} 个物品的详情数据...")
        all_details = []

        try:
            # 分组处理，使用配置的批处理大小，减少并发压力
            batch_size = settings.CRAWLER_BATCH_SIZE
            grouped = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

            for group_idx, group in tqdm.tqdm(enumerate(grouped)):
                logger.info(f"处理第 {group_idx + 1}/{len(grouped)} 组物品")
                results = []
                for item in group:
                    results.append(
                        await fetch_item_details(item['id'], "YOUPIN")
                    )

                for item, result in zip(group, results):
                    if result is None:
                        logger.warning(f"物品 {item['name']} 详情请求失败，跳过")
                        continue

                    if result['success'] and result['data']:
                        # 为每条记录添加物品信息
                        for detail in result['data']:
                            detail_dict = detail.model_dump()
                            detail_dict['item_id'] = item['id']
                            detail_dict['item_name'] = item['name']
                            all_details.append(detail_dict)
                    else:
                        logger.warning(f"物品 {item['name']} 详情数据为空或请求失败: {result['errorMsg']}")

                # 组间延迟，避免请求过快
                if group_idx < len(grouped) - 1:
                    await asyncio.sleep(settings.CRAWLER_DELAY_BETWEEN_REQUESTS)

        except Exception as e:
            logger.error(f"爬取物品详情数据失败: {e}")
            traceback.print_exc()

        logger.info(f"成功爬取 {len(all_details)} 条详情数据")
        return all_details

    def save_market_data(self, market_data: List[Dict[str, Any]]) -> bool:
        """保存市场数据到数据库"""
        if not market_data:
            logger.info("没有数据需要保存")
            return True

        try:
            # 生成文件ID
            file_id = f"market_crawl_{int(time.time())}"
            current_time = datetime.now()

            # 保存主文档
            main_doc = {
                'file_id': file_id,
                'crawl_type': 'market_data',
                'success': True,
                'error_code': 0,
                'error_msg': None,
                'record_count': len(market_data),
                'created_at': current_time,
                'crawl_timestamp': int(time.time())
            }

            main_result = self.collection.insert_one(main_doc)
            logger.info(f"主文档保存成功，ID: {main_result.inserted_id}")

            # 为每条记录添加元数据
            for item in market_data:
                item['file_id'] = file_id
                item['crawl_timestamp'] = int(time.time())
                item['created_at'] = current_time

            # 批量插入市场数据记录
            if market_data:
                records_result = self.records_collection.insert_many(market_data)
                logger.info(f"成功保存 {len(records_result.inserted_ids)} 条市场数据记录")

            return True

        except Exception as e:
            logger.error(f"保存市场数据到数据库失败: {e}")
            return False

    async def run_single_crawl(self):
        """运行一次完整的爬取周期"""
        logger.info("开始执行爬取任务")
        start_time = time.time()

        try:
            # 连接数据库
            self.connect_db()

            # 1. 爬取市场物品列表（一次性获取所有数据）
            market_items = await self.crawl_market_items()
            if not market_items:
                logger.warning("没有获取到市场物品数据，可能是API问题")
                return False

            # 2. 直接保存市场数据到数据库（不再爬取详情）
            # 将MarketItem转换为字典格式以便存储
            market_data = []
            for item in market_items:
                if isinstance(item, dict):
                    market_data.append(item)
                else:
                    # 如果是Pydantic模型，转换为字典
                    market_data.append(item.model_dump() if hasattr(item, 'model_dump') else dict(item))

            # 3. 保存到数据库
            success = self.save_market_data(market_data)

            elapsed_time = time.time() - start_time
            logger.info(f"爬取任务完成，耗时 {elapsed_time:.2f} 秒")

            return success

        except Exception as e:
            logger.error(f"爬取任务执行失败: {e}")
            traceback.print_exc()
            return False
        finally:
            self.close_db()

    async def start_scheduled_crawling(self):
        """启动定时爬取任务"""
        self.is_running = True
        logger.info(f"启动定时爬取任务，间隔 {settings.CRAWLER_INTERVAL} 秒")

        while self.is_running:
            try:
                await self.run_single_crawl()

                if self.is_running:  # 检查是否仍在运行
                    logger.info(f"等待 {settings.CRAWLER_INTERVAL} 秒后进行下次爬取")
                    await asyncio.sleep(settings.CRAWLER_INTERVAL)

            except asyncio.CancelledError:
                logger.info("爬取任务被取消")
                break
            except Exception as e:
                logger.error(f"定时爬取任务异常: {e}")
                # 出错后等待较短时间再重试
                await asyncio.sleep(60)

    def stop_crawling(self):
        """停止爬取任务"""
        self.is_running = False
        logger.info("停止爬取任务")


# 创建全局爬虫实例
csgo_crawler = CSGOCrawler()
crawler_manager = csgo_crawler  # 为了兼容main.py的导入


# 主函数和测试代码


async def main():
    """主函数 - 用于测试"""
    crawler = CSGOCrawler()

    # 运行一次爬取
    success = await crawler.run_single_crawl()

    if success:
        print("✅ 爬取任务执行成功")
    else:
        print("❌ 爬取任务执行失败")


if __name__ == '__main__':
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行爬虫
    asyncio.run(main())
