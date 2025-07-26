"""
CSGO市场数据爬虫模块
基于你提供的爬虫代码，整合到项目中
"""

import asyncio
import json
import traceback
import logging
import time
from typing import Any, Literal, List, Dict, Optional
from datetime import datetime
import aiohttp
import tenacity
from pydantic import BaseModel, model_validator
from tqdm import trange
from pymongo import MongoClient
from config import settings

logger = logging.getLogger(__name__)

# 重试装饰器
myretry = tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
    retry_error_callback=lambda retry_state: None,
)
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
    def parse_trend_list(cls, v):
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


class TrendDetailsDataItem(BaseModel):
    """趋势详情数据项 - 对应你要的数据格式"""
    timestamp: int
    price: float
    onSaleQuantity: int
    seekPrice: float
    seekQuantity: int
    transactionAmount: float | None
    transcationNum: int | None
    surviveNum: int | None

    @model_validator(mode="before")
    def parse_trend_details_list(cls, v):
        if isinstance(v, list):
            assert len(v) == 8, "Invalid trend details list"
            return {
                "timestamp": int(v[0]),
                "price": float(v[1]),
                "onSaleQuantity": int(v[2]),
                "seekPrice": float(v[3]),
                "seekQuantity": int(v[4]),
                "transactionAmount": float(v[5]) if v[5] is not None else None,
                "transcationNum": int(v[6]) if v[6] is not None else None,
                "surviveNum": int(v[7]) if v[7] is not None else None,
            }
        return v


class TypeTrendDetailsResponse(BaseModel):
    success: bool
    data: list[TrendDetailsDataItem]
    errorCode: int
    errorMsg: str | None
    errorData: Any | None
    errorCodeStr: str | None

# API请求函数


@myretry
async def fetch_skin_market_data(next_id: str | None = None):
    """获取皮肤市场数据"""
    url = "https://sdt-api.ok-skins.com/skin/market/v3/page?timestamp=1753372632751"
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

    body = {
        "dataField": "pvNums",
        "dataRange": "",
        "sortType": "desc",
        "nextId": next_id or "",
        "queryName": "",
        "pageSize": 8000,
        "timestamp": str(int(time.time() * 1000))
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body) as response:
            response.raise_for_status()
            json_response = await response.json()
            return MarketPageResponse.model_validate(json_response)


@myretry
async def fetch_item_details(item_id: str, platform: Literal["YOUPIN"]):
    """获取物品详情数据"""
    url = "https://sdt-api.ok-skins.com/user/steam/type-trend/v2/item/details?timestamp=1753373125434"
    headers = {
        "accept": "application/json",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "access-token": "",
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

    body = {
        "platform": platform,
        "typeDay": "5",
        "dateType": 3,
        "specialStyle": "",
        "itemId": item_id,
        "timestamp": str(int(time.time() * 1000))
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body) as response:
            response.raise_for_status()
            json_response = await response.json()
            return TypeTrendDetailsResponse.model_validate(json_response)


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

    async def crawl_market_items(self) -> List[MarketItem]:
        """爬取市场物品列表"""
        logger.info("开始爬取市场物品数据...")
        full_list: List[MarketItem] = []
        next_id = None

        try:
            # 限制爬取页数，避免无限循环
            for page in trange(64, desc="爬取市场数据"):
                skin_data = await fetch_skin_market_data(next_id=next_id)

                data = skin_data.data
                if data is None:
                    logger.warning(f"第{page+1}页数据为空: {skin_data}")
                    break

                full_list += data.list
                next_id = data.nextId

                if next_id is None:
                    logger.info(f"已爬取完所有页面，共{page+1}页")
                    break

                # 添加延迟避免请求过快
                await asyncio.sleep(settings.CRAWLER_DELAY_BETWEEN_REQUESTS)

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
            # 分组处理，每组20个物品
            grouped = [items[i:i + 20] for i in range(0, len(items), 20)]

            for group_idx, group in enumerate(grouped):
                logger.info(f"处理第 {group_idx + 1}/{len(grouped)} 组物品")

                # 并发请求这一组的物品详情
                tasks = [
                    asyncio.create_task(fetch_item_details(item.id, "YOUPIN"))
                    for item in group
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for item, result in zip(group, results):
                    if isinstance(result, Exception):
                        logger.error(f"获取物品 {item.name} 详情失败: {result}")
                        continue

                    if result.success and result.data:
                        # 为每条记录添加物品信息
                        for detail in result.data:
                            detail_dict = detail.model_dump()
                            detail_dict['item_id'] = item.id
                            detail_dict['item_name'] = item.name
                            all_details.append(detail_dict)
                    else:
                        logger.warning(f"物品 {item.name} 详情数据为空或请求失败")

                # 组间延迟
                if group_idx < len(grouped) - 1:
                    await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"爬取物品详情数据失败: {e}")
            traceback.print_exc()

        logger.info(f"成功爬取 {len(all_details)} 条详情数据")
        return all_details

    def save_to_database(self, details: List[Dict[str, Any]]) -> bool:
        """保存数据到数据库"""
        if not details:
            logger.info("没有数据需要保存")
            return True

        try:
            # 生成文件ID
            file_id = f"crawl_{int(time.time())}"

            # 保存主文档
            main_doc = {
                'file_id': file_id,
                'success': True,
                'error_code': 0,
                'error_msg': None,
                'error_data': None,
                'error_code_str': None,
                'record_count': len(details),
                'created_at': datetime.now()
            }

            main_result = self.collection.insert_one(main_doc)
            logger.info(f"主文档保存成功，ID: {main_result.inserted_id}")

            # 为每条记录添加file_id
            for detail in details:
                detail['file_id'] = file_id

            # 批量插入详情记录
            if details:
                records_result = self.records_collection.insert_many(details)
                logger.info(f"成功保存 {len(records_result.inserted_ids)} 条详情记录")

            return True

        except Exception as e:
            logger.error(f"保存数据到数据库失败: {e}")
            return False

    async def run_single_crawl(self):
        """运行一次完整的爬取周期"""
        logger.info("开始执行爬取任务")
        start_time = time.time()

        try:
            # 连接数据库
            self.connect_db()

            # 1. 爬取市场物品列表
            market_items = await self.crawl_market_items()
            if not market_items:
                logger.warning("没有获取到市场物品数据")
                return False

            # 2. 爬取物品详情
            item_details = await self.crawl_item_details(market_items)
            if not item_details:
                logger.warning("没有获取到物品详情数据")
                return False

            # 3. 保存到数据库
            success = self.save_to_database(item_details)

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
