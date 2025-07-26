import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException

from config import settings

market_router = APIRouter()


class MarketDataService:
    """市场数据服务类"""

    def __init__(self):
        self.timeout = settings.REQUEST_TIMEOUT
        self.max_retries = settings.MAX_RETRIES

    async def get_buff_data(self, item_name: str):
        """获取BUFF平台数据"""
        try:
            # 这里可以实现具体的BUFF API调用逻辑
            # 暂时返回模拟数据
            return {
                "platform": "buff",
                "item_name": item_name,
                "sell_price": 100.0,
                "buy_price": 95.0,
                "quantity": 50
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取BUFF数据失败: {str(e)}")

    async def get_steam_data(self, item_name: str):
        """获取Steam市场数据"""
        try:
            # 实现Steam API调用逻辑
            return {
                "platform": "steam",
                "item_name": item_name,
                "sell_price": 98.0,
                "buy_price": 93.0,
                "quantity": 30
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取Steam数据失败: {str(e)}")

    async def get_yyyp_data(self, item_name: str):
        """获取悠悠有品数据"""
        try:
            # 实现悠悠有品API调用逻辑
            return {
                "platform": "yyyp",
                "item_name": item_name,
                "sell_price": 102.0,
                "buy_price": 97.0,
                "quantity": 25
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取悠悠有品数据失败: {str(e)}")


market_service = MarketDataService()


@market_router.get("/price/{item_name}")
async def get_item_price(item_name: str, platforms: Optional[str] = "all"):
    """
    获取指定物品的价格信息

    - **item_name**: 物品名称
    - **platforms**: 平台选择 (all, buff, steam, yyyp)
    """
    try:
        results = {}

        if platforms == "all" or "buff" in platforms:
            results["buff"] = await market_service.get_buff_data(item_name)

        if platforms == "all" or "steam" in platforms:
            results["steam"] = await market_service.get_steam_data(item_name)

        if platforms == "all" or "yyyp" in platforms:
            results["yyyp"] = await market_service.get_yyyp_data(item_name)

        return {
            "status": "success",
            "item_name": item_name,
            "data": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@market_router.get("/compare/{item_name}")
async def compare_prices(item_name: str):
    """
    比较不同平台的价格差异
    """
    try:
        # 并发获取所有平台数据
        buff_data, steam_data, yyyp_data = await asyncio.gather(
            market_service.get_buff_data(item_name),
            market_service.get_steam_data(item_name),
            market_service.get_yyyp_data(item_name)
        )

        # 计算价格差异
        prices = [
            buff_data["sell_price"],
            steam_data["sell_price"],
            yyyp_data["sell_price"]
        ]

        max_price = max(prices)
        min_price = min(prices)
        price_diff = max_price - min_price
        price_diff_percentage = (price_diff / min_price) * 100 if min_price > 0 else 0

        return {
            "status": "success",
            "item_name": item_name,
            "platforms": {
                "buff": buff_data,
                "steam": steam_data,
                "yyyp": yyyp_data
            },
            "analysis": {
                "max_price": max_price,
                "min_price": min_price,
                "price_diff": price_diff,
                "price_diff_percentage": round(price_diff_percentage, 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@market_router.get("/trending")
async def get_trending_items(limit: int = 10):
    """
    获取热门/趋势物品列表
    """
    try:
        # 这里可以实现获取热门物品的逻辑
        trending_items = [
                             {
                                 "name": "AK-47 | Redline (Field-Tested)",
                                 "current_price": 125.0,
                                 "change_24h": 5.2,
                                 "volume_24h": 450
                             },
                             {
                                 "name": "AWP | Dragon Lore (Factory New)",
                                 "current_price": 8500.0,
                                 "change_24h": -2.1,
                                 "volume_24h": 12
                             }
                         ][:limit]

        return {
            "status": "success",
            "trending_items": trending_items,
            "total": len(trending_items)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
