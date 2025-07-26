from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, timedelta
from config import settings

analysis_router = APIRouter()

class DataAnalyzer:
    """数据分析服务类"""

    def __init__(self):
        pass

    def calculate_price_statistics(self, prices: List[float]) -> Dict:
        """计算价格统计信息"""
        if not prices:
            return {}

        prices_array = np.array(prices)
        return {
            "mean": round(float(np.mean(prices_array)), 2),
            "median": round(float(np.median(prices_array)), 2),
            "std": round(float(np.std(prices_array)), 2),
            "min": round(float(np.min(prices_array)), 2),
            "max": round(float(np.max(prices_array)), 2),
            "variance": round(float(np.var(prices_array)), 2)
        }

    def calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """计算两个序列的相关性"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        correlation = np.corrcoef(x, y)[0, 1]
        return round(float(correlation), 3) if not np.isnan(correlation) else 0.0

    def detect_anomalies(self, prices: List[float], threshold: float = 2.0) -> List[int]:
        """检测价格异常值"""
        if len(prices) < 3:
            return []

        prices_array = np.array(prices)
        mean = np.mean(prices_array)
        std = np.std(prices_array)

        anomalies = []
        for i, price in enumerate(prices):
            z_score = abs((price - mean) / std) if std > 0 else 0
            if z_score > threshold:
                anomalies.append(i)

        return anomalies

analyzer = DataAnalyzer()

@analysis_router.get("/statistics/{item_name}")
async def get_item_statistics(item_name: str, days: int = 30):
    """
    获取物品的统计分析数据

    - **item_name**: 物品名称
    - **days**: 分析天数
    """
    try:
        # 模拟生成历史数据
        base_price = 100.0
        prices = []
        volumes = []

        for i in range(days):
            price = base_price + np.random.normal(0, 10) + (i * 0.2)
            volume = np.random.randint(20, 200)
            prices.append(max(price, 1.0))
            volumes.append(volume)

        price_stats = analyzer.calculate_price_statistics(prices)
        volume_stats = analyzer.calculate_price_statistics(volumes)

        # 计算价格与交易量的相关性
        price_volume_correlation = analyzer.calculate_correlation(prices, volumes)

        # 检测异常值
        price_anomalies = analyzer.detect_anomalies(prices)

        return {
            "status": "success",
            "item_name": item_name,
            "analysis_period": f"{days} days",
            "price_statistics": price_stats,
            "volume_statistics": volume_stats,
            "price_volume_correlation": price_volume_correlation,
            "anomalies": {
                "count": len(price_anomalies),
                "indices": price_anomalies
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@analysis_router.get("/market-overview")
async def get_market_overview():
    """
    获取市场整体概况分析
    """
    try:
        # 模拟市场数据
        market_data = {
            "total_items": 15420,
            "active_traders": 8930,
            "daily_volume": 2850000.0,
            "avg_price_change_24h": 2.3,
            "top_categories": [
                {"name": "Rifles", "volume": 850000, "change": 3.2},
                {"name": "Knives", "volume": 650000, "change": -1.8},
                {"name": "Pistols", "volume": 420000, "change": 5.1},
                {"name": "Gloves", "volume": 380000, "change": 1.9}
            ],
            "volatility_index": 0.15,
            "market_sentiment": "bullish"
        }

        return {
            "status": "success",
            "market_overview": market_data,
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@analysis_router.get("/price-distribution/{category}")
async def get_price_distribution(category: str, min_price: float = 0, max_price: float = 10000):
    """
    获取特定类别的价格分布分析

    - **category**: 物品类别
    - **min_price**: 最低价格
    - **max_price**: 最高价格
    """
    try:
        # 模拟生成价格分布数据
        price_ranges = [
            "0-50", "50-100", "100-250", "250-500",
            "500-1000", "1000-2500", "2500-5000", "5000+"
        ]

        distribution_data = []
        for price_range in price_ranges:
            count = np.random.randint(10, 500)
            percentage = round(count / 2000 * 100, 1)
            distribution_data.append({
                "price_range": price_range,
                "count": count,
                "percentage": percentage
            })

        return {
            "status": "success",
            "category": category,
            "price_filter": {
                "min_price": min_price,
                "max_price": max_price
            },
            "distribution": distribution_data,
            "total_items": sum(item["count"] for item in distribution_data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@analysis_router.get("/volatility/{item_name}")
async def analyze_volatility(item_name: str, period: int = 30):
    """
    分析物品价格波动性

    - **item_name**: 物品名称
    - **period**: 分析周期（天）
    """
    try:
        # 模拟生成价格数据
        base_price = 100.0
        prices = []
        returns = []

        for i in range(period):
            if i == 0:
                price = base_price
            else:
                daily_return = np.random.normal(0, 0.02)  # 2% 日波动率
                price = prices[i-1] * (1 + daily_return)
                returns.append(daily_return)

            prices.append(max(price, 1.0))

        # 计算波动性指标
        if returns:
            volatility = np.std(returns) * np.sqrt(365)  # 年化波动率
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0

        # 计算价格波动范围
        max_price = max(prices)
        min_price = min(prices)
        price_range = max_price - min_price
        range_percentage = (price_range / min_price) * 100 if min_price > 0 else 0

        return {
            "status": "success",
            "item_name": item_name,
            "analysis_period": period,
            "volatility_metrics": {
                "annualized_volatility": round(volatility, 4),
                "sharpe_ratio": round(sharpe_ratio, 4),
                "price_range": round(price_range, 2),
                "range_percentage": round(range_percentage, 2),
                "max_price": round(max_price, 2),
                "min_price": round(min_price, 2)
            },
            "risk_level": "High" if volatility > 0.3 else "Medium" if volatility > 0.15 else "Low"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@analysis_router.get("/arbitrage-opportunities")
async def find_arbitrage_opportunities(min_profit_margin: float = 5.0):
    """
    寻找套利机会

    - **min_profit_margin**: 最小利润率（百分比）
    """
    try:
        # 模拟套利机会数据
        opportunities = [
            {
                "item_name": "AK-47 | Redline (Field-Tested)",
                "buy_platform": "Steam",
                "sell_platform": "BUFF",
                "buy_price": 95.0,
                "sell_price": 110.0,
                "profit": 15.0,
                "profit_margin": 15.8,
                "risk_score": 0.3
            },
            {
                "item_name": "AWP | Asiimov (Field-Tested)",
                "buy_platform": "YYYP",
                "sell_platform": "Steam",
                "buy_price": 280.0,
                "sell_price": 320.0,
                "profit": 40.0,
                "profit_margin": 14.3,
                "risk_score": 0.4
            }
        ]

        # 过滤符合最小利润率要求的机会
        filtered_opportunities = [
            opp for opp in opportunities
            if opp["profit_margin"] >= min_profit_margin
        ]

        return {
            "status": "success",
            "arbitrage_opportunities": filtered_opportunities,
            "total_opportunities": len(filtered_opportunities),
            "filters": {
                "min_profit_margin": min_profit_margin
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
