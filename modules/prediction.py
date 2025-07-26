from datetime import datetime, timedelta
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException

from modules.models import TrendDetailsDataItem, PredictionRequest, PredictionResponse

prediction_router = APIRouter()


class PricePredictor:
    """价格预测服务类"""

    def __init__(self):
        self.model_loaded = False

    def prepare_features(self, data: List[TrendDetailsDataItem]) -> np.ndarray:
        """准备机器学习特征"""
        features = []
        for item in data:
            feature_vector = [
                item.price,
                item.onSaleQuantity,
                item.seekPrice,
                item.seekQuantity,
                item.transactionAmount,
                item.transcationNum,
                item.surviveNum
            ]
            features.append(feature_vector)
        return np.array(features)

    def simple_linear_prediction(self, prices: List[float], steps: int = 5) -> List[float]:
        """简单线性预测"""
        if len(prices) < 2:
            return [prices[-1]] * steps if prices else [0.0] * steps

        # 计算价格变化趋势
        price_changes = []
        for i in range(1, len(prices)):
            price_changes.append(prices[i] - prices[i - 1])

        # 计算平均变化率
        avg_change = sum(price_changes) / len(price_changes)

        # 生成预测
        predictions = []
        last_price = prices[-1]
        for i in range(steps):
            next_price = last_price + avg_change * (i + 1)
            predictions.append(max(0, next_price))  # 确保价格不为负

        return predictions

    def calculate_mse(self, actual: List[float], predicted: List[float]) -> float:
        """计算均方误差"""
        if len(actual) != len(predicted):
            return 0.0

        mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
        return mse

    def predict_prices(self, data: List[TrendDetailsDataItem]) -> PredictionResponse:
        """预测价格"""
        try:
            prices = [item.price for item in data]

            # 使用简单线性预测
            predictions = self.simple_linear_prediction(prices)

            # 计算置信度（基于历史数据的稳定性）
            if len(prices) > 1:
                price_volatility = np.std(prices)
                avg_price = np.mean(prices)
                confidence = max(0.1, 1.0 - (price_volatility / avg_price))
            else:
                confidence = 0.5

            # 计算MSE（这里使用模拟值）
            mse = price_volatility if len(prices) > 1 else 0.0
            
            # 分析趋势
            if len(prices) >= 2:
                recent_change = prices[-1] - prices[0]
                if recent_change > avg_price * 0.05:  # 上涨超过5%
                    trend = "up"
                elif recent_change < -avg_price * 0.05:  # 下跌超过5%
                    trend = "down"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            return PredictionResponse(
                predictions=predictions,
                mse=mse,
                confidence=round(confidence, 3),
                trend=trend
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


predictor = PricePredictor()


@prediction_router.post("/predict")
async def predict_item_price(request: PredictionRequest):
    """
    根据历史数据预测物品价格

    - **data**: 历史趋势数据列表
    """
    try:
        if not request.data:
            raise HTTPException(status_code=400, detail="数据不能为空")

        result = predictor.predict_prices(request.data)

        return {
            "status": "success",
            "predictions": result.predictions,
            "mse": result.mse,
            "confidence": result.confidence,
            "data_points": len(request.data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@prediction_router.get("/trend/{item_name}")
async def get_price_trend(item_name: str, days: int = 30):
    """
    获取物品价格趋势分析

    - **item_name**: 物品名称
    - **days**: 分析天数
    """
    try:
        # 模拟生成历史价格数据
        base_price = 100.0
        trend_data = []

        for i in range(days):
            date = datetime.now() - timedelta(days=days - i)
            # 添加一些随机波动
            price_variation = np.random.normal(0, 5)
            price = max(base_price + price_variation + (i * 0.5), 1.0)

            trend_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(price, 2),
                "volume": np.random.randint(10, 100)
            })

        # 计算趋势指标
        prices = [item["price"] for item in trend_data]
        price_change = prices[-1] - prices[0]
        price_change_percentage = (price_change / prices[0]) * 100

        return {
            "status": "success",
            "item_name": item_name,
            "trend_data": trend_data,
            "analysis": {
                "start_price": prices[0],
                "end_price": prices[-1],
                "price_change": round(price_change, 2),
                "price_change_percentage": round(price_change_percentage, 2),
                "volatility": round(np.std(prices), 2),
                "avg_price": round(np.mean(prices), 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@prediction_router.get("/forecast/{item_name}")
async def forecast_price(item_name: str, horizon: int = 7):
    """
    预测未来价格走势

    - **item_name**: 物品名称
    - **horizon**: 预测天数
    """
    try:
        # 模拟获取历史数据并进行预测
        # 这里应该从数据库或缓存中获取真实的历史数据
        historical_prices = [100 + i + np.random.normal(0, 2) for i in range(30)]

        # 生成预测
        predictions = predictor.simple_linear_prediction(historical_prices, horizon)

        # 生成预测日期
        forecast_dates = []
        for i in range(horizon):
            future_date = datetime.now() + timedelta(days=i + 1)
            forecast_dates.append(future_date.strftime("%Y-%m-%d"))

        forecast_data = [
            {"date": date, "predicted_price": round(price, 2)}
            for date, price in zip(forecast_dates, predictions)
        ]

        return {
            "status": "success",
            "item_name": item_name,
            "forecast_horizon": horizon,
            "forecast_data": forecast_data,
            "confidence_level": 0.75
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
