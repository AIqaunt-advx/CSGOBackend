"""
数据模型定义
包含项目中使用的所有Pydantic模型
"""

from typing import List

from pydantic import BaseModel, model_validator


class TrendDetailsDataItem(BaseModel):
    """趋势详情数据项"""
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
        """解析API返回的数组格式数据"""
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


class PredictionRequest(BaseModel):
    """预测请求模型"""
    data: List[TrendDetailsDataItem]


class PredictionResponse(BaseModel):
    """预测响应模型"""
    predictions: List[float]
    mse: float
    confidence: float
    trend: str = "stable"  # "up", "down", "stable"
