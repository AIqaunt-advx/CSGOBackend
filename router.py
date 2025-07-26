from fastapi import APIRouter
from modules.market import market_router
from modules.prediction import prediction_router
from modules.analysis import analysis_router

# 创建主路由器
api_router = APIRouter()

# 包含各个模块的路由
api_router.include_router(
    market_router,
    prefix="/market",
    tags=["Market Data"]
)

api_router.include_router(
    prediction_router,
    prefix="/prediction",
    tags=["Price Prediction"]
)

api_router.include_router(
    analysis_router,
    prefix="/analysis",
    tags=["Data Analysis"]
)
