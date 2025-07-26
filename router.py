from fastapi import APIRouter

from modules.analysis import analysis_router
from modules.crawler_api import crawler_router
from modules.market import market_router
from modules.prediction import prediction_router

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

api_router.include_router(
    crawler_router,
    prefix="/crawler",
    tags=["Crawler Management"]
)
