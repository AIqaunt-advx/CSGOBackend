#!/usr/bin/env python3
"""启动服务器的简单脚本"""

import uvicorn

from config import settings

if __name__ == "__main__":
    print(f"🚀 启动 {settings.APP_NAME}")
    print(f"📍 地址: http://{settings.HOST}:{settings.PORT}")
    print(f"📖 API文档: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"🔮 预测端点: http://{settings.HOST}:{settings.PORT}/predict")
    print("按 Ctrl+C 停止服务器")

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
