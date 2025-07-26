#!/usr/bin/env python3
"""测试预测API"""

import json
import os
import sys

import requests

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings


def test_predict_api():
    """测试预测API"""
    print("🚀 测试预测API...")

    # 准备测试数据
    test_data = {
        "data": [
            {
                "timestamp": 1640995200,
                "price": 100.5,
                "onSaleQuantity": 50,
                "seekPrice": 95.0,
                "seekQuantity": 10,
                "transactionAmount": 1000.0,
                "transcationNum": 10,
                "surviveNum": 5
            },
            {
                "timestamp": 1640995260,
                "price": 102.0,
                "onSaleQuantity": 45,
                "seekPrice": 97.0,
                "seekQuantity": 12,
                "transactionAmount": 1200.0,
                "transcationNum": 12,
                "surviveNum": 6
            },
            {
                "timestamp": 1640995320,
                "price": 98.5,
                "onSaleQuantity": 55,
                "seekPrice": 93.0,
                "seekQuantity": 8,
                "transactionAmount": 800.0,
                "transcationNum": 8,
                "surviveNum": 4
            }
        ]
    }

    try:
        print(f"📡 发送请求到: {settings.PREDICT_API_URL}")

        response = requests.post(
            settings.PREDICT_API_URL,
            json=test_data,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )

        print(f"📊 响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ 预测API响应成功")
            print("📋 预测结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ 预测API响应失败: {response.status_code}")
            print(f"错误信息: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到预测API服务")
        print("💡 请确保服务器正在运行: uv run main.py")
    except requests.exceptions.Timeout:
        print("❌ 预测API请求超时")
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    test_predict_api()
