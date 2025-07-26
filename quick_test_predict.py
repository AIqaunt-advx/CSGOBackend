#!/usr/bin/env python3
"""快速测试7小时数据预测"""

import os
import sys

import requests

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings


def quick_test():
    """快速测试预测API"""
    print("⚡ 快速预测测试")
    print("-" * 30)

    # 获取7小时内的数据
    print("1. 获取7小时内的数据...")
    try:
        from tools.llm_test_cli import QuickDataRetriever

        retriever = QuickDataRetriever()
        raw_data = retriever.get_data(method="hours", hours=7, limit=10)
        retriever.close()

        if not raw_data:
            print("❌ 没有7小时内的数据")
            return False

        print(f"✅ 获取到 {len(raw_data)} 条数据")

    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        return False

    # 准备预测数据
    print("2. 准备预测数据...")
    predict_data = []
    for record in raw_data:
        predict_record = {
            "timestamp": record.get("timestamp", 0),
            "price": float(record.get("price", 0)),
            "onSaleQuantity": record.get("onSaleQuantity", 0),
            "seekPrice": float(record.get("seekPrice", 0)),
            "seekQuantity": record.get("seekQuantity", 0),
            "transactionAmount": float(record.get("transactionAmount") or 0),
            "transcationNum": record.get("transcationNum") or 0,
            "surviveNum": record.get("surviveNum") or 0
        }
        predict_data.append(predict_record)

    print(f"✅ 准备了 {len(predict_data)} 条预测数据")

    # 发送到预测API
    print("3. 调用预测API...")
    try:
        payload = {"data": predict_data}
        response = requests.post(settings.PREDICT_API_URL, json=payload, timeout=15)

        if response.status_code == 200:
            result = response.json()
            print("✅ 预测成功")

            # 显示关键结果
            predictions = result.get("predictions", [])
            confidence = result.get("confidence", 0)
            trend = result.get("trend", "unknown")

            print(f"📈 预测结果: {predictions}")
            print(f"🎯 置信度: {confidence:.3f}")
            print(f"📊 趋势: {trend}")

            return True
        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"错误: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到预测API")
        print("💡 请先启动服务器: uv run cli.py server")
        return False
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return False


if __name__ == "__main__":
    success = quick_test()
    print("\n" + ("🎉 测试通过" if success else "💥 测试失败"))
