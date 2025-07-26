#!/usr/bin/env python3
"""测试使用最新7小时内的数据进行预测"""

import json
import os
import sys
from datetime import datetime

import requests

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from tools.llm_test_cli import QuickDataRetriever, format_for_llm


def get_7h_data():
    """获取最新7小时内的数据"""
    print("📊 获取最新7小时内的数据...")

    retriever = QuickDataRetriever()

    try:
        # 使用hours方法获取7小时内的数据
        raw_data = retriever.get_data(method="hours", hours=7, limit=50)

        if not raw_data:
            print("❌ 没有找到7小时内的数据")
            return None

        # 格式化数据
        formatted_data = format_for_llm(raw_data)

        print(f"✅ 成功获取 {formatted_data['count']} 条7小时内的数据")
        print(f"📅 时间范围: {formatted_data['time_range']['earliest']} 到 {formatted_data['time_range']['latest']}")
        print(
            f"💰 价格范围: {formatted_data['statistics']['price_range'][0]:.2f} - {formatted_data['statistics']['price_range'][1]:.2f}")
        print(f"📈 平均价格: {formatted_data['statistics']['avg_price']:.2f}")

        return formatted_data

    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        return None
    finally:
        retriever.close()


def prepare_predict_data(formatted_data):
    """准备预测API所需的数据格式"""
    predict_data = []

    for record in formatted_data.get("data", []):
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

    return predict_data


def send_to_predict_api(predict_data):
    """发送数据到预测API"""
    print(f"\n🤖 发送 {len(predict_data)} 条数据到预测API...")

    payload = {"data": predict_data}

    try:
        print(f"📡 请求地址: {settings.PREDICT_API_URL}")

        response = requests.post(
            settings.PREDICT_API_URL,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )

        print(f"📊 响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ 预测API响应成功")
            return result
        else:
            print(f"❌ 预测API响应失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到预测API服务")
        print("💡 请确保服务器正在运行: uv run cli.py server")
        return None
    except requests.exceptions.Timeout:
        print("❌ 预测API请求超时")
        return None
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return None


def display_prediction_result(result):
    """显示预测结果"""
    if not result:
        return

    print("\n🔮 预测结果:")
    print("=" * 50)

    # 显示预测值
    predictions = result.get("predictions", [])
    if predictions:
        print(f"📈 预测价格: {predictions}")
        print(f"🎯 预测数量: {len(predictions)} 个未来价格点")

        # 显示价格趋势
        if len(predictions) > 1:
            trend_direction = "上涨" if predictions[-1] > predictions[0] else "下跌" if predictions[-1] < predictions[
                0] else "稳定"
            change_percent = ((predictions[-1] - predictions[0]) / predictions[0] * 100) if predictions[0] != 0 else 0
            print(f"📊 价格趋势: {trend_direction} ({change_percent:+.2f}%)")

    # 显示其他指标
    mse = result.get("mse", 0)
    confidence = result.get("confidence", 0)
    trend = result.get("trend", "unknown")

    print(f"📏 均方误差 (MSE): {mse:.4f}")
    print(f"🎯 置信度: {confidence:.3f}")
    print(f"📈 趋势判断: {trend}")

    # 如果有错误信息
    if "error" in result:
        print(f"⚠️ 错误信息: {result['error']}")


def save_test_result(formatted_data, predict_data, result):
    """保存测试结果到文件"""
    test_result = {
        "test_time": datetime.now().isoformat(),
        "input_data_summary": {
            "count": formatted_data.get("count", 0),
            "time_range": formatted_data.get("time_range", {}),
            "statistics": formatted_data.get("statistics", {})
        },
        "predict_data_sample": predict_data[:3],  # 只保存前3条作为样本
        "prediction_result": result,
        "test_config": {
            "hours": 7,
            "api_url": settings.PREDICT_API_URL,
            "data_limit": 50
        }
    }

    filename = f"test_7h_predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 测试结果已保存到: {filename}")
    except Exception as e:
        print(f"⚠️ 保存测试结果失败: {e}")


def main():
    """主测试函数"""
    print("🧪 7小时数据预测测试")
    print("=" * 50)

    # 1. 获取7小时内的数据
    formatted_data = get_7h_data()
    if not formatted_data:
        print("💥 无法获取数据，测试终止")
        return False

    # 2. 准备预测数据
    predict_data = prepare_predict_data(formatted_data)
    if not predict_data:
        print("💥 无法准备预测数据，测试终止")
        return False

    print(f"\n📋 准备发送的数据样本 (前3条):")
    for i, record in enumerate(predict_data[:3], 1):
        time_str = datetime.fromtimestamp(record['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i}. 时间: {time_str}, 价格: {record['price']}, 数量: {record['onSaleQuantity']}")

    # 3. 发送到预测API
    result = send_to_predict_api(predict_data)
    if not result:
        print("💥 预测API调用失败，测试终止")
        return False

    # 4. 显示结果
    display_prediction_result(result)

    # 5. 保存测试结果
    save_test_result(formatted_data, predict_data, result)

    print("\n🎉 测试完成！")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试异常: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
