#!/usr/bin/env python3
"""LLM测试CLI工具 - 快速获取数据用于LLM请求测试"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta

from pymongo import MongoClient

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import settings
except ImportError:
    print("❌ 无法导入config模块，请确保在项目根目录运行", file=sys.stderr)
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.WARNING)  # 减少日志输出
logger = logging.getLogger(__name__)

# MongoDB配置
MONGODB_CONFIG = {
    "uri": settings.MONGODB_URL,
    "database": settings.MONGODB_DATABASE,
    "collection": settings.MONGODB_COLLECTION_MARKET_DATA
}


class QuickDataRetriever:
    def __init__(self):
        """快速数据检索器"""
        try:
            self.client = MongoClient(MONGODB_CONFIG["uri"], serverSelectionTimeoutMS=5000)
            self.db = self.client[MONGODB_CONFIG["database"]]
            self.records_collection = self.db[f"{MONGODB_CONFIG['collection']}_records"]
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}", file=sys.stderr)
            sys.exit(1)

    def get_data(self, method: str = "latest", limit: int = 10, hours: int = 7,
                 min_price: float = None, max_price: float = None) -> list:
        """统一的数据获取方法"""
        try:
            projection = {
                "_id": 0,
                "timestamp": 1,
                "price": 1,
                "onSaleQuantity": 1,
                "seekPrice": 1,
                "seekQuantity": 1,
                "transactionAmount": 1,
                "transcationNum": 1,
                "surviveNum": 1
            }

            if method == "sample":
                # 随机样本
                pipeline = [
                    {"$sample": {"size": limit}},
                    {"$project": projection}
                ]
                return list(self.records_collection.aggregate(pipeline))

            elif method == "latest":
                # 最新数据
                return list(self.records_collection.find({}, projection)
                            .sort("timestamp", -1).limit(limit))

            elif method == "hours":
                # 最近N小时
                now = datetime.now()
                hours_ago = now - timedelta(hours=hours)
                query = {
                    "timestamp": {
                        "$gte": int(hours_ago.timestamp()),
                        "$lte": int(now.timestamp())
                    }
                }
                return list(self.records_collection.find(query, projection)
                            .sort("timestamp", -1).limit(limit))

            elif method == "price":
                # 价格范围
                query = {}
                if min_price is not None or max_price is not None:
                    price_query = {}
                    if min_price is not None:
                        price_query["$gte"] = min_price
                    if max_price is not None:
                        price_query["$lte"] = max_price
                    query["price"] = price_query

                return list(self.records_collection.find(query, projection).limit(limit))

        except Exception as e:
            print(f"❌ 获取数据失败: {e}", file=sys.stderr)
            return []

    def close(self):
        """关闭连接"""
        self.client.close()


def format_for_llm(data: list) -> dict:
    """格式化数据用于LLM请求"""
    if not data:
        return {"error": "没有找到数据", "data": []}

    # 清理和格式化数据
    formatted_data = []
    for record in data:
        formatted_record = {
            "timestamp": record.get('timestamp', 0),
            "datetime": datetime.fromtimestamp(record.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
            "price": float(record.get('price', 0)),
            "onSaleQuantity": record.get('onSaleQuantity', 0),
            "seekPrice": float(record.get('seekPrice', 0)),
            "seekQuantity": record.get('seekQuantity', 0),
            "transactionAmount": float(record.get('transactionAmount') or 0),
            "transcationNum": record.get('transcationNum') or 0,
            "surviveNum": record.get('surviveNum') or 0
        }
        formatted_data.append(formatted_record)

    # 添加统计信息
    prices = [r['price'] for r in formatted_data]
    quantities = [r['onSaleQuantity'] for r in formatted_data]

    return {
        "data": formatted_data,
        "count": len(formatted_data),
        "statistics": {
            "price_range": [min(prices), max(prices)] if prices else [0, 0],
            "avg_price": sum(prices) / len(prices) if prices else 0,
            "quantity_range": [min(quantities), max(quantities)] if quantities else [0, 0],
            "avg_quantity": sum(quantities) / len(quantities) if quantities else 0
        },
        "time_range": {
            "earliest": formatted_data[-1]['datetime'] if formatted_data else None,
            "latest": formatted_data[0]['datetime'] if formatted_data else None
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLM测试数据获取工具')
    parser.add_argument('--method', '-m', choices=['sample', 'latest', 'hours', 'price'],
                        default='latest', help='数据获取方法')
    parser.add_argument('--limit', '-l', type=int, default=10, help='获取数量限制')
    parser.add_argument('--hours', type=int, default=7, help='小时数 (用于hours方法)')
    parser.add_argument('--min-price', type=float, help='最小价格 (用于price方法)')
    parser.add_argument('--max-price', type=float, help='最大价格 (用于price方法)')
    parser.add_argument('--format', '-f', choices=['json', 'pretty'], default='json',
                        help='输出格式')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式，只输出数据')

    args = parser.parse_args()

    if not args.quiet:
        print(f"🚀 获取{args.method}数据 (限制: {args.limit})", file=sys.stderr)

    retriever = QuickDataRetriever()

    try:
        # 获取数据
        data = retriever.get_data(
            method=args.method,
            limit=args.limit,
            hours=args.hours,
            min_price=args.min_price,
            max_price=args.max_price
        )

        # 格式化输出
        if args.format == 'json':
            result = format_for_llm(data)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # 简单表格格式
            if data:
                print(f"{'时间':<20} {'价格':<8} {'数量':<8} {'求购价':<8}")
                print("-" * 50)
                for record in data[:10]:  # 只显示前10条
                    time_str = datetime.fromtimestamp(record.get('timestamp', 0)).strftime('%m-%d %H:%M:%S')
                    price = record.get('price', 0)
                    qty = record.get('onSaleQuantity', 0)
                    seek_price = record.get('seekPrice', 0)
                    print(f"{time_str:<20} {price:<8.2f} {qty:<8} {seek_price:<8.2f}")
            else:
                print("❌ 没有找到数据")

    except KeyboardInterrupt:
        if not args.quiet:
            print("\n👋 用户中断", file=sys.stderr)
    except Exception as e:
        print(f"❌ 执行出错: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
