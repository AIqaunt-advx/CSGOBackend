"""LLM请求测试脚本 - 从MongoDB获取数据并美化输出"""
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any

from pymongo import MongoClient

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import settings
except ImportError:
    print("❌ 无法导入config模块，请确保在项目根目录运行", file=sys.stderr)
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 兼容性配置
MONGODB_CONFIG = {
    "uri": settings.MONGODB_URL,
    "database": settings.MONGODB_DATABASE,
    "collection": settings.MONGODB_COLLECTION_MARKET_DATA
}


class DataRetriever:
    def __init__(self):
        """初始化数据检索器"""
        self.client = MongoClient(
            MONGODB_CONFIG["uri"],
            serverSelectionTimeoutMS=settings.MONGODB_CONNECTION_TIMEOUT,
            maxPoolSize=settings.MONGODB_MAX_POOL_SIZE
        )
        self.db = self.client[MONGODB_CONFIG["database"]]
        self.records_collection = self.db[f"{MONGODB_CONFIG['collection']}_records"]

    def get_sample_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """从数据库获取样本数据"""
        try:
            # 获取随机样本数据
            pipeline = [
                {"$sample": {"size": limit}},
                {"$project": {
                    "_id": 0,
                    "timestamp": 1,
                    "price": 1,
                    "onSaleQuantity": 1,
                    "seekPrice": 1,
                    "seekQuantity": 1,
                    "transactionAmount": 1,
                    "transcationNum": 1,
                    "surviveNum": 1
                }}
            ]
            cursor = self.records_collection.aggregate(pipeline)
            data = list(cursor)
            logger.info(f"成功获取 {len(data)} 条样本数据")
            return data
        except Exception as e:
            logger.error(f"获取样本数据时出错: {e}")
            return []

    def get_latest_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最新的数据"""
        try:
            cursor = self.records_collection.find(
                {},
                {
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
            ).sort("timestamp", -1).limit(limit)
            data = list(cursor)
            logger.info(f"成功获取 {len(data)} 条最新数据")
            return data
        except Exception as e:
            logger.error(f"获取最新数据时出错: {e}")
            return []

    def get_latest_hours_data(self, hours: int = 7) -> List[Dict[str, Any]]:
        """获取最近N小时的数据"""
        try:
            # 计算N小时前的时间戳
            now = datetime.now()
            hours_ago = now - timedelta(hours=hours)
            hours_ago_timestamp = int(hours_ago.timestamp())

            logger.info(
                f"获取时间范围: {hours_ago.strftime('%Y-%m-%d %H:%M:%S')} 到 {now.strftime('%Y-%m-%d %H:%M:%S')}")

            # 查询最近N小时的数据
            query = {
                "timestamp": {
                    "$gte": hours_ago_timestamp,
                    "$lte": int(now.timestamp())
                }
            }

            projection = {
                "_id": 0,
                "timestamp": 1,
                "price": 1,
                "onSaleQuantity": 1,
                "seekPrice": 1,
                "seekQuantity": 1,
                "transactionAmount": 1,
                "transcationNum": 1,
                "surviveNum": 1,
                "file_id": 1
            }

            cursor = self.records_collection.find(query, projection).sort("timestamp", -1)
            data = list(cursor)
            logger.info(f"成功获取最近{hours}小时的 {len(data)} 条数据")
            return data
        except Exception as e:
            logger.error(f"获取最近{hours}小时数据时出错: {e}")
            return []

    def get_price_range_data(self, min_price: float = None, max_price: float = None, limit: int = 10) -> List[
        Dict[str, Any]]:
        """获取指定价格范围的数据"""
        try:
            query = {}
            if min_price is not None or max_price is not None:
                price_query = {}
                if min_price is not None:
                    price_query["$gte"] = min_price
                if max_price is not None:
                    price_query["$lte"] = max_price
                query["price"] = price_query

            cursor = self.records_collection.find(
                query,
                {
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
            ).limit(limit)
            data = list(cursor)
            logger.info(f"成功获取价格范围 {min_price}-{max_price} 的 {len(data)} 条数据")
            return data
        except Exception as e:
            logger.error(f"获取价格范围数据时出错: {e}")
            return []

    def save_single_record(self, record_data: Dict[str, Any]) -> str:
        """保存单条记录到数据库"""
        try:
            result = self.records_collection.insert_one(record_data)
            logger.info(f"成功插入记录，ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"插入记录时出错: {e}")
            return None

    def save_multiple_records(self, records_list: List[Dict[str, Any]]) -> List[str]:
        """批量保存多条记录到数据库"""
        try:
            result = self.records_collection.insert_many(records_list)
            logger.info(f"成功批量插入 {len(result.inserted_ids)} 条记录")
            return [str(id) for id in result.inserted_ids]
        except Exception as e:
            logger.error(f"批量插入记录时出错: {e}")
            return None

    def close(self):
        """关闭数据库连接"""
        self.client.close()


def format_timestamp(timestamp: int) -> str:
    """格式化时间戳"""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(timestamp)


def print_beautiful_data(data: List[Dict[str, Any]], title: str = "数据样本"):
    """美化打印数据"""
    if not data:
        print(f"\n❌ {title}: 没有找到数据")
        return

    print(f"\n🎯 {title}")
    print("=" * 80)

    # 打印表头
    print(
        f"{'序号':<4} {'时间':<20} {'价格':<8} {'在售量':<8} {'求购价':<8} {'求购量':<8} {'交易额':<10} {'交易数':<8} {'存活数':<8}")
    print("-" * 80)

    # 打印数据行
    for i, record in enumerate(data, 1):
        timestamp_str = format_timestamp(record.get('timestamp', 0))
        price = record.get('price', 0)
        on_sale_qty = record.get('onSaleQuantity', 0)
        seek_price = record.get('seekPrice', 0)
        seek_qty = record.get('seekQuantity', 0)
        trans_amount = record.get('transactionAmount') or 0
        trans_num = record.get('transcationNum') or 0
        survive_num = record.get('surviveNum') or 0

        print(
            f"{i:<4} {timestamp_str:<20} {price:<8.2f} {on_sale_qty:<8} {seek_price:<8.2f} {seek_qty:<8} {trans_amount:<10.2f} {trans_num:<8} {survive_num:<8}")

    print("-" * 80)
    print(f"📊 总计: {len(data)} 条记录")


def print_json_format(data: List[Dict[str, Any]], title: str = "JSON格式数据"):
    """以JSON格式美化打印数据"""
    if not data:
        print(f"\n❌ {title}: 没有找到数据")
        return

    print(f"\n📋 {title}")
    print("=" * 60)

    # 转换为更适合LLM请求的格式
    formatted_data = []
    for record in data:
        formatted_record = {
            "timestamp": record.get('timestamp', 0),
            "price": float(record.get('price', 0)),
            "onSaleQuantity": record.get('onSaleQuantity', 0),
            "seekPrice": float(record.get('seekPrice', 0)),
            "seekQuantity": record.get('seekQuantity', 0),
            "transactionAmount": float(record.get('transactionAmount') or 0),
            "transcationNum": record.get('transcationNum') or 0,
            "surviveNum": record.get('surviveNum') or 0
        }
        formatted_data.append(formatted_record)

    # 美化打印JSON
    json_str = json.dumps(formatted_data, indent=2, ensure_ascii=False)
    print(json_str)
    print("\n" + "=" * 60)
    print(f"🔗 可直接用于LLM请求的数据格式")


def print_statistics(data: List[Dict[str, Any]]):
    """打印数据统计信息"""
    if not data:
        return

    prices = [record.get('price', 0) for record in data]
    quantities = [record.get('onSaleQuantity', 0) for record in data]

    print(f"\n📈 数据统计")
    print("-" * 40)
    print(f"价格范围: {min(prices):.2f} - {max(prices):.2f}")
    print(f"平均价格: {sum(prices) / len(prices):.2f}")
    print(f"数量范围: {min(quantities)} - {max(quantities)}")
    print(f"平均数量: {sum(quantities) / len(quantities):.1f}")


def main():
    """主函数"""
    print("🚀 LLM请求数据获取工具")
    print("=" * 50)

    retriever = DataRetriever()

    try:
        # 显示菜单
        print("\n请选择数据获取方式:")
        print("1. 获取随机样本数据")
        print("2. 获取最新数据")
        print("3. 获取指定价格范围数据")
        print("4. 获取最近N小时数据")
        print("5. 获取所有类型的数据展示")

        choice = input("\n请输入选择 (1-5, 默认5): ").strip()
        if not choice:
            choice = "5"

        if choice == "1":
            limit = int(input("请输入获取数量 (默认10): ") or "10")
            data = retriever.get_sample_data(limit)
            print_beautiful_data(data, "随机样本数据")
            print_json_format(data, "随机样本数据 (JSON格式)")
            print_statistics(data)

        elif choice == "2":
            limit = int(input("请输入获取数量 (默认10): ") or "10")
            data = retriever.get_latest_data(limit)
            print_beautiful_data(data, "最新数据")
            print_json_format(data, "最新数据 (JSON格式)")
            print_statistics(data)

        elif choice == "3":
            min_price = input("请输入最小价格 (可选): ").strip()
            max_price = input("请输入最大价格 (可选): ").strip()
            limit = int(input("请输入获取数量 (默认10): ") or "10")

            min_p = float(min_price) if min_price else None
            max_p = float(max_price) if max_price else None

            data = retriever.get_price_range_data(min_p, max_p, limit)
            print_beautiful_data(data, f"价格范围数据 ({min_p}-{max_p})")
            print_json_format(data, f"价格范围数据 (JSON格式)")
            print_statistics(data)

        elif choice == "4":
            hours = int(input("请输入小时数 (默认7): ") or "7")
            data = retriever.get_latest_hours_data(hours)
            print_beautiful_data(data, f"最近{hours}小时数据")
            print_json_format(data[:10], f"最近{hours}小时数据 (JSON格式, 前10条)")
            print_statistics(data)

        elif choice == "5":
            # 展示所有类型的数据
            print("\n🎪 完整数据展示")

            # 1. 随机样本
            sample_data = retriever.get_sample_data(5)
            print_beautiful_data(sample_data, "随机样本数据 (5条)")

            # 2. 最新数据
            latest_data = retriever.get_latest_data(5)
            print_beautiful_data(latest_data, "最新数据 (5条)")

            # 3. 高价格数据
            high_price_data = retriever.get_price_range_data(min_price=4.0, limit=5)
            print_beautiful_data(high_price_data, "高价格数据 (≥4.0, 5条)")

            # 4. 最近7小时数据
            recent_data = retriever.get_latest_hours_data(7)
            print_beautiful_data(recent_data[:5], "最近7小时数据 (前5条)")

            # 5. JSON格式输出 (用于LLM请求)
            print_json_format(sample_data, "LLM请求样本数据")

            # 6. 统计信息
            all_data = sample_data + latest_data + high_price_data
            print_statistics(all_data)

        else:
            print("❌ 无效选择")

    except KeyboardInterrupt:
        print("\n\n👋 用户中断操作")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
    finally:
        retriever.close()
        print("\n✅ 数据库连接已关闭")


if __name__ == "__main__":
    main()
