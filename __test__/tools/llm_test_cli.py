#!/usr/bin/env python3
"""
CSGO数据获取和格式化工具

提供快速数据获取和LLM格式化功能
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from modules.database import DatabaseManager
    from config import settings
except ImportError as e:
    print(f"❌ 导入模块失败: {e}", file=sys.stderr)
    print("请确保在项目根目录运行", file=sys.stderr)
    sys.exit(1)


class QuickDataRetriever:
    """快速数据获取器"""
    
    def __init__(self):
        """初始化数据获取器"""
        self.db_manager = DatabaseManager()
        self.db_manager.connect()
    
    def close(self):
        """关闭数据库连接"""
        if self.db_manager:
            self.db_manager.disconnect()
    
    def get_sample_data(self, limit: int = 20) -> List[Dict]:
        """获取样本数据"""
        # 生成模拟数据
        sample_data = []
        base_price = 124.5358
        base_time = datetime.now()
        
        for i in range(limit):
            # 模拟价格波动
            price_variation = (i % 10 - 5) * 2.5  # -12.5 到 +12.5 的波动
            current_price = base_price + price_variation
            
            # 模拟时间序列
            timestamp = base_time - timedelta(minutes=i * 15)
            
            record = {
                "timestamp": int(timestamp.timestamp()),
                "price": round(current_price, 4),
                "onSaleQuantity": 50 + (i % 20),
                "seekPrice": round(current_price * 0.95, 4),
                "seekQuantity": 30 + (i % 15),
                "transactionAmount": round(current_price * (10 + i % 5), 2),
                "transcationNum": 5 + (i % 8),
                "surviveNum": 100 + (i % 30),
                "item_name": "★ Butterfly Knife"
            }
            sample_data.append(record)
        
        return sample_data
    
    def get_latest_data(self, limit: int = 20) -> List[Dict]:
        """获取最新数据"""
        try:
            # 尝试从数据库获取数据
            data = self.db_manager.get_market_data(limit=limit)
            if data:
                return list(data)
            else:
                # 如果数据库没有数据，返回样本数据
                print("⚠️ 数据库无数据，使用样本数据", file=sys.stderr)
                return self.get_sample_data(limit)
        except Exception as e:
            print(f"⚠️ 数据库查询失败: {e}，使用样本数据", file=sys.stderr)
            return self.get_sample_data(limit)
    
    def get_hours_data(self, hours: int = 7, limit: int = 50) -> List[Dict]:
        """获取指定小时数内的数据"""
        try:
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # 尝试从数据库获取数据
            data = self.db_manager.get_market_data(
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            if data:
                return list(data)
            else:
                # 如果数据库没有数据，返回样本数据
                print(f"⚠️ 数据库无{hours}小时内数据，使用样本数据", file=sys.stderr)
                return self.get_sample_data(limit)
        except Exception as e:
            print(f"⚠️ 数据库查询失败: {e}，使用样本数据", file=sys.stderr)
            return self.get_sample_data(limit)
    
    def get_price_range_data(self, min_price: float, max_price: float, limit: int = 50) -> List[Dict]:
        """获取指定价格范围内的数据"""
        try:
            # 尝试从数据库获取数据
            # 注意：这里需要根据实际数据库结构调整查询逻辑
            data = self.db_manager.get_market_data(limit=limit * 2)  # 获取更多数据用于筛选
            
            if data:
                # 筛选价格范围内的数据
                filtered_data = []
                for record in data:
                    price = record.get('price', 0)
                    if min_price <= price <= max_price:
                        filtered_data.append(record)
                        if len(filtered_data) >= limit:
                            break
                
                if filtered_data:
                    return filtered_data
            
            # 如果没有找到合适的数据，生成模拟数据
            print(f"⚠️ 数据库无价格范围 {min_price}-{max_price} 的数据，使用样本数据", file=sys.stderr)
            sample_data = self.get_sample_data(limit)
            
            # 调整样本数据的价格到指定范围
            for record in sample_data:
                # 将价格调整到指定范围内
                price_ratio = (record['price'] - 100) / 50  # 假设原始价格在100-150范围
                new_price = min_price + (max_price - min_price) * abs(price_ratio % 1)
                record['price'] = round(new_price, 4)
                record['seekPrice'] = round(new_price * 0.95, 4)
            
            return sample_data
            
        except Exception as e:
            print(f"⚠️ 数据库查询失败: {e}，使用样本数据", file=sys.stderr)
            return self.get_sample_data(limit)
    
    def get_data(self, method: str = 'latest', **kwargs) -> List[Dict]:
        """统一数据获取接口"""
        if method == 'sample':
            return self.get_sample_data(kwargs.get('limit', 20))
        elif method == 'latest':
            return self.get_latest_data(kwargs.get('limit', 20))
        elif method == 'hours':
            return self.get_hours_data(
                hours=kwargs.get('hours', 7),
                limit=kwargs.get('limit', 50)
            )
        elif method == 'price':
            min_price = kwargs.get('min_price')
            max_price = kwargs.get('max_price')
            if min_price is None or max_price is None:
                raise ValueError("价格范围方法需要指定 min_price 和 max_price")
            return self.get_price_range_data(min_price, max_price, kwargs.get('limit', 50))
        else:
            raise ValueError(f"不支持的数据获取方法: {method}")


def format_for_llm(raw_data: List[Dict]) -> Dict[str, Any]:
    """格式化数据为LLM分析所需的格式"""
    if not raw_data:
        return {
            "count": 0,
            "data": [],
            "time_range": {},
            "statistics": {}
        }
    
    # 提取数值数据
    prices = []
    quantities = []
    timestamps = []
    
    for record in raw_data:
        price = record.get('price', 0)
        quantity = record.get('onSaleQuantity', 0)
        timestamp = record.get('timestamp', 0)
        
        if price > 0:
            prices.append(float(price))
        if quantity > 0:
            quantities.append(int(quantity))
        if timestamp > 0:
            timestamps.append(timestamp)
    
    # 计算统计信息
    statistics = {}
    if prices:
        statistics['avg_price'] = sum(prices) / len(prices)
        statistics['price_range'] = [min(prices), max(prices)]
    else:
        statistics['avg_price'] = 0
        statistics['price_range'] = [0, 0]
    
    if quantities:
        statistics['avg_quantity'] = sum(quantities) / len(quantities)
        statistics['quantity_range'] = [min(quantities), max(quantities)]
    else:
        statistics['avg_quantity'] = 0
        statistics['quantity_range'] = [0, 0]
    
    # 计算时间范围
    time_range = {}
    if timestamps:
        earliest_ts = min(timestamps)
        latest_ts = max(timestamps)
        time_range = {
            'earliest': datetime.fromtimestamp(earliest_ts).strftime('%Y-%m-%d %H:%M:%S'),
            'latest': datetime.fromtimestamp(latest_ts).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    return {
        "count": len(raw_data),
        "data": raw_data,
        "time_range": time_range,
        "statistics": statistics
    }


def main():
    """测试函数"""
    retriever = QuickDataRetriever()
    
    try:
        # 测试不同的数据获取方法
        print("测试样本数据获取:")
        sample_data = retriever.get_data('sample', limit=5)
        formatted_sample = format_for_llm(sample_data)
        print(json.dumps(formatted_sample, indent=2, ensure_ascii=False))
        
        print("\n测试最新数据获取:")
        latest_data = retriever.get_data('latest', limit=5)
        formatted_latest = format_for_llm(latest_data)
        print(json.dumps(formatted_latest, indent=2, ensure_ascii=False))
        
    finally:
        retriever.close()


if __name__ == "__main__":
    main()