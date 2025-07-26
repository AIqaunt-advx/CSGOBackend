#!/usr/bin/env python3
"""测试curl版本的API调用"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.crawler import fetch_skin_market_data, fetch_item_details

async def test_curl_api():
    """测试curl版本的API调用"""
    print("🚀 测试curl版本的API调用...")
    
    try:
        # 测试获取市场数据
        print("1. 测试获取市场数据...")
        market_data = await fetch_skin_market_data()
        
        if market_data is not None:
            print(f"✅ 成功获取市场数据")
            print(f"   成功状态: {market_data.success}")
            
            if market_data.data and market_data.data.list:
                print(f"   物品数量: {len(market_data.data.list)}")
                
                # 测试获取第一个物品的详情
                first_item = market_data.data.list[0]
                print(f"2. 测试获取物品详情: {first_item.name}")
                
                details = await fetch_item_details(first_item.id, "YOUPIN")
                
                if details is not None:
                    print(f"✅ 成功获取物品详情")
                    print(f"   成功状态: {details.success}")
                    
                    if details.data:
                        print(f"   详情记录数: {len(details.data)}")
                        
                        # 显示第一条记录
                        if details.data:
                            first_record = details.data[0]
                            print(f"   样本数据: 时间戳={first_record.timestamp}, 价格={first_record.price}")
                    else:
                        print("   详情数据为空")
                else:
                    print("❌ 获取物品详情失败")
            else:
                print("   市场数据为空")
        else:
            print("❌ 获取市场数据失败")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_curl_api())