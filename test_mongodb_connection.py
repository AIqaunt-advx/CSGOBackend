#!/usr/bin/env python3
"""测试MongoDB连接"""

from pymongo import MongoClient
from config import settings
import sys

def test_connection():
    """测试MongoDB连接"""
    print("🔍 测试MongoDB连接...")
    print(f"连接地址: {settings.MONGODB_URL}")
    print(f"数据库: {settings.MONGODB_DATABASE}")
    print(f"集合前缀: {settings.MONGODB_COLLECTION_MARKET_DATA}")
    
    try:
        # 创建连接
        client = MongoClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000
        )
        
        # 测试连接
        client.admin.command('ping')
        print("✅ MongoDB连接成功")
        
        # 获取数据库
        db = client[settings.MONGODB_DATABASE]
        
        # 列出所有集合
        collections = db.list_collection_names()
        print(f"📋 数据库中的集合: {collections}")
        
        # 检查目标集合
        target_collection = f"{settings.MONGODB_COLLECTION_MARKET_DATA}_records"
        if target_collection in collections:
            collection = db[target_collection]
            count = collection.count_documents({})
            print(f"✅ 找到目标集合 '{target_collection}'，包含 {count} 条记录")
            
            # 获取一条样本数据
            if count > 0:
                sample = collection.find_one({}, {"_id": 0})
                print(f"📊 样本数据: {sample}")
            else:
                print("⚠️ 集合为空")
        else:
            print(f"❌ 未找到目标集合 '{target_collection}'")
            print("可用的集合:", collections)
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ MongoDB连接失败: {e}")
        return False

if __name__ == "__main__":
    if test_connection():
        print("\n🎉 数据库连接测试通过！")
    else:
        print("\n💥 数据库连接测试失败！")
        sys.exit(1)