#!/usr/bin/env python3
"""预测工具 - 获取数据并发送到预测API进行分析"""

import json
import requests
import argparse
import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import settings
    from tools.llm_test_cli import QuickDataRetriever, format_for_llm
except ImportError as e:
    print(f"❌ 导入模块失败: {e}", file=sys.stderr)
    print("请确保在项目根目录运行", file=sys.stderr)
    sys.exit(1)

class PredictTool:
    def __init__(self):
        """初始化预测工具"""
        self.predict_url = settings.PREDICT_API_URL
        self.timeout = 30
        
    def prepare_prediction_data(self, data: dict) -> list:
        """准备预测API所需的数据格式"""
        prediction_data = []
        
        for record in data.get("data", []):
            prediction_record = {
                "timestamp": record.get("timestamp", 0),
                "price": float(record.get("price", 0)),
                "onSaleQuantity": record.get("onSaleQuantity", 0),
                "seekPrice": float(record.get("seekPrice", 0)),
                "seekQuantity": record.get("seekQuantity", 0),
                "transactionAmount": float(record.get("transactionAmount") or 0),
                "transcationNum": record.get("transcationNum") or 0,
                "surviveNum": record.get("surviveNum") or 0
            }
            prediction_data.append(prediction_record)
        
        return prediction_data
    
    def send_prediction_request(self, data: list) -> dict:
        """发送预测请求"""
        try:
            payload = {"data": data}
            
            response = requests.post(
                self.predict_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": f"预测请求失败: {e}"}
        except Exception as e:
            return {"error": f"处理失败: {e}"}
    
    def predict_prices(self, data: dict) -> dict:
        """预测价格"""
        if not data.get("data"):
            return {"error": "没有可预测的数据"}
        
        # 准备数据
        prediction_data = self.prepare_prediction_data(data)
        
        # 发送请求
        result = self.send_prediction_request(prediction_data)
        
        return {
            "input_data_summary": {
                "count": data.get("count", 0),
                "time_range": data.get("time_range", {}),
                "statistics": data.get("statistics", {})
            },
            "prediction_result": result,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CSGO价格预测工具')
    parser.add_argument('--method', '-m', choices=['sample', 'latest', 'hours', 'price'], 
                       default='latest', help='数据获取方法')
    parser.add_argument('--limit', '-l', type=int, default=20, help='获取数量限制')
    parser.add_argument('--hours', type=int, default=7, help='小时数 (用于hours方法)')
    parser.add_argument('--min-price', type=float, help='最小价格 (用于price方法)')
    parser.add_argument('--max-price', type=float, help='最大价格 (用于price方法)')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print(f"🚀 开始价格预测分析...", file=sys.stderr)
        print(f"📊 获取{args.method}数据 (限制: {args.limit})", file=sys.stderr)
    
    # 获取数据
    retriever = QuickDataRetriever()
    
    try:
        # 获取原始数据
        raw_data = retriever.get_data(
            method=args.method,
            limit=args.limit,
            hours=args.hours,
            min_price=args.min_price,
            max_price=args.max_price
        )
        
        if not raw_data:
            print("❌ 没有获取到数据", file=sys.stderr)
            sys.exit(1)
        
        # 格式化数据
        formatted_data = format_for_llm(raw_data)
        
        if not args.quiet:
            print(f"✅ 获取到 {formatted_data['count']} 条数据", file=sys.stderr)
            print(f"🤖 发送到预测API进行分析...", file=sys.stderr)
        
        # 预测分析
        predict_tool = PredictTool()
        result = predict_tool.predict_prices(formatted_data)
        
        # 输出结果
        output_data = json.dumps(result, indent=2, ensure_ascii=False)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_data)
            if not args.quiet:
                print(f"📁 结果已保存到: {args.output}", file=sys.stderr)
        else:
            print(output_data)
        
        if not args.quiet:
            print("✅ 预测分析完成", file=sys.stderr)
            
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