#!/usr/bin/env python3
"""
CSGO价格预测分析工具

功能说明:
    这个工具用于获取CSGO市场的历史价格数据，并将数据发送到预测API进行价格预测分析。
    
主要特性:
    - 支持多种数据获取方式：最新数据、指定时间范围、价格范围筛选
    - 自动格式化数据为预测API所需的格式
    - 返回详细的预测结果和模型性能指标
    - 支持JSON格式输出，便于后续处理

数据格式:
    输入数据包含以下字段：
    - timestamp: 时间戳
    - price: 当前价格
    - onSaleQuantity: 在售数量
    - seekPrice: 求购价格
    - seekQuantity: 求购数量
    - transactionAmount: 交易金额
    - transcationNum: 交易次数
    - surviveNum: 存活数量

预测API接口:
    POST /predict
    请求体: {"data": [数据数组]}
    响应: {"predictions": [预测值数组], "mse": 均方误差}

使用方法:
    python llm_request_tool.py --method hours --hours 7 --limit 50
"""

import argparse
import json
import os
import sys
from datetime import datetime

import requests

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from config import settings
    from __test__.tools.llm_test_cli import QuickDataRetriever, format_for_llm
except ImportError as e:
    print(f"❌ 导入模块失败: {e}", file=sys.stderr)
    print("请确保在项目根目录运行", file=sys.stderr)
    sys.exit(1)


class PredictRequestTool:
    def __init__(self):
        """初始化预测请求工具"""
        self.predict_url = settings.PREDICT_API_URL
        self.timeout = 30

    def create_analysis_prompt(self, data: dict, analysis_type: str = "trend") -> str:
        """创建分析提示词"""
        prompts = {
            "trend": """
请分析以下CSGO市场数据的趋势：

数据统计：
- 数据条数: {count}
- 价格范围: {price_min:.2f} - {price_max:.2f}
- 平均价格: {avg_price:.2f}
- 数量范围: {qty_min} - {qty_max}
- 平均数量: {avg_qty:.1f}
- 时间范围: {time_earliest} 到 {time_latest}

详细数据：
{data_json}

请分析：
1. 价格趋势和波动情况
2. 供需关系变化
3. 交易活跃度
4. 市场预测建议

请用中文回答，并提供具体的数据支撑。
""",
            "prediction": """
基于以下CSGO市场数据，请进行价格预测分析：

{data_json}

请分析：
1. 短期价格走势预测（1-3天）
2. 影响价格的关键因素
3. 买入/卖出建议
4. 风险评估

请提供具体的价格区间预测和置信度。
""",
            "summary": """
请总结以下CSGO市场数据的关键信息：

{data_json}

请提供：
1. 市场现状概述
2. 关键数据指标
3. 异常情况识别
4. 简要结论

请保持简洁明了。
"""
        }

        template = prompts.get(analysis_type, prompts["trend"])

        # 格式化数据
        stats = data.get("statistics", {})
        time_range = data.get("time_range", {})

        return template.format(
            count=data.get("count", 0),
            price_min=stats.get("price_range", [0, 0])[0],
            price_max=stats.get("price_range", [0, 0])[1],
            avg_price=stats.get("avg_price", 0),
            qty_min=stats.get("quantity_range", [0, 0])[0],
            qty_max=stats.get("quantity_range", [0, 0])[1],
            avg_qty=stats.get("avg_quantity", 0),
            time_earliest=time_range.get("earliest", "未知"),
            time_latest=time_range.get("latest", "未知"),
            data_json=json.dumps(
                data["data"][:10], indent=2, ensure_ascii=False)  # 只显示前10条
        )

    def send_request(self, data: list) -> dict:
        """发送预测请求"""
        try:
            # 按照API要求的格式准备payload
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
            return {"error": f"请求失败: {e}"}
        except Exception as e:
            return {"error": f"处理失败: {e}"}

    def analyze_data(self, data: dict, analysis_type: str = "trend") -> dict:
        """分析数据"""
        if not data.get("data"):
            return {"error": "没有可分析的数据"}

        # 准备预测API所需的数据格式
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

        # 发送预测请求
        result = self.send_request(prediction_data)

        return {
            "analysis_type": analysis_type,
            "data_summary": {
                "count": data.get("count", 0),
                "time_range": data.get("time_range", {}),
                "statistics": data.get("statistics", {})
            },
            "prediction_result": result,
            "timestamp": datetime.now().isoformat()
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CSGO价格预测分析工具 - 获取历史数据并发送到预测API进行分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用示例:
  # 获取最近7小时的50条数据进行趋势分析
  python llm_request_tool.py --method hours --hours 7 --limit 50 --analysis trend
  
  # 获取最新100条数据进行波动性分析
  python llm_request_tool.py --method latest --limit 100 --analysis volatility
  
  # 获取特定价格范围的数据进行模式分析
  python llm_request_tool.py --method price --min-price 50 --max-price 200 --analysis pattern

输入参数说明:
  --method: 数据获取方法
    - sample: 获取样本数据
    - latest: 获取最新数据
    - hours: 获取指定小时数内的数据
    - price: 获取指定价格范围内的数据
  
  --limit: 数据条数限制 (默认20条)
  --hours: 小时数，仅在method=hours时使用 (默认7小时)
  --min-price/--max-price: 价格范围，仅在method=price时使用
  --analysis: 分析类型 (trend/volatility/pattern)

输出结果说明:
  返回JSON格式的预测结果，包含:
  - analysis_type: 分析类型
  - data_summary: 数据摘要 (数量、时间范围、统计信息)
  - prediction_result: 预测结果
    - predictions: 预测价格数组
    - mse: 模型均方误差
  - timestamp: 分析时间戳
        """)

    parser.add_argument('--method', '-m', choices=['sample', 'latest', 'hours', 'price'],
                        default='latest', help='数据获取方法')
    parser.add_argument('--limit', '-l', type=int, default=20, help='获取数量限制')
    parser.add_argument('--hours', type=int, default=7, help='小时数 (用于hours方法)')
    parser.add_argument('--min-price', type=float, help='最小价格 (用于price方法)')
    parser.add_argument('--max-price', type=float, help='最大价格 (用于price方法)')
    parser.add_argument('--analysis', '-a', choices=['trend', 'volatility', 'pattern'],
                        default='trend', help='分析类型')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')

    args = parser.parse_args()

    if not args.quiet:
        print(f"🚀 开始{args.analysis}分析...", file=sys.stderr)
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
            print(f"🤖 发送到LLM进行{args.analysis}分析...", file=sys.stderr)

        # LLM分析
        llm_tool = PredictRequestTool()
        result = llm_tool.analyze_data(formatted_data, args.analysis)

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
            print("✅ 分析完成", file=sys.stderr)

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
