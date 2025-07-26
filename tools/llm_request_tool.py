#!/usr/bin/env python3
"""预测请求工具 - 获取数据并发送到预测API进行分析"""

import argparse
import json
import os
import sys
from datetime import datetime

import requests

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import settings
    from tools.llm_test_cli import QuickDataRetriever, format_for_llm
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

    def send_request(self, prompt: str) -> dict:
        """发送LLM请求"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
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

        # 创建提示词
        prompt = self.create_analysis_prompt(data, analysis_type)

        # 发送请求
        result = self.send_request(prompt)

        return {
            "analysis_type": analysis_type,
            "data_summary": {
                "count": data.get("count", 0),
                "time_range": data.get("time_range", {}),
                "statistics": data.get("statistics", {})
            },
            "llm_response": result,
            "timestamp": datetime.now().isoformat()
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLM数据分析工具')
    parser.add_argument('--method', '-m', choices=['sample', 'latest', 'hours', 'price'],
                        default='latest', help='数据获取方法')
    parser.add_argument('--limit', '-l', type=int, default=20, help='获取数量限制')
    parser.add_argument('--hours', type=int, default=7, help='小时数 (用于hours方法)')
    parser.add_argument('--min-price', type=float, help='最小价格 (用于price方法)')
    parser.add_argument('--max-price', type=float, help='最大价格 (用于price方法)')
    parser.add_argument('--analysis', '-a', choices=['trend', 'prediction', 'summary'],
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
        llm_tool = LLMRequestTool()
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
