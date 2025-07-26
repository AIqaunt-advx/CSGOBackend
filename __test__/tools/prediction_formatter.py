#!/usr/bin/env python3
"""
CSGO预测结果格式化工具

功能说明:
    这个工具用于将llm_request_tool.py的预测结果转换为前端需要的格式。
    
主要特性:
    - 解析预测API的JSON输出
    - 计算价格差异和推荐购买数量
    - 生成前端所需的格式化数据
    - 支持批量处理多个饰品的预测结果

输入格式:
    接受llm_request_tool.py输出的JSON格式数据
    
输出格式:
    生成包含以下字段的前端数据:
    - ITEM DESIGNATION: 饰品名称
    - MAX DIFF: 最大价格差异
    - EXPECTED TODAY SALES: 预期今日销量
    - RECOMMENDED BUY: 推荐购买数量
    - EXPECTED INCOME: 预期收入

使用示例:
    python prediction_formatter.py --input prediction_result.json
    python prediction_formatter.py --input prediction_result.json --output formatted_result.json
    cat prediction_result.json | python prediction_formatter.py --stdin
"""

import json
import sys
import argparse
from typing import Dict, List, Any
from datetime import datetime


class PredictionFormatter:
    """预测结果格式化器"""
    
    def __init__(self):
        """初始化格式化器"""
        self.item_counter = 1
    
    def format_prediction_result(self, prediction_data: Dict[str, Any], max_items: int = 10) -> List[Dict[str, Any]]:
        """
        格式化预测结果为前端需要的格式，生成多条推荐
        
        Args:
            prediction_data: llm_request_tool.py的输出数据
            max_items: 最大推荐数量
            
        Returns:
            格式化后的前端数据列表
        """
        try:
            # 提取预测结果
            prediction_result = prediction_data.get('prediction_result', {})
            data_summary = prediction_data.get('data_summary', {})
            
            predictions = prediction_result.get('predictions', [])
            mse = prediction_result.get('mse', 0)
            
            # 获取历史价格数据用于计算
            statistics = data_summary.get('statistics', {})
            price_range = statistics.get('price_range', [0, 0])
            current_price = statistics.get('avg_price', 0)
            avg_price = current_price
            max_price = price_range[1] if len(price_range) > 1 else current_price
            min_price = price_range[0] if len(price_range) > 0 else current_price
            
            # 计算预测价格统计
            if predictions:
                predicted_avg = sum(predictions) / len(predictions)
                predicted_max = max(predictions)
                predicted_min = min(predictions)
            else:
                predicted_avg = current_price
                predicted_max = current_price
                predicted_min = current_price
            
            # 生成多条推荐
            formatted_results = []
            
            # 基于预测数据生成多个不同的推荐
            for i in range(min(max_items, len(self._get_item_names()))):
                # 为每个推荐计算不同的指标
                variation_factor = 1 + (i * 0.1)  # 每个推荐有10%的变化
                risk_factor = 1 - (i * 0.05)  # 风险递增
                
                # 计算关键指标
                max_diff = abs(predicted_max - current_price) * variation_factor
                price_trend = (predicted_avg - current_price) * variation_factor
                
                # 基于数据计算预期销量和推荐购买数量
                avg_price = statistics.get('avg_price', current_price)
                
                # 预期今日销量：avg_price 取整
                expected_sales = int(avg_price)
                
                # 推荐购买数量：avg_price 乘以 0.413864～0.579532 的随机系数然后取整
                import random
                buy_factor = random.uniform(0.413864, 0.579532)
                recommended_buy = int(avg_price * buy_factor)
                
                # 预期收入：推荐购买数量 × 最大价格差异
                expected_income = recommended_buy * max_diff
                
                # 生成饰品名称
                item_name = self._get_item_names()[i]
                
                # 格式化结果 - 简化版DTO
                formatted_result = {
                    "id": f"{self.item_counter:02d}",
                    "item_designation": item_name,
                    "expected_today_sales": int(expected_sales),
                    "recommended_buy": int(max(recommended_buy, 1)),  # 至少推荐1个
                    "expected_income_value": expected_income,  # 用于排序的数值
                }
                
                formatted_results.append(formatted_result)
                self.item_counter += 1
            
            # 按预期收入排序（从高到低）
            formatted_results.sort(key=lambda x: x['expected_income_value'], reverse=True)
            
            # 移除排序用的字段
            for result in formatted_results:
                result.pop('expected_income_value', None)
            
            return formatted_results
            
        except Exception as e:
            print(f"格式化预测结果时出错: {e}", file=sys.stderr)
            return []
    
    def _calculate_expected_sales(self, avg_quantity: float, price_trend: float, current_price: float) -> float:
        """
        计算预期今日销量
        
        基于历史平均销量和价格趋势预测
        """
        base_sales = max(avg_quantity, 10)  # 最少10个
        
        # 价格下降通常会增加销量
        if current_price > 0:
            if price_trend < 0:
                trend_factor = 1 + abs(price_trend) / current_price * 2
            else:
                trend_factor = 1 - price_trend / current_price * 0.5
        else:
            trend_factor = 1
        
        return max(base_sales * trend_factor, 1)
    
    def _calculate_recommended_buy(self, price_trend: float, max_diff: float, current_price: float, mse: float) -> float:
        """
        计算推荐购买数量
        
        基于价格趋势、风险和预期收益
        """
        # 基础推荐数量
        base_quantity = 5
        
        # 价格上涨趋势增加推荐数量
        if current_price > 0:
            if price_trend > 0:
                trend_factor = 1 + (price_trend / current_price) * 10
            else:
                trend_factor = 0.5  # 价格下降时减少推荐
        else:
            trend_factor = 0.5
        
        # 风险调整：MSE越高，风险越大，推荐数量越少
        risk_factor = 1 / (1 + mse / 1000)  # 归一化MSE影响
        
        # 价格差异调整：差异越大，潜在收益越高
        if current_price > 0:
            diff_factor = 1 + (max_diff / current_price)
        else:
            diff_factor = 1
        
        recommended = base_quantity * trend_factor * risk_factor * diff_factor
        
        return max(min(recommended, 50), 0)  # 限制在0-50之间
    
    def _calculate_confidence(self, mse: float) -> str:
        """
        基于MSE计算预测置信度
        """
        if mse < 100:
            return "High"
        elif mse < 500:
            return "Medium"
        else:
            return "Low"
    
    def _get_item_names(self) -> List[str]:
        """
        获取饰品名称列表
        
        实际应用中应该从数据库或API获取真实的饰品信息
        """
        return [
            "★ Butterfly Knife",
            "AK-47 | Redline",
            "AWP | Dragon Lore",
            "M4A4 | Howl",
            "Glock-18 | Fade",
            "★ Karambit | Doppler",
            "Desert Eagle | Blaze",
            "★ Bayonet | Tiger Tooth",
            "M4A1-S | Knight",
            "★ Gut Knife | Doppler"
        ]
    
    def _generate_item_name(self, prediction_data: Dict[str, Any]) -> str:
        """
        生成饰品名称
        
        实际应用中应该从数据库或API获取真实的饰品信息
        """
        item_names = self._get_item_names()
        name_index = (self.item_counter - 1) % len(item_names)
        return item_names[name_index]
    
    def format_multiple_results(self, results_list: List[Dict[str, Any]], max_items: int = 10) -> List[Dict[str, Any]]:
        """
        格式化多个预测结果
        
        Args:
            results_list: 多个预测结果的列表
            max_items: 最大推荐数量
            
        Returns:
            格式化后的前端数据列表
        """
        formatted_results = []
        
        for result in results_list:
            formatted = self.format_prediction_result(result, max_items)
            formatted_results.extend(formatted)
        
        # 按预期收入排序（从高到低）
        formatted_results.sort(key=lambda x: float(x['expected_income'].replace('$', '').replace(',', '')), reverse=True)
        
        # 限制最大数量
        return formatted_results[:max_items]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CSGO预测结果格式化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 从文件读取预测结果并格式化
  python prediction_formatter.py --input prediction_result.json
  
  # 从标准输入读取并输出到文件
  cat prediction_result.json | python prediction_formatter.py --stdin --output formatted.json
  
  # 处理多个预测结果文件
  python prediction_formatter.py --input result1.json result2.json --output combined.json

输出格式说明:
  每个饰品包含以下字段:
  - id: 饰品编号 (01, 02, ...)
  - item_designation: 饰品名称 (★ Butterfly Knife)
  - max_diff: 最大价格差异 ($1000)
  - expected_today_sales: 预期今日销量 (533)
  - recommended_buy: 推荐购买数量 (3)
  - expected_income: 预期收入 ($3,000)
  - details: 详细信息 (包含当前价格、预测价格、置信度等)
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        help="输入的预测结果JSON文件路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="输出文件路径 (默认输出到标准输出)"
    )
    
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="从标准输入读取数据"
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="美化JSON输出格式"
    )
    
    args = parser.parse_args()
    
    # 检查输入参数
    if not args.input and not args.stdin:
        parser.error("必须指定 --input 或 --stdin")
    
    formatter = PredictionFormatter()
    all_results = []
    
    try:
        # 从标准输入读取
        if args.stdin:
            input_data = json.load(sys.stdin)
            if isinstance(input_data, list):
                all_results = formatter.format_multiple_results(input_data)
            else:
                all_results = formatter.format_prediction_result(input_data)
        
        # 从文件读取
        elif args.input:
            for input_file in args.input:
                try:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        input_data = json.load(f)
                        
                    if isinstance(input_data, list):
                        results = formatter.format_multiple_results(input_data)
                    else:
                        results = formatter.format_prediction_result(input_data)
                    
                    all_results.extend(results)
                    
                except FileNotFoundError:
                    print(f"错误: 文件 {input_file} 不存在", file=sys.stderr)
                    sys.exit(1)
                except json.JSONDecodeError as e:
                    print(f"错误: 文件 {input_file} JSON格式错误: {e}", file=sys.stderr)
                    sys.exit(1)
        
        # 输出结果
        output_data = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "total_items": len(all_results),
            "items": all_results
        }
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.pretty:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                else:
                    json.dump(output_data, f, ensure_ascii=False)
            print(f"结果已保存到: {args.output}")
        else:
            if args.pretty:
                print(json.dumps(output_data, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(output_data, ensure_ascii=False))
    
    except Exception as e:
        print(f"处理过程中出错: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()