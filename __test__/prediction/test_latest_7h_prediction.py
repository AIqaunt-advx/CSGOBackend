#!/usr/bin/env python3
"""
测试最新7小时预测数据

这个脚本用于测试获取最新7小时的预测数据并格式化为前端需要的格式
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def get_latest_7h_prediction():
    """获取最新7小时预测数据"""
    print("🚀 开始获取最新7小时预测数据...")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 运行llm_request_tool.py获取最新7小时预测数据
        print("📊 执行预测分析命令:")
        command = [
            "uv", "run", "python", "__test__/tools/llm_request_tool.py",
            "--method", "hours",
            "--hours", "7",
            "--limit", "100",  # 增加数据量以获得更多推荐
            "--analysis", "trend"
        ]
        print(f"   {' '.join(command)}")
        print()
        
        result = subprocess.run(command, capture_output=True, text=True, cwd=Path.cwd())
        
        print("📥 原始输出:")
        print("-" * 40)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        print("-" * 40)
        print()
        
        if result.returncode != 0:
            print(f"❌ 预测分析失败，退出码: {result.returncode}")
            return None
            
        # 解析JSON输出
        output_lines = result.stdout.strip().split('\n')
        json_content = []
        in_json = False
        brace_count = 0
        
        for line in output_lines:
            line = line.strip()
            if line.startswith('{'):
                in_json = True
                json_content = [line]
                brace_count = line.count('{') - line.count('}')
            elif in_json:
                json_content.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    break
        
        if not json_content:
            print("❌ 无法找到JSON输出")
            return None
        
        json_line = '\n'.join(json_content)
            
        prediction_data = json.loads(json_line)
        
        print("✅ 预测数据获取成功")
        print(f"📈 分析类型: {prediction_data.get('analysis_type', 'Unknown')}")
        
        data_summary = prediction_data.get('data_summary', {})
        if data_summary:
            print(f"📊 数据统计:")
            print(f"   - 数据点数量: {data_summary.get('count', 0)}")
            time_range = data_summary.get('time_range', {})
            if time_range:
                print(f"   - 时间范围: {time_range.get('earliest', 'N/A')} ~ {time_range.get('latest', 'N/A')}")
            
            statistics = data_summary.get('statistics', {})
            if statistics:
                print(f"   - 平均价格: ${statistics.get('avg_price', 0):.2f}")
                print(f"   - 平均数量: {statistics.get('avg_quantity', 0):.2f}")
        
        prediction_result = prediction_data.get('prediction_result', {})
        if prediction_result:
            predictions = prediction_result.get('predictions', [])
            mse = prediction_result.get('mse', 0)
            print(f"🔮 预测结果:")
            print(f"   - 预测数量: {len(predictions)}")
            print(f"   - MSE: {mse:.6f}")
            if predictions:
                print(f"   - 预测范围: ${min(predictions):.2f} ~ ${max(predictions):.2f}")
                print(f"   - 平均预测价格: ${sum(predictions)/len(predictions):.2f}")
        
        print()
        return prediction_data
        
    except Exception as e:
        print(f"❌ 获取预测数据时出错: {e}")
        return None

def format_prediction_data(prediction_data):
    """格式化预测数据为前端格式"""
    print("🔄 格式化预测数据为前端格式...")
    
    try:
        # 将预测数据写入临时文件
        temp_file = Path("__test__/temp_latest_prediction.json")
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 临时文件已保存: {temp_file}")
        
        # 运行格式化工具
        print("📊 执行格式化命令:")
        command = [
            "uv", "run", "python", "__test__/tools/prediction_formatter.py",
            "--input", str(temp_file),
            "--pretty"
        ]
        print(f"   {' '.join(command)}")
        print()
        
        result = subprocess.run(command, capture_output=True, text=True, cwd=Path.cwd())
        
        print("📤 格式化输出:")
        print("-" * 40)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        print("-" * 40)
        print()
        
        # 清理临时文件
        if temp_file.exists():
            temp_file.unlink()
            print(f"🗑️  临时文件已清理: {temp_file}")
        
        if result.returncode != 0:
            print(f"❌ 格式化失败，退出码: {result.returncode}")
            return None
            
        formatted_data = json.loads(result.stdout)
        print("✅ 数据格式化成功")
        
        return formatted_data
        
    except Exception as e:
        print(f"❌ 格式化数据时出错: {e}")
        return None

def display_frontend_data(formatted_data):
    """展示前端数据格式"""
    print("=" * 80)
    print("📋 前端数据格式展示")
    print("=" * 80)
    
    if not formatted_data or not formatted_data.get('items'):
        print("❌ 没有可显示的前端数据")
        return
    
    print(f"✅ 成功: {formatted_data.get('success', False)}")
    print(f"⏰ 时间戳: {formatted_data.get('timestamp', 'N/A')}")
    print(f"📊 总物品数: {formatted_data.get('total_items', 0)}")
    print()
    
    # 表格头部 - 简化版
    print(f"{'ID':<4} {'ITEM DESIGNATION':<30} {'EXPECTED TODAY SALES':<20} {'RECOMMENDED BUY':<15}")
    print("-" * 70)
    
    # 显示每个物品
    for item in formatted_data['items']:
        print(f"{item['id']:<4} {item['item_designation']:<30} {item['expected_today_sales']:<20} {item['recommended_buy']:<15}")

def save_test_results(prediction_data, formatted_data):
    """保存测试结果"""
    print("\n" + "=" * 80)
    print("💾 保存测试结果")
    print("=" * 80)
    
    try:
        # 保存原始预测数据
        raw_file = Path("__test__/latest_7h_prediction_raw.json")
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 原始预测数据已保存: {raw_file}")
        
        # 保存格式化后的前端数据
        frontend_file = Path("__test__/latest_7h_prediction_frontend.json")
        with open(frontend_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 前端格式数据已保存: {frontend_file}")
        
        # 创建测试报告
        report_file = Path("__test__/latest_7h_test_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"最新7小时预测数据测试报告\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")
            
            f.write(f"原始数据文件: {raw_file}\n")
            f.write(f"前端数据文件: {frontend_file}\n\n")
            
            if prediction_data:
                f.write(f"预测数据概要:\n")
                f.write(f"- 分析类型: {prediction_data.get('analysis_type', 'Unknown')}\n")
                
                data_summary = prediction_data.get('data_summary', {})
                if data_summary:
                    f.write(f"- 数据点数量: {data_summary.get('count', 0)}\n")
                    time_range = data_summary.get('time_range', {})
                    if time_range:
                        f.write(f"- 时间范围: {time_range.get('earliest', 'N/A')} ~ {time_range.get('latest', 'N/A')}\n")
                
                prediction_result = prediction_data.get('prediction_result', {})
                if prediction_result:
                    predictions = prediction_result.get('predictions', [])
                    mse = prediction_result.get('mse', 0)
                    f.write(f"- 预测数量: {len(predictions)}\n")
                    f.write(f"- MSE: {mse:.6f}\n")
            
            if formatted_data:
                f.write(f"\n前端数据概要:\n")
                f.write(f"- 总物品数: {formatted_data.get('total_items', 0)}\n")
                f.write(f"- 处理时间: {formatted_data.get('timestamp', 'N/A')}\n")
                f.write(f"- 处理状态: {'成功' if formatted_data.get('success', False) else '失败'}\n")
        
        print(f"✅ 测试报告已保存: {report_file}")
        
    except Exception as e:
        print(f"❌ 保存测试结果时出错: {e}")

def main():
    """主函数"""
    print("🧪 最新7小时预测数据测试")
    print("=" * 60)
    
    # 步骤1: 获取最新7小时预测数据
    prediction_data = get_latest_7h_prediction()
    if not prediction_data:
        print("❌ 测试失败：无法获取预测数据")
        sys.exit(1)
    
    # 步骤2: 格式化为前端格式
    formatted_data = format_prediction_data(prediction_data)
    if not formatted_data:
        print("❌ 测试失败：无法格式化数据")
        sys.exit(1)
    
    # 步骤3: 展示前端数据
    display_frontend_data(formatted_data)
    
    # 步骤4: 保存测试结果
    save_test_results(prediction_data, formatted_data)
    
    print("\n" + "=" * 80)
    print("🎉 测试完成！")
    print("\n📁 生成的文件:")
    print("   - __test__/latest_7h_prediction_raw.json (原始预测数据)")
    print("   - __test__/latest_7h_prediction_frontend.json (前端格式数据)")
    print("   - __test__/latest_7h_test_report.txt (测试报告)")
    print("\n💡 使用说明:")
    print("   - 原始数据包含完整的预测结果和统计信息")
    print("   - 前端数据已格式化为表格显示格式")
    print("   - 测试报告包含本次测试的详细信息")

if __name__ == "__main__":
    main()