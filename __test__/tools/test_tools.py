#!/usr/bin/env python3
"""测试LLM工具的快速脚本"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.llm_test_cli import QuickDataRetriever, format_for_llm
import json


def test_connection():
    """测试数据库连接"""
    print("🔍 测试数据库连接...")
    try:
        retriever = QuickDataRetriever()
        print("✅ 数据库连接成功")
        retriever.close()
        return True
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False


def test_data_retrieval():
    """测试数据获取"""
    print("\n📊 测试数据获取...")
    retriever = QuickDataRetriever()

    try:
        # 测试获取最新数据
        data = retriever.get_data(method="latest", limit=5)
        if data:
            print(f"✅ 成功获取 {len(data)} 条最新数据")

            # 测试数据格式化
            formatted = format_for_llm(data)
            print(f"✅ 数据格式化成功，包含 {formatted['count']} 条记录")

            # 显示样本数据
            if formatted['data']:
                sample = formatted['data'][0]
                print(f"📋 样本数据: 时间={sample['datetime']}, 价格={sample['price']}")

            return True
        else:
            print("❌ 没有获取到数据")
            return False

    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        return False
    finally:
        retriever.close()


def test_cli_tool():
    """测试CLI工具"""
    print("\n🛠️ 测试CLI工具...")
    try:
        # 测试命令行工具
        import subprocess
        result = subprocess.run([
            sys.executable, "tools/llm_test_cli.py",
            "--method", "latest",
            "--limit", "3",
            "--quiet"
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                print(f"✅ CLI工具正常，返回 {data.get('count', 0)} 条数据")
                return True
            except json.JSONDecodeError:
                print("❌ CLI工具输出格式错误")
                return False
        else:
            print(f"❌ CLI工具执行失败: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ CLI工具执行超时")
        return False
    except Exception as e:
        print(f"❌ CLI工具测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 LLM工具测试套件")
    print("=" * 50)

    tests = [
        ("数据库连接", test_connection),
        ("数据获取", test_data_retrieval),
        ("CLI工具", test_cli_tool)
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"⚠️ {name} 测试失败")
        except Exception as e:
            print(f"❌ {name} 测试异常: {e}")

    print(f"\n📈 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！工具可以正常使用")
        print("\n💡 使用建议:")
        print("- 交互式使用: python tools/llm_data_retriever.py")
        print("- 命令行使用: python tools/llm_test_cli.py --help")
        print("- 价格预测: python tools/predict_tool.py --help")
    else:
        print("⚠️ 部分测试失败，请检查配置和数据库连接")
        sys.exit(1)


if __name__ == "__main__":
    main()
