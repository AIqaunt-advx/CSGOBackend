#!/usr/bin/env python3
"""
测试运行脚本
使用方法：
    python run_tests.py              # 运行所有测试
    python run_tests.py --cov        # 运行测试并生成覆盖率报告
"""

import sys
import subprocess

def run_tests(with_coverage=False):
    """运行测试"""
    if with_coverage:
        cmd = [
            sys.executable, "-m", "pytest", 
            "test_main.py", 
            "--cov=main", 
            "--cov-report=html", 
            "--cov-report=term",
            "-v"
        ]
    else:
        cmd = [sys.executable, "-m", "pytest", "test_main.py", "-v"]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ 所有测试通过!")
        if with_coverage:
            print("📊 覆盖率报告已生成在 htmlcov/ 目录中")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 测试失败，退出码: {e.returncode}")
        return False

if __name__ == "__main__":
    with_cov = "--cov" in sys.argv
    success = run_tests(with_coverage=with_cov)
    sys.exit(0 if success else 1)