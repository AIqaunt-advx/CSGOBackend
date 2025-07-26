#!/usr/bin/env python3
"""CSGO Backend 统一CLI工具"""

import asyncio
import threading
import time
import signal
import sys
import os
import subprocess
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.crawler import csgo_crawler
from config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrawlerService:
    """爬虫服务管理"""
    
    def __init__(self):
        self.crawler_thread = None
        self.is_running = False
        self.loop = None
    
    def _run_crawler_loop(self):
        """在新线程中运行爬虫循环"""
        try:
            # 创建新的事件循环
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # 运行爬虫
            self.loop.run_until_complete(csgo_crawler.start_scheduled_crawling())
            
        except Exception as e:
            logger.error(f"爬虫线程异常: {e}")
        finally:
            if self.loop:
                self.loop.close()
    
    def start(self):
        """启动爬虫服务"""
        if self.is_running:
            print("⚠️ 爬虫服务已在运行")
            return False
        
        print("🚀 启动爬虫服务...")
        self.is_running = True
        
        # 在新线程中启动爬虫
        self.crawler_thread = threading.Thread(
            target=self._run_crawler_loop,
            daemon=True,
            name="CrawlerThread"
        )
        self.crawler_thread.start()
        
        print(f"✅ 爬虫服务已启动，间隔 {settings.CRAWLER_INTERVAL} 秒")
        return True
    
    def stop(self):
        """停止爬虫服务"""
        if not self.is_running:
            print("⚠️ 爬虫服务未运行")
            return False
        
        print("🛑 停止爬虫服务...")
        
        # 停止爬虫
        csgo_crawler.stop_crawling()
        self.is_running = False
        
        # 等待线程结束
        if self.crawler_thread and self.crawler_thread.is_alive():
            self.crawler_thread.join(timeout=10)
        
        print("✅ 爬虫服务已停止")
        return True
    
    def status(self):
        """获取爬虫状态"""
        if self.is_running and self.crawler_thread and self.crawler_thread.is_alive():
            return "运行中"
        else:
            return "已停止"

# 全局爬虫服务实例
crawler_service = CrawlerService()

def run_test_command(cmd, description, timeout=60):
    """运行测试命令"""
    print(f"\n{'='*50}")
    print(f"🧪 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - 通过")
            if result.stdout.strip():
                # 只显示关键输出
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:  # 显示最后10行
                    if any(marker in line for marker in ['✅', '❌', '📊', '🎉', '⚠️']):
                        print(line)
        else:
            print(f"❌ {description} - 失败")
            if result.stderr.strip():
                print("错误信息:")
                print(result.stderr[-500:])  # 只显示最后500字符
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - 超时")
        return False
    except Exception as e:
        print(f"💥 {description} - 异常: {e}")
        return False

def cmd_test(args):
    """测试命令"""
    print("🚀 CSGO Backend 测试套件")
    
    # 检查项目根目录
    if not os.path.exists("config.py"):
        print("❌ 请在项目根目录运行此脚本")
        return False
    
    # 定义测试
    all_tests = {
        'db': ("uv run __test__/database/test_mongodb_connection.py", "数据库连接测试"),
        'tools': ("uv run __test__/tools/test_tools.py", "LLM工具测试"),
        'crawler': ("uv run __test__/crawler/test_simple_crawler.py", "爬虫基础功能测试"),
        'api': ("uv run __test__/crawler/simple_crawler_test.py", "爬虫API调用测试"),
    }
    
    # 根据参数选择测试
    if args.module == 'all':
        tests_to_run = list(all_tests.items())
    elif args.module in all_tests:
        tests_to_run = [(args.module, all_tests[args.module])]
    else:
        print(f"❌ 未知的测试模块: {args.module}")
        print(f"可用模块: {', '.join(all_tests.keys())}, all")
        return False
    
    # 运行测试
    passed = 0
    total = len(tests_to_run)
    
    for test_name, (cmd, description) in tests_to_run:
        if run_test_command(cmd, description):
            passed += 1
    
    # 显示结果
    print(f"\n{'='*50}")
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 测试通过！")
    else:
        print("⚠️ 部分测试失败")
    
    return passed == total

def cmd_crawler(args):
    """爬虫命令"""
    if args.action == 'start':
        return crawler_service.start()
    elif args.action == 'stop':
        return crawler_service.stop()
    elif args.action == 'status':
        status = crawler_service.status()
        print(f"爬虫状态: {status}")
        return True
    elif args.action == 'restart':
        crawler_service.stop()
        time.sleep(2)
        return crawler_service.start()
    else:
        print(f"❌ 未知的爬虫操作: {args.action}")
        return False

def cmd_tools(args):
    """工具命令"""
    if args.tool == 'data':
        # 获取数据
        cmd = f"uv run tools/llm_test_cli.py --method {args.method} --limit {args.limit}"
        if args.format:
            cmd += f" --format {args.format}"
        if args.quiet:
            cmd += " --quiet"
        
        os.system(cmd)
        return True
    
    elif args.tool == 'interactive':
        # 交互式工具
        os.system("uv run tools/llm_data_retriever.py")
        return True
    
    else:
        print(f"❌ 未知的工具: {args.tool}")
        return False

def cmd_server(args):
    """服务器命令"""
    print("🚀 启动CSGO Backend服务")
    
    # 启动爬虫服务
    if args.with_crawler:
        crawler_service.start()
    
    try:
        print("服务器运行中... (按 Ctrl+C 停止)")
        
        # 这里可以添加实际的服务器启动代码
        # 比如FastAPI应用等
        
        # 现在只是保持运行状态
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 收到停止信号")
    finally:
        if args.with_crawler:
            crawler_service.stop()
        print("✅ 服务已停止")

def setup_signal_handlers():
    """设置信号处理器"""
    def signal_handler(signum, frame):
        print(f"\n收到信号 {signum}，正在停止服务...")
        crawler_service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CSGO Backend CLI工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='运行测试')
    test_parser.add_argument('module', choices=['all', 'db', 'tools', 'crawler', 'api'], 
                           default='all', nargs='?', help='测试模块')
    
    # 爬虫命令
    crawler_parser = subparsers.add_parser('crawler', help='爬虫管理')
    crawler_parser.add_argument('action', choices=['start', 'stop', 'status', 'restart'], 
                               help='爬虫操作')
    
    # 工具命令
    tools_parser = subparsers.add_parser('tools', help='数据工具')
    tools_subparsers = tools_parser.add_subparsers(dest='tool', help='工具类型')
    
    # 数据获取工具
    data_parser = tools_subparsers.add_parser('data', help='获取数据')
    data_parser.add_argument('--method', '-m', choices=['sample', 'latest', 'hours', 'price'], 
                           default='latest', help='数据获取方法')
    data_parser.add_argument('--limit', '-l', type=int, default=10, help='获取数量')
    data_parser.add_argument('--format', '-f', choices=['json', 'pretty'], help='输出格式')
    data_parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    
    # 交互式工具
    tools_subparsers.add_parser('interactive', help='交互式数据浏览')
    
    # 服务器命令
    server_parser = subparsers.add_parser('server', help='启动服务器')
    server_parser.add_argument('--with-crawler', action='store_true', help='同时启动爬虫')
    
    args = parser.parse_args()
    
    # 设置信号处理器
    setup_signal_handlers()
    
    # 执行命令
    if args.command == 'test':
        success = cmd_test(args)
    elif args.command == 'crawler':
        success = cmd_crawler(args)
    elif args.command == 'tools':
        success = cmd_tools(args)
    elif args.command == 'server':
        success = cmd_server(args)
    else:
        parser.print_help()
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()