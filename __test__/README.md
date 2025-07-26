# 测试目录结构说明

本目录包含了CSGO Backend项目的所有测试文件，按功能模块进行分类组织。

## 📁 目录结构

```
__test__/
├── api/                    # API相关测试
│   ├── test_predict_api.py     # 预测API测试
│   ├── test_curl_api.py        # curl版本API调用测试
│   └── requests_test.py        # HTTP请求测试
├── crawler/                # 爬虫相关测试
│   ├── simple_crawler_test.py  # 简单爬虫测试
│   ├── test_crawler.py         # 爬虫功能测试
│   ├── test_simple_crawler.py  # 简化爬虫测试
│   ├── test_crawler_thread.py  # 爬虫线程测试
│   └── test_crawler_with_mock.py # 模拟数据爬虫测试
├── database/               # 数据库相关测试
│   └── test_mongodb_connection.py # MongoDB连接测试
├── demo/                   # 演示和示例
│   ├── demo_prediction_workflow.py # 完整预测工作流演示
│   └── demo_simple.py             # 简化演示脚本
├── prediction/             # 预测相关测试
│   ├── test_7h_predict.py         # 7小时数据预测测试
│   ├── test_latest_7h_prediction.py # 最新7小时预测测试
│   └── quick_test_predict.py      # 快速预测测试
├── tools/                  # 工具相关测试和脚本
│   ├── llm_request_tool.py        # LLM请求工具
│   ├── llm_test_cli.py            # LLM测试CLI工具
│   ├── prediction_formatter.py    # 预测结果格式化工具
│   └── test_tools.py              # 工具测试脚本
└── [测试数据文件...]        # 测试生成的数据文件
```

## 🧪 测试分类说明

### API测试 (`api/`)

包含所有API相关的测试，验证各种API接口的功能和性能。

- **test_predict_api.py**: 测试预测API的基本功能
- **test_curl_api.py**: 测试使用curl方式调用API
- **requests_test.py**: HTTP请求相关测试

### 爬虫测试 (`crawler/`)

包含爬虫功能的各种测试，从基础功能到高级特性。

- **simple_crawler_test.py**: 基础爬虫API调用测试
- **test_crawler.py**: 完整爬虫功能测试
- **test_simple_crawler.py**: 简化版爬虫测试
- **test_crawler_thread.py**: 爬虫多线程测试
- **test_crawler_with_mock.py**: 使用模拟数据的爬虫测试

### 数据库测试 (`database/`)

包含数据库连接和操作相关的测试。

- **test_mongodb_connection.py**: MongoDB数据库连接测试

### 演示脚本 (`demo/`)

包含各种演示和示例脚本，用于展示系统功能。

- **demo_prediction_workflow.py**: 完整的预测工作流演示
- **demo_simple.py**: 简化版演示，使用示例数据

### 预测测试 (`prediction/`)

包含所有预测功能相关的测试。

- **test_7h_predict.py**: 使用7小时数据进行预测测试
- **test_latest_7h_prediction.py**: 最新7小时预测数据测试
- **quick_test_predict.py**: 快速预测功能测试

### 工具脚本 (`tools/`)

包含各种工具脚本和工具测试。

- **llm_request_tool.py**: LLM数据请求工具
- **llm_test_cli.py**: LLM测试命令行工具
- **prediction_formatter.py**: 预测结果格式化工具
- **test_tools.py**: 工具功能测试脚本

## 🚀 快速开始

### 运行所有测试

```bash
# 从项目根目录运行
uv run cli.py test all
```

### 运行特定分类的测试

```bash
# 数据库测试
uv run cli.py test db

# 爬虫测试
uv run cli.py test crawler

# 工具测试
uv run cli.py test tools

# API测试
uv run cli.py test api
```

### 运行单个测试

```bash
# 预测测试
uv run python __test__/prediction/test_7h_predict.py

# 演示脚本
uv run python __test__/demo/demo_simple.py

# 爬虫测试
uv run python __test__/crawler/test_simple_crawler.py
```

## 📝 添加新测试

当添加新的测试文件时，请按照以下规则：

1. **命名规范**: 测试文件以 `test_` 开头，演示文件以 `demo_` 开头
2. **分类放置**: 根据功能将文件放入对应的子目录
3. **文档更新**: 更新本README文件，说明新测试的用途
4. **依赖管理**: 确保测试文件的导入路径正确

## 🔧 注意事项

- 所有测试都应该从项目根目录运行
- 确保已安装所有依赖: `uv sync`
- 某些测试需要数据库连接或API服务运行
- 测试数据文件会自动生成在 `__test__` 目录中

## 📊 测试数据

测试过程中生成的数据文件包括：

- `*.json`: 测试结果和数据文件
- `*.txt`: 测试报告文件
- `frontend_data.json`: 前端格式化数据

这些文件可以用于验证测试结果和调试问题。