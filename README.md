# CSGO Backend

CSGO市场数据分析后端系统，包含数据爬取、存储、分析和LLM工具。

## 🎯 功能特性

- 🕷️ **数据爬取**: 自动爬取CSGO市场数据并存储到MongoDB
- 📊 **数据分析**: 提供多种数据获取和分析工具
- 🤖 **LLM集成**: 支持LLM数据分析和预测
- 🔧 **统一CLI**: 通过命令行工具管理所有功能
- 📈 **价格分析**: 计算不同平台间的价格差异
- 💰 **套利识别**: 自动识别套利机会

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用uv（推荐）
uv sync

# 或使用pip
pip install -r requirements.txt
```

### 2. 配置环境

复制 `.env.example` 到 `.env` 并配置MongoDB连接：

```env
MONGODB_URL=mongodb://root:password@host:port
MONGODB_DATABASE=market_data
MONGODB_COLLECTION_MARKET_DATA=price_history
```

### 3. 运行测试

```bash
# 运行所有测试
uv run cli.py test

# 运行特定模块测试
uv run cli.py test db          # 数据库测试
uv run cli.py test tools       # LLM工具测试
uv run cli.py test crawler     # 爬虫测试
```

### 4. 启动服务

```bash
# 启动完整服务（包含爬虫）
uv run cli.py server --with-crawler

# 或单独启动爬虫
uv run cli.py crawler start
```

## 📋 CLI命令详解

### 测试命令

```bash
uv run cli.py test [module]
```

- `all` - 运行所有测试（默认）
- `db` - 数据库连接测试
- `tools` - LLM工具测试
- `crawler` - 爬虫功能测试
- `api` - API调用测试

### 爬虫管理

```bash
uv run cli.py crawler [action]
```

- `start` - 启动爬虫（后台线程运行）
- `stop` - 停止爬虫
- `status` - 查看爬虫状态
- `restart` - 重启爬虫

### 数据工具

```bash
# 获取数据
uv run cli.py tools data [options]
  --method {sample,latest,hours,price}  # 数据获取方法
  --limit N                             # 获取数量
  --format {json,pretty}                # 输出格式
  --quiet                               # 静默模式

# 交互式数据浏览
uv run cli.py tools interactive
```

### 服务器

```bash
uv run cli.py server [options]
  --with-crawler    # 同时启动爬虫
```

## 📊 数据格式

系统存储的数据格式：

```json
{
  "timestamp": 1640995200,
  "price": 100.5,
  "onSaleQuantity": 50,
  "seekPrice": 95.0,
  "seekQuantity": 10,
  "transactionAmount": 1000.0,
  "transcationNum": 10,
  "surviveNum": 5,
  "file_id": "crawl_1640995200",
  "item_id": "item_12345",
  "item_name": "AK-47 | 红线"
}
```

## 🛠️ 使用示例

### 开发调试流程

```bash
# 1. 测试所有功能
uv run cli.py test

# 2. 启动爬虫收集数据
uv run cli.py crawler start

# 3. 查看收集的数据
uv run cli.py tools data --format pretty --limit 10

# 4. 获取JSON格式数据用于分析
uv run cli.py tools data --method latest --limit 100 --quiet > data.json

# 5. 停止爬虫
uv run cli.py crawler stop
```

### 生产部署

```bash
# 启动完整服务
uv run cli.py server --with-crawler
```

### 数据分析

```bash
# 获取最新数据
uv run cli.py tools data --method latest --limit 20

# 获取随机样本
uv run cli.py tools data --method sample --limit 10

# 获取价格范围数据
uv run cli.py tools data --method price --limit 50

# 交互式浏览
uv run cli.py tools interactive
```

## 🏗️ 项目结构

```
.
├── cli.py                      # 统一CLI工具
├── config.py                   # 配置管理
├── main.py                     # FastAPI应用
├── modules/
│   ├── crawler.py             # 爬虫模块
│   └── database.py            # 数据库模块
├── tools/
│   ├── llm_test_cli.py        # CLI数据工具
│   ├── llm_data_retriever.py  # 交互式数据工具
│   └── predict_tool.py       # 价格预测工具
├── __test__/                   # 测试文件
│   ├── crawler/               # 爬虫测试
│   ├── tools/                 # 工具测试
│   └── database/              # 数据库测试
└── README.md                  # 项目说明
```

## 🧪 测试

### 测试模块

- **数据库测试**: 验证MongoDB连接和数据操作
- **工具测试**: 测试LLM数据获取和格式化工具
- **爬虫测试**: 验证数据爬取和存储功能
- **API测试**: 测试外部API调用

### 测试覆盖

- ✅ 数据库连接和操作
- ✅ 数据获取和格式化
- ✅ 爬虫功能和错误处理
- ✅ CLI工具功能
- ✅ 数据统计和分析

### 测试输出示例

```
🚀 CSGO Backend 测试套件
==================================================
🧪 数据库连接测试
==================================================
✅ 数据库连接测试 - 通过
📊 测试结果: 4/4 通过
🎉 测试通过！
```

## ⚙️ 配置说明

### 环境变量

在 `.env` 文件中配置以下参数：

```env
# MongoDB配置
MONGODB_URL=mongodb://root:password@host:port
MONGODB_DATABASE=market_data
MONGODB_COLLECTION_MARKET_DATA=price_history

# 爬虫配置
CRAWLER_INTERVAL=3600                    # 爬取间隔（秒）
CRAWLER_DELAY_BETWEEN_REQUESTS=1.0       # 请求间延迟（秒）
CRAWLER_BATCH_SIZE=100                   # 批处理大小

# LLM配置
LLM_API_BASE_URL=http://localhost:8001
LLM_API_KEY=your-llm-api-key
LLM_MODEL=gpt-3.5-turbo
```

## 🔧 技术栈

- **Python 3.12+** - 编程语言
- **FastAPI** - Web框架
- **MongoDB** - 数据库
- **aiohttp** - 异步HTTP客户端
- **Pydantic** - 数据验证
- **tenacity** - 重试机制
- **tqdm** - 进度条
- **uv** - 包管理器

## 🚨 注意事项

1. **数据库**: 确保MongoDB服务正在运行
2. **网络**: API调用可能因网络问题失败
3. **爬虫**: 注意设置合适的爬取间隔避免被限制
4. **资源**: 爬虫会在后台持续运行，记得停止
5. **权限**: 确保有足够的数据库读写权限

## 🔍 故障排除

### 常见问题

- **数据库连接失败**: 检查MongoDB服务和连接字符串
- **API调用失败**: 验证网络连接和API端点
- **测试失败**: 确保在项目根目录运行
- **爬虫异常**: 查看日志文件排查问题

### 调试命令

```bash
# 测试数据库连接
uv run cli.py test db

# 查看爬虫状态
uv run cli.py crawler status

# 获取最新数据验证
uv run cli.py tools data --limit 5 --format pretty
```

## 📈 开发路线图

- [ ] 添加更多数据源
- [ ] 实现实时数据推送
- [ ] 增强LLM分析功能
- [ ] 添加Web界面
- [ ] 支持更多数据格式
- [ ] 性能优化和监控

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

### 开发流程

1. Fork项目
2. 创建功能分支
3. 运行测试确保通过
4. 提交Pull Request

### 代码规范

- 使用Python类型提示
- 遵循PEP 8代码风格
- 添加适当的文档字符串
- 确保测试覆盖率