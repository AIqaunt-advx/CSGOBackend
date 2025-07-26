# LLM数据获取工具

这个目录包含了用于获取和测试LLM请求数据的工具。

## 文件说明

### 1. `llm_data_retriever.py`

完整的交互式数据获取工具，提供友好的用户界面。

**功能特性：**

- 随机样本数据获取
- 最新数据获取
- 价格范围筛选
- 时间范围筛选（最近N小时）
- 美化的表格输出
- JSON格式输出（适合LLM请求）
- 数据统计信息
- 数据保存功能

**使用方法：**

```bash
python tools/llm_data_retriever.py
```

### 2. `llm_test_cli.py`

命令行工具，适合快速获取数据和脚本集成。

**使用示例：**

```bash
# 获取最新10条数据（JSON格式）
python tools/llm_test_cli.py --method latest --limit 10

# 获取随机5条样本数据
python tools/llm_test_cli.py -m sample -l 5

# 获取最近24小时的数据
python tools/llm_test_cli.py -m hours --hours 24 -l 50

# 获取价格在4.0-10.0之间的数据
python tools/llm_test_cli.py -m price --min-price 4.0 --max-price 10.0

# 简单表格格式输出
python tools/llm_test_cli.py -f pretty

# 静默模式（只输出数据，无额外信息）
python tools/llm_test_cli.py -q
```

**命令行参数：**

- `--method, -m`: 数据获取方法 (sample/latest/hours/price)
- `--limit, -l`: 获取数量限制
- `--hours`: 小时数（用于hours方法）
- `--min-price`: 最小价格（用于price方法）
- `--max-price`: 最大价格（用于price方法）
- `--format, -f`: 输出格式 (json/pretty)
- `--quiet, -q`: 静默模式

### 3. `llm_request_tool.py`

LLM请求工具，用于直接向LLM服务发送请求。

## 数据格式

### 输入数据结构

```json
{
  "timestamp": 1640995200,
  "price": 100.5,
  "onSaleQuantity": 50,
  "seekPrice": 95.0,
  "seekQuantity": 10,
  "transactionAmount": 1000.0,
  "transcationNum": 10,
  "surviveNum": 5
}
```

### LLM请求格式输出

```json
{
  "data": [
    {
      "timestamp": 1640995200,
      "datetime": "2022-01-01 12:00:00",
      "price": 100.5,
      "onSaleQuantity": 50,
      "seekPrice": 95.0,
      "seekQuantity": 10,
      "transactionAmount": 1000.0,
      "transcationNum": 10,
      "surviveNum": 5
    }
  ],
  "count": 1,
  "statistics": {
    "price_range": [95.0, 105.0],
    "avg_price": 100.25,
    "quantity_range": [10, 100],
    "avg_quantity": 55.0
  },
  "time_range": {
    "earliest": "2022-01-01 10:00:00",
    "latest": "2022-01-01 12:00:00"
  }
}
```

## 配置要求

确保 `config.py` 中包含以下MongoDB配置：

- `MONGODB_URL`: MongoDB连接URL
- `MONGODB_DATABASE`: 数据库名称
- `MONGODB_COLLECTION_MARKET_DATA`: 集合名称前缀

## 依赖项

```bash
pip install pymongo python-dotenv pydantic-settings
```

## 使用场景

1. **LLM数据分析**: 获取格式化的市场数据用于LLM分析
2. **API测试**: 快速获取测试数据
3. **数据探索**: 交互式浏览数据库内容
4. **脚本集成**: 在其他脚本中获取数据

## 快速开始

### 1. 测试工具是否正常工作

```bash
python test_tools.py
```

### 2. 快速获取数据

```bash
# 获取最新10条数据
python tools/llm_test_cli.py

# 获取随机样本
python tools/llm_test_cli.py -m sample -l 5

# 获取最近24小时数据
python tools/llm_test_cli.py -m hours --hours 24
```

### 3. 交互式数据浏览

```bash
python tools/llm_data_retriever.py
```

### 4. LLM分析（需要配置LLM服务）

```bash
python tools/llm_request_tool.py -a trend
```

## 注意事项

- 确保MongoDB服务正在运行
- 检查数据库连接配置
- 大量数据查询时注意性能影响
- 使用适当的limit参数避免内存问题
- LLM功能需要正确配置LLM服务端点