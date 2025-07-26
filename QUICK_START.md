# LLM工具快速使用指南

## 🎉 测试结果
✅ MongoDB连接成功  
✅ 数据获取正常  
✅ 所有工具可用  

## 📊 数据概况
- **数据库**: market_data
- **集合**: price_history_records  
- **记录数**: 51条
- **数据时间范围**: 2022年7月 - 2025年7月

## 🚀 快速使用

### 1. 获取最新数据（JSON格式，适合LLM）
```bash
uv run tools/llm_test_cli.py --method latest --limit 10
```

### 2. 获取数据（表格格式，便于查看）
```bash
uv run tools/llm_test_cli.py --method latest --limit 5 --format pretty
```

### 3. 获取随机样本数据
```bash
uv run tools/llm_test_cli.py -m sample -l 5
```

### 4. 获取指定价格范围的数据
```bash
uv run tools/llm_test_cli.py -m price --min-price 100 --max-price 105 -l 10
```

### 5. 静默模式（只输出数据，无额外信息）
```bash
uv run tools/llm_test_cli.py -q -l 3
```

### 6. 交互式数据浏览
```bash
uv run tools/llm_data_retriever.py
```

## 📋 数据格式说明

### 输出的JSON数据包含：
- **data**: 具体的数据记录数组
- **count**: 记录总数
- **statistics**: 统计信息（价格范围、平均值等）
- **time_range**: 时间范围

### 每条记录包含：
- **timestamp**: Unix时间戳
- **datetime**: 格式化的时间字符串
- **price**: 当前价格
- **onSaleQuantity**: 在售数量
- **seekPrice**: 求购价格
- **seekQuantity**: 求购数量
- **transactionAmount**: 交易金额
- **transcationNum**: 交易数量
- **surviveNum**: 存活数量

## 💡 使用建议

1. **日常数据查看**: 使用 `--format pretty` 获得易读的表格格式
2. **LLM分析**: 使用默认JSON格式，数据结构化程度高
3. **大量数据**: 使用 `--quiet` 减少输出噪音
4. **数据探索**: 先用小的 `--limit` 值了解数据结构

## 🔧 故障排除

如果遇到问题：
1. 确保在项目根目录运行命令
2. 检查MongoDB连接：`uv run test_mongodb_connection.py`
3. 运行完整测试：`uv run test_tools.py`

## 📈 下一步

- 配置LLM服务后可以使用 `tools/llm_request_tool.py` 进行自动分析
- 根据需要修改 `.env` 文件中的配置
- 可以扩展工具添加更多数据筛选功能