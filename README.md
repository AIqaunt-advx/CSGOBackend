# CSGO Market Price Difference API

这是一个用于分析CSGO饰品在不同平台价格差异的API服务。

## 功能特性

- 🔍 获取全量饰品价格数据
- 📊 计算不同平台间的价格差异
- 💰 识别套利机会
- 📈 提供详细的价格分析

## API接口

### GET /
健康检查接口

### POST /api/max_diff
获取饰品价格差异分析

**响应格式：**
```json
{
  "status": "success",
  "total_items": 1,
  "data": [
    {
      "market_hash_name": "★ Sport Gloves | Vice (Minimal Wear)",
      "name": "运动手套（★） | 迈阿密风云 (略有磨损)",
      "platforms": {
        "buff": {
          "sell_price": 29000.0,
          "buy_price": 28100.0,
          "sell_num": 760,
          "buy_num": 42
        },
        "steam": {...},
        "yyyp": {...},
        "r8": {...}
      },
      "price_analysis": {
        "sell_price_diff": 221.0,
        "buy_price_diff": 47678.73,
        "max_sell": {"platform": "yyyp", "price": 29221.0, "quantity": 811},
        "min_sell": {"platform": "buff", "price": 29000.0, "quantity": 760},
        "arbitrage_opportunity": 0,
        "arbitrage_profit_rate": 0
      },
      "statistic": 4581,
      "updated_at": "2025-07-25T16:52:57"
    }
  ],
  "summary": {
    "top_arbitrage_opportunities": [...]
  }
}
```

## 安装和运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行服务：
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. 访问API文档：
```
http://localhost:8000/docs
```

## 测试

### 运行所有测试
```bash
python -m pytest test_main.py -v
```

### 运行测试并生成覆盖率报告
```bash
python run_tests.py --cov
```

### 测试用例说明

我们的测试套件包含以下测试用例：

1. **test_root_endpoint** - 测试根路径健康检查
2. **test_max_diff_success** - 测试正常获取价格数据
3. **test_max_diff_upstream_api_error** - 测试上游API错误处理
4. **test_max_diff_upstream_api_business_error** - 测试业务逻辑错误处理
5. **test_max_diff_network_error** - 测试网络错误处理
6. **test_max_diff_empty_data** - 测试空数据处理
7. **test_max_diff_price_calculation** - 测试价格计算逻辑
8. **test_max_diff_arbitrage_opportunity** - 测试套利机会计算
9. **test_max_diff_json_decode_error** - 测试JSON解析错误处理

### 测试覆盖的场景

- ✅ 正常数据流程
- ✅ 各种错误情况处理
- ✅ 价格计算逻辑验证
- ✅ 套利机会识别
- ✅ 边界条件测试

## 配置

在使用前，请确保在 `main.py` 中配置正确的API Token：

```python
headers = {
    "ApiToken": "your_actual_api_token_here"  # 替换为实际的API Token
}
```

## 项目结构

```
.
├── main.py              # 主应用文件
├── test_main.py         # 单元测试
├── run_tests.py         # 测试运行脚本
├── requirements.txt     # 依赖列表
├── pytest.ini          # pytest配置
└── README.md           # 项目说明
```

## 技术栈

- **FastAPI** - 现代、快速的Web框架
- **Requests** - HTTP客户端库
- **Pytest** - 测试框架
- **Uvicorn** - ASGI服务器

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！