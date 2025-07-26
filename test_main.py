import pytest
import json
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestMaxDiff:
    """测试 max_diff 接口的单元测试"""
    
    def test_root_endpoint(self):
        """测试根路径接口"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "message": "Welcome to the DaoGO Backend!"}
    
    @patch('main.requests.post')
    def test_max_diff_success(self, mock_post):
        """测试成功获取价格差价数据"""
        # 模拟上游API返回的数据
        mock_response_data = {
            "code": 200,
            "msg": "Success",
            "data": [
                {
                    "market_hash_name": "★ Sport Gloves | Vice (Minimal Wear)",
                    "name": "运动手套（★） | 迈阿密风云 (略有磨损)",
                    "buff_sell_price": 29000.0,
                    "buff_sell_num": 760,
                    "buff_buy_price": 28100.0,
                    "buff_buy_num": 42,
                    "steam_sell_price": 0.0,
                    "steam_sell_num": 0,
                    "steam_buy_price": 17321.27,
                    "steam_buy_num": 345,
                    "yyyp_sell_price": 29221.0,
                    "yyyp_sell_num": 811,
                    "yyyp_buy_price": 65000.0,
                    "yyyp_buy_num": 48,
                    "yyyp_lease_num": 44,
                    "yyyp_lease_price": 3.5,
                    "yyyp_long_lease_price": 3.7,
                    "r8_sell_price": 0.0,
                    "r8_sell_num": 0,
                    "statistic": 4581,
                    "statistic_at": "2025-07-25T16:29:05",
                    "updated_at": "2025-07-25T16:52:57"
                }
            ]
        }
        
        # 配置mock响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response
        
        # 发送请求
        response = client.post("/api/max_diff")
        
        # 验证响应
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["total_items"] == 1
        assert len(data["data"]) == 1
        
        # 验证第一个商品的数据结构
        item = data["data"][0]
        assert item["market_hash_name"] == "★ Sport Gloves | Vice (Minimal Wear)"
        assert item["name"] == "运动手套（★） | 迈阿密风云 (略有磨损)"
        
        # 验证平台数据
        assert "platforms" in item
        assert "buff" in item["platforms"]
        assert "steam" in item["platforms"]
        assert "yyyp" in item["platforms"]
        assert "r8" in item["platforms"]
        
        # 验证价格分析
        assert "price_analysis" in item
        price_analysis = item["price_analysis"]
        assert "sell_price_diff" in price_analysis
        assert "buy_price_diff" in price_analysis
        assert "arbitrage_opportunity" in price_analysis
        assert "arbitrage_profit_rate" in price_analysis
        
        # 验证API调用
        mock_post.assert_called_once_with(
            "https://api.csqaq.com/api/v1/goods/get_all_goods_info",
            headers={"ApiToken": "your_api_token_here"}
        )
    
    @patch('main.requests.post')
    def test_max_diff_upstream_api_error(self, mock_post):
        """测试上游API返回错误状态码"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        response = client.post("/api/max_diff")
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"] == "Failed to fetch data from upstream API"
        assert data["status_code"] == 500
    
    @patch('main.requests.post')
    def test_max_diff_upstream_api_business_error(self, mock_post):
        """测试上游API返回业务错误"""
        mock_response_data = {
            "code": 400,
            "msg": "Invalid request"
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response
        
        response = client.post("/api/max_diff")
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"] == "Upstream API returned error"
        assert data["message"] == "Invalid request"
    
    @patch('main.requests.post')
    def test_max_diff_network_error(self, mock_post):
        """测试网络错误"""
        import requests
        mock_post.side_effect = requests.RequestException("Network error")
        
        response = client.post("/api/max_diff")
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"] == "Network error"
        assert "Network error" in data["details"]
    
    @patch('main.requests.post')
    def test_max_diff_empty_data(self, mock_post):
        """测试空数据响应"""
        mock_response_data = {
            "code": 200,
            "msg": "Success",
            "data": []
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response
        
        response = client.post("/api/max_diff")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_items"] == 0
        assert len(data["data"]) == 0
        assert len(data["summary"]["top_arbitrage_opportunities"]) == 0
    
    @patch('main.requests.post')
    def test_max_diff_price_calculation(self, mock_post):
        """测试价格计算逻辑"""
        mock_response_data = {
            "code": 200,
            "msg": "Success",
            "data": [
                {
                    "market_hash_name": "Test Item",
                    "name": "测试物品",
                    "buff_sell_price": 100.0,
                    "buff_sell_num": 10,
                    "buff_buy_price": 90.0,
                    "buff_buy_num": 5,
                    "steam_sell_price": 120.0,
                    "steam_sell_num": 8,
                    "steam_buy_price": 85.0,
                    "steam_buy_num": 3,
                    "yyyp_sell_price": 110.0,
                    "yyyp_sell_num": 12,
                    "yyyp_buy_price": 95.0,
                    "yyyp_buy_num": 7,
                    "yyyp_lease_price": 2.0,
                    "yyyp_long_lease_price": 2.5,
                    "yyyp_lease_num": 5,
                    "r8_sell_price": 105.0,
                    "r8_sell_num": 6,
                    "statistic": 1000,
                    "updated_at": "2025-07-25T16:52:57"
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response
        
        response = client.post("/api/max_diff")
        
        assert response.status_code == 200
        data = response.json()
        
        item = data["data"][0]
        price_analysis = item["price_analysis"]
        
        # 验证售价差价计算 (最高120 - 最低100 = 20)
        assert price_analysis["sell_price_diff"] == 20.0
        
        # 验证求购价差价计算 (最高95 - 最低85 = 10)
        assert price_analysis["buy_price_diff"] == 10.0
        
        # 验证最高售价平台
        assert price_analysis["max_sell"]["platform"] == "steam"
        assert price_analysis["max_sell"]["price"] == 120.0
        
        # 验证最低售价平台
        assert price_analysis["min_sell"]["platform"] == "buff"
        assert price_analysis["min_sell"]["price"] == 100.0
        
        # 验证套利机会计算 (无套利机会，因为最高求购价95 < 最低售价100)
        assert price_analysis["arbitrage_opportunity"] == 0
        assert price_analysis["arbitrage_profit_rate"] == 0
    
    @patch('main.requests.post')
    def test_max_diff_arbitrage_opportunity(self, mock_post):
        """测试套利机会计算"""
        mock_response_data = {
            "code": 200,
            "msg": "Success",
            "data": [
                {
                    "market_hash_name": "Arbitrage Item",
                    "name": "套利物品",
                    "buff_sell_price": 100.0,
                    "buff_sell_num": 10,
                    "buff_buy_price": 90.0,
                    "buff_buy_num": 5,
                    "steam_sell_price": 80.0,  # 最低售价
                    "steam_sell_num": 8,
                    "steam_buy_price": 85.0,
                    "steam_buy_num": 3,
                    "yyyp_sell_price": 110.0,
                    "yyyp_sell_num": 12,
                    "yyyp_buy_price": 95.0,  # 最高求购价
                    "yyyp_buy_num": 7,
                    "yyyp_lease_price": 2.0,
                    "yyyp_long_lease_price": 2.5,
                    "yyyp_lease_num": 5,
                    "r8_sell_price": 105.0,
                    "r8_sell_num": 6,
                    "statistic": 1000,
                    "updated_at": "2025-07-25T16:52:57"
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_post.return_value = mock_response
        
        response = client.post("/api/max_diff")
        
        assert response.status_code == 200
        data = response.json()
        
        item = data["data"][0]
        price_analysis = item["price_analysis"]
        
        # 验证套利机会 (95 - 80 = 15)
        assert price_analysis["arbitrage_opportunity"] == 15.0
        
        # 验证套利利润率 (15/80 * 100 = 18.75%)
        assert price_analysis["arbitrage_profit_rate"] == 18.75
    
    @patch('main.requests.post')
    def test_max_diff_json_decode_error(self, mock_post):
        """测试JSON解析错误"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response
        
        response = client.post("/api/max_diff")
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"] == "Internal server error"


if __name__ == "__main__":
    pytest.main([__file__])