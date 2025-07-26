from fastapi import FastAPI

import requests
import time
import datetime


app = FastAPI()

@app.get("/")
def main():
    return {"status": "ok", "message": "Welcome to the DaoGO Backend!"}


@app.post("/api/frontend/max_diff") # 不同平台的差价
async def max_diff():
    """
    直接去拿上游域名的饰品信息
    然后去做对比
    你说这不导购吗？
    """
    try:
        # 调用上游API获取全量饰品数据
        api_url = "https://api.csqaq.com/api/v1/goods/get_all_goods_info"
        headers = {
            "ApiToken": "MFKMX1M7V5O3R5F2W5S5N6Y1"  # 需要替换为实际的API Token
        }
        
        response = requests.post(api_url, headers=headers)
        
        if response.status_code != 200:
            return {"error": "Failed to fetch data from upstream API", "status_code": response.status_code}
        
        data = response.json()
        
        if data.get("code") != 200:
            return {"error": "Upstream API returned error", "message": data.get("msg", "Unknown error")}
        
        # 解析价格数据并计算差价
        items = data.get("data", [])
        price_analysis = []
        
        for item in items:
            # 提取各平台价格信息
            platforms = {
                "buff": {
                    "sell_price": item.get("buff_sell_price", 0),
                    "buy_price": item.get("buff_buy_price", 0),
                    "sell_num": item.get("buff_sell_num", 0),
                    "buy_num": item.get("buff_buy_num", 0)
                },
                "steam": {
                    "sell_price": item.get("steam_sell_price", 0),
                    "buy_price": item.get("steam_buy_price", 0),
                    "sell_num": item.get("steam_sell_num", 0),
                    "buy_num": item.get("steam_buy_num", 0)
                },
                "yyyp": {
                    "sell_price": item.get("yyyp_sell_price", 0),
                    "buy_price": item.get("yyyp_buy_price", 0),
                    "sell_num": item.get("yyyp_sell_num", 0),
                    "buy_num": item.get("yyyp_buy_num", 0),
                    "lease_price": item.get("yyyp_lease_price", 0),
                    "long_lease_price": item.get("yyyp_long_lease_price", 0),
                    "lease_num": item.get("yyyp_lease_num", 0)
                },
                "r8": {
                    "sell_price": item.get("r8_sell_price", 0),
                    "sell_num": item.get("r8_sell_num", 0)
                }
            }
            
            # 计算最大差价
            sell_prices = []
            buy_prices = []
            
            for platform, prices in platforms.items():
                if prices.get("sell_price", 0) > 0:
                    sell_prices.append({
                        "platform": platform,
                        "price": prices["sell_price"],
                        "quantity": prices.get("sell_num", 0)
                    })
                if prices.get("buy_price", 0) > 0:
                    buy_prices.append({
                        "platform": platform,
                        "price": prices["buy_price"],
                        "quantity": prices.get("buy_num", 0)
                    })
            
            # 找出最高和最低售价
            max_sell = max(sell_prices, key=lambda x: x["price"]) if sell_prices else None
            min_sell = min(sell_prices, key=lambda x: x["price"]) if sell_prices else None
            
            # 找出最高和最低求购价
            max_buy = max(buy_prices, key=lambda x: x["price"]) if buy_prices else None
            min_buy = min(buy_prices, key=lambda x: x["price"]) if buy_prices else None
            
            # 计算差价和利润率
            sell_diff = (max_sell["price"] - min_sell["price"]) if max_sell and min_sell else 0
            buy_diff = (max_buy["price"] - min_buy["price"]) if max_buy and min_buy else 0
            
            # 计算潜在套利机会（最高求购价 vs 最低售价）
            arbitrage_opportunity = 0
            arbitrage_profit_rate = 0
            if max_buy and min_sell and max_buy["price"] > min_sell["price"]:
                arbitrage_opportunity = max_buy["price"] - min_sell["price"]
                arbitrage_profit_rate = (arbitrage_opportunity / min_sell["price"]) * 100
            
            analysis = {
                "market_hash_name": item.get("market_hash_name", ""),
                "name": item.get("name", ""),
                "platforms": platforms,
                "price_analysis": {
                    "sell_price_diff": sell_diff,
                    "buy_price_diff": buy_diff,
                    "max_sell": max_sell,
                    "min_sell": min_sell,
                    "max_buy": max_buy,
                    "min_buy": min_buy,
                    "arbitrage_opportunity": arbitrage_opportunity,
                    "arbitrage_profit_rate": round(arbitrage_profit_rate, 2)
                },
                "statistic": item.get("statistic", 0),
                "updated_at": item.get("updated_at", "")
            }
            
            price_analysis.append(analysis)
        
        # 按套利机会排序
        price_analysis.sort(key=lambda x: x["price_analysis"]["arbitrage_opportunity"], reverse=True)
        
        return {
            "status": "success",
            "total_items": len(price_analysis),
            "data": price_analysis,
            "summary": {
                "top_arbitrage_opportunities": price_analysis[:10] if price_analysis else []
            }
        }
        
    except requests.RequestException as e:
        return {"error": "Network error", "details": str(e)}
    except Exception as e:
        return {"error": "Internal server error", "details": str(e)}



@app.patch("/api/developer/ItemDetail")
async def item_detail():
    """
    利用现有的市场情况，发送给AI Agent分析
    分析完毕后交给 /api/frontend/recommendation
    """

    timestamp = datetime.datetime.strptime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "%Y-%m-%d %H:%M:%S")



@app.post("/api/frontend/recommendation")
async def recommendation(data: dict):
    """
    AI Agent分析完毕，推送给前端
    """
    ...


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)