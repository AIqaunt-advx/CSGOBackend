"""
LLM服务模块
负责向LLM服务器发送请求进行数据分析和预测
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import aiohttp

from config import settings
from modules.database import db_manager

logger = logging.getLogger(__name__)


class LLMService:
    """LLM服务类"""

    def __init__(self):
        self.base_url = settings.LLM_API_BASE_URL
        self.api_key = settings.LLM_API_KEY
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=settings.LLM_REQUEST_TIMEOUT)
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'CSGO-Backend/1.0'
        }
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_llm_request(self, prompt: str, **kwargs) -> Optional[Dict[str, Any]]:
        """向LLM服务器发送请求"""
        try:
            payload = {
                "model": settings.LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的CSGO市场数据分析师，擅长分析市场趋势和价格预测。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": settings.LLM_MAX_TOKENS,
                "temperature": settings.LLM_TEMPERATURE,
                **kwargs
            }

            async with self.session.post(f"{self.base_url}/chat/completions",
                                         json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"LLM request failed with status {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error making LLM request: {e}")
            return None

    async def analyze_market_trend(self, item_name: str, hours: int = 7) -> Dict[str, Any]:
        """分析指定物品的市场趋势"""
        try:
            # 获取最近7小时的数据
            data = await db_manager.get_recent_data(item_name=item_name, hours=hours)

            if not data:
                return {
                    "success": False,
                    "message": "没有找到相关数据",
                    "analysis": None
                }

            # 准备数据摘要
            data_summary = self._prepare_data_summary(data)

            # 构建分析提示词
            prompt = f"""
            请分析以下CSGO物品"{item_name}"的市场数据（最近{hours}小时）：

            数据摘要：
            {json.dumps(data_summary, ensure_ascii=False, indent=2)}

            请提供以下分析：
            1. 价格趋势分析（上升/下降/稳定）
            2. 交易量变化分析
            3. 市场活跃度评估
            4. 短期价格预测（未来1-3小时）
            5. 风险评估和建议

            请以JSON格式返回分析结果，包含以下字段：
            - trend: 趋势（"上升"/"下降"/"稳定"）
            - price_change_percent: 价格变化百分比
            - volume_analysis: 交易量分析
            - market_activity: 市场活跃度（"高"/"中"/"低"）
            - prediction: 短期预测
            - confidence: 预测置信度（0-1）
            - risk_level: 风险等级（"低"/"中"/"高"）
            - recommendation: 投资建议
            """

            # 发送LLM请求
            response = await self._make_llm_request(prompt)

            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']

                # 尝试解析JSON响应
                try:
                    analysis = json.loads(content)
                    return {
                        "success": True,
                        "item_name": item_name,
                        "analysis_time": datetime.utcnow().isoformat(),
                        "data_points": len(data),
                        "analysis": analysis
                    }
                except json.JSONDecodeError:
                    # 如果无法解析JSON，返回原始文本
                    return {
                        "success": True,
                        "item_name": item_name,
                        "analysis_time": datetime.utcnow().isoformat(),
                        "data_points": len(data),
                        "analysis": {"raw_response": content}
                    }
            else:
                return {
                    "success": False,
                    "message": "LLM服务响应异常",
                    "analysis": None
                }

        except Exception as e:
            logger.error(f"Error analyzing market trend: {e}")
            return {
                "success": False,
                "message": f"分析过程中发生错误: {str(e)}",
                "analysis": None
            }

    async def analyze_multiple_items(self, item_names: List[str], hours: int = 7) -> Dict[str, Any]:
        """分析多个物品的市场数据"""
        try:
            results = {}

            # 并发分析多个物品
            tasks = []
            for item_name in item_names:
                task = self.analyze_market_trend(item_name, hours)
                tasks.append(task)

            analyses = await asyncio.gather(*tasks, return_exceptions=True)

            for item_name, analysis in zip(item_names, analyses):
                if isinstance(analysis, Exception):
                    results[item_name] = {
                        "success": False,
                        "message": f"分析失败: {str(analysis)}",
                        "analysis": None
                    }
                else:
                    results[item_name] = analysis

            return {
                "success": True,
                "analysis_time": datetime.utcnow().isoformat(),
                "items_analyzed": len(item_names),
                "results": results
            }

        except Exception as e:
            logger.error(f"Error analyzing multiple items: {e}")
            return {
                "success": False,
                "message": f"批量分析失败: {str(e)}",
                "results": {}
            }

    async def generate_market_summary(self, hours: int = 7) -> Dict[str, Any]:
        """生成市场总体概况"""
        try:
            # 获取所有最近数据
            all_data = await db_manager.get_recent_data(hours=hours)

            if not all_data:
                return {
                    "success": False,
                    "message": "没有找到市场数据",
                    "summary": None
                }

            # 准备市场概况数据
            market_summary = self._prepare_market_summary(all_data)

            prompt = f"""
            请分析以下CSGO市场整体情况（最近{hours}小时）：

            市场概况：
            {json.dumps(market_summary, ensure_ascii=False, indent=2)}

            请提供以下分析：
            1. 市场整体趋势
            2. 热门物品分析
            3. 价格波动情况
            4. 交易活跃度评估
            5. 市场风险提示
            6. 投资机会建议

            请以简洁易懂的方式回答，重点突出关键信息。
            """

            response = await self._make_llm_request(prompt)

            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                return {
                    "success": True,
                    "analysis_time": datetime.utcnow().isoformat(),
                    "data_period_hours": hours,
                    "total_items": market_summary.get("total_items", 0),
                    "summary": content
                }
            else:
                return {
                    "success": False,
                    "message": "LLM服务响应异常",
                    "summary": None
                }

        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return {
                "success": False,
                "message": f"生成市场概况失败: {str(e)}",
                "summary": None
            }

    def _prepare_data_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """准备数据摘要用于LLM分析"""
        if not data:
            return {}

        # 按时间排序
        sorted_data = sorted(data, key=lambda x: x.get('timestamp', datetime.min))

        # 计算价格统计
        prices = [item.get('price', 0) for item in sorted_data if item.get('price')]
        volumes = [item.get('volume', 0) for item in sorted_data if item.get('volume')]

        if not prices:
            return {"error": "没有有效的价格数据"}

        summary = {
            "item_count": len(sorted_data),
            "time_range": {
                "start": sorted_data[0].get('timestamp'),
                "end": sorted_data[-1].get('timestamp')
            },
            "price_stats": {
                "current": prices[-1] if prices else 0,
                "initial": prices[0] if prices else 0,
                "min": min(prices) if prices else 0,
                "max": max(prices) if prices else 0,
                "avg": sum(prices) / len(prices) if prices else 0,
                "change_percent": ((prices[-1] - prices[0]) / prices[0] * 100) if prices and prices[0] > 0 else 0
            },
            "volume_stats": {
                "current": volumes[-1] if volumes else 0,
                "total": sum(volumes) if volumes else 0,
                "avg": sum(volumes) / len(volumes) if volumes else 0
            },
            "sources": list(set(item.get('source', 'unknown') for item in sorted_data))
        }

        return summary

    def _prepare_market_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """准备市场总体数据摘要"""
        if not data:
            return {}

        # 按物品分组
        items_data = {}
        for item in data:
            item_name = item.get('item_name', 'unknown')
            if item_name not in items_data:
                items_data[item_name] = []
            items_data[item_name].append(item)

        # 计算每个物品的统计信息
        item_stats = {}
        for item_name, item_data in items_data.items():
            prices = [d.get('price', 0) for d in item_data if d.get('price')]
            if prices:
                item_stats[item_name] = {
                    "data_points": len(item_data),
                    "price_range": {"min": min(prices), "max": max(prices)},
                    "latest_price": prices[-1],
                    "price_change": ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 and prices[
                        0] > 0 else 0
                }

        # 整体统计
        all_prices = [item.get('price', 0) for item in data if item.get('price')]

        summary = {
            "total_items": len(items_data),
            "total_data_points": len(data),
            "price_overview": {
                "min_price": min(all_prices) if all_prices else 0,
                "max_price": max(all_prices) if all_prices else 0,
                "avg_price": sum(all_prices) / len(all_prices) if all_prices else 0
            },
            "top_items": dict(sorted(item_stats.items(),
                                     key=lambda x: x[1]["latest_price"],
                                     reverse=True)[:10]),
            "sources": list(set(item.get('source', 'unknown') for item in data))
        }

        return summary


# 创建全局LLM服务实例
llm_service = LLMService()
