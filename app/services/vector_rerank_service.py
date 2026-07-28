"""向量重排服务模块 - 基于 DashScope TextReRank API 实现精排"""

from typing import List

import dashscope
from loguru import logger

from app.config import config

# 设定 dashscope 业务空间的 api 根路径（rerank模型会接受到私有知识库数据，必须以业务空间为单位隔离知识和防泄密）
dashscope.base_http_api_url = config.dashscope_biz_space_api_base

class VectorRerankService:
    """
    向量重排服务 - 使用 DashScope Rerank API 对候选文档进行精排

    工作流程：
    1. 接收 query 和候选文档列表（由粗排阶段产生）
    2. 调用 DashScope TextReRank API 计算每个文档与 query 的相关性分数
    3. 按分数降序排序，返回 top_k 个最相关的文档
    """

    def __init__(self):
        """初始化向量重排服务"""
        self.model = config.dashscope_rerank_model
        self.api_key = config.dashscope_api_key
        logger.info(f"向量重排服务初始化完成, 模型: {self.model}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3,
    ) -> List[dict]:
        """
        对候选文档进行重排序

        Args:
            query: 查询文本
            documents: 候选文档文本列表
            top_k: 返回前 K 个最相关文档

        Returns:
            List[dict]: 重排后的文档列表，每项包含:
                - index: 原始文档在输入列表中的索引
                - text: 文档文本
                - relevance_score: 相关性分数（越高越相关）
        """
        if not documents:
            logger.warning("候选文档列表为空，跳过重排")
            return []

        if not query or not query.strip():
            logger.warning("查询文本为空，跳过重排")
            return []

        try:
            logger.info(
                f"开始 Rerank: query='{query[:50]}...', "
                f"候选数: {len(documents)}, top_k: {top_k}"
            )

            # 调用 DashScope TextReRank API
            response = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k,
                api_key=self.api_key,
                return_documents=True,
            )

            # 解析响应
            if response.status_code != 200:
                logger.error(
                    f"Rerank API 调用失败: "
                    f"status_code={response.status_code}, "
                    f"message={response.message}"
                )
                raise RuntimeError(
                    f"Rerank API 调用失败: {response.status_code} - {response.message}"
                )

            # 提取重排结果
            rerank_results = []
            for item in response.output.results:
                rerank_results.append({
                    "index": item.index,
                    "text": item.text if hasattr(item, "text") else documents[item.index],
                    "relevance_score": item.relevance_score,
                })

            logger.info(
                f"Rerank 完成, 返回 {len(rerank_results)} 个结果, "
                f"最高分: {rerank_results[0]['relevance_score']:.4f}"
                if rerank_results else "Rerank 完成, 无结果"
            )

            return rerank_results

        except Exception as e:
            logger.error(f"Rerank 重排失败: {e}")
            raise RuntimeError(f"Rerank 重排失败: {e}") from e


# 全局单例
vector_rerank_service = VectorRerankService()
