"""向量检索服务单元测试"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.services.vector_search_service import VectorSearchService


class TestVectorSearchService:
    """VectorSearchService 测试类"""

    @pytest.fixture
    def search_service(self):
        """创建测试用的 search 服务实例"""
        return VectorSearchService()

    def test_search_without_rerank(self, search_service):
        """测试不使用 rerank 的搜索"""
        query = "什么是 Kubernetes"

        # Mock 粗排结果
        mock_docs = [
            Document(page_content="K8s 是容器编排平台", metadata={"_file_name": "k8s.md"}),
            Document(page_content="Docker 是容器运行时", metadata={"_file_name": "docker.md"}),
        ]

        with patch(
            "app.services.vector_search_service.vector_store_manager"
        ) as mock_manager, patch(
            "app.services.vector_search_service.config"
        ) as mock_config:
            mock_manager.similarity_search.return_value = mock_docs
            mock_config.rerank_enabled = False
            mock_config.rag_top_k = 2

            results = search_service.search(query, top_k=2)

        assert len(results) == 2
        assert results[0].metadata["score"] == 1.0  # 未精排时默认分数

    def test_search_with_rerank(self, search_service):
        """测试使用 rerank 的搜索"""
        query = "什么是 Kubernetes"

        # Mock 粗排结果
        mock_docs = [
            Document(page_content="K8s 是容器编排平台", metadata={"_file_name": "k8s.md"}),
            Document(page_content="Python 是编程语言", metadata={"_file_name": "python.md"}),
            Document(page_content="Docker 是容器运行时", metadata={"_file_name": "docker.md"}),
        ]

        # Mock rerank 结果
        mock_rerank_results = [
            {"index": 0, "relevance_score": 0.95, "text": "K8s 是容器编排平台"},
            {"index": 2, "relevance_score": 0.80, "text": "Docker 是容器运行时"},
        ]

        with patch(
            "app.services.vector_search_service.vector_store_manager"
        ) as mock_manager, patch(
            "app.services.vector_search_service.config"
        ) as mock_config, patch(
            "app.services.vector_search_service.vector_rerank_service"
        ) as mock_rerank:
            mock_manager.similarity_search.return_value = mock_docs
            mock_config.rerank_enabled = True
            mock_config.rerank_candidate_count = 30
            mock_rerank.rerank.return_value = mock_rerank_results

            results = search_service.search(query, top_k=2)

        assert len(results) == 2
        assert results[0].metadata["score"] == 0.95
        assert results[1].metadata["score"] == 0.80
        assert results[0].page_content == "K8s 是容器编排平台"
        assert results[1].page_content == "Docker 是容器运行时"

    def test_search_empty_results(self, search_service):
        """测试空结果"""
        query = "不存在的内容"

        with patch(
            "app.services.vector_search_service.vector_store_manager"
        ) as mock_manager, patch(
            "app.services.vector_search_service.config"
        ) as mock_config:
            mock_manager.similarity_search.return_value = []
            mock_config.rerank_enabled = False

            results = search_service.search(query, top_k=3)

        assert results == []

    def test_search_rerank_fallback(self, search_service):
        """测试 rerank 失败时降级"""
        query = "什么是 Kubernetes"

        mock_docs = [
            Document(page_content="K8s 是容器编排平台", metadata={"_file_name": "k8s.md"}),
            Document(page_content="Docker 是容器运行时", metadata={"_file_name": "docker.md"}),
        ]

        with patch(
            "app.services.vector_search_service.vector_store_manager"
        ) as mock_manager, patch(
            "app.services.vector_search_service.config"
        ) as mock_config, patch(
            "app.services.vector_search_service.vector_rerank_service"
        ) as mock_rerank:
            mock_manager.similarity_search.return_value = mock_docs
            mock_config.rerank_enabled = True
            mock_config.rerank_candidate_count = 30
            mock_rerank.rerank.side_effect = RuntimeError("API error")

            results = search_service.search(query, top_k=2)

        # 降级为原始排序
        assert len(results) == 2
        assert results[0].metadata["score"] == 1.0  # 降级时默认分数

    def test_search_recall_count_calculation(self, search_service):
        """测试召回数量计算"""
        query = "测试"

        mock_docs = [Document(page_content=f"文档{i}", metadata={}) for i in range(30)]

        with patch(
            "app.services.vector_search_service.vector_store_manager"
        ) as mock_manager, patch(
            "app.services.vector_search_service.config"
        ) as mock_config:
            mock_manager.similarity_search.return_value = mock_docs
            mock_config.rerank_enabled = True
            mock_config.rerank_candidate_count = 30

            search_service.search(query, top_k=3)

            # 验证召回数量：min(top_k * 10, rerank_candidate_count) = min(30, 30) = 30
            mock_manager.similarity_search.assert_called_once_with(query, k=30)
