"""向量重排服务单元测试"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.vector_rerank_service import VectorRerankService


class TestVectorRerankService:
    """VectorRerankService 测试类"""

    @pytest.fixture
    def rerank_service(self):
        """创建测试用的 rerank 服务实例"""
        with patch("app.services.vector_rerank_service.config") as mock_config:
            mock_config.dashscope_rerank_model = "text-rerank-v1"
            mock_config.dashscope_api_key = "test-api-key"
            yield VectorRerankService()

    def test_rerank_success(self, rerank_service):
        """测试正常 rerank 调用"""
        query = "什么是 Kubernetes"
        documents = [
            "Kubernetes 是一个容器编排平台",
            "Python 是一种编程语言",
            "Docker 是容器运行时",
        ]

        # Mock API 响应
        mock_result_1 = MagicMock()
        mock_result_1.index = 0
        mock_result_1.text = documents[0]
        mock_result_1.relevance_score = 0.95

        mock_result_2 = MagicMock()
        mock_result_2.index = 2
        mock_result_2.text = documents[2]
        mock_result_2.relevance_score = 0.80

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output.results = [mock_result_1, mock_result_2]

        with patch(
            "app.services.vector_rerank_service.TextReRank.call",
            return_value=mock_response,
        ):
            results = rerank_service.rerank(query, documents, top_k=2)

        assert len(results) == 2
        assert results[0]["index"] == 0
        assert results[0]["relevance_score"] == 0.95
        assert results[1]["index"] == 2
        assert results[1]["relevance_score"] == 0.80

    def test_rerank_empty_documents(self, rerank_service):
        """测试空文档列表"""
        results = rerank_service.rerank("query", [], top_k=3)
        assert results == []

    def test_rerank_empty_query(self, rerank_service):
        """测试空查询文本"""
        results = rerank_service.rerank("", ["doc1", "doc2"], top_k=3)
        assert results == []

    def test_rerank_whitespace_query(self, rerank_service):
        """测试空白字符查询"""
        results = rerank_service.rerank("   ", ["doc1", "doc2"], top_k=3)
        assert results == []

    def test_rerank_api_error(self, rerank_service):
        """测试 API 调用失败"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.message = "Internal Server Error"

        with patch(
            "app.services.vector_rerank_service.TextReRank.call",
            return_value=mock_response,
        ):
            with pytest.raises(RuntimeError, match="Rerank API 调用失败"):
                rerank_service.rerank("query", ["doc1"], top_k=1)

    def test_rerank_network_error(self, rerank_service):
        """测试网络异常"""
        with patch(
            "app.services.vector_rerank_service.TextReRank.call",
            side_effect=ConnectionError("Network error"),
        ):
            with pytest.raises(RuntimeError, match="Rerank 重排失败"):
                rerank_service.rerank("query", ["doc1"], top_k=1)

    def test_rerank_topk大于文档数(self, rerank_service):
        """测试 top_k 大于文档数量的情况"""
        query = "测试"
        documents = ["文档1"]

        mock_result = MagicMock()
        mock_result.index = 0
        mock_result.text = "文档1"
        mock_result.relevance_score = 0.9

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output.results = [mock_result]

        with patch(
            "app.services.vector_rerank_service.TextReRank.call",
            return_value=mock_response,
        ):
            # top_k=5 但只有 1 个文档
            results = rerank_service.rerank(query, documents, top_k=5)

        assert len(results) == 1
        assert results[0]["index"] == 0
