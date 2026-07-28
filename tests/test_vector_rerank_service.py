"""向量重排服务单元测试"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.vector_rerank_service import VectorRerankService


class TestVectorRerankService:
    """VectorRerankService 测试类"""

    @pytest.fixture
    def rerank_service(self):
        """创建测试用的 rerank 服务实例"""
        with patch("app.services.vector_rerank_service.config") as mock_config:
            mock_config.dashscope_rerank_model = "qwen3-rerank"
            mock_config.dashscope_api_key = "test-api-key"
            mock_config.dashscope_biz_space_api_base = "https://test.cn-beijing.maas.aliyuncs.com"
            yield VectorRerankService()

    def test_rerank_success(self, rerank_service):
        """测试正常 rerank 调用"""
        query = "什么是 Kubernetes"
        documents = [
            "Kubernetes 是一个容器编排平台",
            "Python 是一种编程语言",
            "Docker 是容器运行时",
        ]

        # Mock HTTP JSON 响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {
                "results": [
                    {"document": {"text": documents[0]}, "index": 0, "relevance_score": 0.95},
                    {"document": {"text": documents[2]}, "index": 2, "relevance_score": 0.80},
                ]
            }
        }

        with patch("app.services.vector_rerank_service.requests.post", return_value=mock_response):
            results = rerank_service.rerank(query, documents, top_k=2)

        assert len(results) == 2
        assert results[0]["index"] == 0
        assert results[0]["relevance_score"] == 0.95
        assert results[0]["text"] == documents[0]
        assert results[1]["index"] == 2
        assert results[1]["relevance_score"] == 0.80
        assert results[1]["text"] == documents[2]

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
        mock_response.text = "Internal Server Error"

        with patch("app.services.vector_rerank_service.requests.post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Rerank API 调用失败"):
                rerank_service.rerank("query", ["doc1"], top_k=1)

    def test_rerank_network_error(self, rerank_service):
        """测试网络异常"""
        with patch(
            "app.services.vector_rerank_service.requests.post",
            side_effect=requests.RequestException("Network error"),
        ):
            with pytest.raises(RuntimeError, match="Rerank HTTP 请求失败"):
                rerank_service.rerank("query", ["doc1"], top_k=1)

    def test_rerank_topk大于文档数(self, rerank_service):
        """测试 top_k 大于文档数量的情况"""
        query = "测试"
        documents = ["文档1"]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {
                "results": [
                    {"document": {"text": documents[0]}, "index": 0, "relevance_score": 0.9},
                ]
            }
        }

        with patch("app.services.vector_rerank_service.requests.post", return_value=mock_response):
            # top_k=5 但只有 1 个文档
            results = rerank_service.rerank(query, documents, top_k=5)

        assert len(results) == 1
        assert results[0]["index"] == 0
