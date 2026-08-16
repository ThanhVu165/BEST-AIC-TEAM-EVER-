from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_contract_kis():
    response = client.post(
        "/api/v1/search",
        json={
            "query_id": "integration-kis",
            "task": "KIS",
            "description": "find the described event",
            "raw_text": "find the described event",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "KIS"
    assert len(body["candidates"]) == 1
    assert body["candidates"][0]["video_id"] == "MOCK_VIDEO_001"


def test_search_contract_qa():
    response = client.post(
        "/api/v1/search",
        json={
            "query_id": "integration-qa",
            "task": "QA",
            "description": "what is shown?",
            "question": "What is shown?",
        },
    )
    assert response.status_code == 200
    assert response.json()["candidates"][0]["answer"] == "mock answer"
