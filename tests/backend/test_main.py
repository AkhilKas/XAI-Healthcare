import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)
# Separate client that converts server-side exceptions into 500 responses
# instead of re-raising them in the test process.
unsafe_client = TestClient(app, raise_server_exceptions=False)


def test_root_returns_running_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Backend running!"}


def test_go_requires_patient_and_task():
    response = client.post("/go", json={})
    assert response.status_code == 422  # FastAPI validation error


def test_go_rejects_unknown_task():
    # /go does a dict lookup on task name; unknown task → KeyError → 500
    response = unsafe_client.post("/go", json={"patient": "patient_1", "task": "Skydiving"})
    assert response.status_code == 500


def test_go_raises_keyerror_for_unknown_task_in_strict_mode():
    # Confirms the underlying root cause is a KeyError on the task mapping.
    with pytest.raises(KeyError):
        client.post("/go", json={"patient": "patient_1", "task": "Skydiving"})
