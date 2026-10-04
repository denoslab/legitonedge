import json
from unittest.mock import patch, MagicMock

from legit_edge.cli.ollama_pull import pull_model


def test_pull_model_streams_events_and_returns_final_status():
    lines = [
        json.dumps({"status": "pulling manifest"}),
        json.dumps({"status": "downloading", "completed": 50, "total": 100}),
        json.dumps({"status": "downloading", "completed": 100, "total": 100}),
        json.dumps({"status": "success"}),
    ]
    fake_response = MagicMock()
    fake_response.iter_lines.return_value = (l.encode() for l in lines)
    fake_response.__enter__.return_value = fake_response
    fake_response.raise_for_status.return_value = None

    with patch("legit_edge.cli.ollama_pull.requests.post", return_value=fake_response) as p:
        events = list(pull_model("http://x:11434", "llama3.1:8b-instruct-q4_K_M"))

    assert events[-1]["status"] == "success"
    assert any(e.get("status") == "downloading" for e in events)
    p.assert_called_once()
