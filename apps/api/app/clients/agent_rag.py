import json
import os
import urllib.error
import urllib.request


class AgentRagClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: float = 5.0) -> None:
        self.base_url = (base_url or os.getenv("AGENT_RAG_URL", "http://127.0.0.1:8013/agent-api/v1")).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def draft_report(self, payload: dict[str, object]) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self.base_url}/draft-report",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise TimeoutError(str(exc)) from exc
