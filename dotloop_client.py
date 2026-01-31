import httpx
from typing import Any, Dict, Optional


class DotloopClient:
    """Minimal Dotloop Loop-It client for create/refresh flows."""

    def __init__(self, api_token: str, base_url: str = "https://api-gw.dotloop.com/public/v2", timeout: float = 15.0):
        if not api_token:
            raise ValueError("Dotloop API token is required")
        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def find_existing_loop(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the first loop whose name matches (best effort)."""
        resp = self._client.get("/loops", params={"name": name, "limit": 1})
        resp.raise_for_status()
        data = resp.json()
        loops = data.get("loops") if isinstance(data, dict) else None
        if loops and isinstance(loops, list):
            return loops[0]
        return None

    def create_loop(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new loop with the provided payload."""
        resp = self._client.post("/loops", json=payload)
        resp.raise_for_status()
        return resp.json()

    def refresh_loop(self, loop_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing loop's details."""
        resp = self._client.patch(f"/loops/{loop_id}", json=payload)
        resp.raise_for_status()
        return resp.json()
