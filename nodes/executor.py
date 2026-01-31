import os
from typing import Any, Dict, List, Optional

import httpx

from state import DealState
from dotloop_client import DotloopClient


def _normalize_loop_id(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    try:
        return str(raw)
    except Exception:
        return None


def api_executor_node(state: DealState) -> Dict[str, Any]:
    """
    Node G: API Executor
    Pushes data to Dotloop and refreshes if a loop already exists.
    """
    print("--- NODE: API Executor ---")

    system = state.get("target_system", "dotloop")
    if system != "dotloop":
        return {"status": "Skipped", "sync_errors": []}

    payload = state.get("dotloop_payload")
    if not payload:
        return {
            "status": "Failed",
            "sync_errors": ["dotloop_payload is missing; cannot sync"],
        }

    api_token = os.environ.get("DOTLOOP_API_TOKEN")
    if not api_token:
        return {
            "status": "Failed",
            "sync_errors": ["DOTLOOP_API_TOKEN is not configured"],
        }

    loop_id = state.get("dotloop_loop_id")
    loop_url = state.get("dotloop_loop_url")
    errors: List[str] = []

    try:
        with DotloopClient(api_token=api_token) as client:
            if loop_id:
                print(f"   Refreshing existing Dotloop loop {loop_id}...")
                refreshed = client.refresh_loop(loop_id, payload)
                loop_url = loop_url or refreshed.get("loopUrl") or refreshed.get("url")
                status_label = "Refreshed"
            else:
                loop_name = payload.get("name", "")
                existing = client.find_existing_loop(loop_name) if loop_name else None
                if existing:
                    loop_id = _normalize_loop_id(
                        existing.get("id")
                        or existing.get("loopId")
                        or existing.get("loop_id")
                    )
                    loop_url = existing.get("loopUrl") or existing.get("url")
                    print(f"   Found existing Dotloop loop {loop_id}; refreshing...")
                    client.refresh_loop(loop_id, payload)
                    status_label = "Refreshed"
                else:
                    print("   Creating Dotloop loop...")
                    created = client.create_loop(payload)
                    loop_id = _normalize_loop_id(
                        created.get("id")
                        or created.get("loopId")
                        or created.get("loop_id")
                    )
                    loop_url = created.get("loopUrl") or created.get("url")
                    status_label = "Created"

    except httpx.HTTPError as exc:
        errors.append(f"Dotloop API error: {exc}")
    except Exception as exc:
        errors.append(f"Unexpected sync error: {exc}")

    if errors:
        return {
            "status": "Failed",
            "dotloop_loop_id": loop_id,
            "dotloop_loop_url": loop_url,
            "sync_errors": errors,
        }

    return {
        "status": "Synced",
        "dotloop_loop_id": loop_id,
        "dotloop_loop_url": loop_url,
        "sync_errors": [],
        "dotloop_sync_action": status_label,
    }