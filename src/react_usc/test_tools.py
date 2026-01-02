
from typing import Any, Dict, cast
import time
from .models import ToolSpec

def make_flaky_tool() -> ToolSpec:
    """
    A simulated API client tool that mimics HTTP requests with various failure modes.
    
    Commands simulate endpoints:
    - "GET /api/v1/users/{id}": Requires a specific query param 'include_profile=true' (Arg Fix scenario).
    - "POST /api/v1/sync/data": Simulates a flaky upstream service (Wait/Retry scenario).
    - "DELETE /api/v1/admin/system": Simulates a forbidden endpoint (Abort scenario).
    - "GET /health": Succeeds immediately.
    """
    
    # State to track attempts per command type
    state = {
        "sync_attempts": 0
    }

    def _http_client_mock(args: Dict[str, Any]) -> Any:
        endpoint = cast(str, args.get("endpoint", "")).strip()
        method = cast(str, args.get("method", "GET")).upper()
        params = args.get("params", {})
        
        # Scenario 1: RETRY (Arg Fix)
        # "GET /api/v1/users/123" fails unless "include_profile" is True
        if method == "GET" and "/api/v1/users/" in endpoint:
            if params.get("include_profile") is True:
                 return {"status": 200, "data": {"id": "123", "name": "Alice", "profile": "active"}}
            
            # Simulate a 400 Bad Request with a helpful message
            raise ValueError("400 Bad Request: Missing required query parameter 'include_profile' for user fetch.")

        # Scenario 2: WAIT (Transient 503)
        # "POST /api/v1/sync/data" fails twice with 503, then succeeds
        elif method == "POST" and endpoint == "/api/v1/sync/data":
            state["sync_attempts"] += 1
            if state["sync_attempts"] < 3:
                raise RuntimeError("503 Service Unavailable: Upstream data sync service is overloaded. Retry-After: 1s")
            return {"status": 201, "message": f"Data synced successfully on attempt {state['sync_attempts']}"}

        # Scenario 3: ABORT (Fatal 403/405)
        # "DELETE /api/v1/admin/system" is strictly forbidden
        elif method == "DELETE" and "/admin/system" in endpoint:
            raise ValueError("403 Forbidden: API credentials lack 'system.delete' scope. This action is not allowed.")

        # Scenario 4: Success
        elif endpoint == "/health":
            return {"status": 200, "status": "ok"}
            
        return {"status": 404, "error": f"Endpoint not found: {method} {endpoint}"}

    return ToolSpec(
        name="api_client",
        description="A generic HTTP API client for internal services. Supports GET, POST, DELETE.",
        input_schema={
            "type": "object",
            "required": ["endpoint", "method"],
            "properties": {
                "endpoint": {"type": "string", "description": "API endpoint path (e.g. /api/v1/users/123)"},
                "method": {"type": "string", "enum": ["GET", "POST", "DELETE"]},
                "params": {"type": "object", "description": "Query parameters or body data"}
            },
        },
        func=_http_client_mock,
    )
