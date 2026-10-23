"""Private MCP media service.

``config``, ``types``, ``jobs`` and ``runtime`` own settings, wire results, the
SQLite store and the planner/worker lifecycle. ``auth`` and ``app`` add the
transport: JWT issuance/verification, the FastMCP tools and resource, and the
protected custom artifact route. The runtime and the agent package never import
FastMCP, Starlette or Uvicorn, so the durable core can be exercised without a
server.
"""
