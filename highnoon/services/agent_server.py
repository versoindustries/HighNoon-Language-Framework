# highnoon/services/agent_server.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase B2: HighNoon MCP Server.

Model Context Protocol (MCP) server for exposing HighNoon tools and
resources to external agents (Claude Code, etc.).

Supports multiple transports:
    - stdio: Standard input/output for CLI integration
    - SSE: Server-Sent Events for web integration

Key Features:
    - Tool discovery and invocation
    - Resource exposure (model info, training status)
    - Async request handling
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from highnoon.cli.manifest import ToolManifest, create_default_manifest

log = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """Incoming MCP request."""

    id: str | int
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResponse:
    """Outgoing MCP response."""

    id: str | int
    result: Any = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-RPC format."""
        response = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            response["error"] = self.error
        else:
            response["result"] = self.result
        return response


class MCPServer:
    """Model Context Protocol server.

    Implements the MCP protocol for tool discovery and invocation.
    Can be used with stdio transport for CLI or SSE for web.
    """

    def __init__(
        self,
        manifest: ToolManifest | None = None,
        server_name: str = "highnoon-mcp",
        server_version: str = "1.0.0",
    ) -> None:
        """Initialize MCP server.

        Args:
            manifest: Tool manifest for available tools.
            server_name: Server identification name.
            server_version: Server version string.
        """
        self.manifest = manifest or create_default_manifest()
        self.server_name = server_name
        self.server_version = server_version
        self._handlers: dict[str, Callable] = {}
        self._register_handlers()

        log.info(f"Initialized MCPServer: {server_name} v{server_version}")

    def _register_handlers(self) -> None:
        """Register MCP method handlers."""
        self._handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "notifications/initialized": self._handle_initialized,
        }

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an incoming MCP request.

        Args:
            request: The MCP request to handle.

        Returns:
            MCP response.
        """
        handler = self._handlers.get(request.method)

        if handler is None:
            log.warning(f"Unknown method: {request.method}")
            return MCPResponse(
                id=request.id,
                error={"code": -32601, "message": f"Method not found: {request.method}"},
            )

        try:
            result = await handler(request.params)
            return MCPResponse(id=request.id, result=result)
        except Exception as e:
            log.error(f"Error handling {request.method}: {e}")
            return MCPResponse(id=request.id, error={"code": -32603, "message": str(e)})

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version,
            },
        }

    async def _handle_initialized(self, params: dict) -> None:
        """Handle initialized notification."""
        log.info("Client initialized")
        return None

    async def _handle_tools_list(self, params: dict) -> dict:
        """Handle tools/list request."""
        tools = []
        for tool_info in self.manifest.list_tools():
            tools.append(
                {
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                }
            )
        return {"tools": tools}

    async def _handle_tools_call(self, params: dict) -> dict:
        """Handle tools/call request."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if not name:
            raise ValueError("Tool name is required")

        # Dispatch tool call
        result = self.manifest.dispatch(name, arguments)

        return {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
            "isError": result.get("status") == "error",
        }

    async def _handle_resources_list(self, params: dict) -> dict:
        """Handle resources/list request."""
        return {
            "resources": [
                {
                    "uri": "highnoon://model/info",
                    "name": "Model Information",
                    "description": "Current model configuration and status",
                    "mimeType": "application/json",
                },
                {
                    "uri": "highnoon://training/status",
                    "name": "Training Status",
                    "description": "Current training run status",
                    "mimeType": "application/json",
                },
            ]
        }

    async def _handle_resources_read(self, params: dict) -> dict:
        """Handle resources/read request."""
        uri = params.get("uri", "")

        if uri == "highnoon://model/info":
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(
                            {
                                "framework": "HighNoon",
                                "version": self.server_version,
                                "qsg_enabled": True,
                            }
                        ),
                    }
                ]
            }
        elif uri == "highnoon://training/status":
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(
                            {
                                "status": "idle",
                                "last_run": None,
                            }
                        ),
                    }
                ]
            }

        raise ValueError(f"Unknown resource URI: {uri}")


class StdioTransport:
    """Stdio transport for MCP server."""

    def __init__(self, server: MCPServer) -> None:
        """Initialize stdio transport.

        Args:
            server: MCP server instance.
        """
        self.server = server
        self._running = False

    async def run(self) -> None:
        """Run the stdio transport loop."""
        self._running = True
        log.info("Starting stdio transport")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(
            writer_transport, writer_protocol, reader, asyncio.get_event_loop()
        )

        while self._running:
            try:
                line = await reader.readline()
                if not line:
                    break

                data = json.loads(line.decode())
                request = MCPRequest(
                    id=data.get("id", 0),
                    method=data.get("method", ""),
                    params=data.get("params", {}),
                )

                response = await self.server.handle_request(request)
                writer.write((json.dumps(response.to_dict()) + "\n").encode())
                await writer.drain()

            except json.JSONDecodeError as e:
                log.error(f"Invalid JSON: {e}")
            except Exception as e:
                log.error(f"Transport error: {e}")

    def stop(self) -> None:
        """Stop the transport."""
        self._running = False


async def run_mcp_server(
    transport: str = "stdio",
    manifest: ToolManifest | None = None,
) -> None:
    """Run the MCP server.

    Args:
        transport: Transport type ("stdio" or "sse").
        manifest: Tool manifest.
    """
    server = MCPServer(manifest=manifest)

    if transport == "stdio":
        transport_handler = StdioTransport(server)
        await transport_handler.run()
    else:
        raise ValueError(f"Unsupported transport: {transport}")


def main() -> None:
    """Entry point for MCP server CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="HighNoon MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_mcp_server(transport=args.transport))


if __name__ == "__main__":
    main()
