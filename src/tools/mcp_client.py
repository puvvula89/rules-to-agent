import asyncio
from typing import Dict, Any, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    """
    Tier 2: The Tool Retriever.
    Connects to the Mock MCP Server via STDIO to fetch schemas and execute calls.
    """
    def __init__(self, server_script_path: str = "mock_mcp_server/server.py"):
        self.server_params = StdioServerParameters(
            command="python",
            args=[server_script_path]
        )
        self.session = None
        self._exit_stack = None

    async def connect(self):
        """Initializes the STDIO connection and MCP Session."""
        if self.session:
             return
             
        from contextlib import AsyncExitStack
        self._exit_stack = AsyncExitStack()
        
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(self.server_params))
        read, write = stdio_transport
        
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    async def disconnect(self):
        if self._exit_stack:
            await self._exit_stack.aclose()
            self.session = None

