"""MCP integration: load tools from an MCP server as FunctionTools."""

from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents_bundle.tools.base_tool import BaseTool, FunctionTool
from agents_bundle.tools.tool_schema import format_tool_definition

# The MCP server's stderr goes here. stdio_client needs a real file: in Jupyter
# on Windows the default sys.stderr has no fileno() and the server fails to start.
MCP_ERRLOG = "mcp_server.log"


def _extract_text_content(result) -> str:
    """Extract plain text from an MCP CallToolResult."""
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


def _create_mcp_tool(mcp_tool, connection: dict) -> FunctionTool:
    """Create a FunctionTool that wraps an MCP tool."""

    async def call_mcp(**kwargs):
        with open(MCP_ERRLOG, "a") as errlog:
            async with stdio_client(StdioServerParameters(**connection), errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(mcp_tool.name, kwargs)
                    return _extract_text_content(result)

    tool_definition = format_tool_definition(
        mcp_tool.name, mcp_tool.description, mcp_tool.inputSchema
    )

    return FunctionTool(
        func=call_mcp,
        name=mcp_tool.name,
        description=mcp_tool.description,
        tool_definition=tool_definition,
    )


async def load_mcp_tools(connection: dict) -> list[BaseTool]:
    """Load tools from an MCP server and convert them to FunctionTools.

    Each MCP tool becomes a FunctionTool that re-establishes the connection
    on each invocation.
    """
    tools: list[BaseTool] = []

    with open(MCP_ERRLOG, "a") as errlog:
        async with stdio_client(StdioServerParameters(**connection), errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = await session.list_tools()

                for mcp_tool in mcp_tools.tools:
                    tools.append(_create_mcp_tool(mcp_tool, connection))

    return tools


def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI tool format."""
    return [
        format_tool_definition(
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
        )
        for tool in mcp_tools.tools
    ]


@asynccontextmanager
async def mcp_connection(connection: dict):
    """Context manager for maintaining an MCP server connection.

    Usage:
        async with mcp_connection({"command": "npx", "args": [...]}) as session:
            tools = await session.list_tools()
            result = await session.call_tool("tool_name", arguments={...})
    """
    server_params = StdioServerParameters(
        command=connection["command"],
        args=connection.get("args", []),
        env=connection.get("env"),
    )

    with open(MCP_ERRLOG, "a") as errlog:
        async with stdio_client(server_params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
