import asyncio
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


def _print_tool_result(result: Any) -> None:
    # MCP call_tool results can come back as TextContent blocks and/or structuredContent
    if hasattr(result, "content") and result.content:
        for block in result.content:
            if isinstance(block, types.TextContent):
                print(block.text)
            else:
                print(block)
    if hasattr(result, "structuredContent") and result.structuredContent is not None:
        print("\n[structuredContent]")
        print(result.structuredContent)


async def run():
    # Use the current interpreter (your .venv python) to run the server
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["scripts/mcp_server.py"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1) handshake
            await session.initialize()

            # 2) list tools
            tools = await session.list_tools()
            print("Tools exposed by server:", [t.name for t in tools.tools])

            # 3) call search
            print("\nCalling tool: search(query='Wi-Fi is slow but connected')\n")
            search_res = await session.call_tool(
                "search",
                arguments={"query": "Wi-Fi is slow but connected"},
            )
            _print_tool_result(search_res)

            # Try to pull the first result id from structuredContent (if present)
            first_id = None
            if getattr(search_res, "structuredContent", None):
                sc = search_res.structuredContent
                if isinstance(sc, dict) and isinstance(sc.get("result"), list) and sc["result"]:
                    first_id = sc["result"][0].get("id")

            # If we got an id, call fetch
            if first_id:
                print(f"\nCalling tool: fetch(id='{first_id}')\n")
                fetch_res = await session.call_tool("fetch", arguments={"id": first_id})
                _print_tool_result(fetch_res)
            else:
                print("\nCould not auto-detect a result id from structuredContent. "
                      "If your search tool returns text-only, we can adjust the server to return structured JSON.")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
