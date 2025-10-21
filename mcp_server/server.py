from fastmcp import FastMCP
from tools.execution import (
    crossval_stamp,
    deploy_stamp,
    encode_patients_stamp,
    encode_slides_stamp,
    heatmaps_stamp,
    preprocess_stamp,
    statistics_stamp,
)
from tools.utils import (
    check_available_devices,
    list_files,
    manage_workspace_dirs_,
    read_file,
)

# Initialize the FastMCP server
mcp = FastMCP("STAMP MCP Server")

# STAMP executions
mcp.tool(preprocess_stamp)
mcp.tool(crossval_stamp)
mcp.tool(deploy_stamp)
mcp.tool(statistics_stamp)
mcp.tool(heatmaps_stamp)
mcp.tool(encode_slides_stamp)
mcp.tool(encode_patients_stamp)

# Utils
mcp.tool(read_file)
mcp.tool(list_files)
mcp.tool(check_available_devices)
mcp.tool(manage_workspace_dirs_)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
