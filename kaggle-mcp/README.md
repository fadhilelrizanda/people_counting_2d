# Kaggle MCP Server Directory

This directory (`kaggle-mcp`) implements a Model Context Protocol (MCP) server that wraps the `kaggle` CLI.

## Purpose
It allows AI agents to perform automated Kaggle operations such as downloading datasets, submitting competition results, and managing Kaggle kernels remotely. It manages user credentials and converts MCP tool calls into Kaggle CLI subprocess executions.
