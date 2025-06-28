# Copilot Configuration for Agentic Trader

This directory contains configuration files for GitHub Copilot integration with the Agentic Trader project.

## Files

- `strategy_context.json`: Main configuration for strategy context API
- `templates/`: Strategy templates with embedded context

## Usage

The configuration enables GitHub Copilot to access historical strategy performance data and provide intelligent code suggestions based on:

- Historical parameter optimization results
- Strategy performance patterns
- Common failure modes
- Successful code patterns

## API Endpoints

When the Copilot API server is running (on localhost:8000), the following endpoints are available:

- `GET /api/strategy-context/{strategy_type}`: Get historical context for a strategy
- `GET /api/parameters/{strategy_type}`: Get optimal parameters  
- `GET /api/code-patterns/{strategy_type}`: Get successful code patterns
- `POST /api/add-result`: Add new strategy results to the database

## Starting the API Server

```bash
cd copilot_integration
python api_endpoints.py
```

Or using uvicorn directly:

```bash
uvicorn copilot_integration.api_endpoints:app --reload --host 0.0.0.0 --port 8000
```