# Freight Disruption Agent (`freight-reroute-agent`)

Enterprise-grade agentic system in LangGraph that monitors freight shipping corridors, detects weather bottlenecks via the National Weather Service (NOAA) API, calculates delays and demurrage penalties, deterministically evaluates alternative intermodal routes via OpenStreetMap/Overpass, and enforces a human-in-the-loop approval gate for expensive rerouting actions.

## Project Structure

```
freight-reroute-agent/
├── pyproject.toml
├── README.md
├── src/
│   └── freight_agent/
│       ├── __init__.py
│       ├── models/          # Immutable Pydantic v2 domain schemas & contracts
│       │   ├── __init__.py
│       │   └── domain.py
│       ├── tools/           # NOAA API, OSM/Overpass, Demurrage calculators
│       │   └── __init__.py
│       ├── graph/           # LangGraph state machine & nodes
│       │   └── __init__.py
│       └── utils/           # Geo / spatial utilities and telemetry
│           └── __init__.py
├── tests/                   # Pytest test suite
│   ├── __init__.py
│   └── test_models.py
└── evals/                   # Benchmark datasets and evaluation harnesses
    └── __init__.py
```

## Quickstart

```bash
# Using uv / pip with Python 3.11+
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Linting
ruff check .
```
