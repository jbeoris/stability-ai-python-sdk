# Stability AI Python SDK

A Python library to easily access the Stability AI API.

## Installation

### PIP
```bash
pip install stability-ai
```

## Authentication
This environment variable must be set for authentication to take place.
```bash
export STABILITY_AI_API_KEY=<your key>
```

## Table of Contents

### Setup
- [Initialization](#initialization)

### Engines (v1)
- [List](#list)

### User (v1)
- [Account](#account)
- [Balance](#balance)

## Setup

### Initialization
To begin using the Stability AI SDK, just import at the top of your python file.
```python
import stability_ai
```

## Engines (v1)

### List
Fetches a list of all available engines.

```python
results = stability_ai.v1.engines.list()

if not results.engines or len(results.engines) == 0:
  raise Exception("No engines found")
engine = results.engines[0]

print(f"Engine found: {engine}")
```

## Engines (v1)

### Account
Fetches user account.

```python
result = stability_ai.v1.user.account()

print(f"Account found with id: {result.id}")
```

### List
Fetches a list of all available engines.

```python
result = stability_ai.v1.user.balance()

print(f"Balance: {result.credits}")
```