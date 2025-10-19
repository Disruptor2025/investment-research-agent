# Multi-Agent Financial Analysis System"

Name: Idrees Khan
Course : AAI 520
Date:   10/19/2025

## What It Does

An AI agent that researches stocks automatically. Gets data from Yahoo Finance, analyzes news, and makes investment recommendations. Uses multiple specialist agents that work together and learns from past analyses.

## Project Requirements

We had to build an agent with:
- Planning capability
- Dynamic tool usage (APIs, data sources)
- Self-reflection on quality
- Learning across runs

And three workflow patterns:
1. Prompt Chaining - 5-step news processing
2. Routing - directs tasks to specialists
3. Evaluator-Optimizer - generate, evaluate, refine

## Setup

Install packages:
```bash
pip install yfinance requests pandas
```

Run it:
```bash
python investment_agent.py
```

Analyzes AAPL and MSFT, saves results to research_reports.json.

## Code Structure

- **AgentMemory** - stores learnings between runs
- **FinancialTools** - gets Yahoo Finance data
- **NewsChainProcessor** - 5-step chaining workflow
- **TaskRouter** - routes to specialist agents
- **AnalysisEvaluator** - evaluates and improves analysis
- **InvestmentAgent** - main coordinator

## Data Sources

Using Yahoo Finance API (yfinance) for stock data. News analysis demonstrates the workflow patterns required for the assignment.


## 📄 File 2: requirements.txt
```
yfinance
requests
pandas
