# AI Audit Risk Scanner — LangGraph Multi-Agent System

A production-grade 3-agent audit risk scanner built with LangGraph and Claude API. 
Identifies financial risks from documents in under 2 minutes with human approval gates.

## 🎯 Business Problem
Manual audit risk review takes days and costs tens of thousands of euros at Big 4 firms. 
This system automates the entire process — from document ingestion to formal CFO report.

## 🤖 How It Works

**Agent 1 — Data Agent**
Reads financial documents and extracts all risk indicators with evidence citations.

**Agent 2 — Analyst Agent**
Scores each risk (likelihood × impact), determines severity and confidence level.
If confidence < 70% → routes back to Agent 1 for more data automatically.

**Agent 3 — Controller Agent**
Writes a formal CFO audit report. If severity is HIGH or CRITICAL → pauses for human approval before saving.

## 📊 Results on Test Data
- 19 risks identified from 3 financial documents
- 4 critical risks flagged including going concern risk
- Human rejection loop tested — Agent 2 re-analysed with reviewer feedback
- 82% confidence score with detailed reasoning

## 🛠️ Tools
- Claude Sonnet API (Anthropic)
- LangGraph — state management and agent orchestration
- Python 3.12 + python-dotenv

## 🚀 How to Run
```bash
git clone https://github.com/Ayman-Kassar/audit-risk-scanner
cd audit-risk-scanner
pip install anthropic langgraph python-dotenv
cp .env.example .env  # add your ANTHROPIC_API_KEY
python audit_langgraph.py
```

## 💼 Portfolio Context
Built by a senior finance professional (15+ years FP&A, controlling, audit) 
transitioning to AI Finance. This system demonstrates:
- Multi-agent orchestration with LangGraph
- Human-in-the-loop approval workflows
- Confidence-based routing and safety brakes
- Production-grade error handling

## 📁 Files
- `audit_langgraph.py` — LangGraph version with full persistence and routing
- `audit_risk_scanner.py` — Simple version showing the core logic
- `finance_policy.txt` — Sample finance policy document for testing
