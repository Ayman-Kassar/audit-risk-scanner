import anthropic
import json
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()

# ── STATE DEFINITION ──────────────────────────────
class AuditState(TypedDict):
    documents: List[str]
    extracted_risks: List[dict]
    risk_scores: List[dict]
    severity: str
    confidence: float
    report: str
    human_approved: bool
    iteration_count: int
    needs_human_review: bool
    warning_flags: List[str]
    rejection_reason: str

# ── SAMPLE FINANCIAL DOCUMENTS ─────────────────────
documents = [
    """
    Accounts Payable — Sweden Q1 2026
    Outstanding supplier invoices: 2.8M SEK
    Average payment period: 67 days (industry standard: 30 days)
    Three suppliers have issued late payment notices
    One key supplier threatening to suspend deliveries
    """,    
    """
    Q1 2026 Financial Report — Sweden
    Revenue: 4.2M SEK vs budget 4.5M SEK (-6.7%)
    Operating costs: 3.1M SEK vs budget 2.9M SEK (+6.9%)
    Cash position: 0.8M SEK (down from 2.1M SEK in Q4 2025)
    Accounts receivable: 45 days (policy limit: 30 days)
    Three largest customers represent 78% of revenue
    One customer (32% of revenue) has requested extended payment terms
    """,
    """
    Internal Audit Notes — Sweden Q1 2026
    Expense reports submitted late for 3 senior managers
    Travel expenses 40% above policy limits in February
    Two invoices approved without second signatory (policy violation)
    IT systems upgrade delayed — legacy system running past end-of-life date
    Staff turnover in finance team: 2 of 5 analysts left in Q1
    """,
    """
    Market Context — Sweden Q1 2026
    Key competitor launched aggressive pricing in Nordic market
    Raw material costs up 12% year on year
    SEK weakened 8% against EUR affecting import costs
    Regulatory change: new financial reporting requirements from July 2026
    Customer satisfaction score dropped from 8.2 to 7.1 out of 10
    """
]

# ── AGENT 1 — DATA AGENT ──────────────────────────
def data_agent(state: AuditState) -> AuditState:
    print("\n" + "="*60)
    print("🔍 AGENT 1 — DATA AGENT: Extracting risks")
    print("="*60)

    all_documents = "\n\n---\n\n".join(state["documents"])

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        temperature=0,
        system="""You are a senior audit risk specialist.
Extract specific risk indicators from financial documents.
Be precise — only extract what is explicitly stated.""",
        messages=[{
            "role": "user",
            "content": f"""
<documents>{all_documents}</documents>
<instructions>
Extract all risk indicators. Return ONLY a JSON array:
[{{"risk_id": "R001", "category": "Financial|Operational|Compliance|Strategic", 
"description": "description", "evidence": "exact quote", "area": "source"}}]
</instructions>"""
        }]
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    extracted_risks = json.loads(raw.strip())
    print(f"✅ Extracted {len(extracted_risks)} risk indicators")

    return {
        **state,
        "extracted_risks": extracted_risks,
        "iteration_count": state["iteration_count"] + 1
    }

# ── AGENT 2 — ANALYST AGENT ───────────────────────
def analyst_agent(state: AuditState) -> AuditState:
    print("\n" + "="*60)
    print("📊 AGENT 2 — ANALYST AGENT: Scoring risks")
    print("="*60)

    # Include rejection reason if re-running after rejection
    rejection_context = ""
    if state.get("rejection_reason"):
        rejection_context = f"""
<rejection_reason>
The previous report was rejected for this reason: {state["rejection_reason"]}
Please take this into account when re-scoring risks.
</rejection_reason>"""

    risks_json = json.dumps(state["extracted_risks"], indent=2)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        temperature=0,
        system="""You are a senior audit risk analyst at a Big 4 firm.
Score and assess risks from financial documents.
Be rigorous and conservative.""",
        messages=[{
            "role": "user",
            "content": f"""
<extracted_risks>{risks_json}</extracted_risks>
{rejection_context}
<instructions>
Return ONLY a JSON object:
{{"scored_risks": [{{"risk_id": "R001", "likelihood": 1-5, "impact": 1-5, 
"risk_score": 0, "priority": "Critical|High|Medium|Low", 
"recommendation": "action"}}],
"overall_severity": "LOW|MEDIUM|HIGH|CRITICAL",
"confidence": 0.0-1.0,
"confidence_reasoning": "reasoning",
"critical_risks": ["R001"],
"immediate_actions": ["action"]}}
</instructions>"""
        }]
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    analysis = json.loads(raw.strip())

    print(f"✅ Severity: {analysis['overall_severity']}")
    print(f"✅ Confidence: {analysis['confidence']:.0%}")

    warning_flags = list(state.get("warning_flags", []))
    if analysis["confidence"] < 0.7 and state["iteration_count"] < 3:
        warning_flags.append(f"Low confidence ({analysis['confidence']:.0%})")

    return {
        **state,
        "risk_scores": analysis["scored_risks"],
        "severity": analysis["overall_severity"],
        "confidence": analysis["confidence"],
        "warning_flags": warning_flags
    }

# ── AGENT 3 — CONTROLLER AGENT ────────────────────
def controller_agent(state: AuditState) -> AuditState:
    print("\n" + "="*60)
    print("📋 AGENT 3 — CONTROLLER AGENT: Writing report")
    print("="*60)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        temperature=0,
        system="""You are a senior audit controller writing formal audit risk reports.
Reports are read by CFOs and audit committees.
Be direct, professional, and specific.""",
        messages=[{
            "role": "user",
            "content": f"""
<extracted_risks>{json.dumps(state["extracted_risks"], indent=2)}</extracted_risks>
<scored_risks>{json.dumps(state["risk_scores"], indent=2)}</scored_risks>
<overall_severity>{state["severity"]}</overall_severity>
<confidence>{state["confidence"]}</confidence>
<instructions>
Write a formal audit risk report with:
1. EXECUTIVE SUMMARY (3 sentences)
2. OVERALL RISK RATING with justification
3. CRITICAL FINDINGS (Critical priority only)
4. HIGH PRIORITY FINDINGS (top 5)
5. IMMEDIATE ACTIONS REQUIRED
6. RECOMMENDED TIMELINE
Use actual numbers. Professional tone. No fluff.
</instructions>"""
        }]
    )

    report = message.content[0].text
    needs_human_review = state["severity"] in ["HIGH", "CRITICAL"]

    print(f"\n{'⚠️ Routing to human approval' if needs_human_review else '✅ Auto-approving'}")
    print("\n" + "="*60)
    print("📄 DRAFT REPORT PREVIEW (first 500 chars):")
    print("="*60)
    print(report[:5000] + "...")

    return {
        **state,
        "report": report,
        "needs_human_review": needs_human_review,
        "rejection_reason": ""
    }

# ── HUMAN APPROVAL NODE ────────────────────────────
def human_approval_node(state: AuditState) -> AuditState:
    print("\n" + "="*60)
    print("👤 HUMAN APPROVAL REQUIRED")
    print("="*60)
    print(f"Severity: {state['severity']}")
    print("\n1 — Approve and save")
    print("2 — Reject — send back to Agent 2")
    print("3 — Approve with modification")

    choice = input("\nYour decision (1/2/3): ").strip()

    if choice == "1":
        print("✅ Approved")
        return {**state, "human_approved": True, "rejection_reason": ""}

    elif choice == "2":
        reason = input("Reason for rejection: ").strip()
        print(f"❌ Rejected — reason: {reason}")
        return {**state, "human_approved": False, "rejection_reason": reason}

    elif choice == "3":
        note = input("Your modification note: ").strip()
        updated_report = state["report"] + f"\n\n[REVIEWER NOTE]: {note}"
        print("✅ Approved with modification")
        return {**state, "human_approved": True, "report": updated_report, "rejection_reason": ""}

    return {**state, "human_approved": True}

# ── SAVE REPORT NODE ───────────────────────────────
def save_report_node(state: AuditState) -> AuditState:
    print("\n" + "="*60)
    print("💾 SAVING FINAL REPORT")
    print("="*60)

    with open("audit_risk_report.txt", "w", encoding="utf-8") as f:
        f.write("AUDIT RISK REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Severity: {state['severity']}\n")
        f.write(f"Confidence: {state['confidence']:.0%}\n")
        f.write(f"Risks: {len(state['extracted_risks'])}\n")
        f.write(f"Approved: {state['human_approved']}\n")
        f.write(f"Iterations: {state['iteration_count']}\n")
        f.write("="*60 + "\n\n")
        f.write(state["report"])

    print(f"✅ Report saved to audit_risk_report.txt")
    print(f"📊 Total risks: {len(state['extracted_risks'])}")
    print(f"📊 Severity: {state['severity']}")
    print(f"📊 Iterations: {state['iteration_count']}")
    return state

# ── CONDITIONAL EDGE FUNCTIONS ─────────────────────
def route_after_analyst(state: AuditState) -> Literal["controller", "data_agent"]:
    if state["confidence"] < 0.7 and state["iteration_count"] < 3:
        print(f"\n🔄 Low confidence — routing back to Agent 1")
        return "data_agent"
    print(f"\n✅ Confidence sufficient — proceeding to Agent 3")
    return "controller"

def route_after_human(state: AuditState) -> Literal["save_report", "analyst_agent"]:
    if state["human_approved"]:
        return "save_report"
    print(f"\n🔄 Rejected — routing back to Agent 2 with reason: {state['rejection_reason']}")
    return "analyst_agent"

def route_after_controller(state: AuditState) -> Literal["human_approval", "save_report"]:
    if state["needs_human_review"]:
        return "human_approval"
    return "save_report"

# ── BUILD THE GRAPH ────────────────────────────────
def build_graph():
    graph = StateGraph(AuditState)

    # Add nodes
    graph.add_node("data_agent", data_agent)
    graph.add_node("analyst_agent", analyst_agent)
    graph.add_node("controller", controller_agent)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("save_report", save_report_node)

    # Set entry point
    graph.set_entry_point("data_agent")

    # Add edges
    graph.add_edge("data_agent", "analyst_agent")

    # Conditional edges
    graph.add_conditional_edges(
        "analyst_agent",
        route_after_analyst,
        {"controller": "controller", "data_agent": "data_agent"}
    )

    graph.add_conditional_edges(
        "controller",
        route_after_controller,
        {"human_approval": "human_approval", "save_report": "save_report"}
    )

    graph.add_conditional_edges(
        "human_approval",
        route_after_human,
        {"save_report": "save_report", "analyst_agent": "analyst_agent"}
    )

    graph.add_edge("save_report", END)

    # Add memory for persistence
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# ── RUN THE AGENT ──────────────────────────────────
print("="*60)
print("🚀 AUDIT RISK SCANNER — LangGraph Version")
print("="*60)

app = build_graph()

# Thread ID — unique per audit run
# Change this to run a different audit
thread_id = "sweden_q1_2026_audit_v3"

initial_state: AuditState = {
    "documents": documents,
    "extracted_risks": [],
    "risk_scores": [],
    "severity": "",
    "confidence": 0.0,
    "report": "",
    "human_approved": False,
    "iteration_count": 0,
    "needs_human_review": False,
    "warning_flags": [],
    "rejection_reason": ""
}

config = {"configurable": {"thread_id": thread_id}}

# Run the graph
final_state = app.invoke(initial_state, config=config)

print("\n" + "="*60)
print("✅ PIPELINE COMPLETE")
print("="*60)
print(f"Thread ID: {thread_id}")
print(f"Final severity: {final_state['severity']}")
print(f"Human approved: {final_state['human_approved']}")
print(f"Iterations: {final_state['iteration_count']}")