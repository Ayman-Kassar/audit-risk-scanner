import anthropic
import json
from typing import TypedDict, List
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

print("✅ Foundation loaded — state and documents ready")
print(f"📄 {len(documents)} documents loaded for analysis")

# ── AGENT 1 — DATA AGENT ──────────────────────────
def data_agent(state: AuditState) -> AuditState:
    print("\n" + "="*60)
    print("🔍 AGENT 1 — DATA AGENT: Extracting risks from documents")
    print("="*60)
    
    # Combine all documents into one context
    all_documents = "\n\n---\n\n".join(state["documents"])
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        temperature=0,
        system="""You are a senior audit risk specialist. 
Your job is to extract specific risk indicators from financial documents.
Be precise and factual — only extract what is explicitly stated in the documents.
Never invent or assume risks that are not clearly indicated.""",
        messages=[
            {
                "role": "user",
                "content": f"""
<documents>
{all_documents}
</documents>

<instructions>
Extract all risk indicators from these documents.
Return ONLY a JSON array with this exact structure — no explanation, no markdown:
[
  {{
    "risk_id": "R001",
    "category": "Financial|Operational|Compliance|Strategic",
    "description": "clear description of the risk",
    "evidence": "exact quote or data point from the document",
    "area": "which document/area this came from"
  }}
]
Extract every risk you can find. Be thorough.
</instructions>
"""
            }
        ]
    )
    
    # Parse the response
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    extracted_risks = json.loads(raw.strip())
    
    # Update state
    state["extracted_risks"] = extracted_risks
    state["iteration_count"] = state["iteration_count"] + 1
    
    print(f"\n✅ Extracted {len(extracted_risks)} risk indicators")
    for risk in extracted_risks:
        print(f"  • [{risk['category']}] {risk['description'][:60]}...")
    
    return state

# ── TEST AGENT 1 ───────────────────────────────────
# Initialise state
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
    "warning_flags": []
}

# Run Agent 1
state_after_agent1 = data_agent(initial_state)

print(f"\n📊 State after Agent 1:")
print(f"  Risks extracted: {len(state_after_agent1['extracted_risks'])}")
print(f"  Iteration count: {state_after_agent1['iteration_count']}")

# ── AGENT 2 — ANALYST AGENT ───────────────────────
def analyst_agent(state: AuditState) -> AuditState:
    print("\n" + "="*60)
    print("📊 AGENT 2 — ANALYST AGENT: Scoring and assessing risks")
    print("="*60)

    risks_json = json.dumps(state["extracted_risks"], indent=2)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        temperature=0,
        system="""You are a senior audit risk analyst at a Big 4 firm.
Your job is to score and assess risks extracted from financial documents.
Be rigorous and conservative — in audit, it is better to flag too much than too little.
Base your scores only on the evidence provided.""",
        messages=[
            {
                "role": "user",
                "content": f"""
<extracted_risks>
{risks_json}
</extracted_risks>

<instructions>
Score each risk and provide an overall assessment.
Return ONLY a JSON object with this exact structure — no explanation, no markdown:
{{
  "scored_risks": [
    {{
      "risk_id": "R001",
      "likelihood": 1-5,
      "impact": 1-5,
      "risk_score": likelihood x impact,
      "priority": "Critical|High|Medium|Low",
      "recommendation": "specific action required"
    }}
  ],
  "overall_severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "confidence": 0.0-1.0,
  "confidence_reasoning": "why you are this confident",
  "critical_risks": ["list of risk_ids that are critical"],
  "immediate_actions": ["list of actions needed immediately"]
}}
</instructions>
"""
            }
        ]
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    analysis = json.loads(raw.strip())

    # Update state
    state["risk_scores"] = analysis["scored_risks"]
    state["severity"] = analysis["overall_severity"]
    state["confidence"] = analysis["confidence"]

    # Safety brake — if low confidence and iterations < 3, flag for more data
    if analysis["confidence"] < 0.7 and state["iteration_count"] < 3:
        state["warning_flags"].append(
            f"Low confidence ({analysis['confidence']}) — may need more data"
        )

    print(f"\n✅ Risk scoring complete")
    print(f"  Overall severity: {analysis['overall_severity']}")
    print(f"  Confidence: {analysis['confidence']:.0%}")
    print(f"  Confidence reasoning: {analysis['confidence_reasoning']}")
    print(f"\n  Risk breakdown:")

    priority_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for risk in analysis["scored_risks"]:
        priority_counts[risk["priority"]] = priority_counts.get(risk["priority"], 0) + 1

    for priority, count in priority_counts.items():
        if count > 0:
            print(f"    {priority}: {count} risks")

    print(f"\n  Critical risks: {analysis['critical_risks']}")
    print(f"\n  Immediate actions required:")
    for action in analysis["immediate_actions"]:
        print(f"    → {action}")

    if state["warning_flags"]:
        print(f"\n  ⚠️ Warning flags: {state['warning_flags']}")

    return state

# ── CONDITIONAL EDGE LOGIC ─────────────────────────
def should_get_more_data(state: AuditState) -> str:
    if state["confidence"] < 0.7 and state["iteration_count"] < 3:
        print(f"\n🔄 Confidence too low ({state['confidence']:.0%}) — routing back to Agent 1")
        return "get_more_data"
    elif state["iteration_count"] >= 3:
        print(f"\n⚠️ Safety brake triggered — max iterations reached, proceeding anyway")
        return "proceed"
    else:
        print(f"\n✅ Confidence sufficient ({state['confidence']:.0%}) — proceeding to Agent 3")
        return "proceed"

# ── TEST AGENT 2 ───────────────────────────────────
state_after_agent2 = analyst_agent(state_after_agent1)
routing_decision = should_get_more_data(state_after_agent2)

print(f"\n📊 State after Agent 2:")
print(f"  Severity: {state_after_agent2['severity']}")
print(f"  Confidence: {state_after_agent2['confidence']:.0%}")
print(f"  Routing decision: {routing_decision}")


# ── AGENT 3 — CONTROLLER AGENT ────────────────────
def controller_agent(state: AuditState) -> AuditState:
    print("\n" + "="*60)
    print("📋 AGENT 3 — CONTROLLER AGENT: Writing audit report")
    print("="*60)

    scored_risks = json.dumps(state["risk_scores"], indent=2)
    extracted_risks = json.dumps(state["extracted_risks"], indent=2)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        temperature=0,
        system="""You are a senior audit controller writing a formal audit risk report.
Your reports are read by CFOs and audit committees.
Be direct, professional, and specific.
Every finding must have evidence and a recommended action.""",
        messages=[
            {
                "role": "user",
                "content": f"""
<extracted_risks>
{extracted_risks}
</extracted_risks>

<scored_risks>
{scored_risks}
</scored_risks>

<overall_severity>{state["severity"]}</overall_severity>
<confidence>{state["confidence"]}</confidence>

<instructions>
Write a formal audit risk report for the CFO and audit committee.
Structure it exactly as follows:

1. EXECUTIVE SUMMARY (3 sentences maximum)
2. OVERALL RISK RATING with justification
3. CRITICAL FINDINGS (only Critical priority risks)
4. HIGH PRIORITY FINDINGS (top 5 High priority risks)
5. IMMEDIATE ACTIONS REQUIRED (numbered list, specific owners)
6. RECOMMENDED TIMELINE for remediation

Be specific. Use the actual numbers from the documents.
Professional tone. Direct language. No fluff.
</instructions>
"""
            }
        ]
    )

    report = message.content[0].text
    state["report"] = report

    # Determine if human review needed
    if state["severity"] in ["HIGH", "CRITICAL"]:
        state["needs_human_review"] = True
        print(f"\n⚠️ Severity is {state['severity']} — routing to human approval")
    else:
        state["needs_human_review"] = False
        print(f"\n✅ Severity is {state['severity']} — auto-approving report")

    print("\n" + "="*60)
    print("📄 DRAFT AUDIT REPORT")
    print("="*60)
    print(report)

    return state

# ── HUMAN IN THE LOOP ──────────────────────────────
def human_approval(state: AuditState) -> AuditState:
    print("\n" + "="*60)
    print("👤 HUMAN APPROVAL REQUIRED")
    print("="*60)
    print(f"Severity: {state['severity']} — This report requires senior review.")
    print("\nOptions:")
    print("  1 — Approve and save report")
    print("  2 — Reject and send back to Agent 2 for re-analysis")
    print("  3 — Approve with modifications")

    choice = input("\nYour decision (1/2/3): ").strip()

    if choice == "1":
        state["human_approved"] = True
        print("✅ Report approved by human reviewer")
    elif choice == "2":
        state["human_approved"] = False
        print("❌ Report rejected — returning to Agent 2")
    elif choice == "3":
        modification = input("Enter your modification note: ")
        state["report"] = state["report"] + f"\n\n[REVIEWER NOTE]: {modification}"
        state["human_approved"] = True
        print("✅ Report approved with modifications")
    else:
        state["human_approved"] = True
        print("✅ Invalid input — defaulting to approved")

    return state

# ── SAVE REPORT ────────────────────────────────────
def save_report(state: AuditState) -> AuditState:
    print("\n" + "="*60)
    print("💾 SAVING FINAL REPORT")
    print("="*60)

    filename = "audit_risk_report.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("AUDIT RISK REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Overall Severity: {state['severity']}\n")
        f.write(f"Confidence: {state['confidence']:.0%}\n")
        f.write(f"Risks Identified: {len(state['extracted_risks'])}\n")
        f.write(f"Human Approved: {state['human_approved']}\n")
        f.write(f"Iterations: {state['iteration_count']}\n")
        f.write("="*60 + "\n\n")
        f.write(state["report"])

    print(f"✅ Report saved to {filename}")
    print(f"📊 Final summary:")
    print(f"  Total risks found: {len(state['extracted_risks'])}")
    print(f"  Overall severity: {state['severity']}")
    print(f"  Confidence: {state['confidence']:.0%}")
    print(f"  Human approved: {state['human_approved']}")
    print(f"  Agent iterations: {state['iteration_count']}")

    return state

# ── RUN THE FULL PIPELINE ──────────────────────────
print("\n" + "="*60)
print("🚀 RUNNING FULL AUDIT RISK SCANNER PIPELINE")
print("="*60)

# Agent 3
state_after_agent3 = controller_agent(state_after_agent2)

# Human in the loop if needed
if state_after_agent3["needs_human_review"]:
    state_after_human = human_approval(state_after_agent3)
else:
    state_after_human = state_after_agent3
    state_after_human["human_approved"] = True

# Save report if approved
if state_after_human["human_approved"]:
    final_state = save_report(state_after_human)
else:
    print("\n❌ Report not approved — pipeline stopped")
    print("Re-run the script to restart from Agent 2")