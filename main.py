import os
import json
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
from rich.prompt import Prompt

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# --- ENVIRONMENT & CLI AESTHETICS ---
load_dotenv()
custom_theme = Theme({"info": "dim cyan", "warning": "magenta", "bot": "bold #c15f3c", "user": "bold white", "debug": "dim yellow"})
console = Console(theme=custom_theme)

# ==========================================
# --- MODEL SWITCHER ---
# ==========================================
from langchain_groq import ChatGroq
if not os.getenv("GROQ_API_KEY"):
    console.print("[red]Error: GROQ_API_KEY not found in .env[/red]")
    exit(1)
    
llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b")

# ==========================================
# --- LOAD SOP DATA ---
# ==========================================
try:
    with open("sop.json", "r") as file:
        sop_json = json.load(file)
        SOP_DATA = json.dumps(sop_json, indent=2)
except FileNotFoundError:
    console.print("[red]Error: sop.json file not found in the directory.[/red]")
    exit(1)
except json.JSONDecodeError:
    console.print("[red]Error: sop.json is empty or contains invalid JSON.[/red]")
    exit(1)

# --- ROUTING SCHEMA ---
class RouteDecision(BaseModel):
    thought: str = Field(description="Step-by-step reasoning.")
    route: str = Field(description="MUST be exactly one of: 'faq', 'qualify', or 'escalate'")
    reason: str = Field(description="If route is escalate, MUST be exactly one of: 'out-of-scope question', 'low confidence', 'angry sentiment', or 'explicit escalation request'. Otherwise, empty.")
    
# --- STATE DEFINITION ---
class AgentState(TypedDict):
    chat_history: List[Dict[str, str]]
    user_input: str
    lead_data: Dict[str, str]
    unanswered_count: int
    escalated: bool
    escalation_reason: str
    all_escalations: List[str]  # Fix: Array to hold per-prompt escalation flags
    summary: str
    bot_response: str
    next_node: str

# --- NODES ---
def router_node(state: AgentState):
    """Hybrid Router matching explicit assignment criteria."""
    user_text = state['user_input'].lower()
    services = [s.get("name", "").lower() for s in sop_json.get("services", [])]

    is_price_q = any(tok in user_text for tok in ["price", "cost", "how much", "starting price", "rate"])
    is_greeting = any(tok in user_text for tok in ["hi", "hello", "hey", "good morning", "good evening"])
    is_angry = any(tok in user_text for tok in ["complaint", "angry", "not happy", "refund", "bad", "poor", "hurt", "upset", "frustrated"])
    is_explicit_escalation = any(tok in user_text for tok in ["human", "agent", "representative", "manager", "doctor", "real person"])

    # 1. Deterministic Checks
    if is_price_q:
        for svc in services:
            if svc and (svc in user_text or svc.rstrip('s') in user_text):
                console.print(f"[debug]System Log: Deterministic match -> faq[/debug]")
                return {"escalation_reason": "", "next_node": "faq"}

    if is_angry:
        reason = "angry sentiment"
        console.print(f"[warning]System Log: Flagged -> {reason}[/warning]")
        return {"escalation_reason": reason, "next_node": "escalate"}

    if is_explicit_escalation:
        reason = "explicit escalation request"
        console.print(f"[warning]System Log: Flagged -> {reason}[/warning]")
        return {"escalation_reason": reason, "next_node": "escalate"}

    if is_greeting and not is_price_q:
        console.print("[debug]System Log: Greeting detected -> qualify[/debug]")
        return {"escalation_reason": "", "next_node": "qualify"}

    # 2. LLM Fallback (Semantic Routing)
    prompt = f"""
    You are the triage router for Bloom Aesthetics.
    SOP Data: {SOP_DATA}
    User says: {state['user_input']}
    
    Task: Choose the correct route.
    - "faq": User asks about something explicitly listed in the SOP.
    - "escalate": User asks about something NOT in the SOP, or you cannot confidently answer.
    - "qualify": User is answering a qualification question.
    """
    
    structured_llm = llm.with_structured_output(RouteDecision)
    
    try:
        decision = structured_llm.invoke([HumanMessage(content=prompt)])
        route = decision.route.lower()
        reason = decision.reason
        
        if route == "escalate":
            console.print(f"[warning]System Log: Flagged -> {reason}[/warning]")
            
        if route not in ["faq", "qualify", "escalate"]:
            route = "escalate"
            reason = "low confidence"
            console.print(f"[warning]System Log: Flagged -> {reason}[/warning]")
            
    except Exception as e:
        route = "escalate"
        reason = "low confidence"
        console.print(f"[warning]System Log: Flagged -> {reason}[/warning]")

    if state['unanswered_count'] > 2:
        route = "escalate"
        reason = "out-of-scope question"
        console.print(f"[warning]System Log: Flagged -> {reason}[/warning]")

    return {"escalation_reason": reason, "next_node": route}

def faq_node(state: AgentState):
    user_text = state['user_input'].lower()
    
    for s in sop_json.get("services", []):
        name = s.get("name", "").lower()
        if name and (name in user_text or name.rstrip('s') in user_text):
            resp = f"{s.get('name')} from {s.get('starting_price')}. Hours: {sop_json.get('operating_hours')}. Booking: {sop_json.get('booking_policy')}."
            return {"bot_response": resp, "unanswered_count": 0}
            
    prompt = f"""
    You are a polite assistant for Bloom Aesthetics.
    Answer using ONLY this SOP data: {SOP_DATA}
    User: {state['user_input']}
    If you cannot find the answer in the SOP, say you don't know. Keep it short.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"bot_response": response.content, "unanswered_count": 0}

def qualify_node(state: AgentState):
    lead_data = state['lead_data']
    
    if not lead_data.get("interest"):
        resp = "Are you interested in Botox, Fillers, or a general consultation today?"
    elif not lead_data.get("timeline"):
        resp = "Great! Are you looking to book something this week, or just exploring options?"
    else:
        resp = "Thank you! I have everything I need to qualify your profile. Can I help with anything else?"
        
    return {"bot_response": resp}

def escalate_node(state: AgentState):
    resp = "Sorry I can't assist you with this, I'm going to connect you with a human specialist right away. While we wait, is there anything else I can help you with from our standard services?"
    return {"bot_response": resp, "escalated": True}

def summarizer_node(state: AgentState):
    """Generates the final structured summary using the ACTUAL chat history."""
    
    # Format the chat history into a readable transcript string
    transcript = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in state['chat_history']])
    
    prompt = f"""
    Analyze this customer support chat transcript and generate a structured summary.
    
    TRANSCRIPT:
    {transcript}
    
    Lead Data Collected: {state['lead_data']}
    All Escalation Flags Triggered During Session: {state['all_escalations']}
    
    Output a clean text summary detailing:
    1. Customer Intent
    2. Key details collected
    3. SOP gaps identified
    4. Recommended next action
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": response.content}

# --- EDGE ROUTING ---
def edge_router(state: AgentState):
    return state.get("next_node", "escalate")

# --- BUILD GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("faq", faq_node)
workflow.add_node("qualify", qualify_node)
workflow.add_node("escalate", escalate_node)
workflow.add_node("summarizer", summarizer_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", edge_router, {"faq": "faq", "qualify": "qualify", "escalate": "escalate"})
workflow.add_edge("faq", END)
workflow.add_edge("qualify", END)
workflow.add_edge("escalate", END) 
workflow.add_edge("summarizer", END)

app = workflow.compile()

# --- CLI EXECUTION ---
def run_chat():
    console.print(Panel.fit("[bold white]Closira AI Agent Active (Powered by GPT-OSS-120B)[/bold white]\nType 'quit' to end session and generate summary.", border_style="#c15f3c"))
    
    state = {
        "chat_history": [],
        "user_input": "",
        "lead_data": {},
        "unanswered_count": 0,
        "escalated": False,
        "escalation_reason": "",
        "all_escalations": [], # Array to store all flags triggered
        "summary": "",
        "bot_response": "",
        "next_node": ""
    }

    while True:
        user_text = Prompt.ask("\n[user]User[/user]")
        
        if user_text.lower() in ['quit', 'exit']:
            console.print("\n[info]Session ended by user. Generating summary...[/info]")
            final_summary = summarizer_node(state)
            console.print(Panel(final_summary['summary'], title="Final Session Summary", border_style="#c15f3c"))
            break

        state["user_input"] = user_text
        state["chat_history"].append({"role": "user", "content": user_text})
        
        if any(word in user_text.lower() for word in ["botox", "filler", "consultation"]):
            state["lead_data"]["interest"] = user_text
        elif any(word in user_text.lower() for word in ["week", "exploring", "today"]):
            state["lead_data"]["timeline"] = user_text

        # Run Graph
        result = app.invoke(state)
        state.update(result)
        
        # Log the escalation flag if one occurred on this turn
        if state.get("escalated") and state.get("escalation_reason"):
            if state["escalation_reason"] not in state["all_escalations"]:
                state["all_escalations"].append(state["escalation_reason"])
            state["escalated"] = False # Reset for the next turn

        bot_reply = state.get("bot_response", "")
        state["chat_history"].append({"role": "bot", "content": bot_reply})
        console.print(f"\n[bot]AI:[/bot] {bot_reply}")

if __name__ == "__main__":
    run_chat()