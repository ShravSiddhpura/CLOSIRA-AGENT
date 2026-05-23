# Closira AI Agent - Bloom Aesthetics

A robust, multi-stage AI customer support workflow built with Python, LangGraph, and Groq (OpenAI GPT-OSS-120B). It deterministically routes inbound customer inquiries, answers FAQs strictly from a provided SOP, qualifies leads, and gracefully escalates when necessary.

---

## Architecture & Engineering Decisions

This system utilizes a Hybrid Routing Architecture via a LangGraph State Machine:

- **Deterministic Pre-Checks:** Explicit price/service queries, complaints, and greetings are caught via O(1) keyword matching to guarantee zero latency and zero hallucination for critical business bounds.

- **Semantic Fallback Router (LLM):** Ambiguous queries are routed using GPT-OSS-120B constrained by LangChain’s `with_structured_output` (Pydantic), enforcing strict schema compliance to prevent arbitrary branching and model hallucination.

- **Isolated Worker Nodes:** To maintain safety, the FAQ node is isolated from the conversational LLM. It attempts a deterministic lookup first, only using the LLM to format the grounded SOP text if necessary.

---

## Trade-offs & Known Limitations

### Model Selection

While smaller MoE models (20B) are fast, they occasionally suffer from attention drift during complex JSON generation. Upgrading to the 120B frontier model and enforcing Pydantic schemas resolved this, ensuring highly reliable routing.

### State Schema Rigidity

LangGraph enforces strict validation. The `next_node` key must be maintained in the `AgentState` `TypedDict`, or the graph will silently drop routing instructions and hit the default escalation failsafe.

---

## Setup & Execution (Windows via uv)

### Clone the repository and setup the environment:

```bash
uv venv
.venv\Scripts\activate
uv pip install langchain-groq langgraph langchain-core python-dotenv rich pydantic
```

### Environment Variables

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_api_key_here
```

### Run the Agent

```bash
python main.py
```
