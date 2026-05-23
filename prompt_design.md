# Prompt Design & Architecture: Closira AI Agent

## 1. System Architecture Overview

The intelligence layer of this agent is built using a Hybrid Routing Architecture managed by a LangGraph state machine. Rather than relying entirely on a single monolithic LLM prompt to handle conversation, logic, and data retrieval simultaneously, the system separates concerns:

- **Deterministic Pre-Router:** Catches explicit keywords (prices, greetings, angry sentiment) in O(1) time.
- **Semantic Fallback Router (LLM):** Classifies complex intents using GPT-OSS-120B constrained by strict Pydantic schemas.
- **Isolated Worker Nodes:** Specific tasks (FAQ answering, Lead Qualification, Summarization) are handled by dedicated prompts injected with only the necessary context.

---

## 2. Full System Prompts

### The Semantic Router Prompt

Used when deterministic pre-checks fall through, forcing the model into a strict schema evaluation.

```text
You are the triage router for Bloom Aesthetics.

SOP Data: {SOP_DATA}

User says: {user_input}

Task: Choose the correct route.

"faq": User asks about something explicitly listed in the SOP.

"escalate": User asks about something NOT in the SOP, or you cannot confidently answer.

"qualify": User is answering a qualification question.

CRITICAL INSTRUCTION FOR ESCALATION REASON:
If routing to "escalate", your reason MUST be exactly "Out-of-scope question" OR "Low confidence".
```

### The FAQ Worker Prompt

Used exclusively when a query is confirmed to be within the SOP bounds.

```text
You are a polite assistant for Bloom Aesthetics.

Answer using ONLY this SOP data: {SOP_DATA}

User: {user_input}

If you cannot find the answer in the SOP, say you don't know. Keep it short.
```

---

## 3. Reasoning for Key Design Choices

### Hybrid Routing over Pure LLM

Smaller open-weight MoE models sometimes suffer from attention drift during complex routing. Adding a deterministic pre-check for explicit prices/services guarantees 100% accuracy and zero-latency routing for core business queries.

### Structured Output (Pydantic)

Forcing the LLM to output a strictly typed JSON object (with keys thought, route, and reason) implements a built-in Chain-of-Thought (CoT). By forcing the model to generate a thought before selecting a route, it actively cross-references the SOP in its context window, drastically reducing false escalations.

---

## 4. Hallucination Prevention

Hallucinations are prevented through three specific structural guardrails:

- **Context Injection:** The LLM is never allowed to rely on its pre-trained weights to answer client questions. The exact sop.json data is injected directly into the prompt at runtime.
- **Prompt Boundaries:** By decoupling the Router from the FAQ generator, the LLM generating the final text is given an extremely narrow task. It does not have the conversational freedom to guess.
- **Explicit Fallback:** The FAQ prompt contains a hard failsafe:

```text
"If you cannot find the answer in the SOP, say you don't know."
```

---

## 5. Confidence-Based Escalation Logic

Escalation is not left to LLM vibes. It is triggered deterministically and semantically across four specific vectors defined by the assignment:

### Out-of-Scope (Semantic)

If the user asks for a service not in the SOP (e.g., "Lasik"), the router's CoT evaluation triggers an escalation with the exact flag `"out-of-scope question"`.

### Angry Sentiment (Deterministic)

Keywords associated with frustration or medical issues (`complaint`, `hurt`, `doctor`, `angry`) immediately short-circuit the router and escalate.

### Explicit Request (Deterministic)

Requests for a `"human"`, `"manager"`, or `"agent"` are instantly routed to the escalation node.

### Low Confidence / Failsafe (Systemic)

If the LangGraph router fails to parse the model's output, or if the user asks >2 unhandled questions, the system triggers a `"low confidence"` or loop-breaker escalation to protect the user experience.

---

## 6. Tone and Persona

### Persona

The AI acts as a digital concierge for an SMB (Bloom Aesthetics Clinic).

### Tone Design

- **Concise & Direct:** SMB customers on WhatsApp or SMS expect fast answers, not paragraphs. Prompts explicitly instruct the model to `"Keep it short."`

- **Empathetic & Professional:** During an escalation, the AI does not just coldly terminate the chat or act like a robot. It apologizes, acknowledges the handoff, but keeps the conversation alive to maintain lead engagement:

```text
"Sorry I can't assist you with this, I'm going to connect you with a human specialist right away. While we wait, is there anything else I can help you with from our standard services?"
```