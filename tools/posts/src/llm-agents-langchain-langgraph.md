---
title: LLM agents: the loop, the graph, and LangChain and LangGraph
summary: An agent is a language model in a loop that may call tools. This post builds the loop, a safe tool dispatcher and a small state graph with checkpoints and human approval in plain Python, shows how the same pieces look in LangChain and LangGraph, and ends with how to evaluate an agent's trajectory.
group: LLMs and generative AI
order: 10
tags: LLM · agents · tool calling · LangGraph · evaluation
---
A chat model on its own can only produce text. Give it a list of functions it may ask for, run the ones it asks for, feed the results back, and repeat until it answers: that loop is what people call an agent. Libraries such as LangChain and LangGraph package the loop, but the loop itself is short, and most of the engineering is in what surrounds it: validating tool calls, bounding the number of steps, saving state, and pausing for a person before anything irreversible.

This post builds each of those pieces in plain Python with a scripted stand-in for the model, so every claim can be checked with an assert. Then it shows the same structure in the two libraries, which are not installed here, so those snippets are marked as not executed.

## Chains and agents: who picks the next step
A **chain** is a fixed sequence: retrieve documents, fill a prompt, call the model, parse the output. The programmer wrote the control flow and it is the same on every input. Retrieval-augmented generation is usually a chain; see [RAG architecture](../rag-architecture/).

An **agent** hands part of the control flow to the model. At each step the model either answers or requests one or more tool calls, and what happens next depends on what it chose. That buys flexibility on open-ended tasks and costs predictability, latency and money: every step is another model call. The pattern of interleaving reasoning with actions and observations was named ReAct by Yao et al. [1]; Toolformer [2] showed a model can learn to insert API calls into its own text, and the tool-calling interfaces of today's chat models are the structured form of the same idea.

## Messages and a stand-in model
A conversation is a list of messages. A tool-calling model returns an assistant message that holds either text or a list of tool calls, each a name plus JSON arguments. Each tool result goes back as a `tool` message tied to the call's id.

The stand-in below plays the model's part for a small order-support task. It is a plain function of the conversation, so runs are deterministic; a real model makes the same kind of decision from the same inputs, just less predictably.

```python
import ast
import inspect
import json
import operator
import re

ORDERS = {"A17": {"order_id": "A17", "subtotal": 120.0, "status": "delivered"}}
REFUNDS = []


def call(name, **args):
    return {"name": name, "args": args}


def toy_model(messages):
    """Stand-in for a tool-calling chat model: reads the conversation, returns an assistant message."""
    toy_model.calls += 1
    question, last = messages[0]["content"], messages[-1]
    if last["role"] == "user":
        calls = [call("lookup_order", order_id=i) for i in re.findall(r"\b[A-Z]\d+\b", question)]
    elif last["content"].startswith("error"):
        return {"role": "assistant", "content": f"Sorry, I could not complete that: {last['content']}"}
    elif last["name"] == "lookup_order":
        order = json.loads(last["content"])
        if "refund" in question.lower():
            calls = [call("issue_refund", order_id=order["order_id"], amount=order["subtotal"])]
        else:
            calls = [call("calculate", expression=f"{order['subtotal']} * 1.08")]
    elif last["name"] == "calculate":
        return {"role": "assistant", "content": f"The total with 8% tax is {float(last['content']):.2f}."}
    else:
        done = json.loads(last["content"])
        return {"role": "assistant", "content": f"Refunded {done['amount']:.2f} for order {done['order_id']}."}
    for k, c in enumerate(calls):
        c["id"] = f"call_{len(messages)}_{k}"
    return {"role": "assistant", "content": "", "tool_calls": calls}


toy_model.calls = 0
```

## Tools: schemas, validation and safe execution
A tool is a function plus a description the model reads. The description is generated from the signature and docstring, so the two cannot drift apart. Every call is checked against that signature before anything runs, because the model's arguments are untrusted input: it can misspell a name, invent an argument, or pass a string where a number belongs.

The obvious calculator tool, `eval(expression)`, is a remote code execution hole: whoever controls the text the model reads (a user, a retrieved web page) controls the expression. The version below parses the expression and walks the syntax tree, allowing numbers and arithmetic only. A cap on exponents alone is not enough: in `(((9 ** 64) ** 64) ** 64) ** 64` every exponent is 64, yet the result is \(9^{64^4} = 9^{16777216}\), an integer of about 53 million bits that takes well over a minute to compute. So the evaluator also rejects any intermediate result larger than \(10^{100}\), which bounds every operand and therefore the cost of every step.

```python
TOOLS = {}


def tool(fn):
    params = {n: p.annotation for n, p in inspect.signature(fn).parameters.items()}
    TOOLS[fn.__name__] = {"fn": fn, "params": params,
                          "schema": {"name": fn.__name__, "description": inspect.getdoc(fn),
                                     "parameters": {n: t.__name__ for n, t in params.items()}}}
    return fn


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def arithmetic(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](arithmetic(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = arithmetic(node.left), arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 64:
            raise ValueError("exponent too large")
        result = OPS[type(node.op)](left, right)
        if abs(result) > 1e100:
            raise ValueError("result too large")
        return result
    raise ValueError(f"not allowed: {type(node).__name__}")


@tool
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression with + - * / ** and parentheses."""
    return arithmetic(ast.parse(expression, mode="eval").body)


@tool
def lookup_order(order_id: str) -> dict:
    """Return the id, subtotal and status of an order."""
    return ORDERS[order_id]


@tool
def issue_refund(order_id: str, amount: float) -> dict:
    """Refund an amount to the customer for an order. Irreversible."""
    REFUNDS.append((order_id, amount))
    return {"order_id": order_id, "amount": amount}


def dispatch(tool_call):
    """Run one tool call; every failure becomes a message the model can read and react to."""
    spec = TOOLS.get(tool_call["name"])
    if spec is None:
        return f"error: unknown tool {tool_call['name']}"
    args = tool_call["args"]
    if set(args) != set(spec["params"]):
        return f"error: {tool_call['name']} takes {sorted(spec['params'])}"
    args = dict(args)
    for name, typ in spec["params"].items():
        if typ is float and type(args[name]) is int:         # JSON has one number type: 120 is a valid float
            args[name] = float(args[name])
        if not isinstance(args[name], typ) or isinstance(args[name], bool):
            return f"error: {name} must be {typ.__name__}"
    try:
        return json.dumps(spec["fn"](**args))
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


assert TOOLS["issue_refund"]["schema"]["parameters"] == {"order_id": "str", "amount": "float"}
assert dispatch(call("calculate", expression="(120 + 30) * 1.08")) == "162.0"
assert dispatch(call("calculate", expression="__import__('os').getcwd()")).startswith("error: ValueError")
assert dispatch(call("calculate", expression="9 ** 9 ** 9")) == "error: ValueError: exponent too large"
assert dispatch(call("calculate", expression="(((9 ** 64) ** 64) ** 64) ** 64")) == "error: ValueError: result too large"
assert dispatch(call("calculate", expression="1 / 0")) == "error: ZeroDivisionError: division by zero"
assert dispatch(call("lookup_order", id="A17")) == "error: lookup_order takes ['order_id']"
assert dispatch(call("issue_refund", order_id="A17", amount="120")) == "error: amount must be float"
assert dispatch(call("issue_refund", order_id="A17", amount=True)) == "error: amount must be float"
assert dispatch(call("issue_refund", order_id="A17", amount=5)) == '{"order_id": "A17", "amount": 5.0}'
REFUNDS.clear()                                         # that test refund was real; undo it
assert dispatch(call("lookup_order", order_id="Z99")).startswith("error: KeyError") and not REFUNDS
```

JSON has a single number type, so a model that sends `120` for a `float` parameter is not wrong; the dispatcher converts it, while a string or a boolean is still rejected. Returning errors as text instead of raising is deliberate. The model sees "amount must be float" and can retry with a corrected call, which is how a person would recover too.

## The agent loop
The loop appends the model's reply, stops if it contains no tool calls, and otherwise runs each call and appends the results. The step limit is not optional: a model that keeps requesting tools would otherwise run forever, billing for every step.

```python
def run_agent(model, question, max_steps=6):
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        reply = model(messages)
        messages.append(reply)
        if not reply.get("tool_calls"):
            return messages
        for c in reply["tool_calls"]:
            messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": dispatch(c)})
    raise RuntimeError(f"no answer after {max_steps} model calls")


toy_model.calls = 0
loop_messages = run_agent(toy_model, "What does order A17 cost with tax?")
assert loop_messages[-1]["content"] == "The total with 8% tax is 129.60."
assert toy_model.calls == 3                       # two tool rounds, then one more call to write the answer

assert run_agent(toy_model, "What does order Z99 cost?")[-1]["content"].startswith("Sorry")

stubborn = lambda messages: {"role": "assistant", "content": "", "tool_calls": [dict(call("calculate", expression="1 + 1"), id="x")]}
try:
    run_agent(stubborn, "Add one and one.")
    raise AssertionError("expected the step limit to fire")
except RuntimeError as e:
    assert "6 model calls" in str(e)
```

The count matters for latency budgets. An agent that needs \(n\) sequential rounds of tools makes \(n + 1\) model calls, because the last call writes the answer. Several calls requested in one round, which most chat APIs allow, can run concurrently and share one model call.

## The same loop as a state graph
LangGraph's central idea is to write that loop as a graph. Nodes are functions that read a shared state and return an update; edges say which node runs next, either fixed or chosen by a routing function; a **reducer** per state key says how an update is merged (for messages: append, not overwrite). The loop above becomes two nodes and one conditional edge. A minimal engine with the same semantics fits in a few lines:

```python
START, END = "__start__", "__end__"


class StateGraph:
    def __init__(self, reducers):
        self.reducers, self.nodes, self.edges = reducers, {}, {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, src, dst):
        self.edges[src] = lambda state: dst

    def add_conditional_edges(self, src, router):
        self.edges[src] = router

    def merge(self, state, update):
        out = dict(state)
        for key, value in update.items():
            out[key] = self.reducers[key](state.get(key), value) if key in self.reducers else value
        return out

    def invoke(self, state, recursion_limit=25):
        node = self.edges[START](state)
        for _ in range(recursion_limit):
            if node == END:
                return state
            state = self.merge(state, self.nodes[node](state))
            node = self.edges[node](state)
        raise RuntimeError("recursion limit reached")


def agent_node(state):
    return {"messages": [toy_model(state["messages"])]}


def tool_node(state):
    return {"messages": [{"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": dispatch(c)}
                         for c in state["messages"][-1]["tool_calls"]]}


def route(state):
    return "tools" if state["messages"][-1].get("tool_calls") else END


g = StateGraph({"messages": lambda old, new: (old or []) + new})
g.add_node("agent", agent_node)
g.add_node("tools", tool_node)
g.add_edge(START, "agent")
g.add_conditional_edges("agent", route)
g.add_edge("tools", "agent")

graph_state = g.invoke({"messages": [{"role": "user", "content": "What does order A17 cost with tax?"}]})
assert graph_state["messages"] == loop_messages         # same trajectory, message for message
```

Why bother, if the result is identical? Because the graph makes the control flow data. Once the steps are named nodes, the engine can save state between them, pause before one, resume later, and route differently per node, without touching the nodes themselves.

## Checkpoints and a person in the loop
Refunds should not be issued on a model's say-so. The graph below sends refund calls to a separate `refund` node, and the runner pauses before that node. After every step the engine stores the next node and the state under a thread id. The stored checkpoint is plain JSON, so it survives a process restart; here a round trip through `json` stands in for a database.

```python
def route_sensitive(state):
    calls = state["messages"][-1].get("tool_calls")
    if not calls:
        return END
    return "refund" if any(c["name"] == "issue_refund" for c in calls) else "tools"


g.add_node("refund", tool_node)
g.add_conditional_edges("agent", route_sensitive)
g.add_edge("refund", "agent")


def run_thread(graph, store, thread, state=None, pause_before=("refund",)):
    resuming = state is None                                    # resume = the reviewer approved
    node, state = json.loads(store[thread]) if resuming else (graph.edges[START](state), state)
    while node != END:
        if node in pause_before and not resuming:
            store[thread] = json.dumps([node, state])
            return "paused", state
        resuming = False                                        # approval covers one step only
        state = graph.merge(state, graph.nodes[node](state))
        node = graph.edges[node](state)
        store[thread] = json.dumps([node, state])
    return "done", state


store = {}
status, state = run_thread(g, store, "ticket-42", {"messages": [{"role": "user", "content": "Please refund order A17."}]})
assert status == "paused" and REFUNDS == []                             # nothing irreversible has happened
assert state["messages"][-1]["tool_calls"][0]["args"] == {"order_id": "A17", "amount": 120.0}

store = json.loads(json.dumps(store))                                   # "restart the process"
status, state = run_thread(g, store, "ticket-42")
assert status == "done" and REFUNDS == [("A17", 120.0)]
assert state["messages"][-1]["content"] == "Refunded 120.00 for order A17."
```

The reviewer sees the exact pending call, arguments included, before approving. A rejection would be the same resume with an edited state: replace the pending call with a tool message saying the refund was declined, and let the model explain that to the customer.

## Constraining the path
The more of the routing written as code, the less the model can get wrong. For a task with a known procedure (verify the customer, look up the order, check the refund policy, then act), the steps can be graph nodes with deterministic edges, and the model only fills in the parts that need language: extracting the order id from free text, writing the reply. This is a workflow rather than an agent, and for well-understood tasks it is usually the better choice. The model-driven loop earns its cost on open-ended tasks where the sequence of steps cannot be written down in advance. A graph lets the two mix: fixed edges where the procedure is known, a model-chosen edge where it is not.

## The same structure in LangChain and LangGraph
LangChain composes chains with the `|` operator, its expression language (LCEL), and wraps functions as tools. The snippet is not executed here; `llm` stands for any chat-model object from a provider package.

```plaintext
# Not executed: needs langchain-core and a chat-model provider package.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from this context:\n{context}"),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()           # a chain: fixed control flow
answer = chain.invoke({"context": docs_text, "question": "What is the return window?"})

@tool
def lookup_order(order_id: str) -> dict:
    """Return the id, subtotal and status of an order."""
    ...

llm_with_tools = llm.bind_tools([lookup_order, calculate, issue_refund])
reply = llm_with_tools.invoke(messages)            # reply.tool_calls: list of {"name", "args", "id"}
```

LangGraph builds the graph from the previous sections. `MessagesState` declares a `messages` key whose reducer, `add_messages`, appends new messages (a message whose id matches an existing one replaces it instead), `ToolNode` runs the requested tools, and `tools_condition` routes to the node named `tools` or to the end.

```plaintext
# Not executed: needs langgraph.
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver   # called MemorySaver in older releases

def agent(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([lookup_order, calculate, issue_refund]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["tools"])  # pauses before every tool round

config = {"configurable": {"thread_id": "ticket-42"}}
graph.invoke({"messages": [("user", "Please refund order A17.")]}, config)   # stops before tools
graph.invoke(None, config)                                                   # resume after review
```

A few corrections to older tutorials are worth knowing. Classes such as `RetrievalQA` and `ConversationBufferMemory` belong to LangChain's earlier chain API and have been deprecated in favour of LCEL and LangGraph's checkpointed state. The documented way to mark the first node is `add_edge(START, ...)`; `set_entry_point` is an older shorthand for the same edge. Besides static `interrupt_before`, a node can pause itself with `interrupt(...)` and be resumed with a `Command`, which suits approvals that depend on the arguments. Static `interrupt_before=["tools"]` pauses before every tool round; to pause only for refunds, as in the version above, put the refund tool in its own node and interrupt before that one. The names move between releases, so check the current documentation; the structure does not.

## Evaluating an agent
A correct final answer can hide a bad path: an agent that guessed the total without looking up the order, or issued a refund it was not asked for. So agents are scored on two things. The **final answer** is checked as for any model output, against a reference or with a rubric. The **trajectory**, the sequence of tool calls, is compared with a reference path, either exactly or more leniently as an in-order match that tolerates harmless extra calls.

```python
def trajectory(messages):
    return [c["name"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls", [])]


def in_order(pred, ref):
    it = iter(pred)
    return all(any(p == r for p in it) for r in ref)          # ref is a subsequence of pred


cases = [("What does order A17 cost with tax?", ["lookup_order", "calculate"], "129.60"),
         ("What does order Z99 cost with tax?", ["lookup_order"], "could not")]
for question, ref, expected in cases:
    msgs = run_agent(toy_model, question)
    assert trajectory(msgs) == ref and expected in msgs[-1]["content"]

wasteful = ["lookup_order", "lookup_order", "calculate"]       # a repeated lookup: harmless but slow
assert in_order(wasteful, cases[0][1]) and wasteful != cases[0][1]
assert not in_order(["calculate"], cases[0][1])               # skipped the lookup: fails either way

from itertools import combinations, product                   # brute force: try every subset of positions
names = ["lookup_order", "calculate", "issue_refund"]
for n_pred, n_ref in product(range(5), range(4)):
    for pred, ref in product(product(names, repeat=n_pred), product(names, repeat=n_ref)):
        brute = any(list(sub) == list(ref) for sub in combinations(pred, len(ref)))
        assert in_order(pred, ref) == brute
```

Track cost alongside accuracy: model calls, tokens and wall-clock time per task. Comparing a trajectory with a reference is cheap and catches routing regressions on every change; the answers need a reference set or a judge.

## Pitfalls
- **No step or cost limit.** Loops happen, especially when a tool keeps failing. Cap model calls per task and alert on runs that hit the cap.
- **Trusting tool arguments.** Validate types and allowed values, never `eval` model output, and give tools the narrowest permissions that work.
- **Prompt injection through tool results.** A retrieved page can contain instructions. Treat tool output as data, and gate irreversible tools behind approval no matter what the conversation says.
- **Too many tools.** Each tool's description costs context on every call, and similar tools get confused. Prefer a few well-described tools, or route to a sub-graph that holds the relevant ones.
- **An agent where a workflow would do.** If the steps are known, write them as edges. Fewer model decisions means fewer ways to fail and lower latency.
- **Unbounded message history.** Long runs fill the context window. Trim or summarise old turns, keeping the tool results the answer depends on.

For the model underneath, see [Transformers and LLMs: the key engineering ideas](../transformers-and-llms-key-ideas/) and [LLM inference](../llm-inference/), which explains why each extra model call costs what it does. Lilian Weng's overview [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) and the LangGraph documentation were starting points for this post; the explanations and code here are my own. LangChain was created by Harrison Chase and is developed by LangChain, Inc., which also builds LangGraph.

## References
1. Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), ICLR 2023.
2. Schick et al., [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761), NeurIPS 2023.
3. Weng, [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/), 2023.
4. [LangGraph documentation](https://langchain-ai.github.io/langgraph/), LangChain, Inc.
