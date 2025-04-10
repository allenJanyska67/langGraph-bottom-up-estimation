import dotenv

from langchain_openai import ChatOpenAI

from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain_core.messages import trim_messages

from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from typing_extensions import Any, Annotated

# Load api keys from .env file.
dotenv.load_dotenv()

global canvas
canvas = "initial global variable canvas"

# Extra fields in state could be added here.
class State(MessagesState):
    canvas: str = ""

# Set up the finest jokes of all time.
@tool
def store_markdown(tool_call_id: Annotated[str, InjectedToolCallId], markdownToSave: str):
    """Store markdown in the canvas"""
    print(f"Storing markdown: {markdownToSave}")
    global canvas
    canvas = markdownToSave
    return Command(
        update={
            # update the state keys
            "canvas": canvas,
             "messages": [
                ToolMessage(
                    "Successfully looked up user information", tool_call_id=tool_call_id
                )
            ],
        }
    )


tools = [store_markdown]

def make_thinking_agent():
    model = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
    # Limit the size of the context window provided to the LLM.
    trimmer = trim_messages(
        max_tokens=100000,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human")
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
You are a collaborative technical estimation assistant helping a user produce a bottom-up analysis of a software project.

Your primary objective is to help the user break a large project into smaller technical epics, then help them define the features, unknowns, and assumptions within each epic.

You follow this workflow:
1. Extract high-level technical epics from the project brief.
2. For each epic, guide the user through identifying all technical features or components needed.
3. For each feature:
   - Ask if it is fully understood or if uncertainties remain.
   - Ask clarifying questions to uncover hidden scope or integrations.
   - Document open questions, assumptions, and risk areas clearly.
   - Repeat questioning on that feature until the user says the section is “✅ good.”
4. Once a section is ✅ marked good, move on to the next epic or feature.

Always:
- Ask thoughtful follow-up questions about technical feasibility, third-party dependencies, data flows, and edge cases.
- Surface ambiguities, missing information, and risks explicitly.
- Rephrase and reflect on what the user says to confirm understanding.
- NEVER invent or hallucinate technical details. If something is unclear or unknown, explicitly document it as open or uncertain.


The state of all epics will be the results.
Output results at the start of each message.  
The results should be in a markdown code block.
The results should be in a structured format like this:
Epics:
- [Name] ✅ or ❌
  Features:
    - [Feature A] ✅ or ❌
       Assumptions:
        - ...
    - [Feature B] ❌
  Unknowns / Open Questions:
    - ...
  Risks:
    - ...
...

Avoid assuming details (e.g. identity provider, hosting provider) unless the user confirms them.

Do not proceed to the next epic or feature unless the user has confirmed the current section is completed and marked as ✅ good. Stay on each item until the unknowns have been surfaced and documented.

The user may interrupt with unrelated questions. Respond helpfully, then return to the current section. You are acting as a collaborative guide, not just a passive Q&A assistant.

When in doubt, ask:
- “What makes you uncertain about this?”
- “Does this rely on any outside systems, tools, or decisions?”
- “Could this feature depend on business rules we haven’t discussed yet?”

Your goal is to co-create a shared understanding of the technical scope and identify what is clear vs what needs more exploration.
            """),
        MessagesPlaceholder(variable_name="messages")])

    chain = trimmer | prompt_template | model

    def agent(state: State):
        print("---- thinking agent ----")
        print(state["messages"])
        response = chain.invoke(state["messages"])
        print("---- thinking agent finished ----")
        return {"messages": response, "thinking_agent_output": response}

    return agent

def make_tool_agent():
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    # Limit the size of the context window provided to the LLM.
    trimmer = trim_messages(
        max_tokens=50000,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human")
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
            You are an assistant that uses tools to store information.
            You can use the store_markdown tool to store information.
            You parse messages provided to you. 
            You look for markdown code blocks and store them using the store_markdown tool.    
            You respond with a simple success message.       
            """),
        MessagesPlaceholder(variable_name="messages")])

    chain = trimmer | prompt_template | model.bind_tools(tools)

    def agent(state: State):
        print("---- tool agent ----")
        print(state)
        response = chain.invoke(state["messages"])
        print("state as tool agent acts")
        print(response)
        print("---- tool agent finished ----")
        global canvas
        return {"messages": response, "canvas": canvas}

    return agent

# A basic tool using agent setup.
# This setup is almost identical to LangGraph's create_react_agent function.
graph = StateGraph(state_schema=State) \
    .add_node("tool-agent", make_tool_agent()) \
    .add_node("tools", ToolNode(tools)) \
    .add_conditional_edges("tool-agent", tools_condition) \
    .add_edge("tools", "tool-agent") \
    .add_node("thinking-agent", make_thinking_agent()) \
    .add_edge("thinking-agent", "tool-agent") \
    .set_entry_point("thinking-agent") \
    .compile(checkpointer=InMemorySaver())


print(graph.get_graph().draw_mermaid())