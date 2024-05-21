from typing import Sequence, Tuple

from langchain import hub
from langchain.agents.format_scratchpad.tools import _create_tool_message
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.agents import initialize_agent, Tool, AgentType, create_openai_tools_agent, create_openai_functions_agent, AgentExecutor, create_tool_calling_agent
from langchain_core.agents import AgentAction
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


@tool
def get_age(name: str) -> int:
    """Gets age of a person"""
    return 15

@tool
def get_phone_number(name: str) -> str:
    """Gets phone number of a person"""
    return "+7 989 989 98 88"


tools = [get_age, get_phone_number]


import os
from langchain_openai import ChatOpenAI

TOGETHER_API_KEY = "ca4d4859d3c3aa62398f9ee300684d409599edd824df998cc95eb88ebb8618bb"

llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
    # model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model="mistralai/Mistral-7B-Instruct-v0.1",
)

# examples = [
#     HumanMessage(
#         "What's the product of x and y", name="example_user"
#     ),
#     AIMessage(
#         "",
#         name="example_assistant",
#         tool_calls=[
#             {"name": "multiply", "args": {"a": "x", "b": "y"}, "id": "1"}
#         ],
#     ),
#     ToolMessage("<return value>", tool_call_id="1"),
#     AIMessage(
#         "The product of x and y is <return value>",
#         name="example_assistant",
#     ),
# ]


def format_to_agent_scratchpad(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> str:
    if len(intermediate_steps) == 0:
        return []

    messages = []
    for agent_action, observation in intermediate_steps:
        # messages.append(_create_tool_message(agent_action, observation))
        messages.append(AIMessage(f"`{agent_action.tool}` for `{agent_action.tool_input}` is: '{observation}'"))
        # messages.append(HumanMessage(f"Result is: {observation}"))
        # agent_scratchpad += agent_action.log
        # agent_scratchpad += f"Result is: {observation}"

        # agent_scratchpad += f"{agent_action.tool}("
        # agent_scratchpad += ", ".join(f"{key}={val}" for key, val in agent_action.tool_input.items())
        # agent_scratchpad += f") == {observation}\n"

        # agent_scratchpad += f"{agent_action.tool}("
        # agent_scratchpad += ", ".join(f"{val}" for key, val in agent_action.tool_input.items())
        # agent_scratchpad += f") == {observation}\n"

    # agent_scratchpad += "\nPlease don't use more tools and give answer if you have enough data.\n"
    # agent_scratchpad += "If background information is enough, give the final answer.\n"
    # agent_scratchpad += "If background information is enough, give the final answer.\n"
    messages.append(AIMessage("My answer is:"))
    # messages.append(HumanMessage(f"{input}. Use information i've given to you"))
    # print(messages)
    # print(agent_scratchpad)
    # return agent_scratchpad
    return messages


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You can use tools to access additional information"),
        # *examples,
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# agent = create_tool_calling_agent(llm, tools, prompt=prompt)
agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_agent_scratchpad(x["intermediate_steps"])
    )
    | prompt
    | llm.bind_tools(tools)
    | ToolsAgentOutputParser()
)
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

print(agent_executor.invoke({"input": "How old is Alex?"}))
