from typing import Sequence, Tuple

from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.agents import AgentAction
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


@tool
def get_age(name: str) -> int:
    """Gets age of a person"""
    return 15


tools = [get_age]


import os
from langchain_openai import ChatOpenAI

TOGETHER_API_KEY = "ca4d4859d3c3aa62398f9ee300684d409599edd824df998cc95eb88ebb8618bb"

llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
    # model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model="mistralai/Mistral-7B-Instruct-v0.1",
)


def format_to_agent_scratchpad(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> str:
    if len(intermediate_steps) == 0:
        return []

    messages = []
    for agent_action, observation in intermediate_steps:
        messages.append(AIMessage(f"`{agent_action.tool}` for `{agent_action.tool_input}` is: '{observation}'"))

    messages.append(AIMessage("My answer is:"))

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
