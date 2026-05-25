from pathlib import Path
from haystack import Document, Pipeline
from haystack_integrations.components.generators.ollama import OllamaGenerator, OllamaChatGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool


template = """
    Answer the questions based on the given context.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    Question: {{ question }}
    Answer:
"""
rag_pipe = Pipeline()
llm = OllamaGenerator(model="zephyr",generation_kwargs={"num_predict": 500,"temperature": 0.9,},)
prompt_builder = PromptBuilder(template=template)

rag_pipe.add_component("llm", llm)
rag_pipe.add_component("prompt_builder", prompt_builder)

rag_pipe.connect("prompt_builder", "llm")

INTEGRATION_INFO = {
    "ollama": {
        "install": "pip install ollama-haystack",
        "components": ["OllamaGenerator", "OllamaChatGenerator", "OllamaDocumentEmbedder", "OllamaTextEmbedder"],
        "requires_api_key": False,
        "runs_locally": True,
        "recommended_models": ["zephyr", "llama3", "mistral", "nomic-embed-text"],
    },
    "cohere": {
        "install": "pip install cohere-haystack",
        "components": ["CohereGenerator", "CohereChatGenerator", "CohereDocumentEmbedder", "CohereTextEmbedder"],
        "requires_api_key": True,
        "runs_locally": False,
        "recommended_models": ["embed-english-v3.0", "embed-multilingual-light-v3.0", "command-r"],
    },
    "openai": {
        "install": "pip install haystack-ai",
        "components": ["OpenAIGenerator", "OpenAIChatGenerator", "OpenAIDocumentEmbedder", "OpenAITextEmbedder"],
        "requires_api_key": True,
        "runs_locally": False,
        "recommended_models": ["gpt-4o", "gpt-4o-mini", "text-embedding-3-small"],
    },
    "firecrawl": {
        "install": "pip install firecrawl-haystack",
        "components": ["FirecrawlCrawler"],
        "requires_api_key": True,
        "runs_locally": False,
        "recommended_models": [],
    },
}

def get_integration_info(provider: str):
    key = provider.lower()
    if key in INTEGRATION_INFO:
        return INTEGRATION_INFO[key]
    else:
        return {"error": f"No integration info found for '{provider}'. Available: {list(INTEGRATION_INFO.keys())}"}


def rag_pipeline_func(query: str):
    txt_files = [str(f) for f in Path("data/").glob("*.txt")]
    converter = TextFileToDocument()
    splitter = DocumentSplitter(split_by="sentence", split_length=5)

    converted = converter.run(sources=txt_files)
    split = splitter.run(documents=converted["documents"])
    documents = split["documents"]

    result = rag_pipe.run({
        "prompt_builder": {
            "documents": documents,
            "question": query
        }
    })

    return {"reply": result["llm"]["replies"][0]}


rag_tool = Tool(
    name="rag_pipeline_func",
    description="Search Haystack documentation to answer general questions about Haystack concepts, components, pipelines, RAG, embedders, retrievers, generators, document stores, and how things work. Use this for any broad or conceptual question about Haystack.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The question or topic to search for in the Haystack docs"}
        },
        "required": ["query"],
    },
    function=rag_pipeline_func,
)

integration_tool = Tool(
    name="get_integration_info",
    description="Get specific integration details like install command, available components, and recommended models for a specific EXTERNAL provider. Only use this when the user explicitly asks about installing or using one of these providers: 'ollama', 'cohere', 'openai', 'firecrawl'.",
    parameters={
        "type": "object",
        "properties": {
            "provider": {"type": "string", "description": "Must be one of: 'ollama', 'cohere', 'openai', 'firecrawl'"}
        },
        "required": ["provider"],
    },
    function=get_integration_info,
)

chat_generator = OllamaChatGenerator(
    model="llama3.2",
    generation_kwargs={"num_predict": 500, "temperature": 0.9},
    tools=[rag_tool, integration_tool]
)

import json

available_functions = {
    "rag_pipeline_func": rag_pipeline_func,
    "get_integration_info": get_integration_info,
}

def function_caller(messages):
    replies = chat_generator.run(messages=messages)
    reply = replies['replies'][0]

    # If the model wants to call a tool
    if reply.tool_calls:
        tool_call = reply.tool_calls[0]
        func_name = tool_call.tool_name
        func_args = tool_call.arguments

        print(f"Calling function: {func_name} with args: {func_args}")

        # Call the actual function
        func = available_functions[func_name]
        result = func(**func_args)

        # Return result as a tool message
        tool_message = ChatMessage.from_tool(
            tool_result=json.dumps(result),
            origin=tool_call
        )
        return tool_message
    else:
        return reply


user_message = ChatMessage.from_user("How do I install the Cohere integration?")
messages = [user_message]

result = function_caller(messages)
print(result.text)

from typing import List
from haystack import component
from haystack.components.joiners import BranchJoiner

@component
class OllamaFunctionCaller:
    def __init__(self, available_functions: dict):
        self.available_functions = available_functions

    @component.output_types(function_replies=List[ChatMessage], assistant_replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage]):
        last_message = messages[-1]

        # If the model made a tool call
        if last_message.tool_calls:
            tool_replies = []
            for tool_call in last_message.tool_calls:
                func = self.available_functions[tool_call.tool_name]
                result = func(**tool_call.arguments)
                tool_replies.append(
                    ChatMessage.from_tool(
                        tool_result=json.dumps(result),
                        origin=tool_call
                    )
                )
            return {"function_replies": tool_replies}
        else:
            # Final answer — no more tool calls
            return {"assistant_replies": [last_message]}


message_collector = BranchJoiner(List[ChatMessage])
agent_generator = OllamaChatGenerator(
    model="llama3.2",
    generation_kwargs={"num_predict": 500, "temperature": 0.9},
    tools=[rag_tool, integration_tool]
)
agent_function_caller = OllamaFunctionCaller(
    available_functions={
        "rag_pipeline_func": rag_pipeline_func,
        "get_integration_info": get_integration_info,
    }
)

chat_agent = Pipeline(max_runs_per_component=10)
chat_agent.add_component("message_collector", message_collector)
chat_agent.add_component("generator", agent_generator)
chat_agent.add_component("function_caller", agent_function_caller)

chat_agent.connect("message_collector", "generator.messages")
chat_agent.connect("generator.replies", "function_caller.messages")
chat_agent.connect("function_caller.function_replies", "message_collector")

print("\nHaystack Chat Agent ready! Ask anything about Haystack.")
print("Type 'exit' or 'quit' to stop.\n")

messages = [ChatMessage.from_system(
    "You are a helpful Haystack assistant. "
    "For general questions about Haystack (what it is, how components work, pipelines, RAG, etc.) always use the rag_pipeline_func tool. "
    "Only use get_integration_info when the user specifically asks about installing or using ollama, cohere, openai, or firecrawl."
)]

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    messages.append(ChatMessage.from_user(user_input))
    response = chat_agent.run({"message_collector": {"value": messages}})
    assistant_reply = response["function_caller"]["assistant_replies"][0]
    messages.append(assistant_reply)
    print(f"\nAssistant: {assistant_reply.text}\n")







