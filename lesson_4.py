from haystack import Document, Pipeline
from haystack.components.routers import ConditionalRouter
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.websearch import SerperDevWebSearch
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Load all .txt files from data/ folder
txt_files = [str(f) for f in Path("data/").glob("*.txt")]

converter = TextFileToDocument()
splitter = DocumentSplitter(split_by="sentence", split_length=5)

converted = converter.run(sources=txt_files)
split = splitter.run(documents=converted["documents"])

document_store = InMemoryDocumentStore()
document_store.write_documents(documents=split["documents"])

rag_prompt_template = """
    Answer the following query given the documents.
    If the answer is not contained within the documents, reply with 'no_answer'
    Query: {{query}}
    Documents:
    {% for document in documents %}
    {{document.content}}
    {% endfor %}
"""

routes = [
    {
        "condition": "{{'no_answer' in replies[0]|lower}}",
        "output": "{{query}}",
        "output_name": "go_to_websearch",
        "output_type": str,
    },
    {
        "condition": "{{'no_answer' not in replies[0]|lower}}",
        "output": "{{replies[0]}}",
        "output_name": "answer",
        "output_type": str,
    },
]

prompt_for_websearch = """
    Answer the following query given the documents retrieved from the web.
    Your answer should indicate that your answer was generated from websearch.
    You can also reference the URLs that the answer was generated from

    Query: {{query}}
    Documents:
    {% for document in documents %}
    {{document.content}}
    {% endfor %}
"""

rag_or_websearch = Pipeline()
rag_or_websearch.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
rag_or_websearch.add_component("prompt_builder", PromptBuilder(template=rag_prompt_template))
rag_or_websearch.add_component("llm",OllamaGenerator(model="zephyr",generation_kwargs={"num_predict": 500,"temperature": 0.9,},))
rag_or_websearch.add_component("router", ConditionalRouter(routes))
rag_or_websearch.add_component("websearch", SerperDevWebSearch())
rag_or_websearch.add_component("prompt_builder_for_websearch", PromptBuilder(template=prompt_for_websearch))
rag_or_websearch.add_component("llm_for_websearch",OllamaGenerator(model="zephyr",generation_kwargs={"num_predict": 500,"temperature": 0.9,},))

rag_or_websearch.connect("retriever", "prompt_builder.documents")
rag_or_websearch.connect("prompt_builder", "llm")
rag_or_websearch.connect("llm.replies", "router.replies")
rag_or_websearch.connect("router.go_to_websearch", "websearch.query")
rag_or_websearch.connect("router.go_to_websearch", "prompt_builder_for_websearch.query")
rag_or_websearch.connect("websearch.documents", "prompt_builder_for_websearch.documents")
rag_or_websearch.connect("prompt_builder_for_websearch", "llm_for_websearch")

question = "How does Haystack handle document embeddings?"

result = rag_or_websearch.run(
    {
        "retriever": {"query": question},
        "prompt_builder": {"query": question},
        "router": {"query": question},
    }
)

# Print the answer — either from RAG or web search
if "answer" in result["router"]:
    print("Answer from RAG:")
    print(result["router"]["answer"])
elif "llm_for_websearch" in result:
    print("Answer from Web Search:")
    print(result["llm_for_websearch"]["replies"][0])

