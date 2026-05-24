from haystack import Pipeline
from haystack.utils.auth import Secret
from haystack.components.builders import PromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.cohere import CohereDocumentEmbedder, CohereTextEmbedder
from dotenv import load_dotenv
load_dotenv()

document_store = InMemoryDocumentStore()
# Components
fetcher = LinkContentFetcher()
converter = HTMLToDocument()
embedder = CohereDocumentEmbedder(model="embed-english-v3.0")
writer = DocumentWriter(document_store=document_store)

indexing = Pipeline()
indexing.add_component("fetcher", fetcher)
indexing.add_component("converter", converter)
indexing.add_component("embedder", embedder)
indexing.add_component("writer", writer)

indexing.connect("fetcher", "converter.sources")
indexing.connect("converter", "embedder")
indexing.connect("embedder", "writer")

indexing.run(
    {
        "fetcher": {
            "urls": [
                "https://haystack.deepset.ai/integrations/cohere",
                "https://haystack.deepset.ai/integrations/anthropic",
                "https://haystack.deepset.ai/integrations/jina",
                "https://haystack.deepset.ai/integrations/nvidia",
            ]
        }
    }
)
# prompt = """
#     Answer the question based on the provided context.
#     Context:
#     {% for doc in documents %}
#     {{ doc.content }} 
#     {% endfor %}
#     Question: {{ query }}
# """

prompt = """
    You will be provided some context, followed by the URL that this context comes from.
    Answer the question based on the context, and reference the URL from which your answer is generated.
    Your answer should be in {{ language }}.
    Context:
    {% for doc in documents %}
    {{ doc.content }} 
    URL: {{ doc.meta['url']}}
    {% endfor %}
    Question: {{ query }}
    Answer:
"""

query_embedder = CohereTextEmbedder(model="embed-english-v3.0")
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt)
generator = OllamaGenerator(
    model="zephyr",
    generation_kwargs={
        "num_predict": 500,
        "temperature": 0.9,
    },
)

rag = Pipeline()
rag.add_component("query_embedder", query_embedder)
rag.add_component("retriever", retriever)
rag.add_component("prompt", prompt_builder)
rag.add_component("generator", generator)

rag.connect("query_embedder.embedding", "retriever.query_embedding")
rag.connect("retriever.documents", "prompt.documents")
rag.connect("prompt", "generator")

question = "How can I use Cohere with Haystack?"

# result = rag.run(
#     {
#         "query_embedder": {"text": question},
#         "retriever": {"top_k": 1},
#         "prompt": {"query": question},
#     }
# )

question = "How can I use Cohere with Haystack?"

result = rag.run(
    {
        "query_embedder": {"text": question},
        "retriever": {"top_k": 1},
        "prompt": {"query": question, "language": "English"},
    }
)

print(result["generator"]["replies"][0])

print(result["generator"]["replies"][0])
