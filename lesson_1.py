from haystack import Pipeline
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import Document
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

embedder = OllamaDocumentEmbedder(
    model="nomic-embed-text"
)
document_store = InMemoryDocumentStore()
converter = TextFileToDocument()
splitter = DocumentSplitter()
writer = DocumentWriter(document_store=document_store)

indexing_pipeline = Pipeline()

indexing_pipeline.add_component("converter", converter)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("embedder", embedder)
indexing_pipeline.add_component("writer", writer)

indexing_pipeline.connect("converter", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")

indexing_pipeline.run({
    "converter": {"sources": [
    "data/haystack_overview.txt",
    "data/haystack_components.txt",
    "data/haystack_pipelines.txt",
    "data/haystack_document_stores.txt",
    "data/haystack_rag.txt",
    "data/haystack_ollama_integration.txt",
]}})

query_embedder = OllamaTextEmbedder(model = "nomic-embed-text")
retriever = InMemoryEmbeddingRetriever(document_store=document_store)

document_search = Pipeline()

document_search.add_component("query_embedder", query_embedder)
document_search.add_component("retriever", retriever)

document_search.connect("query_embedder.embedding", "retriever.query_embedding")

question = "What is haystack components?"
results = document_search.run({"query_embedder": {"text": question}})

print(f"\nQuery: {question}\n")
print("=" * 60)
for i, doc in enumerate(results["retriever"]["documents"], 1):
    print(f"\n[Result {i}] (score: {doc.score:.4f})")
    print(f"Source: {doc.meta.get('file_path', 'unknown')}")
    print(f"{doc.content.strip()}")
    print("-" * 60)



