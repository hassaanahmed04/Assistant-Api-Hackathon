import os
from openai import OpenAI, APIError
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

try:
    client = OpenAI(
   api_key=""

    )

    assistant = client.beta.assistants.create(
        name="Job Advisor",
        instructions="You are a personal Technical Job Advisor. Write and run code to answer technical questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-3.5-turbo-1106",
    )

    PERSIST_DIR = "./storage"
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader("files").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.80)
    query_engine = RetrieverQueryEngine(
        retriever=retriever, node_postprocessors=[postprocessor]
    )

    response = query_engine.query("What is attention is all you need?")
    pprint_response(response, show_source=True)

    assistant_response = assistant.messages.create(
        assistant_id=assistant.id,
        messages=[
            {"role": "user", "content": "What is attention?"}
        ]
    )
    print(assistant_response.messages[1].content)

except APIError as e:
    print(f"Error: {e}")


