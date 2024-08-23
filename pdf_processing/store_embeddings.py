from langchain_community.vectorstores import Chroma

def store_embeddings(embedding, chunks, folder_path):
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )
    vector_store.persist()
    return vector_store