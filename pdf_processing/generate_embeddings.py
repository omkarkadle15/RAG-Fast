from langchain_community.embeddings import FastEmbedEmbeddings

def generate_embeddings(chunks):
    embedding = FastEmbedEmbeddings()
    return embedding, chunks