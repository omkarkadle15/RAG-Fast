import os
import shutil
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

folder_path = "db"
cached_llm = Ollama(model="llama3.1")
embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant, good at searching documents. If you do not have an answer from the provided information, say so. [/INST] </s>
    [INST] {input}
            Context: {context}
            Answer:
    [/INST]
"""
)

def process_query(query: str) -> str:
    return cached_llm.invoke(query)

def process_pdf_query(query: str):
    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    
    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )
    
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    
    result = chain.invoke({"input": query})
    
    sources = [
        {"source": doc.metadata["source"], "page_content": doc.page_content}
        for doc in result["context"]
    ]
    
    return result["answer"], sources

def save_and_process_pdf(file):
    file_name = file.filename
    save_file = f"pdf/{file_name}"
    
    try:
        with open(save_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise Exception(f"Could not save file: {str(e)}")
    
    print(f"filename: {file_name}")
    
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")
    
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")
    
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )
    
    vector_store.persist()
    
    return {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks)
    }