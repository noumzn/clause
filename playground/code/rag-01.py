import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from pathlib import Path
HANDBOOK_PATH=Path(__file__).resolve().parent /"data"/ "shop_handbook.md"



from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


#INDEXING
#1.LOAD
text=HANDBOOK_PATH.read_text(encoding="utf-8")
doc=Document(
    page_content=text,
    metadata={"source": "data/shop_handbook.md"}
)


#2.CHUNKS
splitter= RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)
chunks=splitter.split_documents(documents=[doc])


#3.EMBEDDING
embeddings=init_embeddings(
    model=os.getenv("EMBEDDINGS_MODEL"),
    provider=os.getenv("EMBEDDINGS_PROVIDER")
)

#4.STORE
vector_store=InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

#RAG
##LLM DEFINING
llm=init_chat_model(
    model=os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("CHAT_PROVIDER"),
    api_key=os.getenv("CHAT_API_KEY") or None,
    temperature=0.1
)
query=input("You:").strip()

#5-7.RETRIEVING
retriver=vector_store.similarity_search(query,k=3)
texts=[]
for doc in retriver:
    texts.append(doc.page_content)
context="\n\n".join(texts)#string

messages=[
    ("system", "You are a helpful Clause shop assistant."),
    ("human", f"Context:\n{context}\n\nQuestion: {query}")
]

response = llm.invoke(messages)
print(response.text)

