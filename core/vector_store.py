from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chroma_dir = "chroma_dir"
COLLECTION_NAME = "transcript"

def build_vector_store(transcript: str)->Chroma:
    print("Building vector store--")

    spliter = RecursiveCharacterTextSplitter(
        chunk_size =  1000,
        chunk_overlap = 150
    )

    chunks = spliter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    vector_store= Chroma.from_documents(
        documents=docs,
        persist_directory=chroma_dir,
        collection_name=COLLECTION_NAME,
        embedding=embeddings

    )

    return vector_store


def load_vector_store()->Chroma:

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=chroma_dir

    )

    return vector_store

def get_retriever(vector_store: Chroma, k: int =5):

    return vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {'k': k}
    )
