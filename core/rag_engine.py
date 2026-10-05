import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

model = ChatOpenAI(model = "gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])
    

def build_rag_pipeline(transcript: str):

    vector_store=build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)



    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a helpful answer giving assistant, answer to user's question ONLY using"
            "the video transcript as CONTEXT provided below"
            "If you don't find any answer just say 'i could not find any relevant information from the video about your question'"
            "always be concide and precise. If quoting someone, mention it clearly."
            "Context for transcript:  {context}"
            ),
            ("human","{question}")
        ]
    )


    ## full LCEL Rag pipeline  - make a chain-- 1. retriever- retrieves DOCs from vector_DB and convert to formatted version

    rag_chain = (
        {"context" : retriever | RunnableLambda(format_docs),
         "question" : RunnablePassthrough()
        }
        | prompt | model | StrOutputParser()
    )

    return rag_chain

def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever()


    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        "Context:\n{context}"),
        ("human", "{question}"),
    ])

    rag_chain = (
        {"context" : retriever | RunnableLambda(format_docs),
         "question" : RunnablePassthrough()  
        }
        | prompt | model | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain, question : str)->str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print( f"answer: {answer}")

    return answer 


