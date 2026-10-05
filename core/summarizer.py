from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
import os
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
parser = StrOutputParser()

def split_transcript(transcript: str)-> list:
    splitter= RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )


    return splitter.split_text(transcript)


def summarize(transcript : str)-> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a transcript summarizer assistant, summarize the given transcript chunks in a concise manner including most important topics "),
            ("human", "this is the transcript:  {text}"),
        ]
    )

    chain = prompt | model | parser

    chunks = split_transcript(transcript)  ## list of transcript chunks
    chunk_summaries = chain.batch( ## runs invoke in parallel keeping batch of 5 chunks at a time- returns a list : batch 1: chunk
    [{"text": chunk} for chunk in chunks],
    config={"max_concurrency": 5},
)

    combined = "\n\n".join(
    f"Part {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries)  ## keeps the order  --> part 1: summary, part 2 : summary...
)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",  "You are an expert summarizer. Combine these partial summaries of one "
     "video into a single final professional summary in bullet points, "
     "keeping the most important information and removing repetition."),
            ("human", "{text}"),
        ]
    )

    combined_chain = combined_prompt | model | parser

    return combined_chain.invoke({"text":combined})

def generate_title(transcript : str)-> str:

    title_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a title generator, form a summary title according to the transcript summary"),
            ("human", "{summary}"),
        ]
    )

    title_chain = title_prompt | model | parser

    return title_chain.invoke({"summary":transcript[:2000]})