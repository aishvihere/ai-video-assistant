## 1. need actionable items 
# 2. decisions MADE in the meeting
## 3. questions asked 
from typing import List,Type
from pydantic import BaseModel,Field, create_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
import os 
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini",temperature=0)


# defining structure of the output we want for actionable item from each chunk batch of TRANSCRIPT
class ActionItem(BaseModel):
    task : str =  Field(default="Not specified",description="Concrete action, starting with a verb")
    owner : str = Field(default="Not specified",description="Person/ Team responsible")
    deadline : str = Field(default="Not specified",description="Last date to submit if stated")
    evidence : str  = Field(default="Not specified", description="Short phrase (<=20 words) supporting this task ")

class Question(BaseModel):
    question: str
    asked_by: str = "Not specified"
    answered: bool = Field(description="Whether an answer was given in the transcript")
    answer: str = Field(default="Not specified", description="Short answer if given")
class Decision(BaseModel):
    decision: str = Field(description="What was decided")
    rationale: str = Field(default="Not specified", description="Why, only if stated")
    decided_by: str = "Not specified"


def list_of(item_cls):
    return create_model(f"{item_cls.__name__}List",
                        items=(list[item_cls], Field(default_factory=list)))


## private functions- working behind the scenes :- ## lets make an overall build_chain function which we can call for invoking without making chain for each action repeatedly
def _chain(system_prompt: str, schema):
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
    return prompt | model.with_structured_output(schema)

def _extract(transcript: str, item_cls, extract_prompt: str, merge_prompt: str, max_chunk_chars: int = 15000,chunk= True )->list:
    schema = list_of(item_cls)
    extractor = _chain(extract_prompt, schema)

    if len(transcript) <= max_chunk_chars:
        return extractor.invoke({"text": transcript}).items

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars, chunk_overlap=500
    ).split_text(transcript)

    results = extractor.batch([{"text": c} for c in chunks], config={"max_concurrency": 5})
    all_items = [i for r in results for i in r.items]
    if not all_items:
        return []

    merger = _chain(merge_prompt, schema)
    merged_input = schema(items=all_items).model_dump_json(indent=2)
    return merger.invoke({"text": merged_input}).items


def extract_actionables(transcript: str) -> list[ActionItem]:
    return _extract(
        transcript, ActionItem,
        "Extract action items from this transcript excerpt. An action item is a task that a "
        "specific person or group commits to do, or is explicitly assigned. General advice, "
        "tips, or recommendations given to an audience are NOT action items. "
        "If there are none, return an empty list. Never invent owners or deadlines; "
        "leave them blank if not stated.",
        "Merge these action items from different parts of one video. Remove duplicates, keep any "
        "owner/deadline details. Do not add new items.",
        chunk=False,
    )

def extract_decisions(transcript: str) -> list[Decision]:
    return _extract(
        transcript, Decision,
        "Extract decisions that the speakers themselves explicitly made or agreed on "
        "(e.g. 'we will go with X'). Do NOT treat recommendations, advice, study results, "
        "or expert opinions as decisions. If there are none, return an empty list. "
        "Only fill in decided_by if the speaker names who decided; otherwise leave it blank.",
        "Merge these decisions from different parts of one video. Remove duplicates; do not add new ones.",
    )

def extract_questions(transcript: str) -> list[Question]:
    return _extract(
        transcript, Question,
        "Extract genuine questions asked by a participant that call for an answer from someone "
        "else. Skip rhetorical questions, questions the speaker asks and immediately answers "
        "as a teaching device, and filler. Mark answered=True only if an answer is given in "
        "this excerpt. If there are none, return an empty list.",
        "Merge these questions from different parts of one video. Remove duplicates. If a question "
        "is answered in any copy, mark it answered with that answer.",
        chunk= False,
    )