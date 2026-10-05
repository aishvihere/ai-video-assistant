from dotenv import load_dotenv
load_dotenv()
from utils.audio_processing import process_input
from core.transcriber import transcribe_all
import os
print("KEY LOADED:", bool(os.getenv("OPENAI_API_KEY")))
from core.summarizer import generate_title,summarize
from core.extractor import extract_actionables, extract_decisions, extract_questions
from utils.formatter import format_items

source = "https://www.youtube.com/watch?v=qnQLZE_jh7I"

chunks =  process_input(source)
transcript = transcribe_all(chunks ,language="english")
print("\n" + "=" * 60)
print("📝 TRANSCRIPT")
print("=" * 60)
print(transcript[:500] + "..." if len(transcript) > 500 else transcript)


summary = summarize(transcript)
title = generate_title(summary)

print("\n" + "=" * 60)
print(f"📌 TITLE: {title}")
print("=" * 60)
print("\n📋 SUMMARY")
print("-" * 60)
print(summary)




action_items = extract_actionables(transcript)
decisions = extract_decisions(transcript)
questions = extract_questions(transcript)

print(format_items(action_items, "ACTION ITEMS"))
print(format_items(decisions, "KEY DECISIONS"))
print(format_items([q for q in questions if not q.answered], "OPEN QUESTIONS"))