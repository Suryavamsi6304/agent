"""
Groq API client.
Handles streaming responses with RAG context.
"""

import os
from typing import Iterator, List, Dict, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an Enterprise AI Knowledge Assistant built for Capgemini.
You answer questions using the provided knowledge base context.

Guidelines:
- Always ground your answers in the provided context.
- If the context does not contain relevant information, say so clearly and provide a general answer.
- Cite sources when referencing specific information (e.g., "[Source: filename.pdf, Page 3]").
- Be concise, professional, and accurate.
- For data/analytics questions, provide structured insights with numbers when available.
- If asked to generate a report or summary, structure it clearly with sections.
"""


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def stream_response(
    query: str,
    context: str,
    chat_history: Optional[List[Dict]] = None,
) -> Iterator[str]:
    """
    Stream a response from Groq given a query and RAG context.
    Yields text chunks as they arrive.
    """
    client = get_client()

    if context:
        user_content = (
            f"KNOWLEDGE BASE CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {query}"
        )
    else:
        user_content = (
            f"Note: No relevant context found in the knowledge base for this query.\n\n"
            f"USER QUESTION: {query}"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=2048,
        stream=True,
    )

    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text


def generate_insights(context: str, topic: str) -> Iterator[str]:
    """
    Generate structured insights/report from the knowledge base on a given topic.
    """
    client = get_client()

    prompt = (
        f"Based on the following knowledge base data, generate a professional structured report on: {topic}\n\n"
        f"Include: Executive Summary, Key Findings, Data Insights, and Recommendations.\n\n"
        f"DATA:\n{context}"
    )

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
        stream=True,
    )

    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text
