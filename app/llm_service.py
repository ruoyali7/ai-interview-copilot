import json
import os
import time
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from app.rag_service import retrieve_context, retrieve_from_text

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def call_openai_json(prompt: str, schema: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "structured_response",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )

            return json.loads(response.output_text)

        except Exception as e:
            print(f"OpenAI API error. Attempt {attempt + 1}/{max_retries}: {e}")

            if attempt == max_retries - 1:
                return {
                    "error": "OpenAI API request failed",
                    "detail": str(e),
                }

            time.sleep(2 ** attempt)


QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "behavioral_questions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "technical_questions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "resume_based_questions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "job_specific_questions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "behavioral_questions",
        "technical_questions",
        "resume_based_questions",
        "job_specific_questions",
    ],
}


EVALUATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "integer"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
        },
        "weaknesses": {
            "type": "array",
            "items": {"type": "string"},
        },
        "improved_answer": {"type": "string"},
        "feedback_summary": {"type": "string"},
    },
    "required": [
        "score",
        "strengths",
        "weaknesses",
        "improved_answer",
        "feedback_summary",
    ],
}


def generate_interview_questions(resume_text: str, job_description: str) -> Dict[str, Any]:
    query = f"{resume_text} {job_description}"

    knowledge_chunks = retrieve_context(query)
    resume_chunks = retrieve_from_text(
        text=resume_text,
        query=job_description,
        top_k=3
    )

    knowledge_context = "\n".join(knowledge_chunks)
    resume_context = "\n".join(resume_chunks)

    prompt = f"""
You are an AI interview coach.

Use the knowledge context, resume context, and job description to generate interview questions.

Rules:
- Make questions specific to the candidate's actual resume.
- Use resume context when creating resume-based questions.
- Use knowledge context when creating AI/backend/system design questions.
- Do not invent experience not shown in the resume.

Knowledge Context:
{knowledge_context}

Resume Context:
{resume_context}

Full Resume:
{resume_text}

Job Description:
{job_description}
"""

    return call_openai_json(prompt, QUESTION_SCHEMA)


def evaluate_interview_answer(question: str, answer: str, job_description: str) -> Dict[str, Any]:
    prompt = f"""
You are an AI interview evaluator.

Evaluate the candidate's answer for the interview question.

Rules:
- Score should be from 1 to 10.
- Be specific and practical.
- Consider relevance to the job description.

Question:
{question}

Candidate Answer:
{answer}

Job Description:
{job_description}
"""

    return call_openai_json(prompt, EVALUATION_SCHEMA)