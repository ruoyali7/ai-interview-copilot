import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


def generate_interview_questions(resume_text: str, job_description: str) -> dict:
    prompt = f"""
You are an AI interview coach.

Given the candidate resume and job description, generate exactly 5 tailored interview questions.

Return ONLY valid JSON in this exact format:
{{
  "questions": [
    {{
      "type": "technical | behavioral | ai_system_design",
      "question": "question text",
      "reason": "why this question is relevant",
      "expected_points": ["point 1", "point 2", "point 3"]
    }}
  ]
}}

Rules:
- Do not use markdown.
- Do not wrap the JSON in ```json.
- Do not invent experience not shown in the resume.
- Make questions specific to the resume and job description.

Resume:
{resume_text}

Job Description:
{job_description}
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "questions": [],
            "raw_response": text,
            "error": "Model did not return valid JSON"
        }


def evaluate_interview_answer(question: str, answer: str, job_description: str) -> dict:
    prompt = f"""
You are an AI interview evaluator.

Evaluate the candidate's answer for the interview question.

Return ONLY valid JSON in this exact format:
{{
  "score": 0,
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "improved_answer": "a stronger version of the answer",
  "feedback_summary": "short overall feedback"
}}

Rules:
- Score should be from 1 to 10.
- Be specific and practical.
- Do not use markdown.
- Do not wrap JSON in ```json.
- Consider relevance to the job description.

Question:
{question}

Candidate Answer:
{answer}

Job Description:
{job_description}
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "score": 0,
            "strengths": [],
            "weaknesses": [],
            "improved_answer": "",
            "feedback_summary": "",
            "raw_response": text,
            "error": "Model did not return valid JSON"
        }
    
    