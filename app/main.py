from fastapi import FastAPI
from pydantic import BaseModel
from app.llm_service import generate_interview_questions, evaluate_interview_answer

app = FastAPI(title="AI Interview Copilot")


class InterviewRequest(BaseModel):
    resume_text: str
    job_description: str

class AnswerEvaluationRequest(BaseModel):
    question: str
    answer: str
    job_description: str

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "AI Interview Copilot is running"
    }


@app.post("/generate-questions")
def generate_questions(request: InterviewRequest):
    result = generate_interview_questions(
        resume_text=request.resume_text,
        job_description=request.job_description
    )

    return result

@app.post("/evaluate-answer")
def evaluate_answer(request: AnswerEvaluationRequest):
    result = evaluate_interview_answer(
        question=request.question,
        answer=request.answer,
        job_description=request.job_description
    )

    return result