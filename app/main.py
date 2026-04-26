from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Interview Copilot")


class InterviewRequest(BaseModel):
    resume_text: str
    job_description: str


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "AI Interview Copilot is running"
    }


@app.post("/generate-questions")
def generate_questions(request: InterviewRequest):
    return {
        "questions": [
            {
                "type": "behavioral",
                "question": "Tell me about a project where you solved a difficult technical problem.",
                "reason": "This question checks problem-solving and communication skills."
            }
        ]
    }