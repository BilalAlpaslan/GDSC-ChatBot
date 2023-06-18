import os
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

load_dotenv()
API_ENDPOINT = os.getenv("API_ENDPOINT")
PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_ID = os.getenv("MODEL_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

app = FastAPI()


class Question(BaseModel):
    q: str
    a: Union[str, None] = None


def get_ai_response(question: Question):
    r = requests.post(
        f"https://{API_ENDPOINT}/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/{MODEL_ID}:predict",
        headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
        json={
            "instances": [
                {
                    "context": "You are a Bot for answering question about Google developer student program and related questions. If question not about GDSC program or related things repply with 'this questions not about GDSC program or related'.",
                    "examples": [
                        {
                            "input": {
                                "author": "user",
                                "content": "Can I apply again after my become a lead "
                            },
                            "output": {
                                "author": "bot",
                                "content": "No you can only once become a leader"
                            }
                        }
                    ],
                    "messages": [
                        {
                            "author": "user",
                            "content": question.q
                        }
                    ]
                }
            ],
            "parameters": {
                "temperature": 0.2,
                "maxOutputTokens": 256,
                "topP": 0.8,
                "topK": 40
            }
        },
    )
    return r.json()


@app.post("/questions")
def get_questions(q: Question):
    try:
        res = get_ai_response(q)
        return res
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', reload=True)
