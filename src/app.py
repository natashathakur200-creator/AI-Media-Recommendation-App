from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ai_service import (
    generate_media_recommendation,
    generate_answer_with_memory,
)

from src.session_store import (
    load_sessions_from_disk,
    save_sessions_to_disk,
    cleanup_sessions,
    get_recent_messages,
    append_message,
    get_or_create_session,
)


app = FastAPI(
    title="AI Media Recommendation System",
    description="AI-powered media planning prototype",
    version="0.1.0",
)


app.mount(
    "/web",
    StaticFiles(directory="web"),
    name="web",
)


class BusinessRequest(BaseModel):
    industry: str
    sub_industry: str = ""
    business_name: str
    business_type: str
    years_operating: str = ""
    city: str
    state: str
    target_audience: str
    marketing_objective: str
    marketing_budget_inr: str
    planning_period: str
    current_media_status: str = ""
    previous_campaign_information: str = ""


@app.on_event("startup")
def startup_event():
    load_sessions_from_disk()
    cleanup_sessions()
    save_sessions_to_disk()


@app.get("/")
def home():
    return FileResponse("web/index.html")


@app.post("/api/recommend")
def recommend(request: BusinessRequest):
    business_data = request.model_dump()

    try:
        recommendation = generate_media_recommendation(
            business_data
        )

        return {
            "ok": True,
            "recommendation": recommendation,
        }

    except Exception as error:
        return {
            "ok": False,
            "error": str(error),
        }


@app.post("/api/chat")
def chat(session_id: str, message: str):
    try:
        # Load the existing conversation
        history = get_recent_messages(session_id)

        # Load business context saved for this session
        business_context = get_or_create_session(
            session_id
        ).get(
            "business_context",
            {}
        )

        answer = generate_answer_with_memory(
            user_text=message,
            history=history,
            business_context=business_context,
        )

        # Save user's message
        append_message(
            session_id=session_id,
            role="user",
            content=message,
        )

        # Save AI's response
        append_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        # Persist everything to disk
        save_sessions_to_disk()

        return {
            "ok": True,
            "session_id": session_id,
            "answer": answer,
        }

    except Exception as error:
        return {
            "ok": False,
            "error": str(error),
        }