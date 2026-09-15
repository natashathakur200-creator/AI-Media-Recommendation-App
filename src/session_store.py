import os
import json
import time
from typing import Dict, List, Any


SESSIONS_PATH = "data/sessions.json"

_sessions: Dict[str, Dict[str, Any]] = {}


def load_sessions_from_disk() -> None:
    global _sessions

    if not os.path.exists(SESSIONS_PATH):
        _sessions = {}
        return

    with open(SESSIONS_PATH, "r", encoding="utf-8") as f:
        _sessions = json.load(f)


def save_sessions_to_disk() -> None:
    os.makedirs(
        os.path.dirname(SESSIONS_PATH),
        exist_ok=True
    )

    with open(
        SESSIONS_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            _sessions,
            f,
            ensure_ascii=False,
            indent=2
        )


def get_or_create_session(
    session_id: str
) -> Dict[str, Any]:

    if session_id not in _sessions:
        current_time = int(time.time())

        _sessions[session_id] = {
            "session_id": session_id,
            "created_at": current_time,
            "updated_at": current_time,
            "business_context": {},
            "messages": []
        }

    return _sessions[session_id]


def set_business_context(
    session_id: str,
    business_context: dict
) -> None:

    session = get_or_create_session(session_id)

    session["business_context"] = business_context

    session["updated_at"] = int(time.time())


def get_business_context(
    session_id: str
) -> dict:

    session = get_or_create_session(session_id)

    return session.get(
        "business_context",
        {}
    )


def append_message(
    session_id: str,
    role: str,
    content: str
) -> None:

    session = get_or_create_session(session_id)

    session["messages"].append({
        "role": role,
        "content": content,
        "ts": int(time.time())
    })

    session["updated_at"] = int(time.time())


def get_recent_messages(
    session_id: str,
    limit: int = 10
) -> List[Dict[str, Any]]:

    session = get_or_create_session(session_id)

    return session["messages"][-limit:]


def cleanup_sessions(
    max_age_seconds: int = 60 * 60 * 24 * 7
) -> int:

    now = int(time.time())

    to_delete = []

    for session_id, session in _sessions.items():

        if (
            now
            - int(
                session.get(
                    "updated_at",
                    now
                )
            )
            > max_age_seconds
        ):
            to_delete.append(session_id)

    for session_id in to_delete:
        del _sessions[session_id]

    return len(to_delete)