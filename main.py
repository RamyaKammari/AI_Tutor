from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.plugins.tutor import TutorAgent
import time
import uuid

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory session storage
sessions: Dict[str, Any] = {}

class Message(BaseModel):
    role: str
    content: str
    caption: Optional[str] = None

class SessionData(BaseModel):
    messages: List[Message] = []
    model_choice: str = "groq"

class InteractionRequest(BaseModel):
    prompt: str
    model_choice: str

class InteractionResponse(BaseModel):
    response: str
    tokens_used: int
    sources: List[str]
    elapsed_time: float

def get_session(session_id: str):
    # Check if the session exists; if not, create a new one
    if session_id not in sessions:
        sessions[session_id] = {
            "data": SessionData(),
            "tutor_agent": TutorAgent()
        }
    return sessions[session_id]

@app.post("/interact/{session_id}", response_model=InteractionResponse)
async def interact_with_ai(session_id: str, request: InteractionRequest, session: Dict[str, Any] = Depends(get_session)):
    # Handle user interaction with the AI tutor
    session_data = session["data"]
    tutor_agent = session["tutor_agent"]
    
    session_data.model_choice = request.model_choice
    session_data.messages.append(Message(role="user", content=request.prompt))
    
    # Update tutor_agent's history
    tutor_agent.history.append(("user", request.prompt))

    start_time = time.time()
    response, tokens_used, sources = tutor_agent.interact_with_ai(request.prompt, "Tutor", request.model_choice)
    elapsed_time = round(time.time() - start_time, 2)
    
    ai_message = Message(
        role="assistant",
        content=response,
        caption=f"Sources: {sources} | Time: {elapsed_time}s | Tokens: {tokens_used} | Model: {request.model_choice}"
    )
    session_data.messages.append(ai_message)
    
    # Update tutor_agent's history with the response
    tutor_agent.history.append(("system", response))

    return InteractionResponse(
        response=response,
        tokens_used=tokens_used,
        sources=sources,
        elapsed_time=elapsed_time
    )

@app.get("/session/{session_id}")
async def get_session_data(session_id: str, session: Dict[str, Any] = Depends(get_session)):
    # Retrieve session data for the given session ID
    return session["data"]

@app.post("/new_session")
async def create_new_session():
    # Create a new session and return the session ID
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "data": SessionData(),
        "tutor_agent": TutorAgent()
    }
    return {"session_id": session_id}

@app.get("/")
async def root():
    # Serve the main HTML file for the application
    return FileResponse("static/index.html")