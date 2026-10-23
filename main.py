from fastapi import FastAPI

from routers import user
from routers import ai
from database import engine, Base
from routers.agent import router as agent_router
from models import user as user_model
from models import chat as chat_model

Base.metadata.create_all(bind=engine)


app = FastAPI()

app.include_router(agent_router)
app.include_router(user.router)
app.include_router(ai.router)

@app.get("/")
def home():
    return {
        "message": "Hello Git dev branch"
    }