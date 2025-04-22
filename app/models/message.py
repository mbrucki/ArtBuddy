from pydantic import BaseModel

class MessageRequest(BaseModel):
    message: str
    session_id: str 