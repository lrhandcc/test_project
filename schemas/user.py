from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    age: int


class UserResponse(BaseModel):
    id: int
    username: str
    age: int

    class Config:
        from_attributes = True


class UserCreateResponse(BaseModel):
    message: str
    user: UserResponse

class MessageResponse(BaseModel):
    message: str