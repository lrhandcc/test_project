from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    age: int = Field(gt=0, le=120)


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