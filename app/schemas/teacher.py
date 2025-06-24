from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr


class TeacherBase(BaseModel):
    email: EmailStr
    name: str


class TeacherCreate(TeacherBase):
    password: str


class TeacherUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None


class TeacherResponse(TeacherBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
