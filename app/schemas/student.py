from typing import Optional
from datetime import datetime, date
from pydantic import BaseModel


class StudentBase(BaseModel):
    student_code: str
    name: str
    birth_date: date
    place_of_birth: Optional[str] = None


class StudentCreate(StudentBase):
    pass


class StudentUpdate(BaseModel):
    student_code: Optional[str] = None
    name: Optional[str] = None
    birth_date: Optional[date] = None
    place_of_birth: Optional[str] = None


class StudentResponse(StudentBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
