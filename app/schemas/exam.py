from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class ExamBase(BaseModel):
    name: Optional[str] = None
    file_format: Literal['docx', 'xlsx']
    exam_code: str
    score: Optional[float] = None
    report: Optional[Dict[str, Any]] = None


class ExamCreate(ExamBase):
    student_id: Optional[int] = None
    student_code: Optional[str] = None
    
    # Optional student info for creating new students
    student_name: Optional[str] = None
    birth_date: Optional[datetime] = None
    place_of_birth: Optional[str] = None


class ExamUpdate(BaseModel):
    name: Optional[str] = None
    file_format: Optional[Literal['docx', 'xlsx']] = None
    exam_code: Optional[str] = None
    score: Optional[float] = None
    report: Optional[Dict[str, Any]] = None
    student_id: Optional[int] = None
    teacher_id: Optional[int] = None


class ExamResponse(ExamBase):
    id: int
    name: Optional[str] = None
    student_id: int
    teacher_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    submission_url: Optional[str] = None

    class Config:
        orm_mode = True


class ExamWithRelations(ExamResponse):
    student_name: str
    student_code: str
    teacher_name: Optional[str] = None

    class Config:
        orm_mode = True
