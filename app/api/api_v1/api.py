from fastapi import APIRouter

from app.api.api_v1.endpoints import auth, teachers, students, exams

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(teachers.router, prefix="/teachers", tags=["teachers"])
api_router.include_router(students.router, prefix="/students", tags=["students"])
api_router.include_router(exams.router, prefix="/exams", tags=["exams"])
