from fastapi import APIRouter


router = APIRouter()

@router.get("/")
def get_epitagram():
    return (lambda a : a)("야호")
