from pydantic import BaseModel

class BaseResponse(BaseModel):
    code : str
    status : str

class EpitagramData(BaseModel):
    epitagram : str

class GetEpitagramResponse(BaseResponse):
    data : EpitagramData
