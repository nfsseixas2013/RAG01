from pydantic import  BaseModel

class health_output(BaseModel):
    status: str
    version: str
    author: str