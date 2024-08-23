from pydantic import BaseModel

class Query(BaseModel):
    query: str
    include_sources: bool = False