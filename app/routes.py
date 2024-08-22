import os
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.models import Query
from app.utils import (
    process_query,
    process_pdf_query,
    save_and_process_pdf,
    folder_path,
)

router = APIRouter()

@router.get("/healthcheck")
async def healthcheck():
    try:
        # Check if the folder_path exists
        if not os.path.exists(folder_path):
            raise Exception("PDF storage folder does not exist")
        
        # You might want to add a simple test for each of your main functions
        test_query = "Test query"
        process_query(test_query)
        process_pdf_query(test_query)
        
        return JSONResponse(content={"status": "healthy"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Healthcheck failed: {str(e)}")

@router.post("/query_pdf")
async def query_pdf_post(query: Query):
    print("Post /ask_pdf is called")
    print(f"query: {query.query}")
    
    result, sources = process_pdf_query(query.query)
    
    print(result)
    
    return JSONResponse(content={"answer": result, "sources": sources})

@router.post("/upload_pdf")
async def pdf_post(file: UploadFile = File(...)):
    try:
        result = save_and_process_pdf(file)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))