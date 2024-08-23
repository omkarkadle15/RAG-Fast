import os
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.models import Query
from app.utils import (
    process_query,
    process_pdf_query,
    save_and_process_pdf,
    get_documents_in_database,
    clear_database,
    folder_path,
)

router = APIRouter()

@router.get("/healthcheck", tags=["Healthcheck"])
async def healthcheck():
    try:
        # Check if the folder_path (database directory) exists
        if not os.path.exists(folder_path):
            raise Exception("PDF storage folder does not exist")
        
        # Check if the pdf directory exists
        pdf_directory = "pdf"
        if not os.path.exists(pdf_directory):
            raise Exception("PDF upload directory does not exist")
        
        # You might want to add a simple test for each of your main functions
        test_query = "Test query"
        process_query(test_query)
        process_pdf_query(test_query)
        
        # Check if we can list files in the pdf directory
        try:
            os.listdir(pdf_directory)
        except Exception as e:
            raise Exception(f"Cannot access PDF directory: {str(e)}")
        
        return JSONResponse(content={"status": "healthy", "message": "All systems operational"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "unhealthy", "message": f"Healthcheck failed: {str(e)}"}, status_code=500)

@router.get("/show_documents", tags=["Documents"])
async def show_documents():
    try:
        documents = get_documents_in_database()
        return JSONResponse(content={"documents": documents})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload_pdf", tags=["Documents"])
async def pdf_post(file: UploadFile = File(...)):
    try:
        result = save_and_process_pdf(file)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query_pdf", tags=["Documents"])
async def query_pdf_post(query: Query):
    print("Post /ask_pdf is called")
    print(f"query: {query.query}")
    print(f"include_sources: {query.include_sources}")
    
    result, sources = process_pdf_query(query.query, query.include_sources)
    
    print(result)
    
    if query.include_sources:
        return JSONResponse(content={"answer": result, "sources": sources})
    else:
        return JSONResponse(content={"answer": result})

@router.post("/clear_database", tags=["Documents"])
async def clear_db():
    try:
        result = clear_database()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))