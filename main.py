
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from pypdf import PdfReader

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/upload-report")
async def upload_report(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File must be 10 MB or smaller."
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file."
        )

    file_path = UPLOAD_DIR / Path(file.filename).name
    file_path.write_bytes(content)

    reader = PdfReader(str(file_path))

    extracted_text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    if not extracted_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No selectable text found. OCR is needed."
        )

    return {
        "filename": file.filename,
        "message": "PDF uploaded and text extracted successfully",
        "extracted_text": extracted_text
    }
