from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from utils.inference import (
    analyze_vegetation,
    analyze_soil,
    analyze_soil_test,
    analyze_vegetation_test
)

app = FastAPI(title="Soil & Vegetation Detection System")

# -------------------------------------------------------------
# CORS setup (allow local React frontend)
# -------------------------------------------------------------
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Vegetation analysis endpoint
# -------------------------------------------------------------
@app.post("/analyze/vegetation")
async def analyze_vegetation_endpoint(image: UploadFile = File(...)):
    print(f"[Backend] Received vegetation analysis request for: {image.filename}")
    try:
        image_bytes = await image.read()
        print("[Backend] Read image bytes, calling analyze_vegetation()")
        result = analyze_vegetation(image_bytes)
        print("[Backend] Analysis result:", result)

        safe = jsonable_encoder(result)
        return JSONResponse(content={"status": "success", "result": safe})

    except Exception as e:
        print("[Backend] Error during vegetation analysis:", str(e))
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


# -------------------------------------------------------------
# Soil analysis endpoint
# -------------------------------------------------------------
@app.post("/analyze/soil")
async def analyze_soil_endpoint(image: UploadFile = File(...)):
    print(f"[Backend] Received soil analysis request for: {image.filename}")
    try:
        image_bytes = await image.read()
        print("[Backend] Calling analyze_soil()")
        result = analyze_soil(image_bytes)
        print("[Backend] Soil analysis result:", result)

        safe = jsonable_encoder(result)
        return JSONResponse(content={"status": "success", "result": safe})

    except Exception as e:
        print("[Backend] Error during soil analysis:", str(e))
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


# -------------------------------------------------------------
# Test-only endpoints (for isolated debugging)
# -------------------------------------------------------------
@app.post("/test-soil")
async def test_soil_endpoint(image: UploadFile = File(...)):
    print(f"[Backend] Received TEST soil request for: {image.filename}")
    try:
        image_bytes = await image.read()
        result = analyze_soil_test(image_bytes)
        safe = jsonable_encoder(result)
        return JSONResponse(content={"status": "success", "result": safe})
    except Exception as e:
        print("[Backend] Test soil error:", str(e))
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


@app.post("/test-veg")
async def test_vegetation_endpoint(image: UploadFile = File(...)):
    print(f"[Backend] Received TEST vegetation request for: {image.filename}")
    try:
        image_bytes = await image.read()
        result = analyze_vegetation_test(image_bytes)
        safe = jsonable_encoder(result)
        return JSONResponse(content={"status": "success", "result": safe})
    except Exception as e:
        print("[Backend] Test vegetation error:", str(e))
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


# -------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------
@app.get("/")
async def root():
    return JSONResponse(content={"status": "ok", "message": "Soil & Vegetation Detection API is running!"})
