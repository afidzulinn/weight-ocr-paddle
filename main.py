from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import numpy as np
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    return {"Message": "Weight OCR"}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:

        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)


        df = pd.read_csv(file_path)


        weights = df["Weight"].tolist()


        os.remove(file_path)

        return {"weights": weights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))