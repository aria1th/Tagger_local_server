from fastapi import FastAPI
import uvicorn
import argparse
from typing import List
import numpy as np
from tag_image import predict_image
from typing import Optional
from pydantic import BaseModel

class Item(BaseModel):
	# image: List[List[List[float]]]
    image: List[List[List[float]]]

class Response(BaseModel):
    tags: List[str]

app = FastAPI()
server = None
model_path = "./models"


@app.post("/predict", response_model=Response)
async def predict(image: Item):
    """
    Predicts tags for an image
    image: image to predict (List of List of List of float)
    return: list of tags (List of str)
    """
    global model_path
    image = np.array(image.image)
    # log image shape
    print("image shape: ", image.shape)
    tags = predict_image(image, model_path=model_path)
    return {"tags": tags}

@app.on_event("shutdown")
def close_server():
    # Add any cleanup code here
    print("Server shutting down")

# basic document
@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10080)
    # ip
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    # model_path
    parser.add_argument("--model_path", type=str, default="./models")
    # ip is not supported yet
    args = parser.parse_args()
    port = args.port
    model_path = args.model_path
    ip = args.ip
    uvicorn.run(app, host=ip, port=port)