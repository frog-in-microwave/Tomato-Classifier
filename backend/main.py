from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


import torch 
from torchvision import transforms

from PIL import Image
import io
from predictor import model
from classes import classes







transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)








@app.get("/")
async def root():
    return {"message": "frogssssssss"}




@app.get("/health")
async def health():
    return {"message": "all good"}




@app.post("/predict")
async def predict(file: UploadFile):

    # read the image file and convert it to a PIL Image
    bytes = await file.read()
    image = Image.open(io.BytesIO(bytes))


    # transform it to a batch of one transformed image
    image = transform(image).unsqueeze(0) # (1, 3, 244, 244)

    with torch.no_grad():
        predictions = model(image)
        print(predictions[0])
    return {"predictions": predictions.tolist()[0],
            "class": classes[predictions.argmax().item()]}


