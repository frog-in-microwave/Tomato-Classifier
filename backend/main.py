from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware

from slowapi import Limiter
from slowapi.util import get_remote_address


import torch 
from torchvision import transforms

from PIL import Image
import io
from predictor import model
from classes import classes






# define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


app = FastAPI()


# creating the middleware for cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# creating the ratelimiter 

limiter = Limiter(key_func=get_remote_address) # creating the Limiter object

app.state.limiter = limiter # adding the limiter to the app state





@app.get("/")
async def root(request: Request):
    return {"message": "frogssssssss"}




@app.get("/health")
async def health():
    return {"message": "all good"}




@app.post("/predict")
@limiter.limit("5/minute")  # limit to 5 requests per minute
async def predict(request: Request, file: UploadFile =  File(...)):

    # read the image file and convert it to a PIL Image
    bytes = await file.read()
    image = Image.open(io.BytesIO(bytes))


    # transform it to a batch of one transformed image
    image = transform(image).unsqueeze(0) # (1, 3, 244, 244)

    with torch.no_grad():
        predictions = model(image)
        probabilities = torch.softmax(predictions, dim=1)
    return {"predictions": predictions.tolist()[0],
            "class": classes[predictions.argmax().item()],
            "probabilities": probabilities.tolist()[0],}


