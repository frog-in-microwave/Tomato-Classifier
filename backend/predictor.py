from model import Tomato_classifier
import torch

model = Tomato_classifier()

checkpoint = torch.load(
    "weights/model_mid_lr.pth",
    map_location="cpu"
)

model.load_state_dict(checkpoint["model_state"])


model.eval()