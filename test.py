from dataset_loader import get_datasets
from model import EyeFixationNetwork
import os
import re
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import ConvertImageDtype
import imageio.v2 as imageio

os.makedirs("final_predictions", exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

_, _, test_dataset = get_datasets()
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
# print(len(test_dataloader))

model = EyeFixationNetwork().to(device)
model.load_state_dict(torch.load('final_model_pth/model_epoch_3.pth'))
model.eval()


for i, batch in enumerate(test_dataloader):

    image = batch["image"].to(device)
    # print(image)

    with torch.no_grad():
        output = model(image)

    output = torch.sigmoid(output)
    output = ConvertImageDtype(torch.uint8)(output)
    output = output.squeeze().cpu().numpy()

    img_name = test_dataset.image_files[i]
    match = re.search(r'(\d+)', img_name)
    image_number = match.group(1)
    filename = f"prediction-{image_number}.png"
    imageio.imwrite(os.path.join("final_predictions", filename), output)

    print(f"Saved prediction for {filename}")