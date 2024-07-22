import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from src.model import UNet
from src.dataset import ToySegmentationDataset

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    dataset = ToySegmentationDataset()
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "unet_toy.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main("experiments/config.yaml")