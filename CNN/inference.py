import torch

from train import test_dataset, test_loader, criterion
from model import simple_cnn

device = "cuda" if(torch.cuda.is_available()) else "cpu"

model = simple_cnn().to(device)
model.load_state_dict(
    torch.load(
        "./CNN/model.pth",
        map_location=device,
        weights_only=True
    )
)


model.eval()


with torch.no_grad():
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        y = model(images)
        loss = criterion(y, labels)
        total_loss += loss.item()*images.size(0)
        predictions = y.argmax(dim=1)
        correct += (predictions==labels).sum().item()
        total += labels.size(0)

    print(f"avg_loss = {total_loss/total:.4f}")
    print(f"correct = {100*correct/total:.4f}%")


