import torch
from monai.losses import DiceCELoss
from model import UNet3D
from dataset import get_loader

def train():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=4, out_channels=3).to(device)
    loader = get_loader(data_dir="./data")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    model.train()
    for epoch in range(50):
        epoch_loss = 0
        for batch_data in loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} - Average Loss: {epoch_loss/len(loader):.4f}")
    
    # Save the model
    torch.save(model.state_dict(), "unet3d_brats.pth")

if __name__ == "__main__":
    train()