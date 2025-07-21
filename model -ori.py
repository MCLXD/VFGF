import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from lat_model import LATModel  # Assumes the LATModel is in lat_model.py (previous artifact)


# Custom Dataset (placeholder, replace with your actual dataset)
class CustomDataset(Dataset):
    def __init__(self, num_samples=1000, num_frames=15, i_rgb=1024, i_obj=20, num_class=10):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.i_rgb = i_rgb
        self.i_obj = i_obj
        self.num_class = num_class
        # Simulate random data (replace with actual data loading)
        self.rgb_data = torch.randn(num_samples, num_frames, i_rgb)
        self.obj_data = torch.randn(num_samples, num_frames, i_obj)
        self.labels = torch.randint(0, num_class, (num_samples, num_frames))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.rgb_data[idx], self.obj_data[idx], self.labels[idx]


# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs, s_enc, s_ant,
                checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (rgb_inputs, obj_inputs, labels) in enumerate(train_loader):
            rgb_inputs, obj_inputs, labels = rgb_inputs.to(device), obj_inputs.to(device), labels.to(device)

            # Get label for the frame at s_enc + s_ant
            target_labels = labels[:, s_enc + s_ant]  # Shape: [batch_size]

            optimizer.zero_grad()
            logits, probs = model((rgb_inputs, obj_inputs), s_enc=s_enc, s_ant=s_ant)

            # Logits shape: [batch_size, 1, num_class], squeeze to [batch_size, num_class]
            logits = logits.squeeze(1)
            loss = criterion(logits, target_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {running_loss / (batch_idx + 1):.4f}, Accuracy: {100 * correct / total:.2f}%")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}] Summary, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


# Main script
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    s_enc = 10
    s_ant = 3
    i_rgb = 1024
    i_obj = 20
    num_class = 10
    embed_dim = 128
    hidden_dim = 64
    dropout = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize dataset and dataloader
    dataset = CustomDataset(num_samples=1000, num_frames=15, i_rgb=i_rgb, i_obj=i_obj, num_class=num_class)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize model, loss, and optimizer
    model = LATModel(
        i_rgb=i_rgb,
        i_obj=i_obj,
        num_class=num_class,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        s_enc=s_enc,
        s_ant=s_ant
    )