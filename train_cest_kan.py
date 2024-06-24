from efficient_kan import KAN
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import time
from MSERegLoss import MSERegLoss

# Load data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
data = pd.read_csv('data\train_zspec.csv', header=None)
print(data.shape)
targets = pd.read_csv('data\train_labels.csv', header=None)
data_arr = np.array(data)
data_arr = np.float32(data_arr)
targets_arr = np.array(targets)
targets_arr = np.float32(targets_arr)
train_data, val_data, train_targets, val_targets = train_test_split(data_arr, targets_arr,
                                                                    test_size=0.2,
                                                                    random_state=42)
train_data_tensor = torch.tensor(train_data)
train_targets_tensor = torch.tensor(train_targets)
val_data_tensor = torch.tensor(val_data)
val_targets_tensor = torch.tensor(val_targets)
train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_targets_tensor)
val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_targets_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(train_data.shape, val_data.shape, train_targets.shape, val_targets.shape)

# Define model
input_size = 43
output_size = 9
hidd_layer_sz = 100
net_layers = [input_size,hidd_layer_sz,output_size]
# net_layers = [input_size,hidd_layer_sz,hidd_layer_sz,hidd_layer_sz,hidd_layer_sz,output_size]
hidd_layer_num = len(net_layers)-2
model = KAN(net_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Define optimizer
# optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define loss
epochs = 150
loss_func = MSERegLoss(alpha=0.01)
# loss_func = nn.MSELoss()

# Train & validation
train_loss_all = []
val_loss_all = []
losses = []
start_time = time.time()
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop_patience = 50
for epoch in range(epochs):
    # Train
    train_loss = 0
    train_num = 0
    model.train()
    with tqdm(train_loader) as pbar:
        running_loss = 0.0
        for i, (zspec, labels) in enumerate(pbar):
            zspec = zspec.view(-1, 43).to(device)
            optimizer.zero_grad()
            output = model(zspec)
            loss = loss_func(output, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*zspec.size(0)
            train_num += zspec.size(0)
            pbar.set_postfix(lr=optimizer.param_groups[0]['lr'])
    train_loss_all.append(train_loss/train_num)
    pbar.set_postfix(loss=train_loss/train_num)
    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    val_num = 0
    with torch.no_grad():
        for zspec, labels in val_loader:
            zspec = zspec.view(-1, 43).to(device)
            output = model(zspec)
            val_loss += loss_func(output, labels.to(device)).item()*zspec.size(0)
            val_num += zspec.size(0)
    val_loss_ave = val_loss/val_num
    val_loss_all.append(val_loss/val_num)
    # Update learning rate
    scheduler.step()
    print(
        f"Epoch {epoch + 1}, Train Loss: {train_loss/train_num}, Val Loss: {val_loss/val_num}"
    )
    # Check for improvement
    if val_loss_ave < best_val_loss:
        best_val_loss = val_loss_ave
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    # Stop training if there have been no improvements for a certain number of epochs
    if epochs_no_improve >= early_stop_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
print(best_val_loss)
end_time = time.time()
train_time = end_time - start_time
# print(train_time)

# Save the trained model
torch.save(model.state_dict(), "D:\projects\cestkan\code\model\kan_" + str(hidd_layer_sz) + '_' + str(hidd_layer_num) + '_' + str(epoch+1) + '_' + str(int(time.time())) + ".pth")

# Save the training parameters
tensor_dict = {
    'epochs': epoch+1,
    'hidd_layer_sz': hidd_layer_sz,
    'hidd_layer_num': hidd_layer_num,
    'best_val_loss': best_val_loss,
    'train_time': train_time,
    'train_loss_all': train_loss_all,
    'val_loss_all': val_loss_all
}
torch.save(tensor_dict,"D:\projects\cestkan\code\labels\kan_" + str(hidd_layer_sz) + '_' + str(hidd_layer_num) + '_' + str(epoch+1) + '_' + str(int(time.time())) + ".pth")

# Plot the loss values against the number of epochs
fig, ax = plt.subplots()
ax.plot(range(1, epoch + 2), train_loss_all, label='Train Loss')
ax.plot(range(1, epoch + 2), val_loss_all, label='Val Loss')
ax.set_title('Loss Curves')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
plt.show()
# plt.savefig("D:\projects\cestkan\code\loss\kan_loss_'+str(hidd_layer_sz) + '_' + str(hidd_layer_num) + '_' + str(epoch+1) + '_' + str(int(time.time())) +'.png")
