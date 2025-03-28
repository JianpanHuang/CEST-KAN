import torch
from efficient_kan.mlp import MLP
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from MSERegLoss import MSERegLoss
import time  # 导入时间库

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = MLP(43, [100], 9)
model.load_state_dict(torch.load(r'New Version/model/mlp_kf1_noise0_r4100_-1_54_1076.pth'))
model.to(device)
model.eval()

data = pd.read_csv('D:\projects\CEST-KAN0725\data27noise\zspec_test_ss2.csv', header=None)
targets = pd.read_csv('D:\projects\CEST-KAN0725\data27noise\param_test_ss2.csv', header=None)

data_arr = np.array(data).astype(np.float32)
targets_arr = np.array(targets).astype(np.float32)
test_data_tensor = torch.tensor(data_arr)
test_targets_tensor = torch.tensor(targets_arr)

test_dataset = TensorDataset(test_data_tensor, test_targets_tensor)
testloader = DataLoader(test_dataset, batch_size=test_data_tensor.size(0), shuffle=False)

loss_func = torch.nn.MSELoss()
test_loss = 0
test_num = 0

labels_all = []
outputs_all = []

# 记录测试开始时间
start_time = time.time()

with torch.no_grad():
    for images, labels in testloader:
        images = images.view(-1, 43).to(device)
        labels = labels.view(-1, 9).to(device)
        outputs = model(images).to(device)
        loss = loss_func(outputs, labels.to(device))
        test_loss += loss.item() * images.size(0)
        test_num += images.size(0)

        labels_all.append(labels.cpu().numpy())
        outputs_all.append(outputs.cpu().numpy())

# 记录测试结束时间
end_time = time.time()
test_duration = end_time - start_time  # 计算测试时长

test_loss_ave = test_loss / test_num
print("Average test loss:", test_loss_ave)
print("Test duration (seconds):", test_duration)  # 输出测试时长

labels_all_tensor = np.squeeze(torch.FloatTensor(labels_all))
outputs_all_tensor = np.squeeze(torch.FloatTensor(outputs_all))

scipy.io.savemat('test_mlp_kf1_loss_test1_20241109.mat', {'test_loss_ave': test_loss_ave})
scipy.io.savemat('test_mlp_kf1_labels_test1_20241109.mat', {'labels_all': labels_all})
scipy.io.savemat('test_mlp_kf1_outputs_test1_20241109.mat', {'outputs_all': outputs_all})

# Calculate the correlation coefficient
fontsz = 10
for n in range(labels_all_tensor.size(1)):
    plt.subplot(3, 3, n + 1)
    correlation = np.corrcoef(labels_all_tensor[:, n], outputs_all_tensor[:, n])[0, 1]
    print("Correlation coefficient:", correlation)
    plt.scatter(labels_all_tensor[:, n], outputs_all_tensor[:, n])
    plt.xlabel("MPLF", fontsize=fontsz, font='arial')
    plt.ylabel("MLP", fontsize=fontsz, font='arial')
    plt.title("Correlation", fontsize=fontsz, font='arial')

plt.tick_params(axis='both', which='minor', labelsize=fontsz, left=False, right=False, top=False, bottom=False)
plt.subplots_adjust(wspace=0.8, hspace=1)
plt.show()
