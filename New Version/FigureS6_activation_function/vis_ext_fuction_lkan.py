import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from efficient_kan.kan_variants import LorentzianKAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 加载模型
num_grids = 1
model = LorentzianKAN([43,4,9], num_grids=num_grids, k1=1, k2=5)
model.load_state_dict(torch.load('D:\projects\CEST-KAN0725\cestkanpc10\code\kan_v3_param_2000_model\lorentkan5_kf1_noise0_hidden1_r2_1_54_1_37_1382.pth', map_location=device))
model.to(device)
model.eval()

data = pd.read_csv(r'D:\projects\CEST-KAN0725\cestkanpc10\zspec_test_2subjects.csv', header=None)
targets = pd.read_csv(r'D:\projects\CEST-KAN0725\cestkanpc10\param_test_2subjects.csv', header=None)
data_arr = np.array(data).astype(np.float32)
targets_arr = np.array(targets).astype(np.float32)
  
# 创建张量并移动到GPU
test_data_tensor = torch.tensor(data_arr).to(device)
test_targets_tensor = torch.tensor(targets_arr).to(device)
test_data_tensor_mean = torch.mean(test_data_tensor, dim=0).view(1, -1).to(device)  # 移动到GPU

# 调用plot_spline
model.plot_loren(test_data_tensor_mean)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a * b)