import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from efficient_kan.kan import KAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 加载模型
num_grids = 9
spline_order = 3
model = KAN([43, 100, 9], num_grids=num_grids, spline_order=spline_order)
model.load_state_dict(torch.load('New Version/model/kan5_kf1_noise0_r3kf1_100_1_48_2409.pth', map_location=device))
model.to(device)
model.eval()

data = pd.read_csv(r'zspec_test_2subjects.csv', header=None)
targets = pd.read_csv(r'param_test_2subjects.csv', header=None)
data_arr = np.array(data).astype(np.float32)
targets_arr = np.array(targets).astype(np.float32)
  
test_data_tensor = torch.tensor(data_arr).to(device)
test_targets_tensor = torch.tensor(targets_arr).to(device)
test_data_tensor_mean = torch.mean(test_data_tensor, dim=0).view(1, -1).to(device)  # 移动到GPU

# 调用plot_spline
model.plot_spline(test_data_tensor_mean)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a * b)

