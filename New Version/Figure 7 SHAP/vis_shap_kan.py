import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from efficient_kan import KAN
import os

# Load model
num_grids = 9
spline_order = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU，如果可用
model = KAN([43, 100, 9], num_grids=num_grids, spline_order=spline_order)
model_path = "New Version/model/kan5_kf1_noise0_r3kf1_100_1_48_2409.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load data
data = pd.read_csv(r'D:\projects\CEST-KAN0725\cestkanpc10\zspec_test_400sample_2.csv', header=None)
targets = pd.read_csv(r'D:\projects\CEST-KAN0725\cestkanpc10\param_test_400sample_2.csv', header=None)
data_arr = np.array(data).astype(np.float32)
targets_arr = np.array(targets).astype(np.float32)

# Use all data for testing
test_data_tensor = torch.tensor(data_arr).to(device)
test_targets_tensor = torch.tensor(targets_arr).to(device)

# Define a function wrapper for SHAP input
def model_predict(data):
    data = torch.tensor(data).float().to(device)  # Move input data to GPU
    with torch.no_grad():
        output = model(data)
    return output.cpu().numpy()  # Move output back to CPU

# Initialize Kernel SHAP explainer
explainer = shap.KernelExplainer(model_predict, test_data_tensor.cpu().numpy())

# Calculate Kernel SHAP
shap_values = explainer.shap_values(test_data_tensor.cpu().numpy())

# Create save path
save_path = r'D:\projects\CEST-KAN0725\cestkanpc10\shap\kan_manu\Bar_2'
os.makedirs(save_path, exist_ok=True)

# Define new feature names
new_x_name = [
    -20, -15, -10, -8, -6, -4, -3.75, -3.5,
    -3.25, -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5,
    -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5,
    0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5,
    2.75, 3, 3.25, 3.5, 3.75, 4, 6, 8, 10, 15, 20
]

# Plot SHAP summary and save
titles = [
    "Water",
    "Contribution of inputs to Water FWHM",
    "$\Delta B_0$",
    "Amide",
    "Contribution of inputs to Amide FWHM",
    "rNOE",
    "Contribution of inputs to rNOE FWHM",
    "MT",
    "Contribution of inputs to MT FWHM",
]
x_limits = {
    1: (0.000, 0.030),
    3: (0.000, 0.040),
    4: (0.000, 0.020),
    6: (0.000, 0.014),
    8: (0.000, 0.030),
}

# Define custom color
custom_color = [0.8500, 0.3250, 0.0980]

for i in range(9):  # Visualize 9 output features
    plt.figure(figsize=(20, 20))
    shap.summary_plot(
        shap_values[:, :, i],
        test_data_tensor.cpu().numpy(),
        feature_names=new_x_name,
        show=False,
        max_display=20,
        plot_type='bar',
        color=custom_color  # 使用自定义颜色
    )
    plt.ylabel("Frequency offsets (ppm)", fontsize=20, labelpad=10)
    plt.xlabel("Mean|Shap value|", fontsize=20)
    plt.title(titles[i] if i < len(titles) else f'Contribution of inputs to output {i + 1}', fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)

    # 设置横坐标范围
    if (i + 1) in x_limits:
        plt.xlim(x_limits[i + 1])

    plt.subplots_adjust(top=0.9, bottom=0.2, left=0.15, right=0.95)
    plt.tight_layout()  # Optimize layout
    plt.savefig(os.path.join(save_path, f'shap_summary_output_{i + 1}.png'))
    plt.close()

print("SHAP visualizations saved successfully.")
