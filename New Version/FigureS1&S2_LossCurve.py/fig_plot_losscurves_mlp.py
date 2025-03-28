import torch 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # 导入 ticker 模块

file_path = 'D:\projects\CEST-KAN0725\cestkanpc10\code\kan_v3_mlp_kf_param'
file_name = 'mlp_kf4_noise0_r4100_-1_69_994.pth'
cont = torch.load(file_path+'/'+file_name)
train_loss_all = cont['train_loss_all']
val_loss_all = cont['val_loss_all']
print(len(train_loss_all))
epoch = cont['epochs']
train_time = cont['train_time']
#best_loss = cont['best_loss']
#print(best_loss)
print(train_time)
print(epoch)

# 设置字体参数
font_tit = {'family': 'Arial', 'weight': 'bold', 'size': 35}
font_xylab = {'family': 'Arial', 'weight': 'bold', 'size': 35}
font_leg = {'family': 'Arial', 'weight': 'bold', 'size': 35}
# 绘制损失曲线
fig, ax = plt.subplots()

# 设置坐标轴范围
plt.xlim((0, 80))
plt.ylim((0.3, 1.6))

ax.set_title('(B) MLP',font_tit)
ax.set_xlabel('Epochs',font_xylab)
ax.set_ylabel('Loss',font_xylab) 


ax.plot(range(1, epoch + 1), train_loss_all, linewidth = 2, label='Training Loss')
ax.plot(range(1, epoch + 1), val_loss_all, linewidth = 2, label='Validation Loss')


# 设置刻度字号
plt.xticks(fontsize=32, fontweight='bold')
plt.yticks(fontsize=32, fontweight='bold')
# 设置纵坐标刻度格式，保留 1 位小数
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

ax.legend(prop=font_leg)
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
# plt.savefig("mlp_losscurves.jpg",dpi=600)
plt.show()