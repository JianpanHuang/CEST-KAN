import torch 
import matplotlib.pyplot as plt

file_path = 'New Version/model_param'
file_name = 'lorentkan5_kf1_noise0_r4_1_5100_1_64_1538.pth'
cont = torch.load(file_path+'/'+file_name)
train_loss_all = cont['train_loss_all']
val_loss_all = cont['val_loss_all']
print(len(train_loss_all))
epoch = cont['epochs']
train_time = cont['train_time']
# best_loss = cont['best_loss']
# print(best_loss)
print(train_time)
print(epoch)


# 设置字体参数
font_tit = {'family': 'Arial', 'weight': 'bold', 'size': 35}
font_xylab = {'family': 'Arial', 'weight': 'bold', 'size': 35}
font_leg = {'family': 'Arial', 'weight': 'bold', 'size': 35}
# 绘制损失曲线
fig, ax = plt.subplots()

# 设置坐标轴范围
plt.xlim((0, 90))
plt.ylim((0.3, 1.6))

ax.set_title('(C) LKAN',font_tit)
ax.set_xlabel('Epochs',font_xylab)
ax.set_ylabel('Loss',font_xylab) 


ax.plot(range(1, epoch + 1), train_loss_all, linewidth = 2, label='Training Loss')
ax.plot(range(1, epoch + 1), val_loss_all, linewidth = 2, label='Validation Loss')


# 设置刻度字号
plt.xticks(fontsize=32, fontweight='bold')
plt.yticks(fontsize=32, fontweight='bold')

ax.legend(prop=font_leg)
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
# plt.savefig("mlp_losscurves.jpg",dpi=600)
plt.show()
