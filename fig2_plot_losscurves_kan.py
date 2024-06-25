import torch 
import matplotlib.pyplot as plt

file_path = 'modelparam'
file_name = 'kan_100_1_102_1717484547.pth'
cont = torch.load(file_path+'/'+file_name)
train_loss_all = cont['train_loss_all']
val_loss_all = cont['val_loss_all']
print(len(train_loss_all))
epoch = cont['epochs']
train_time = cont['train_time']
best_loss = cont['best_loss']
print(best_loss)
print(train_time)
print(epoch)
# Plot the loss values against the number of epochs
font_tit = {'family': 'Arial',
         'weight': 'bold',
         'size': 15}
font_xylab = {'family': 'Arial',
         'weight': 'bold',
         'size': 14}
font_leg = {'family': 'Arial',
         'weight': 'bold',
         'size': 13}
fig, ax = plt.subplots()
ax.plot(range(1, epoch + 1), train_loss_all, linewidth = 2, label='Training Loss')
ax.plot(range(1, epoch + 1), val_loss_all, linewidth = 2, label='Validation Loss')
plt.xlim((0,105))
plt.xticks(fontproperties = font_xylab)
plt.ylim((0.2,1.8))
plt.yticks(fontproperties = font_xylab)
plt.tick_params(width = 2)
ax.set_title('KAN',font_tit)
ax.set_xlabel('Epochs',font_xylab)
ax.set_ylabel('Loss',font_xylab)
ax.legend(prop=font_leg)
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
# plt.savefig("kan_losscurves.jpg",dpi=600)
plt.show()