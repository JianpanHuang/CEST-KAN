# from common.public import public # type: ignore

from efficient_kan import KAN
from efficient_kan.kan_variants import FastKAN, LorentzianKAN
import torch
# import torch.nn as nn
import torch.optim as optim
# import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
# import csv
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
# import scipy.io
from efficient_kan.mlp import MLP
import time
# import datetime
from MSERegLoss import MSERegLoss
from sklearn.model_selection import KFold

import pandas as pd
import datetime
import csv


# Load data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

def generate_layer_configurations(max_layers):
            return [[input_size] + [hidd_layer_sz] * i + [output_size] for i in range(1, max_layers + 1)]
        
def make_time_folder():
    x = datetime.datetime.now()
    dateTimeStr = str(x)
    return "record_"+dateTimeStr[5:7]+'_'+dateTimeStr[8:10]+'_'+dateTimeStr[11:13]+'_'+dateTimeStr[14:16] 


def set_seed(seed): 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 

# 加入噪声的函数 
def add_noise(data, noise_level): 
    noise = noise_level * np.random.randn(*data.shape) 
    return np.float32(data + noise)

seed = 42
set_seed(42)
noise_level = [0]

gs = 5
repeat_model = 5
root = "D:/projects/CEST-KAN0725/cestkanpc10/"
save_name = "kf1_mlp"
record_save_path = root+'code/'+save_name+'_param/'+make_time_folder() +'.csv'

# train & val data
data = pd.read_csv(root+'zspec_train_ss25.csv', header=None)
targets = pd.read_csv(root+'param_train_ss25.csv', header=None)
# test data
test_data = pd.read_csv(r'D:\projects\CEST-KAN0725\cestkanpc10\zspec_test_2subjects.csv', header=None)
test_targets = pd.read_csv(r'D:\projects\CEST-KAN0725\cestkanpc10\param_test_2subjects.csv', header=None)

test_data_arr = np.array(test_data)
test_data_arr = np.float32(test_data_arr)
# test_data_tensor = torch.tensor(test_data_arr)
# test_targets_arr = np.array(test_targets)
# test_targets_arr = np.float32(test_targets_arr)
# test_targets_tensor = torch.tensor(test_targets_arr)

# test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_targets_tensor)
# testloader = DataLoader(test_dataset, batch_size=test_data_tensor.size(0), shuffle=False)


# print("test shape:", test_data_arr.shape, test_targets_arr.shape)


data_arr = np.array(data)
data_arr = np.float32(data_arr)
targets_arr = np.array(targets)
targets_arr = np.float32(targets_arr)

# 设置K
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

all_model_test_loss = []
all_model_test_correlation = []
all_model_name = []




with open(record_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_name', 'test_loss', 'correlation1', 'correlation2', 'correlation3', 'correlation4','correlation5','correlation6','correlation7','correlation8','correlation9'])


k_fold_id = 0

for train_index, val_index in kf.split(data_arr):
   
    #if k_fold_id != 0:
        #break
      
    for noise in noise_level:

        train_data, val_data = data_arr[train_index], data_arr[val_index]
        train_targets, val_targets = targets_arr[train_index], targets_arr[val_index]

           
        train_data = add_noise(train_data, noise)
        val_data=add_noise(val_data,noise)
        test_data_arr = add_noise(test_data_arr,noise)
        
        test_data_tensor = torch.tensor(test_data_arr)
        test_targets_arr = np.array(test_targets)
        test_targets_arr = np.float32(test_targets_arr)
        test_targets_tensor = torch.tensor(test_targets_arr)


        test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_targets_tensor)
        testloader = DataLoader(test_dataset, batch_size=test_data_tensor.size(0), shuffle=False)

        k_fold_id += 1

        # train_data, val_data, train_targets, val_targets = train_test_split(data_arr, targets_arr,
        #                                                                     test_size=0.2,random_state=42)

        train_data_tensor = torch.tensor(train_data)
        train_targets_tensor = torch.tensor(train_targets)
        val_data_tensor = torch.tensor(val_data)
        val_targets_tensor = torch.tensor(val_targets)

        train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_targets_tensor)
        val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_targets_tensor)
        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        
        print("train and val shape:", train_data.shape, val_data.shape, train_targets.shape, val_targets.shape)

        
        for k1 in [1]:  # (5,16) 0.5,4,0.8,1,1.1,1.2,1.25,1.5,1.6,1.875,2,2.5,3,(4/0.75)
            for k2 in [3]:   
                    
                for i in range (1,repeat_model+1):  
                    
                    name = f"mlp_kf{k_fold_id}_noise{noise}_r{i}" # gamma extra   name = f"lorentzkan5_{k1}_{k2}_{i}" # gamma extra
                    
                    save_model_dir = root+"code/"+ save_name +"_model/"+ name 
                    save_param_dir = root+"code/"+save_name + "_param/"+name
                    save_loss_dir = root +"code/loss/"+name
                    layer_loop = 1
                    

                    # Define model
                    input_size = 43
                    output_size = 9
                    hidd_layer_sz = 100
                
                    # net_layers_list = generate_layer_configurations(layer_loop)
                    # net_layers_list = [[100],[100,100],[100,100,100],[100,100,100,100]]
                    net_layers_list = [[100]]
                    
                    for net_layers in net_layers_list:

                        print("net_layers:",net_layers)
                        hidd_layer_num = len(net_layers)-2

                        # model = LorentzianKAN(net_layers,num_grids=gs,k1=k1,k2=k2)
                        # model = KAN(net_layers,num_grids=gs)
                        model = MLP(input_size,net_layers,output_size)
                        
                        
                        total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        print("total param !!!", total_param)
                        
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        print(device)
                        model.to(device)
                        # Define optimizer
                        optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
                        # Define learning rate scheduler
                        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

                        # Define loss
                        epochs = 200
                        loss_func = torch.nn.MSELoss()
                        train_loss_all = []
                        val_loss_all = []
                        losses = []
                        start_time = time.time()
                        best_loss = float('inf')
                        epochs_no_improve = 0
                        early_stop_patience = 20
                        
                        best_model_state = None
                        best_epoch = 0  # To track the epoch of the best model

                        for epoch in range(epochs):
                            # Train
                            train_loss = 0
                            train_num = 0
                            model.train()
                            with tqdm(trainloader) as pbar:
                                running_loss = 0.0
                                for i, (images, labels) in enumerate(pbar):
                                    images = images.view(-1, 43).to(device)
                                    optimizer.zero_grad()
                                    output = model(images)
                                    loss = loss_func(output, labels.to(device))
                                    loss.backward()
                                    optimizer.step()
                                    train_loss += loss.item() * images.size(0)
                                    train_num += images.size(0)
                                    pbar.set_postfix(lr=optimizer.param_groups[0]['lr'])
                            train_loss_all.append(train_loss/train_num)
                            pbar.set_postfix(loss=train_loss/train_num)

                            # Validation
                            model.eval()
                            val_loss = 0
                            val_num = 0
                            with torch.no_grad():
                                for images, labels in valloader:
                                    images = images.view(-1, 43).to(device)
                                    output = model(images)
                                    val_loss += loss_func(output, labels.to(device)).item() * images.size(0)
                                    val_num += images.size(0)
                            val_loss_ave = val_loss / val_num
                            val_loss_all.append(val_loss_ave)

                            # Update learning rate
                            scheduler.step()

                            print(f"Epoch {epoch + 1}, Train Loss: {train_loss/train_num}, Val Loss: {val_loss_ave}")

                            # Check for improvement
                            if val_loss_ave < best_loss:
                                best_loss = val_loss_ave
                                epochs_no_improve = 0
                                best_model_state = model.state_dict()  # Save the best model state
                                best_epoch = epoch + 1  # Track the best epoch
                            else:
                                epochs_no_improve += 1

                            # Stop training if there have been no improvements for a certain number of epochs
                            if epochs_no_improve >= early_stop_patience:
                                print(f"Early stopping at epoch {epoch + 1}")
                                break

                        end_time = time.time()
                        train_time = end_time - start_time
                        print("best epoch:",best_epoch, " best_loss:",best_loss, " current epoch:",epoch+1, " current loss:", val_loss_ave)

                        # Save the best trained model
                        if best_model_state is not None:
                            torch.save(best_model_state, save_model_dir + str(hidd_layer_sz) + '_' + str(hidd_layer_num) + '_' + str(best_epoch) + '_' + str(int(train_time)) + ".pth")
                        else:
                            torch.save(model.state_dict(), save_model_dir + str(hidd_layer_sz) + '_' + str(hidd_layer_num) + '_' + str(epoch+1) + '_' + str(int(train_time)) + ".pth")


                        tensor_dict = {
                            'epochs': epoch+1,
                            'best epoch': best_epoch,
                            'hidd_layer_sz': hidd_layer_sz,
                            'hidd_layer_num': hidd_layer_num,
                            'best_val_loss': best_loss,
                            'train_time': train_time,
                            'train_loss_all': train_loss_all,
                            'val_loss_all': val_loss_all
                        }

                        torch.save(tensor_dict, save_param_dir + str(hidd_layer_sz) + '_' + str(hidd_layer_num) + '_' + str(epoch+1) + '_' + str(int(train_time)) + ".pth")
                        print(f"training time for {epoch+1} epoch:",train_time)


                        print("start test!")
                        model.load_state_dict(best_model_state)
                        model.eval()
                    
                        test_loss = 0
                        test_num = 0
                        labels_all = []
                        outputs_all = []
                        
                        with torch.no_grad():
                            for images, labels in testloader:
                                images = images.view(-1, 43).to(device)
                                labels = labels.view(-1, 9).to(device)
                                outputs = model(images).to(device)
                                loss = loss_func(outputs, labels.to(device))
                                test_loss += loss.item()*images.size(0)
                                test_num += images.size(0)
                                labels_all.append(labels.cpu().numpy())
                                outputs_all.append(outputs.cpu().numpy())
                                
                        test_loss_ave = test_loss/test_num    
                        labels_all_tensor = np.squeeze(torch.FloatTensor(labels_all))
                        outputs_all_tensor = np.squeeze(torch.FloatTensor(outputs_all))
                        
                        
                        correlation = []
                        for n in range(labels_all_tensor.size(1)): #labels_all_tensor.size(1)
                            correlation.append(np.corrcoef(labels_all_tensor[:,n], outputs_all_tensor[:,n])[0, 1])   
                        
                        correlation = np.array(correlation)
                        
                        
                        all_model_test_loss.append(test_loss_ave)
                        all_model_test_correlation.append(correlation)
                        all_model_name.append(name)
                        
                        with open(record_save_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([name, test_loss_ave, correlation[0],correlation[1],correlation[2],correlation[3],correlation[4],correlation[5],correlation[6],correlation[7],correlation[8]])
                
            
        
          
all_model_test_correlation = np.array(all_model_test_correlation)             
df = pd.DataFrame(columns = ['model_name', 'test_loss', 'correlation1', 'correlation2', 'correlation3', 'correlation4','correlation5','correlation6','correlation7','correlation8','correlation9'])
df['model_name'] = all_model_name
df['test_loss'] = all_model_test_loss
df['correlation1'] = all_model_test_correlation[:,0]
df['correlation2'] = all_model_test_correlation[:,1]
df['correlation3'] = all_model_test_correlation[:,2]
df['correlation4'] = all_model_test_correlation[:,3]
df['correlation5'] = all_model_test_correlation[:,4]
df['correlation6'] = all_model_test_correlation[:,5]
df['correlation7'] = all_model_test_correlation[:,6]
df['correlation8'] = all_model_test_correlation[:,7]
df['correlation9'] = all_model_test_correlation[:,8]

df.to_csv('record.csv', index=False)
            
