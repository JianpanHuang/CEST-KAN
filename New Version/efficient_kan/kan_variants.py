import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
import matplotlib.pyplot as plt
import os
import numpy as np
 

# class RadialBasisFunction(nn.Module):
#     def __init__(
#         self,
#         grid_min: float = -2.,
#         grid_max: float = 2.,
#         num_grids: int = 5,
#         omega: float = 1.,
#         extra: int = 3,
#     ):
#         super().__init__()
#         h = (grid_max-grid_min)/num_grids   
#         self.var = omega**2
        
#         grid = torch.linspace(grid_min-extra*h, grid_max+extra*h, num_grids+2*extra+1)
#         self.grid = torch.nn.Parameter(grid, requires_grad=False)

    # def forward(self, x):
    #     return torch.exp(-(x[..., None] - self.grid) ** 2/(2*self.var))

class LorentzianBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 5,
        gamma: float = 1.,
        extra: int = 3,
    ):
        super().__init__()
        
        h = (grid_max-grid_min)/num_grids   
        self.gamma = gamma
        
        grid = torch.linspace(grid_min-h*extra, grid_max+h*extra, num_grids+2*extra+1)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)

    def forward(self, x):
        return self.gamma**2 / ((x[..., None] - self.grid)**2 + self.gamma**2)
 

# class FastKANLayer(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         grid_min: float = -2.,
#         grid_max: float = 2.,
#         num_grids: int = 8,
#         base_activation = nn.SiLU,
#         spline_weight_init_scale: float = 0.1,
#         k1=1,
#         k2 = 3,
#     ) -> None:
#         super().__init__()
#         # self.layernorm = nn.LayerNorm(input_dim)
        
#         h = (grid_max-grid_min)/num_grids
#         omega = k1/num_grids
#         extra = math.ceil(k2*omega/h)  # 99.7%
        
#         self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids,omega,extra)
        
#         self.base_activation = base_activation()
#         self.base_linear = nn.Linear(input_dim, output_dim)
        
#         self.spline_weight = torch.nn.Parameter(
#             torch.Tensor(output_dim, input_dim, num_grids+2*extra+1)
#         )
#         nn.init.trunc_normal_(self.spline_weight, mean=0.0, std=spline_weight_init_scale)

#     def forward(self, x):
#         base = self.base_linear(self.base_activation(x))
#         spline_basis = self.rbf(x)
#         spline = torch.einsum(
#             "...in,oin->...o", spline_basis, self.spline_weight
#         )
#         return base + spline


class LorentzianKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        base_activation = nn.SiLU,
        spline_weight_init_scale: float = 0.1,
        k1 = 1,
        k2 = 5,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        
        h = (grid_max-grid_min)/num_grids
        
        gamma = k1/num_grids
        extra = math.ceil(k2*gamma/h)  # 96.15%
        
        self.lorentzian = LorentzianBasisFunction(grid_min, grid_max, num_grids,gamma,extra)
        
        self.base_activation = base_activation()
        self.base_linear = nn.Linear(input_dim, output_dim)
        
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(output_dim, input_dim, num_grids+2*extra+1)
        )
        # nn.init.trunc_normal_(self.spline_weight, mean=0.0, std=spline_weight_init_scale)
        nn.init.kaiming_uniform_(self.spline_weight, a=0, mode='fan_in', nonlinearity='relu')
        # nn.init.xavier_uniform_(self.spline_weight)


    def forward(self, x):
        base = self.base_linear(self.base_activation(x))
        spline_basis = self.lorentzian(self.layernorm(x)) #add layernorm
        #spline_basis = self.lorentzian(x)
        spline = torch.einsum(
            "...in,oin->...o", spline_basis, self.spline_weight
        )
        return base + spline
    

# class FastKAN(nn.Module):
#     def __init__(
#         self,
#         layers_hidden: List[int],
#         grid_range=[0, 1],
#         num_grids: int = 5,
#         base_activation = nn.SiLU,
#         spline_weight_init_scale: float = 0.1,
#         k1=1,
#         k2 = 3,
#     ) -> None:
#         super().__init__()
#         self.layers = nn.ModuleList([
#             FastKANLayer(
#                 in_dim, out_dim,
#                 grid_min=grid_range[0],
#                 grid_max=grid_range[1],
#                 num_grids=num_grids,
#                 base_activation=base_activation,
#                 spline_weight_init_scale=spline_weight_init_scale,
#                 k1=1,
#                 k2 = 3,
#             ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
#         ])

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x


class LorentzianKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_range=[0, 1],
        num_grids: int = 5,
        base_activation = nn.SiLU,
        spline_weight_init_scale: float = 0.1,
        k1=1,
        k2 = 5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            LorentzianKANLayer(
                in_dim, out_dim,
                grid_min=grid_range[0],
                grid_max=grid_range[1],
                num_grids=num_grids,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
                k1 = k1,
                k2 = k2,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def plot_loren(self,x: torch.Tensor,folder="./ext_fuction_figures_lkan"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        layer_count = 0
        for layer in self.layers:
            layer_count += 1
            for i in range(layer.spline_weight.size(0)):

                ################################## w*base should be calculated here ########################################
                wbase = layer.base_activation(x) * layer.base_linear.weight[i] # w*base
                # print(wbase.shape)
                for j in range(layer.spline_weight.size(1)):
                    spline_weight_node = layer.spline_weight[i][j]
                    mul = layer.lorentzian(layer.layernorm(x)) * spline_weight_node.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1),spline_weight_node.size(0))
                    # print(mul.shape)
                    wloren = mul.sum(dim=2) # w*spline
                    # print(wspline.shape)
                    bs = wbase+wloren

                    ######################## can choose to plot w*spline only or w*base+w*spline below ########################################
                    # y = wspline.detach().numpy().T   # plot the w*spline only
                    y = bs.detach().cpu().numpy().T 
                    # y = bs.detach().numpy().T   # plot the w*base+w*spline

                    # plot and save
                    lw = 10
                    color = "black"
                    plt.figure(figsize=(6, 6))
                    plt.gca().patch.set_linewidth(2)
                    plt.plot(range(bs.size(1)), y, color=color, lw=lw)
                    plt.gca().spines[:].set_color(color)
                    plt.xlim(0,bs.size(1))
                    # plt.ylim(np.min(bs.squeeze(0).detach().numpy()),np.max(bs.squeeze(0).detach().numpy()))
                    plt.ylim(-10,10)
                    #print("min & max:", np.min(bs.squeeze(0).detach().numpy()), np.max(bs.squeeze(0).detach().numpy()))
                    print("min & max:", np.min(bs.squeeze(0).detach().cpu().numpy()), np.max(bs.squeeze(0).detach().cpu().numpy()))
                    plt.xticks([])
                    plt.yticks([])
                    ax = plt.gca()
                    ax.spines['top'].set_linewidth(lw)
                    ax.spines['bottom'].set_linewidth(lw)
                    ax.spines['left'].set_linewidth(lw)
                    ax.spines['right'].set_linewidth(lw)
                    # # if sample == True:
                    # #     plt.scatter(self.acts[l][:, i][rank].cpu().detach().numpy(), self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, s=400 * scale ** 2)
                    # # plt.gca().spines[:].set_color(color)
                    plt.savefig(f'{folder}/loren_{layer_count}_{i}_{j}.png', bbox_inches="tight", dpi=400)
                    plt.close()
            x = layer(x)
            # print(x.shape)
            # print(spline_weight.shape)


