clc; close all; clear all;
set(0,'defaultfigurecolor','w')
addpath(genpath(pwd));

Strain1_Mean=[0.46406162	0.399082435	0.447990079	0.594469611]; % MLP
Strain2_Mean=[0.239878407	0.218563744	0.223156486	0.238621384]; % KAN

Strain1_std=[0.014179679	0.014357783	0.057686194	0.024977312]; % MLP
Strain2_std=[0.002079655	0.004728543	0.007045942	0.009031535]; % KAN

barwitherr(cat(3,zeros(4,2),[Strain1_std' Strain2_std'...
    ]),[1 2 3 4],[Strain1_Mean' Strain2_Mean'...
    ],'LineWidth',2,...
    'BarWidth',0.8)

set(gca,'XTickLabel',{'1','2','3','4'})
    legend('MLP','KAN',Location='northwest')
    xlabel('Hidden layer number')
    ylabel('Validation loss')
    grid on
    colormap summer 

set(gca,"FontWeight",'BOLD','FontSize',20,'LineWidth',3)
% export_fig('fig2_val_loss_2024','-jpg','-r300')