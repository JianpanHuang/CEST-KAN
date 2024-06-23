clc; close all; clear all;
set(0,'defaultfigurecolor','w')
addpath(genpath(pwd));

Strain1_Mean=[112.9	110.3	103.2	104.9]; % MLP
Strain2_Mean=[104.1	98.1	99.4	98.7]; % KAN

Strain1_std=[7.445356495	6.864562784	5.473166867	6.854844191]; % MLP
Strain2_std=[5.42524961	4.863697725	3.717824932	4.854551129]; % KAN

barwitherr(cat(3,zeros(4,2),[Strain1_std' Strain2_std'...
    ]),[1 2 3 4],[Strain1_Mean' Strain2_Mean'...
    ],'LineWidth',2,...
    'BarWidth',0.8)

set(gca,'XTickLabel',{'1','2','3','4'})
    legend('MLP','KAN',Location='northwest')
    xlabel('Hidden layer number')
    ylabel('Training epoch')
    ylim([0,180])
    grid on
    colormap summer

set(gca,"FontWeight",'BOLD','FontSize',20,'LineWidth',3)
% export_fig('fig2_epoch','-jpg','-r300')