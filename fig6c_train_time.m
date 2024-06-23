clc; close all; clear all;
set(0,'defaultfigurecolor','w')
addpath(genpath(pwd));

Strain1_Mean=[18.27552443	19.86126504	 20.41229726	21.79406858]; % MLP
Strain2_Mean=[40.16283357	54.31492074	 71.53414051	86.95580303]; % KAN

Strain1_std=[1.475797905	1.282441055	 1.077583078	1.33467682]; % MLP
Strain2_std=[1.641527364	3.211071502	 3.863015831	4.327614965]; % KAN

barwitherr(cat(3,zeros(4,2),[Strain1_std' Strain2_std'...
    ]),[1 2 3 4],[Strain1_Mean' Strain2_Mean'...
    ],'LineWidth',2,...
    'BarWidth',0.8)

set(gca,'XTickLabel',{'1','2','3','4'})
    legend('MLP','KAN',Location='northwest')
    xlabel('Hidden layer number')
    ylabel('Training time (min)')
    grid on
    colormap summer

set(gca,"FontWeight",'BOLD','FontSize',20,'LineWidth',3)
% export_fig('fig2_train_time','-jpg','-r300')