clc; close all; clear all;
set(0,'defaultfigurecolor','w')
addpath(genpath(pwd));

Strain1_Mean=[0.964048319  0.856250855  0.719549926  0.681013116  0.618766981 ]; % MLP
Strain2_Mean=[0.993757802  0.971575411  0.965359852  0.909588332  0.704969137 ]; % KAN
Strain3_Mean=[0.993589854  0.985430372  0.977177032  0.949378186  0.909369688 ]; % LKAN

Strain1_std=[0.00689504   0.042189181  0.141706288   0.070783636  0.048419994 ]; % MLP
Strain2_std=[0.001037225  0.00521505   0.002906629   0.047642218  0.143376549 ]; % KAN
Strain3_std=[0.000421259  0.000619481  0.00108101    0.009587122  0.025864194 ]; % LKAN

% 确认数据维度一致
nGroups = length(Strain1_Mean); % 柱状图的组数
nBars = 5; % 栏数

% 使用 cat 函数串联标准差数据
stdData = cat(2, Strain1_std', Strain2_std', Strain3_std'); % 第二维串联
meanData = cat(2, Strain1_Mean', Strain2_Mean', Strain3_Mean'); % 第二维串联

% 绘制带误差条的柱状图
barwitherr(stdData, meanData, 'LineWidth', 2, 'BarWidth', 0.8);
set(gca,'XTickLabel',{'1','2','3','4','5'})
    legend('MLP','KAN','LKAN',Location='northwest')
    xlabel('Hidden layer number')     
    ylabel('R_a_v_e')
    ylim([0,1.5])
    grid on
    colormap summer

set(gca,"FontWeight",'BOLD','FontSize',40,'LineWidth',3)
% export_fig('fig2_epoch','-jpg','-r300')