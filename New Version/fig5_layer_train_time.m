clc; close all; clear all;
set(0,'defaultfigurecolor','w')
addpath(genpath(pwd));

Strain1_Mean=[1114.8  1073.8  1198  1208.4  1327.6    ]; % MLP
Strain2_Mean=[2293.2  3068.8  3661.2  4145.4  3649.5 ]; % KAN
Strain3_Mean=[1643.8  3345.2  3443.8  4081.4  4756    ]; % LKAN

Strain1_std=[116.3752551  53.34510287  92.26862956  35.19659074  64.04919984  ]; % MLP
Strain2_std=[109.1842479  146.0126707  1161.338323  1356.759116  1998.301028 ]; % KAN
Strain3_std=[87.93577202  672.7872621  128.1939936  292.7358878  346.9423583  ]; % LKAN

% 确认数据维度一致
nGroups = length(Strain1_Mean); % 柱状图的组数
nBars = 5; % 栏数

% 使用 cat 函数串联标准差数据
stdData = cat(2, Strain1_std', Strain2_std', Strain3_std'); % 第二维串联
meanData = cat(2, Strain1_Mean', Strain2_Mean', Strain3_Mean'); % 第二维串联

% 绘制带误差条的柱状图
barwitherr(stdData, meanData, 'LineWidth', 2, 'BarWidth', 0.8);

set(gca,'XTickLabel',{'1','2','3','4','5','6','7'})
    legend('MLP','KAN','LKAN',Location='northwest')
    xlabel('Hidden layer number')
    ylabel('Training time (s)')
    ylim([0,5900])
    grid on
    colormap summer

set(gca,"FontWeight",'BOLD','FontSize',40,'LineWidth',3)
% export_fig('fig2_train_time','-jpg','-r300')