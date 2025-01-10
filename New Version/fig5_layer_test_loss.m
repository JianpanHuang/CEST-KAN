clc; close all; clear all;
set(0,'defaultfigurecolor','w')
addpath(genpath(pwd));

Strain1_Mean = [0.303786325  0.297001827  0.416918582  0.431118798  0.477969724]; % MLP
Strain2_Mean = [0.236233547  0.22729069   0.22371006   0.23666825   0.276506979]; % KAN
Strain3_Mean = [0.203476799  0.194361815  0.195245695  0.199661693  0.219216108]; % LKAN

Strain1_std = [0.00385608   0.004286318  0.016722986  0.022204665  0.040374626]; % MLP
Strain2_std = [0.00379287   0.005825404  0.002227273  0.010252188  0.086364211]; % KAN
Strain3_std = [0.001628868  0.000596181  0.000983205  0.003599778  0.013655956]; % LKAN

% 确认数据维度一致
nGroups = length(Strain1_Mean); % 柱状图的组数
nBars = 5; % 栏数

% 使用 cat 函数串联标准差数据
stdData = cat(2, Strain1_std', Strain2_std', Strain3_std'); % 第二维串联
meanData = cat(2, Strain1_Mean', Strain2_Mean', Strain3_Mean'); % 第二维串联

% 绘制带误差条的柱状图
barwitherr(stdData, meanData, 'LineWidth', 2, 'BarWidth', 0.8);

set(gca, 'XTickLabel', {'1','2', '3', '4','5'}); % 修改XTickLabel
legend('MLP', 'KAN', 'LKAN', 'Location', 'northwest');
xlabel('Hidden layer number');
ylabel('Test loss');
grid on;
colormap summer;

set(gca, 'FontWeight', 'BOLD', 'FontSize', 40, 'LineWidth', 3);
outputPath = 'C:\Users\85267\OneDrive\桌面\KAN_new\layer_testloss.jpg';
export_fig(outputPath, '-jpg', '-r300');