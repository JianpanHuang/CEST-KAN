clc; close all; clear all;
set(0,'defaultfigurecolor','w')
addpath(genpath(pwd));

Strain1_Mean = [0.427697259  0.445732099  0.496796954  0.544447017  0.670553231  0.719729674]; % MLP
Strain2_Mean = [0.390697795  0.423966479  0.466515428  0.523419046  0.640120065  0.720528603]; % KAN
Strain3_Mean = [0.372836328  0.420498234  0.467553061  0.524512744  0.640226555  0.716791666]; % LKAN

Strain1_std = [0.004286318  0.016722986  0.022204665  0.006977467  0.025178275  0.000441128]; % MLP
Strain2_std = [0.001716021  0.0009434    0.00057032   0.000238644  0.000281017  0.000795995]; % KAN
Strain3_std = [0.000742633  0.001011382  0.000344215  0.000841567  0.000427055  0.000581468]; % LKAN

% 确认数据维度一致
nGroups = length(Strain1_Mean); % 柱状图的组数
nBars = 6; % 栏数

% 使用 cat 函数串联标准差数据
stdData = cat(2, Strain1_std', Strain2_std', Strain3_std'); % 第二维串联
meanData = cat(2, Strain1_Mean', Strain2_Mean', Strain3_Mean'); % 第二维串联

% 绘制带误差条的柱状图
barwitherr(stdData, meanData, 'LineWidth', 2, 'BarWidth', 0.8);

set(gca, 'XTickLabel', {'0.01', '0.02', '0.05', '0.1', '0.2', '0.3'}); % 修改XTickLabel
legend('MLP', 'KAN', 'LKAN', 'Location', 'northwest');
xlabel('Noise level');
ylabel('Test loss');
grid on;
colormap summer;

set(gca, 'FontWeight', 'BOLD', 'FontSize', 40, 'LineWidth', 3);
outputPath = 'C:\Users\85267\OneDrive\桌面\Kan\draw_figure\Noise level2.jpg';
export_fig(outputPath, '-jpg', '-r300');