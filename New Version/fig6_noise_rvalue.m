clc; close all; clear all;
set(0,'defaultfigurecolor','w')
addpath(genpath(pwd));

Strain1_Mean = [0.950918291  0.936622296  0.893286426  0.829670309  0.674261706  0.61254806]; % MLP
Strain2_Mean = [0.963034133  0.944929901  0.904378349  0.840490701  0.727646165  0.623457204]; % KAN
Strain3_Mean = [0.960820609  0.94247135   0.90252578   0.839697631  0.728055225  0.627817685]; % LKAN

Strain1_std = [0.000573027  0.001398994  0.005262603  0.003034667  0.068108855  0.001116457]; % MLP
Strain2_std = [0.000675549  0.000333526  0.000553208  0.000922734  0.000425387  0.001906552]; % KAN
Strain3_std = [0.000793975  0.000773573  0.000311436  0.000531878  0.001009486  0.000740082]; % LKAN

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
ylabel('R_a_v_e');
ylim([0,1.5])
grid on;
colormap summer;

set(gca, 'FontWeight', 'BOLD', 'FontSize', 40, 'LineWidth', 3);
outputPath = 'C:\Users\85267\OneDrive\桌面\Kan\draw_figure\Noise level2.jpg';
export_fig(outputPath, '-jpg', '-r300');