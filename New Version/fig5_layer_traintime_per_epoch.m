clc; close all; clear all;
set(0,'defaultfigurecolor','w')
addpath(genpath(pwd));

Strain1_Mean=[14.80303127	15.88407699   17.06442139	18.47845792  19.8714205   ]; % MLP
Strain2_Mean=[34.22972578	47.79355093   61.02952931	74.86161708  92.09141795  ]; % KAN
Strain3_Mean=[25.60185125	36.41164975	  37.76638792	44.94498136  50.23341081  ]; % LKAN

Strain1_std=[0.559523814	0.079199160   0.088118411	0.092009707  0.11631499  ]; % MLP
Strain2_std=[0.186781421	0.249601253   0.454317194	0.561412005  1.505649077 ]; % KAN
Strain3_std=[0.171291137	6.978444655   0.395801151	2.810155112  0.2442383   ]; % LKAN

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
    ylabel('Training time per epoch (s)')
    ylim([0,100])
    grid on
    colormap summer

set(gca,"FontWeight",'BOLD','FontSize',40,'LineWidth',3)
% export_fig('fig2_epoch','-jpg','-r300')
