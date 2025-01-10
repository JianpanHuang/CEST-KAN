clear all; 
close all; 
clc;
addpath(genpath(pwd));

% 文件名和数据读取
filename = 'zspec_test_ss1';
data = csvread([filename, '.csv']);

% 噪声添加
sigma = 0.01; % 噪声方差
noise = 0 + sigma * randn(size(data));
data_noise = data + noise;

% 保存带噪声的数据
csvwrite([filename, '_', num2str(sigma), '.csv'], data_noise);

% 创建图形并设置大小
figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.5, 0.65]); % 调整图形大小

% 绘制数据
plot(data_noise(100, :),'LineWidth', 2);


% 完整的 x 坐标映射
full_x_labels = [
    -20.0, -15.0, -10.0, -8.0, -6.0, -4.0, -3.75, ...
    -3.5, -3.25, -3.0, -2.75, -2.50, -2.25, -2.0, ...
    -1.75, -1.50, -1.25, -1.0, -0.75, -0.50, -0.25, ...
    0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, ...
    2.0, 2.25, 2.50, 2.75, 3.0, 3.25, 3.5, 3.75, ...
    4.0, 6.0, 8.0, 10.0, 15.0, 20.0
];

% 需要展示的特定 x 坐标值
selected_x_labels = [
    -20.0, -8.0, -3.5, -2.0, ...
    0.0, 2.0, 3.5, 8, 20.0
];

% 对应的索引
selected_indices = find(ismember(full_x_labels, selected_x_labels));

% 设置 x 轴的刻度和标签
xticks(selected_indices); % 选择的 x 轴刻度
xticklabels(selected_x_labels); % 使用特定的 x 轴标签


% 设置坐标刻度的字体大小
% 添加纵坐标标题（如果需要）
ylabel('Z', 'FontSize', 16); % 增加字体大小
set(gca, 'FontSize', 22, 'linewidth',2); % 增加坐标轴字体大小
% 添加横坐标标题
xlabel('Frequency offsets (ppm)', 'FontSize', 24); % 增加字体大小
%title('(F) noise level = 0.3', 'FontSize', 24); % 更新标题并设置字号

% 调整 x 轴范围 (可选)
xlim([1, length(data_noise(100, :))]);