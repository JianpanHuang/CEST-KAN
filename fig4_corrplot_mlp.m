clear all, close all, clc
set(0,'defaultfigurecolor','w')

load('test_mlp_labels_2Subjects.mat');
load('test_mlp_outputs_2Subjects.mat');
mksz = 100;
mkcl = 'b';
mktp = 'o';

linecl = 'r';
linewth = 2;
ftsz = 15;
xlb = 'MPLF';
ylb = 'MLP';
figure, set(gcf,'unit','normalized','Position',[0.01,0.3,0.98,0.5])

% Subplot 1
ttl = '\DeltaB_0';
mt_mplf = (labels_all(1,:,3))';
mt_kan = (outputs_all(1,:,3))';
subplot(1,5,1),[corr_coef] = corrplot(mt_mplf,mt_kan,mksz,mkcl,mktp,linecl,linewth,ftsz,xlb,ylb,ttl,[-0.22, 0.8]);
set(gca,"FontWeight",'BOLD','FontSize',16,'LineWidth',3); box on;
set(gca, 'XTick', [0.28, 0.8]);
set(gca, 'YTick', [0.28, 0.8]);

% Subplot 2
ttl = 'A_w_a_t_e_r';
water_mplf = (labels_all(1,:,1))';
water_kan = (outputs_all(1,:,1))';
subplot(1,5,2),[corr_coef] = corrplot(water_mplf,water_kan,mksz,mkcl,mktp,linecl,linewth,ftsz,xlb,ylb,ttl,[0.6190, 0.8634]);
set(gca,"FontWeight",'BOLD','FontSize',16,'LineWidth',3); box on;
set(gca, 'XTick', [0.74, 0.86]);
set(gca, 'YTick', [0.74, 0.86]);


% Subplot 3
ttl = 'A_a_m_i_d_e';
amide_mplf = (labels_all(1,:,4))';
amide_kan = (outputs_all(1,:,4))';
subplot(1,5,3),[corr_coef] = corrplot(amide_mplf,amide_kan,mksz,mkcl,mktp,linecl,linewth,ftsz,xlb,ylb,ttl,[0 0.12]);
set(gca,"FontWeight",'BOLD','FontSize',16,'LineWidth',3); box on;
set(gca, 'XTick', [0.06, 0.12]);
set(gca, 'YTick', [0, 0.06, 0.12]);


% Subplot 4
ttl = 'A_r_N_O_E';
rnoe_mplf = (labels_all(1,:,6))';
rnoe_kan = (outputs_all(1,:,6))';
subplot(1,5,4),[corr_coef] = corrplot(rnoe_mplf,rnoe_kan,mksz,mkcl,mktp,linecl,linewth,ftsz,xlb,ylb,ttl,[0.0093, 0.1510]);
set(gca,"FontWeight",'BOLD','FontSize',16,'LineWidth',3); box on;
set(gca, 'XTick', [0, 0.08, 0.15]);
set(gca, 'YTick', [0, 0.08, 0.15]);

% Subplot 5
ttl = 'A_M_T';
mt_mplf = (labels_all(1,:,8))';
mt_kan = (outputs_all(1,:,8))';
subplot(1,5,5),[corr_coef] = corrplot(mt_mplf,mt_kan,mksz,mkcl,mktp,linecl,linewth,ftsz,xlb,ylb,ttl,[-0.0027, 0.2354]);
set(gca,"FontWeight",'BOLD','FontSize',16,'LineWidth',3); box on;
set(gca, 'XTick', [0.1, 0.23]);
set(gca, 'YTick', [0.1, 0.23]);
