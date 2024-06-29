clear all, close all, clc
set(0,'defaultfigurecolor','w')
addpath(genpath(pwd));

load(['test_res',filesep,'test_mlp_labels_2subjects.mat']);
% labels_all = double(labels_all);
load(['test_res',filesep,'test_mlp_outputs_2subjects.mat']);
% outputs_all = double(outputs_all);

mksz = 100;
mkcl = 'b';
mktp = 'o';

linecl = 'r';
linewth = 2;
ftsz = 20;
xlb = 'MPLF';
ylb = 'MLP';
figure, set(gcf,'unit','normalized','Position',[0.01,0.3,0.98,0.5])

ttl = 'A_w_a_t_e_r';
water_mplf = (labels_all(1,:,1))';
water_kan = (outputs_all(1,:,1))';
water_both = [water_mplf;water_kan];
% xyrg = [min(water_both),max(water_both)]
xyrg = [0.600    0.860];
subplot(1,4,1),[corr_coef] = corrplot(water_mplf,water_kan,mksz,mkcl,mktp,linecl,linewth,ftsz,xlb,ylb,ttl,xyrg);
set(gca,"FontWeight",'BOLD','FontSize',20,'LineWidth',3); box on;

ttl = 'A_a_m_i_d_e';
amide_mplf = (labels_all(1,:,4))';
amide_kan = (outputs_all(1,:,4))';
amide_both = [amide_mplf;amide_kan];
% xyrg = [min(amide_both),max(amide_both)]
xyrg = [0.000    0.125];
subplot(1,4,2),[corr_coef] = corrplot(amide_mplf,amide_kan,mksz,mkcl,mktp,linecl,linewth,ftsz,xlb,ylb,ttl,xyrg);
set(gca,"FontWeight",'BOLD','FontSize',20,'LineWidth',3); box on;

ttl = 'A_r_N_O_E';
rnoe_mplf = (labels_all(1,:,6))';
rnoe_kan = (outputs_all(1,:,6))';
rnoe_both = [rnoe_mplf;rnoe_kan];
% xyrg = [min(rnoe_both),max(rnoe_both)]
xyrg = [0.000    0.15];
subplot(1,4,3),[corr_coef] = corrplot(rnoe_mplf,rnoe_kan,mksz,mkcl,mktp,linecl,linewth,ftsz,xlb,ylb,ttl,xyrg);
set(gca,"FontWeight",'BOLD','FontSize',20,'LineWidth',3); box on;

ttl = 'A_M_T';
mt_mplf = (labels_all(1,:,8))';
mt_kan = (outputs_all(1,:,8))';
mt_both = [mt_mplf;mt_kan];
% xyrg = [min(mt_both),max(mt_both)]
xyrg = [ 0.00    0.240];
subplot(1,4,4),[corr_coef] = corrplot(mt_mplf,mt_kan,mksz,mkcl,mktp,linecl,linewth,ftsz,xlb,ylb,ttl,xyrg);
set(gca,"FontWeight",'BOLD','FontSize',20,'LineWidth',3,'XTick',[min(xyrg):0.05:max(xyrg)],'XTickLabelRotation',0); box on;

% export_fig('fig4_corrplot_mlp','-jpg','-r200')
