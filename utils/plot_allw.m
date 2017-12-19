% function plot_allw()
figure(1),clf

%% RIDGE
subplot(521),cla,
plot(1:length(kridge),w_true,'b')
hold on, plot(1:length(kridge),kridge,'r','linewidth',1.5),hold off
title(sprintf('ridge: wr2=%.2f, trr2=%.2f, ter2=%.2f, nz=%.2f%%', ...
    mse_w(kridge),mse_tr(Xtrain*kridge),mse_te(Xtest*kridge),100*sum(abs(kridge)>1e-4)/prod(nd)));
drawnow
set(gca,'fontsize',9)

%% LASSO
subplot(522),cla,
plot(1:length(klasso),w_true,'b')
hold on, plot(1:length(klasso),klasso,'r','linewidth',1.5),hold off
title(sprintf('lasso: wr2=%.2f, trr2=%.2f, ter2=%.2f, nz=%.2f%%', ...
    mse_w(klasso),mse_tr(Xtrain*klasso),mse_te(Xtest*klasso),100*sum(abs(klasso)>1e-4)/prod(nd)));
drawnow
set(gca,'fontsize',9)

%% ARD
% fixed point ARD
subplot(523),cla,
plot(1:length(kard_fp),w_true,'b')
hold on, plot(1:length(kard_fp),kard_fp,'r','linewidth',1.5),hold off
title(sprintf('fp ard: wr2=%.2f, trr2=%.2f, ter2=%.2f, nz=%.2f%%', ...
    mse_w(kard_fp),mse_tr(Xtrain*kard_fp),mse_te(Xtest*kard_fp),100*sum(abs(kard_fp)>1e-4)/prod(nd)));
drawnow
set(gca,'fontsize',9)

% SBL ARD
subplot(524),cla,
plot(1:length(kard_sbl),w_true,'b')
hold on, plot(1:length(kard_sbl),kard_sbl,'r','linewidth',1.5),hold off
title(sprintf('SBL ard: wr2=%.2f, trr2=%.2f, ter2=%.2f, nz=%.2f%%', ...
    mse_w(kard_sbl),mse_tr(Xtrain*kard_sbl),mse_te(Xtest*kard_sbl),100*sum(abs(kard_sbl)>1e-4)/prod(nd)));
drawnow
set(gca,'fontsize',9)

%% DRD
subplot(525),cla,
plot(1:length(kdrd),w_true,'b')
hold on, plot(1:length(kdrd),kdrd,'r','linewidth',1.5),hold off
title(sprintf('drd: wr2=%.2f, trr2=%.2f, ter2=%.2f, nz=%.2f%%', ...
    mse_w(kdrd),mse_tr(Xtrain*kdrd),mse_te(Xtest*kdrd),100*sum(abs(kdrd)>1e-4)/prod(nd)));
drawnow
set(gca,'fontsize',9)

%% DRD convex
subplot(526),cla,
plot(1:length(kdrd_convex),w_true,'b')
hold on, plot(1:length(kdrd_convex),kdrd_convex,'r','linewidth',1.5),hold off
title(sprintf('drd convex: wr2=%.2f, trr2=%.2f, ter2=%.2f, nz=%.2f%%', ...
    mse_w(kdrd_convex),mse_tr(Xtrain*kdrd_convex),mse_te(Xtest*kdrd_convex),100*sum(abs(kdrd_convex)>1e-4)/prod(nd)));
drawnow
set(gca,'fontsize',9)

%% DRD asd
subplot(527),cla,
plot(1:length(kasd),w_true,'b')
hold on, plot(1:length(kasd),kasd,'r','linewidth',1.5),hold off
title(sprintf('drd asd: wr2=%.2f, trr2=%.2f, ter2=%.2f, nz=%.2f%%', ...
    mse_w(kasd),mse_tr(Xtrain*kasd),mse_te(Xtest*kasd),100*sum(abs(kasd)>1e-4)/prod(nd)));
drawnow
set(gca,'fontsize',9)

%% sDRD
subplot(528),cla,
plot(1:length(ksdrd),w_true,'b')
hold on, plot(1:length(ksdrd),ksdrd,'r','linewidth',1.5),hold off
title(sprintf('sdrd: wr2=%.2f, trr2=%.2f, ter2=%.2f, nz=%.2f%%', ...
    mse_w(ksdrd),mse_tr(Xtrain*ksdrd),mse_te(Xtest*ksdrd),100*sum(abs(ksdrd)>1e-4)/prod(nd)));
drawnow
set(gca,'fontsize',9)

%% DRD mcmc
subplot(5,2,9),cla,
plot(1:length(kdrd_mcmc),w_true,'b')
hold on, plot(1:length(kdrd_mcmc),kdrd_mcmc,'r','linewidth',1.5),hold off
title(sprintf('drd mcmc: wr2=%.2f, trr2=%.2f, ter2=%.2f, nz=%.2f%%', ...
    mse_w(kdrd_mcmc),mse_tr(Xtrain*kdrd_mcmc),mse_te(Xtest*kdrd_mcmc),100*sum(abs(kdrd_mcmc)>1e-4)/prod(nd)));
drawnow
set(gca,'fontsize',9)

%% sDRD mcmc
subplot(5,2,10),cla,
plot(1:length(ksdrd_mcmc),w_true,'b')
hold on, plot(1:length(ksdrd_mcmc),ksdrd_mcmc,'r','linewidth',1.5),hold off
title(sprintf('sdrd mcmc: wr2=%.2f, trr2=%.2f, ter2=%.2f, nz=%.2f%%', ...
    mse_w(ksdrd_mcmc),mse_tr(Xtrain*ksdrd_mcmc),mse_te(Xtest*ksdrd_mcmc),100*sum(abs(ksdrd_mcmc)>1e-4)/prod(nd)));
drawnow
set(gca,'fontsize',9)
