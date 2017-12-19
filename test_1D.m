%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Test script for 1D smooth simulation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc; clf;
addpath(genpath(pwd)); warning off

%% Set up underlying hyperparameters
nsample = 500;
nd = 1000; % number of dims
sqrt_rho_true = 6; rho_true = sqrt_rho_true^2; % marginal variance
delta_true = 50; % length scale (stdev of GP kernel)
b_true = -8; % mean (DC term)
log_nsevar_true = 3; % log of noise variance
nsevar_true = exp(log_nsevar_true);
l_true = 50; % length scale for smoothness
hyper_true = [rho_true; delta_true; log_nsevar_true; l_true]; % underlying hyperparameter set

% bounds for hypers
mindelta = 1; maxdelta = min(nd); minl = 2; maxl = min(nd);
lb = [0.001;mindelta;-50;minl]; ub = [1e5;maxdelta;10;maxl]; % bounds for [rho, delta, log_nsevar len]

%% Generate u from GP kernel
% Generate diagonal of Fourier-defined SE covariance
cond = 1e12^(1/numel(nd)); % condthresh for small eigenvalues
[logkdiag, wvec_true, Bfft_true] = mkcov_logASDfactored_nD(rho_true,delta_true,nd,max([mindelta,delta_true*0.8]),nd(:),cond);
kdiag = exp(logkdiag);
ndcirc_true = length(logkdiag);
iiDC = find(wvec_true==0);
DCmult = sqrt(prod(nd)); % factor to multiply by dc term

% Generate u from the kernel covariance
v = randn(ndcirc_true,1);  % sample v from Normal distribution
bvec = sparse(ndcirc_true,1);
bvec(iiDC) = b_true*DCmult;
ufreq = v.*sqrt(kdiag)+bvec; % get ufreq from v by ufreq=sqrt(kdiag).*v+bp
u_true = fft2real(ufreq, Bfft_true); % transform u from freq to real

% Get c from nonlinear transform of u
opt.nonlinearity = 'rec'; % set the nonlinearity to be log(1+exp(x))
c_true = nonlinear_u(u_true,opt); % get c

% Hard threshold small c_true to be 0 to achieve strict sparsity
iikeep = abs(c_true)>5e-3;
c_true(~iikeep) = 0;

% Plot u and c
subplot(231), plot(u_true), title('u\_true'), drawnow
subplot(232), plot(c_true), title('c\_true'), drawnow

%% gen smooth kernel
cond = 1e12^(1/numel(nd));
[logfdiag, ~, Bfft_f_true] = mkcov_logASDfactored_nD(1,l_true,nd,max([minl,l_true*0.8]),nd(:),cond);
fdiag = exp(logfdiag);
cf_true = fdiag;
d_f = length(cf_true);
% Kf = fft2real(fft2real(diag(fdiag), Bfft_f_true)',Bfft_f_true);
subplot(233), plot(cf_true), title('cf\_true'), axis tight, drawnow

%% Generate data samples
x_true = randn(prod(nd),nsample)'; % generate x from Normal distribution
noise = randn(nsample,1)*0.2; % generate noise from Normal distribution
w_true_f = randn(prod(d_f),1).*sqrt(cf_true);
w_true = fft2real(w_true_f,Bfft_f_true).*sqrt(c_true);
y_true = x_true*w_true+noise*sqrt(nsevar_true);

truth.w_true = w_true;
truth.c_true = c_true;
truth.u_true = u_true;

% Plot w and data
subplot(223), plot(w_true), title('w\_true'), drawnow
subplot(224), plot(y_true, x_true*w_true, 'o');
xlabel('y\_true'), xlabel('x\_true*w\_true'), drawnow
title('y vs x*w'), drawnow

%%
% Split data into training set and test set
ind = randperm(length(y_true));
c = cvpartition(ind,'HoldOut',0.2);
ytrain = y_true(c.training,:);
Xtrain = x_true(c.training,:);
ytest = y_true(c.test,:);
Xtest = x_true(c.test,:);

% Collect data into datastruct
datastruct.x = Xtrain;
datastruct.y = ytrain;
datastruct.xtest = Xtest;
datastruct.ytest = ytest;
datastruct.xx = Xtrain*Xtrain';
datastruct.xy = Xtrain'*ytrain;
datastruct.yy = ytrain'*ytrain;
datastruct.nd = nd;
datastruct.ny = size(Xtrain,1);

% Construct measurement for evaluation
mse_tr = @(a) 1-sum((datastruct.y-a).^2)/sum((datastruct.y-mean(datastruct.y)).^2); % train r2
mse_te = @(a) 1-sum((datastruct.ytest-a).^2)/sum((datastruct.ytest-mean(datastruct.ytest)).^2); % test r2
mse_w = @(a) 1-sum((w_true-a).^2)/sum((w_true-mean(w_true)).^2); % w r2

%% RIDGE
tic;
[kridge,hprs] = autoRidgeRegress_graddual(datastruct);
toc;
figure(1),subplot(521),cla,
plot(1:length(kridge),w_true,'b')
hold on, plot(1:length(kridge),kridge,'r','linewidth',1.5),hold off
title(sprintf('ridge: wR2=%.2f, trainR2=%.2f, testR2=%.2f, nonzero=%.2f%%', ...
    mse_w(kridge),mse_tr(Xtrain*kridge),mse_te(Xtest*kridge),100*sum(abs(kridge)>1e-4)/prod(nd)));
drawnow

%% LASSO
tic;
[klasso,lambda_lasso] = runLASSO(Xtrain, ytrain);
toc;
figure(1),subplot(522),cla,
plot(1:length(klasso),w_true,'b')
hold on, plot(1:length(klasso),klasso,'r','linewidth',1.5),hold off
title(sprintf('lasso: wR2=%.2f, trainR2=%.2f, testR2=%.2f, nonzero=%.2f%%', ...
    mse_w(klasso),mse_tr(Xtrain*klasso),mse_te(Xtest*klasso),100*sum(abs(klasso)>1e-4)/prod(nd)));
drawnow

%% ARD
% fixed point ARD
kard_fp = runARDfull_prior(1e4,Xtrain,ytrain,0,'gamma');
figure(1),subplot(523),cla,
plot(1:length(kard_fp),w_true,'b')
hold on, plot(1:length(kard_fp),kard_fp,'r','linewidth',1.5),hold off
title(sprintf('fix point ard: wR2=%.2f, trainR2=%.2f, testR2=%.2f, nonzero=%.2f%%', ...
    mse_w(kard_fp),mse_tr(Xtrain*kard_fp),mse_te(Xtest*kard_fp),100*sum(abs(kard_fp)>1e-4)/prod(nd)));
drawnow

% SBL ARD
[kard_sbl,dmu,k,gamma3,nsevar3] = sparse_learning_lambda(Xtrain,ytrain,1e3,1e3,0,0,1,1);
figure(1),subplot(524),cla,
plot(1:length(kard_sbl),w_true,'b')
hold on, plot(1:length(kard_sbl),kard_sbl,'r','linewidth',1.5),hold off
title(sprintf('SBL ard: wR2=%.2f, trainR2=%.2f, testR2=%.2f, nonzero=%.2f%%', ...
    mse_w(kard_sbl),mse_tr(Xtrain*kard_sbl),mse_te(Xtest*kard_sbl),100*sum(abs(kard_sbl)>1e-4)/prod(nd)));
drawnow

%% DRD
figure(2)
[kdrd, cdrd, hypers_estimation_drd, w_dif_drd, sq_er_drd] = runDRD([],datastruct,lb,ub,200,0,truth);

figure(1),subplot(525),cla,
plot(1:length(kdrd),w_true,'b')
hold on, plot(1:length(kdrd),kdrd,'r','linewidth',1.5),hold off
title(sprintf('drd: wR2=%.2f, trainR2=%.2f, testR2=%.2f, nonzero=%.2f%%', ...
    mse_w(kdrd),mse_tr(Xtrain*kdrd),mse_te(Xtest*kdrd),100*sum(abs(kdrd)>1e-4)/prod(nd)));
drawnow

%% DRD convex
figure(2)
[kdrd_convex, cdrd_convex, hypers_estimation_drd_convex, w_dif_convex, sq_er_convex] = runDRD_convex([],datastruct,lb,ub,200,0,truth);

figure(1),subplot(526),cla,
plot(1:length(kdrd_convex),w_true,'b')
hold on, plot(1:length(kdrd_convex),kdrd_convex,'r','linewidth',1.5),hold off
title(sprintf('drd convex: wR2=%.2f, trainR2=%.2f, testR2=%.2f, nonzero=%.2f%%', ...
    mse_w(kdrd_convex),mse_tr(Xtrain*kdrd_convex),mse_te(Xtest*kdrd_convex),100*sum(abs(kdrd_convex)>1e-4)/prod(nd)));
drawnow

%% DRD asd
cdrd_half = sqrt(abs(cdrd));
Xtrainf = bsxfun(@times,Xtrain,cdrd_half');
Xtestf = bsxfun(@times,Xtest,cdrd_half');
tic;
[kasd,ASDstats,dd] = fastASD(Xtrainf,ytrain,nd,minl);
toc;
kasd = kasd.*cdrd_half;

figure(1),subplot(527),cla,
plot(1:length(kasd),w_true,'b')
hold on, plot(1:length(kasd),kasd,'r','linewidth',1.5),hold off
title(sprintf('drd asd: wR2=%.2f, trainR2=%.2f, testR2=%.2f, nonzero=%.2f%%', ...
    mse_w(kasd),mse_tr(Xtrain*kasd),mse_te(Xtest*kasd),100*sum(abs(kasd)>1e-4)/prod(nd)));
drawnow

%% sDRD
figure(2)
hh = sum(abs(hypers_estimation_drd),2); ii = find(hh~=0); ii = ii(end);
prs0 = [mean(hypers_estimation_drd(max([2,ii-10]):ii,:),1) 10];
[ksdrd, csdrd, hypers_estimation_sdrd, w_dif_sdrd, sq_er_sdrd] = runsDRD(prs0,datastruct,lb,ub,200,0,truth);

figure(1),subplot(528),cla,
plot(1:length(ksdrd),w_true,'b')
hold on, plot(1:length(ksdrd),ksdrd,'r','linewidth',1.5),hold off
title(sprintf('sdrd: wR2=%.2f, trainR2=%.2f, testR2=%.2f, nonzero=%.2f%%', ...
    mse_w(ksdrd),mse_tr(Xtrain*ksdrd),mse_te(Xtest*ksdrd),100*sum(abs(ksdrd)>1e-4)/prod(nd)));
drawnow

%% DRD mcmc
figure(2)
[wdrd_mcmc, udrd_mcmc, hypers_estimation_drd_mcmc, w_dif_mcmc, sq_er_mcmc] = runDRD_mcmc([],datastruct,lb,ub,2000,0,truth);
st = 1;
kdrd_mcmc = mean(wdrd_mcmc(st:end,:),1)';
cdrd_mcmc = mean(nonlinear_u(udrd_mcmc(st:end,:),opt,inf),1)';

figure(1),subplot(5,2,9),cla,
plot(1:length(kdrd_mcmc),w_true,'b')
hold on, plot(1:length(kdrd_mcmc),kdrd_mcmc,'r','linewidth',1.5),hold off
title(sprintf('drd mcmc: wR2=%.2f, trainR2=%.2f, testR2=%.2f, nonzero=%.2f%%', ...
    mse_w(kdrd_mcmc),mse_tr(Xtrain*kdrd_mcmc),mse_te(Xtest*kdrd_mcmc),100*sum(abs(kdrd_mcmc)>1e-4)/prod(nd)));
drawnow

%% sDRD mcmc
figure(2)
[wsdrd_mcmc, usdrd_mcmc, hypers_estimation_sdrd_mcmc, w_dif_sdrd_mcmc, sq_er_sdrd_mcmc] = runsDRD_mcmc([],datastruct,lb,ub,2000,0,truth);
st = 1;
ksdrd_mcmc = mean(wsdrd_mcmc(st:end,:),1)';
csdrd_mcmc = mean(nonlinear_u(usdrd_mcmc(st:end,:),opt,inf),1)';

figure(1),subplot(5,2,10),cla,
plot(1:length(ksdrd_mcmc),w_true,'b')
hold on, plot(1:length(ksdrd_mcmc),ksdrd_mcmc,'r','linewidth',1.5),hold off
title(sprintf('sdrd mcmc: wR2=%.2f, trainR2=%.2f, testR2=%.2f, nonzero=%.2f%%', ...
    mse_w(ksdrd_mcmc),mse_tr(Xtrain*ksdrd_mcmc),mse_te(Xtest*ksdrd_mcmc),100*sum(abs(ksdrd_mcmc)>1e-4)/prod(nd)));
drawnow

%% plot all weights
plot_allw