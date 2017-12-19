%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Test script for a 3D real fMRI data from Haxby2001. The example data is a
% 2-class classification problem.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc,clear,clf,
addpath(genpath(pwd)); warning off

data = load(['fmri_data.mat']);

Xtrain = double(data.Xtrain0);
ytrain = double(data.Ytrain0');
Xtest = double(data.Xtest0);
ytest = double(data.Ytest0');
maskdata = data.maskdata; % mask for 3D volume
anat = data.anat; % mask for 3D volume

opt.iikeep = logical(maskdata(:));
opt.nonlinearity = 'rec'; % set the nonlinearity to be log(1+exp(x))

Xtrain = permute(Xtrain,[4,1,2,3]);
Xtest = permute(Xtest,[4,1,2,3]);
[n,nx,ny,nz] = size(Xtrain);
Xtrain = reshape(Xtrain,[n,prod([nx,ny,nz])]);
[n,nx,ny,nz] = size(Xtest);
Xtest = reshape(Xtest,[n,prod([nx,ny,nz])]);
Xtrain = Xtrain(:,opt.iikeep);
Xtest = Xtest(:,opt.iikeep);

Xtrain0 = bsxfun(@minus,Xtrain,mean(Xtrain,1));
ytrain0 = bsxfun(@minus,ytrain,mean(ytrain,1))*100;
Xtest0 = bsxfun(@minus,Xtest,mean(Xtest,1));
ytest0 = bsxfun(@minus,ytest,mean(ytest,1));
acc = @(yy,xw) sum(sign(yy)==sign(xw))/length(yy)*100;

[nx,ny,nz] = size(maskdata);
nd = [nx,ny,nz];
datastruct.x = Xtrain0;
datastruct.y = ytrain0;
datastruct.xtest = Xtest0;
datastruct.ytest = ytest0;
datastruct.xy = Xtrain0'*ytrain0;
datastruct.yy = ytrain0'*ytrain0;
datastruct.nd = nd;
datastruct.ny = size(Xtrain0,1);
datastruct.maskdata = maskdata;

%% LASSO
tic;
[klasso,lambda_lasso] = runLASSO(Xtrain0, ytrain0);
toc;

% accuracy
acc_te_lasso = acc(ytest0,Xtest0*klasso)

klasso_3d = zeros(prod(nd),1);
klasso_3d(opt.iikeep) = klasso;
w3d_lasso = reshape(klasso_3d,nd);

% select slice to view, 1:x, 2:y, 3:z
figure(1), view_3d_volume(w3d_lasso,1)
figure(2), view_3d_volume(w3d_lasso,2)
figure(3), view_3d_volume(w3d_lasso,3)

% plot the weight on top of an anatomical structure of the brain
% anat1 = put_on_anat(w3d_lasso, anat, opt.iikeep);
% view_nii_ui(anat1)

%% asd only
Xtrain00 = zeros(size(Xtrain0,1),prod(nd));
Xtrain00(:,opt.iikeep) = Xtrain0;
tic;
minl = 2;
[kasd,ASDstats,dd] = fastASD(Xtrain00,ytrain0,nd,minl);
toc;

% accuracy
acc_te_asd = acc(ytest0,Xtest0*kasd(opt.iikeep))

w3d_asd = reshape(kasd,nd);

% select slice to view, 1:x, 2:y, 3:z
figure(1), view_3d_volume(w3d_asd,1)
figure(2), view_3d_volume(w3d_asd,2)
figure(3), view_3d_volume(w3d_asd,3)

% plot the weight on top of an anatomical structure of the brain
% anat1 = put_on_anat(w3d_asd, anat, opt.iikeep);
% view_nii_ui(anat1)

%% DRD mcmc
maxiter = 2000;
[wdrd_mcmc, udrd_mcmc, hypers_estimation_drd_mcmc, w_dif_mcmc, sq_er_mcmc] = runDRD_fmri([],datastruct,maxiter,0);
kdrd_mcmc = mean(wdrd_mcmc(end-10:end,:),1)';
cdrd_mcmc = mean(nonlinear_u(udrd_mcmc(end-10:end,:),opt,inf),1)';

% accuracy
acc_te_drd = acc(ytest0,Xtest0*kdrd_mcmc(opt.iikeep))

w3d_drd = reshape(kdrd_mcmc,nd);

% select slice to view, 1:x, 2:y, 3:z
figure(1), view_3d_volume(w3d_drd,1)
figure(2), view_3d_volume(w3d_drd,2)
figure(3), view_3d_volume(w3d_drd,3)
figure(4), % plot hist of posterior distributions of hyperparameters
mindelta = 1; maxdelta = min(nd); minl = 2; maxl = min(nd);
lb = [0.001;mindelta;-20;-5;minl]; ub = [1e5;maxdelta;5;10;maxl]; % bounds for [rho, delta, log_nsevar len]
priorfun_r = @(x,pf) gen_gammaprior(x,20,10,lb(1),ub(1),pf); % dgrid,dmean,dstd,lb,ub,plotfig
rgrid = 0.01:0.5:50;
logrprior = priorfun_r(rgrid,1);
priorfun_d = @(x,pf) gen_gammaprior(x,100,50,lb(2),ub(2),pf);
dgrid = 0.01:1:200;
logdprior = priorfun_d(dgrid,1);
priorfun_b = @(x,pf) gen_gaussprior(x,-10,8,lb(3),ub(3),pf);
bgrid = -30:0.5:5;
logbprior = priorfun_b(bgrid,1);
priorfun_n = @(x,pf) gen_gaussprior(x,-2,5,lb(4),ub(4),pf); % log nsevar
ngrid = -10:0.5:5;
lognprior = priorfun_n(ngrid,1);
subplot(221); plot_post(1,hypers_estimation_drd_mcmc,logrprior,rgrid,'rho',maxiter)
subplot(222); plot_post(2,hypers_estimation_drd_mcmc,logdprior,dgrid,'delta',maxiter)
subplot(223); plot_post(3,hypers_estimation_drd_mcmc,logbprior,bgrid,'b',maxiter)
subplot(224); plot_post(4,hypers_estimation_drd_mcmc,lognprior,ngrid,'lognsevar',maxiter)

% plot the weight on top of an anatomical structure of the brain
% anat1 = put_on_anat(w3d_drd, anat, opt.iikeep);
% view_nii_ui(anat1)

%% asd drd
cdrd_half = sqrt(abs(cdrd_mcmc));
Xtrain00 = zeros(size(Xtrain0,1),prod(nd));
Xtrain00(:,opt.iikeep) = Xtrain0;
Xtrain0f = bsxfun(@times,Xtrain00,cdrd_half');
tic;
minl = 2;
[kasd_drd,ASDstats,dd] = fastASD(Xtrain0f,ytrain0,nd,minl);
toc;
kasd_drd = kasd_drd.*cdrd_half;

% accuracy
acc_te_asd_drd = acc(ytest0,Xtest0*kasd_drd(opt.iikeep))

w3d_asd_drd = reshape(kasd_drd,nd);

% select slice to view, 1:x, 2:y, 3:z
figure(1), view_3d_volume(w3d_asd_drd,1)
figure(2), view_3d_volume(w3d_asd_drd,2)
figure(3), view_3d_volume(w3d_asd_drd,3)

% plot the weight on top of an anatomical structure of the brain
% anat1 = put_on_anat(w3d_asd_drd, anat, opt.iikeep);
% view_nii_ui(anat1)

%% sDRD mcmc
% This is slow with the real fMRI data.
% hh = sum(abs(hypers_estimation_drd_mcmc),2); ii = find(hh~=0); ii = ii(end);
% prs0 = [mean(hypers_estimation_drd_mcmc(max([2,ii-10]):ii,:),1) 2];
%
% figure(2)
% [wsdrd_mcmc, usdrd_mcmc, hypers_estimation_sdrd_mcmc, w_dif_sdrd_mcmc, sq_er_sdrd_mcmc] = runsDRD_fmri(prs0,datastruct,1000,0);
% ksdrd_mcmc = mean(wsdrd_mcmc(end-10:end,:),1)';
% csdrd_mcmc = mean(nonlinear_u(usdrd_mcmc(end-10:end,:),opt,inf),1)';
%
% % accuracy
% acc_te_sdrd = acc(ytest0,Xtest0*ksdrd_mcmc(opt.iikeep))
%
% w3d_sdrd = reshape(ksdrd_mcmc,nd);
%
% % select slice to view, 1:x, 2:y, 3:z
% figure(1), view_3d_volume(w3d_sdrd,1)
% figure(2), view_3d_volume(w3d_sdrd,2)
% figure(3), view_3d_volume(w3d_sdrd,3)