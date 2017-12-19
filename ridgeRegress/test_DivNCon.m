%% test_DivNCon.m

% Developing a testing platform to examine convergence and performance
% properties of divide-and-conquor strategies for when size of both p and n
% is intractible in one-shot.  Will borrow heavily from
% test_basicAutoRidgeRegress
clear all;
clc
addpath(genpath('../'))

%% Make data
% Set hyperparameters
nk = 1200;  % number of filter coeffs (1D vector)
rho = 2; % prior variance
alpha = 1/rho;  % prior precision
% Make filter
k = randn(nk,1)*sqrt(rho);

%  Make stimulus and response
nsamps = 5000; % number of stimulus sample
signse = 5;   % stdev of added noise
x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

% plot filter and examine noise level
t = 1:nk;
subplot(211); plot(t,k);
xlabel('index'); ylabel('filter coeff');
title('true filter');

subplot(212); plot(x*k, x*k, 'k.', x*k, y, 'r.');
xlabel('noiseless y'); ylabel('observed y');

%% Compute sufficient statistics

dd.xx = x'*x;  
dd.xy = (x'*y);
dd.yy = y'*y;
dd.nx = nk;
dd.ny = nsamps;

% Use fixed-point algorithm
maxiter = 100;
% tic;
% [kridge1,hprs1] = autoRidgeRegress_fp(dd,maxiter);
% toc; fprintf('\n\n');

% Use gradient-Hessian optimization in primal form as check
tic;
[kridge1,hprs1] = autoRidgeRegress_gradprimal(dd);
toc;


% % break up data into subsets and then estimate from each independently
% K = 10;
% for k = 1:K
% tic
% indk = [(nsamps/K*(k-1) + 1):(nsamps/K*k)]';
% dd.y = y(indk);
% dd.x = x(indk,:);  
% dd.xx = x(indk,:)*x(indk,:)';
% dd.xy = (x(indk,:)'*y(indk,:));
% dd.yy = y(indk,:)'*y(indk,:);
% dd.nx = nk;
% dd.ny = nsamps/K;
% [kridged(:,k),hprsd(k)] = autoRidgeRegress_graddual(dd);
% alphaDC(k) = hprsd(k).alpha;
% signseDC(k) = hprsd(k).nsevar;
% toc
% end
% 
% kridgen = mean(kridged,2);
% alphaN = mean(alphaDC);
% signseN = mean(signseDC);

%% Divide & conquer hyperparams, primal filter coefs
dd.y = y;
dd.x = x;  
dd.xx = x*x';
dd.nx = nk;
dd.ny = nsamps;
K = 6;%number of subsets
tic
[kridgen,hprsd] = autoRidgeHyperparEst_graddual_DNC(dd,K);
toc
alphaN = mean(hprsd(1,:));
signseN = mean(hprsd(2,:));
%%  ---- Make Plots ----

figure;
h = plot(t,k,'k-',t,kridge1,'b--',t,kridgen,'r');
set(h(1),'linewidth',2.5);
title('estimates');
legend('true', 'ridgePrimal', 'ridgeDual-DivNCon');

fprintf('\nHyerparam estimates\n------------------\n');
fprintf(' alpha:  %3.1f  %5.2f %5.2f\n',alpha,hprs1.alpha,alphaN);
fprintf('signse:  %3.1f  %5.2f %5.2f\n',signse,sqrt(hprs1.nsevar),sqrt(signseN));

err = @(khat)(sum((k-khat).^2)); % Define error function
fprintf('\nErrors:\n   RidgeALL = %7.2f\n RidgeDNC = %7.2f\n', [err(kridge1) err(kridgen)]);
