%% test_EB_RidgeReg_allDual

clear all;
close all

Ntrial = 2;% number of trials

% Set hyperparameters
nk = 2000;  % number of filter coeffs (1D vector)
rho = 2; % prior variance
alpha = 1/rho;  % prior precision

%  Set stimulus and sample parameters
nsamps = 5000; % number of stimulus samples
signse = 5;   % stdev of added noise

K = 5;%number of subsets for aggregate estimator
hprsP = zeros(2,Ntrial);
hprsD = hprsP;
kridgep = zeros(nk,Ntrial);
kridged = kridgep;


%% Make data
k = randn(nk,1)*sqrt(rho);% Make filter
x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % dependent variable

%% Estimate by dual
dd  = struct('xx',x'*x,'xy',x'*y,'x',x,'y',y,'yy',y'*y,'nx',nk,'ny',nsamps);

[hprshat, kest]  = autoRidge_dualDaCdualReg(dd,K);


%% Plot dual results

figure(1);plot(k,kest,'.',[min([kest k']) max([kest k'])],[min([kest k']) max([kest k'])])
ylabel('\hat{k}');xlabel('k')
title('dual form DaC solution')

figure;plot(1:nk,k,1:nk,kest)
legend('k','\hat{k}')
title('dual')
%% Estimate by primal
[kest,stats] = autoRidgeRegress_gradprimal(dd,[]);
%% Plot primal results

figure;plot(k,kest,'.',[min([kest k]) max([kest k])],[min([kest k]) max([kest k])])
ylabel('\hat{k}');xlabel('k')
title('primal form solution')

figure;plot(1:nk,k,1:nk,kest)
legend('k','\hat{k}')
title('primal')
