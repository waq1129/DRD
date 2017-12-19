%% test_HyperparamDivNCon.m

% Developing a testing platform to examine convergence and performance
% properties of divide-and-conquor strategies for when size of both p and n
% is intractible in one-shot.  Will borrow heavily from
% test_basicAutoRidgeRegress
% This function will provide simulations estimating the hyperparameters
% using the primal form with all data and with the dual form using subsets
% of the data.
% updated 2015.21.04 (mca)

clear all;

Ntrial = 30;% number of trials

% Set hyperparameters
nk = 2000;  % number of filter coeffs (1D vector)
rho = 2; % prior variance
alpha = 1/rho;  % prior precision

%  Set stimulus and sample parameters
nsamps = 2000; % number of stimulus samples
signse = 5;   % stdev of added noise

K = 5;%number of subsets for aggregate estimator
hprsP = zeros(2,Ntrial);
hprsD1 = hprsP;
hprsD3 = hprsP;
tprimal = zeros(Ntrial,1);
tdual1 = zeros(Ntrial,1);
tdual3 = zeros(Ntrial,1);

% parpool
%% Run trials
for i = 1:Ntrial
    % Make data
    k = randn(nk,1)*sqrt(rho);% filter
    x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
    y = x*k + randn(nsamps,1)*signse;  % dependent variable
    
    % Compute sufficient statistics
    % Make data array to pass to solver
    dd  = struct('xx',x'*x,'xy',x'*y,'yy',y'*y,'nx',nk,'ny',nsamps);
    
    % Use gradient-Hessian optimization in primal form as check
    fprintf('Estimating hyperpar in trial %1.0f by primal form\n',i)
    tic
    hprsP(:,i) = autoRidgeHyperparEst_gradprimal(dd);
    tprimal(i) = toc;
    % Divide & conquer hyperparams
    dd  = struct('x',x,'y',y,'nx',nk,'ny',nsamps);
    fprintf('Estimating hyperpar in trial %1.0f by divide-and-conquer 1\n',i)
    tic
    hprsD1(:,i) = autoRidgeHyperparEst_graddual_DNC(dd,K);
    tdual1(i) = toc;
    
    % Divide & conquer hyperparams3
    fprintf('Estimating hyperpar in trial %1.0f by divide-and-conquer 3\n',i)
    tic
    hprsD3(:,i) = autoRidgeHyperparEst_graddual_DNC3(dd,[],K);
    tdual3(i) = toc;
    
end
% delete(gcp('nocreate'))
save('simout/Hyperpar','hprsD1','hprsD3','hprsP','tprimal','tdual1','tdual3','alpha','signse')
%% Compare results
fprintf('\nHyerparam estimates\n------------------\n');
fprintf(' alpha:  %3.1f  %5.2f %5.2f  %5.2f \n',alpha,1/mean(hprsP(1,:)),1/mean(hprsD1(1,:)),1/mean(hprsD3(1,:)));
fprintf('\nsignse:  %3.1f  %5.2f %5.2f  %5.2f \n',(signse),mean(sqrt(hprsP(2,:))),mean(sqrt(hprsD1(2,:))),mean(sqrt(hprsD3(2,:))));
fprintf('\n');

fprintf('\nAverage time to optimize\n------------------\n');
fprintf(' t=	%5.2f	%5.2f	%5.2f \n',mean(tprimal),mean(tdual1),mean(tdual3));
fprintf('\n');
