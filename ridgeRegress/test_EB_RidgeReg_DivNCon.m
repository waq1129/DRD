clear all;

Ntrial = 2;% number of trials

% Set hyperparameters
nk = 1200;  % number of filter coeffs (1D vector)
rho = 2; % prior variance
alpha = 1/rho;  % prior precision

%  Set stimulus and sample parameters
nsamps = 5000; % number of stimulus samples
signse = 5;   % stdev of added noise

K = 6;%number of subsets for aggregate estimator
hprsP = zeros(2,Ntrial);
hprsD = hprsP;
kridgep = zeros(nk,Ntrial);
kridged = kridgep;
parpool
parfor i = 1:Ntrial
    
    %% Make data
    
    k = randn(nk,1)*sqrt(rho);% Make filter
    x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
    y = x*k + randn(nsamps,1)*signse;  % dependent variable
    
    %% Compute sufficient statistics
    
    % Make data array to pass to solver
    dd  = struct('xx',x'*x,'xy',x'*y,'x',x,'y',y,'yy',y'*y,'nx',nk,'ny',nsamps);
    
    % Use gradient-Hessian optimization in primal form as check
    fprintf('Estimating kernels in batch in trial %2.0f by primal form\n',i)
    [kridgep(:,i),hprsp(:,i)] = autoRidgeRegress_gradprimal(dd);
    
    %% Divide & conquer
    fprintf('Estimating kernels in in trial %2.0f by divide-and-conquer\n',i)
    [kridged(:,i),hprsd(:,i)] = autoRidgeRegress_graddual_DNC(dd,[],K);
    
end
delete(gcp('nocreate'))
save('simout/Nkernels','kridgep','kridged','hprsD','hprsP')

    fprintf('\nHyerparam estimates\n------------------\n');
    fprintf(' alpha:  %3.1f  %5.2f %5.2f\n',alpha,hprsP(1,i),hprsD(1,i));
    fprintf('signse:  %3.1f  %5.2f %5.2f\n',(signse),(hprsP(2,i)),(hprsD(2,i)));
    err = @(khat)(sum((k-khat).^2)); % Define error function
    fprintf('\nErrors:\n   RidgeALL = %7.2f\n RidgeDNC = %7.2f\n', [err(kridge1) err(kridgen)]);
