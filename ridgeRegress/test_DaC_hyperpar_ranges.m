%% test_DaCHyperpar_ranges
%  Examine how speed-up and accuracy of DaC hyperparameter estimation is
%  affected by number of observations, and number of subsets used for
%  estimation.
% updated 2015.21.04 (mca)


clear all;

Ntrial = 50;% number of trials per condition

% Set hyperparameters
nk = 2000;  % number of filter coeffs (1D vector)
rho = 2; % prior variance
alpha = 1/rho;  % prior precision
signse = 5;   % stdev of added noise


% Loop over sampel size and subset size
sampszs = 500:500:5000;
subsets  = [2 5 10 100 500];
nsampszs = numel(sampszs);
nsubsets = numel(subsets);

% Initialize arrays
tprimal = zeros(Ntrial,nsampszs);
hprsP = zeros(2,Ntrial,nsampszs);
tdual = zeros(Ntrial,nsampszs,nsubsets);
hprsD = zeros(2,Ntrial,nsampszs,nsubsets);
flagD = tdual;
parpool(4)
for itrial = 1:Ntrial
    fprintf('Trial %1.0f of %1.0f.\n',itrial,Ntrial)
    
    for isamp = 1:nsampszs% number of stimulus samples
        
        % Generate sample
        nsamps = sampszs(isamp);
        k = randn(nk,1)*sqrt(rho);% filter
        x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
        y = x*k + randn(nsamps,1)*signse;  % dependent variable
        
        % Estimate hyperpars by primal form
        % Make data array to pass to solver
        dd  = struct('xx',x'*x,'xy',x'*y,'yy',y'*y,'nx',nk,'ny',nsamps);
        fprintf('Estimating by primal form with sample size %1.0f.\n',nsamps)
        tic
        hprsP(:,itrial,isamp) = autoRidgeHyperparEst_gradprimal(dd);
        tprimal(itrial,isamp) = toc;
        
        for ik = 1:nsubsets%number of subsets for aggregate estimator
            k = subsets(ik);
            if k<nsamps
                fprintf('Estimating by DaC with %1.0f subsets.\n',k)
                if k<nsamps^(1/3)%flag if it meets sampling criterion
                    flagD(itrial,isamp,ik) = true;
                end
                
                % Estimate hyperpars by dual form with k subsets
                dd  = struct('x',x,'y',y,'nx',nk,'ny',nsamps);
                tic
                hprsD(:,itrial,isamp,ik) = autoRidgeHyperparEst_graddual_DNC(dd,[],k);
                tdual(itrial,isamp,ik) = toc;
            end
        end
    end
end
delete(gcp('nocreate'))
save('simout/Hyperpar_range','tprimal','tdual','hprsP','hprsD','sampszs','subsets','rho','signse','flagD','nk')