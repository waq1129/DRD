function [kArd, cArd, nsevar] = fastARD(x,y)
%fast ARD with RVM and its fast implementation

%% initialize with kmap
[n,d] = size(x);
kmap0 = x'*((x*x' + eye(n))\y);  
nsevar = var(y-x*kmap0);
alpha = 1/nsevar;
[~,mid] = max(abs(kmap0));
SETTINGS = SB2_ParameterSettings('NoiseStd',nsevar,'alpha',alpha,'relevant',mid);
% SETTINGS = SB2_ParameterSettings;
OPTIONS = SB2_UserOptions;

%% main function
[PARAMETER, HYPERPARAMETER] = SparseBayes('gaussian', x, y, OPTIONS, SETTINGS);

%% output
kArd = zeros(d,1);
kArd(PARAMETER.Relevant) = PARAMETER.Value;

cArd = zeros(d,1);
cArd(PARAMETER.Relevant) = 1./HYPERPARAMETER.Alpha;
% time = mean(cArd)/var(kArd);
% time = max(abs(cArd))/var(kArd);
% cArd = cArd/time; %rescale to smaller scale

nsevar = 1/HYPERPARAMETER.beta;