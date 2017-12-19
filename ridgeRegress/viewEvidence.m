% viewEvidence

setpaths; % make sure we have paths set

% Set hyperparameters
nk = 100;  % number of filter coeffs (1D vector)
rho = 2; % prior variance
alpha = 1/rho;  % prior precision

% Make filter
k = randn(nk,1)*sqrt(rho);

%  Make stimulus and response
nsamps = 50; % number of stimulus sample
signse = 5;   % stdev of added noise
x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)
y = x*k + randn(nsamps,1)*signse;  % dependent variable 


%% Compute sufficient statistics
% dd.x = x;
% dd.y = y;
% dd.xx = x*x';%x'*x;  

dd.xx = x'*x;  
dd.xy = (x'*y);
dd.yy = y'*y;
dd.nx = nk;
dd.ny = nsamps;
rhos = [.1:.1:4];
signses = [1:.1:10];
neglogev = zeros(length(rhos),length(signses));
for i = 1:length(rhos)
    for j = 1:length(signses)
    neglogev(i,j) = neglogev_ridgePrimal([rhos(i) signses(j)]',dd);
    end
end

figure;imagesc(signses,rhos,neglogev);set(gca,'ydir','normal') 