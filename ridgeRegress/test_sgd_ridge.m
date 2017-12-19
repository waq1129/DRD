%% test_sgd_ridge

clear all;
close all

% Set hyperparameters
nk = 2;  % number of filter coeffs (1D vector)
rho = 2; % prior variance
alpha = 1/rho;  % prior precision

%  Set stimulus and sample parameters
nsamps = 5000; % number of stimulus samples
signse = 2;   % stdev of added noise

%% Make data
k = randn(nk,1)*sqrt(rho);% Make filter
x = gsmooth(randn(nk,nsamps),1)'; % stimulus (smooth)f
y = x*k + randn(nsamps,1)*signse;  % dependent variable

khat = (x'*x + speye(nk)*rho)\(x'*y);
%% estimate by parameters by stochastic gradient descent

% set sgd parameters
maxit = 8000;% maximum iterations
maxlamb = 1;% step size
minlamb = 1;
gtol = 1e-8;% tolerance for gradient
iter = 1;% initialize iteration count
nsevar = signse*signse;
z = zeros(nk,1);
wi = randn(nk,1);
w = zeros(nk,1);
figure;plot(k,w,'x');hold on
gnorm = inf;
nb = 100;%batchsize;
while and(gnorm>=gtol, iter<=maxit)%conditions for iterating
    %choose dimension to walk down
    
    %return gradient along chosen sample
    dimind = randi([1 nsamps],[nb 1]);
    xi = x(dimind,:);
    g = xi'*y(dimind)/nsevar - alpha*w - xi'*xi*wi/nsevar;
    g = -g;
    gnorm = norm(g);
    %step
    lamb = 1/(iter);
    wnew = w - g*lamb;
    
    dx = norm(wnew-w);
    posteriorW(iter) = -w'*w*alpha/2 - (y-x*w)'*(y-x*w)/2/nsevar;
    w = wnew;
    iter = iter + 1;
end

plot(k,w,'.',k,khat,'o',[min([k;w]) max([k;w])],[min([k;w]) max([k;w])],'k');hold off

figure;plot(-posteriorW)
