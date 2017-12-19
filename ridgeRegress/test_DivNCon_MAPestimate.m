%% test_DivNCon_MAPestimation
%
% Script to test recovery of MAP estimate for weights given either true or
% maximum marginal-likelihood (true or from DAC) hyperparameters

% Set hyperparameters
nk = 500;  % number of filter coeffs (1D vector)
rho = 2; % prior variance
alpha = 1/rho;  % prior precision

% Make filter
k = randn(nk,1)*sqrt(rho);

%  Make stimulus and response
nsamps = 400; % number of stimulus sample
signse = .5;   % stdev of added noise
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

dd.xx = x*x';  
dd.xy = (x'*y);
dd.yy = y'*y;
dd.nx = nk;
dd.ny = nsamps;
dd.y = y;
dd.x = x;  

% Use gradient-Hessian optimization in dual form
tic;
[kridge1,hprs1] = autoRidgeRegress_graddual(dd);
toc;

%% Redo it but with xx reduced to a block-diagonal
nblocks = 4;
blksz = nsamps/nblocks;  

% make sure blksz is integer
if mod(blksz,1)~=0, error('nblocks must evenly divide nsamps'); end

% make mask for XX'
M = mat2cell(ones(blksz,nsamps),blksz,blksz*ones(1,nblocks));
XXmask = sparse(blkdiag(M{:}));

% make struct and replace xx with blocked xx
dd2 = dd;
dd2.xx = sparse(dd2.xx.*XXmask);

% Use gradient-Hessian optimization in dual form
tic;
[kridge2,hprs2] = autoRidgeRegress_graddual(dd2);
toc;


%% Check that we get the same thing when we plug hyperparams into dual MAP formula

% Check dual form equation
lam = (hprs1.alpha*hprs1.nsevar); % ridge penalty
XCX = dd.xx/lam; % XCX/sig^2 term
kridge1b = dd.xy/lam -  (1/lam)*dd.x'*((eye(nsamps)+XCX)\(dd.x*dd.xy/lam));
max(abs(kridge1-kridge1b))  % should agree!

lam = hprs2.alpha*hprs2.nsevar;
XCX2 = dd2.xx/lam;
kridge2b = dd.xy/lam -  (1/lam)*dd.x'*((eye(nsamps)+XCX2)\(XCX2*dd2.y));


%%  ---- Make Plots ----

subplot(221);
h = plot(t,k,'k-',t,kridge1,t,kridge2,'r',t,kridge2b);
set(h(1),'linewidth',2.5);
title('estimates');
legend('true', 'full', 'blocked', 'blocked2b');

fprintf('\nHyerparam estimates\n------------------\n');
fprintf(' alpha:  %3.1f  %5.2f %5.2f\n',alpha,hprs1.alpha,hprs2.alpha);
fprintf('signse:  %3.1f  %5.2f %5.2f\n',signse,sqrt(hprs1.nsevar),sqrt(hprs2.nsevar));

err = @(khat)(sum((k-khat).^2)./norm(k).^2); % Define error function
fprintf(['\nErrors:\n',...
    ' Full Ridge         = %7.2f\n',...
    ' Blocked Ridge      = %7.2f\n',...
    ' Blocked Ridge dual = %7.2f\n'],...
    [err(kridge1) err(kridge2) err(kridge2b)]);

subplot(222);
plot(5*[-1 1], 5*[-1 1], 'k', k,kridge1, 'o');
title('full ridge estimate');

subplot(223);
plot(5*[-1 1], 5*[-1 1], 'k', k,kridge2, 'o');
title('blocked, automatically passed back');

subplot(224);
plot(5*[-1 1], 5*[-1 1], 'k', k,kridge2b, 'o');
title('blocked, using dual formula');