%% test_fastASD_ELLtoepl_1D.m
%
% Short script to illustrate fast automatic smoothness determination (ASD)
% for vector of filter coefficients using ELL approximation in 1D

% add directory with DFT tools 
setpaths;

stimvar = 2;

% Generate true filter vector k
nk1 = 20; % number of rows
nk2 = 4;  % number of columns
nks = [nk1 nk2];
nktot = nk1*nk2; % total number of filter coeffs

% Generate true filter vector k
t1 = (1:nk1)'-nk1/2;
t2 = (1:nk2)'-nk2/2;
l1 = nk1/6;
l2 = nk2/6;
kim = exp(-.5* bsxfun(@plus,(t1./l1).^2,(t2'./l2).^2));
imagesc(kim);
k = kim(:);

%%  Make stimulus and simulate response
nsamps = 200; % number of stimulus sample
signse = 1;   % stdev of added noise

nrpts = 200;
errs = zeros(nrpts,4);

for jj = 1:nrpts
% ==========FOR LOOP ========================
x = randn(nsamps,nktot)*sqrt(stimvar);
y = x*k + randn(nsamps,1)*signse;  % dependent variable 

xx = x'*x;
kml = xx\(x'*y);
kell = (nsamps*stimvar*eye(nktot))\(x'*y);

% ELL-diag
xxd = diag(diag(xx));
kelld = xxd\(x'*y);

%% ELL-kron
xm = reshape(x',nk1,nk2,nsamps);
xc = reshape(xm,nk1,[])';
xxc = (xc'*xc);
xxc = xxc./norm(diag(xxc));
xr = reshape(permute(xm,[2 1 3]),nk2,[])';
xxr = (xr'*xr);
xxr = xxr./(norm(diag(xxr)))*norm(diag(xx)); 

xxk = kron(xxr,xxc);
kellkron = xxk\(x'*y);

%% plot filter and examine noise level
t = 1:nktot;
clf;plot(t,k,t,[kml kell kelld kellkron]);

% Compute errors 
err = @(khat)(sum((k-khat).^2)); % Define error function

errs(jj,:) = [err(kml) err(kell) err(kelld) err(kellkron)];
% ========== END FOR ===================
end
%%

fprintf('Errs:\n ML      ELL    diag   kron\n %6.3f %6.3f %6.3f %6.3f\n',...
mean(errs))

