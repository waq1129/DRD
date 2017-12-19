% test_inspectASDcov.m
%
% Examine original vs. Fourier-domain (diagonlized Fourier-domain) ASD covariance

% add directory with DFT tools 
%setpaths;

% Generate ASD covariance matrix
nk = 200;  % number of filter coeffs (assumed 1D)
rho = 1;
len = 15;
Casd = mkcov_ASD(len,rho,nk); % prior covariance matrix 

nxcirc = nk+len*3+1;

% % Compare to diagonalized version w/o zero-padding (circular boundary condition)
% opts1.nxcirc = nk;
% opts1.condthresh = 1e12;
% [Cdiag1,U1] = mkcov_ASDfactored([len;rho],nk,opts1);
% Casd_approx1 = U1*diag(Cdiag1)*U1';

% Compare to diagonalized version w sufficient zero-padding
opts2.nxcirc = nxcirc;
opts2.condthresh = 1e9;
[Cdiag2,U2,ww] = mkcov_ASDfactored([len;rho],nk,opts2);
nw = length(ww);
Casd_approx2 = U2*diag(Cdiag2)*U2';
B = realfftbasis(nxcirc,nxcirc,ww);

C2 = B'*diag(Cdiag2)*B;


opts3 = opts2;
opts3.condthresh = 1e22;
cd3 = mkcov_ASDfactored([len;rho],nk,opts3);
nw = length(cd3);
%% Make plots
subplot(231);
imagesc(Casd); axis image;
title('ASD covariance');
xt = [100,200];
set(gca,'xtick',xt,'ytick',xt);
%yl = [0.5 nxcirc+.5];
%set(gca,'ylim', yl,'xlim', yl);
xlabel('filter coeff');
ylabel('filter coeff');

subplot(2,3,[2,2.1]);
imagesc(C2);
axis image;
title('extended cov');
set(gca,'xtick',xt,'ytick',xt);

subplot(2,3,3.2);
h = plot(1:nk, Casd(1,:), 1:nk, C2(1,1:nk), '--');
set(h(2),'linewidth', 1.5);
axis square; box off;
set(gca,'ylim', [-.1 1], 'ytick', 0:.5:1);
legend('original', 'extended');

subplot(234);
cd = fftshift(cd3);
nw2 = floor(nw/2);
t = -nw2:nw2;
h = semilogy(t,cd,'.', 30*[-1 1], (max(cd)/opts2.condthresh)*[1 1],'k--');
set(h,'markersize',12);
axis square; box off;
set(gca,'xlim', 35*[-1 1],'ylim',[1e-21 200],'ytick', [1e-20 1e-10 1]);
title('eigenvalue');
xlabel('dft-frequency');


subplot(235);
imagesc(diag(Cdiag2)); axis image;
yt = 20:20:40;
set(gca,'ytick', yt,'xtick',yt); 
title('Fourier domain cov');
ylabel('dft-frequency');
xlabel('dft-frequency');
%set(gca,'ycolor', 'w');


set(gcf,'paperposition',[.25 2.5 6 4.5]);
print -dpdf fftCovFig.pdf

