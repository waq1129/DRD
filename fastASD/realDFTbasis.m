function [B,wvec] = realDFTbasis(nx,nxcirc,wvec)
% [B,wvec] = realDFTbasis(nx,nxcirc,w)
%
% Compute real DFT basis of sines and cosines for nx coefficients 
%
% INPUTS:
%  nx - number of coefficients in basis
%  nxcirc - number of coefficients for FFT (should be >= nx, so FFT is zero-padded)
%  wvec (optional) - frequencies 
%
% OUTPUTS:
%   B [nx x nxcirc] of [nx x length(wvec)] - DFT basis 
%   wvec - frequencies associated with columns of B

if nargin < 3
    % Make frequency vector
    ncos = ceil((nxcirc+1)/2); % number of negative frequencies;
    nsin = floor((nxcirc-1)/2); % number of positive frequencies;
    wvec = [0:ncos-1, -nsin:-1]'; % vector of frequencies
end

% Divide into pos (for cosine) and neg (for sine) frequencies
wcos = wvec(wvec>=0); 
wsin = wvec(wvec<0);  

x = (0:nx-1)'; % spatial pixel indices
if ~isempty(wsin)
    B = [cos(x*(wcos'*2*pi/nxcirc)), sin(x*(wsin'*2*pi/nxcirc))]/sqrt(nxcirc/2);
else
    B = cos(x*(wcos'*2*pi/nxcirc))/sqrt(nxcirc/2);
end

% make DC term into a unit vector
izero = (wvec==0); % index for DC term
B(:,izero) = B(:,izero)./sqrt(2);  

% if nx is even, make highest-freq cosine term into unit vector
if mod(nx,2)==0
    iimax = find(wvec==max(wvec));
    B(:,iimax) = B(:,iimax)./sqrt(2);
end
