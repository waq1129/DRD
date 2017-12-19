function [xhat,wvec] = realDFT(x, nxcirc)
% [xhat,wvec] = realDFT(x, nxcirc);
%
% Compute real-valued version of Discrete Fourier Transform (DFT) using fft, 
% but storing cosine and sine terms separately so that vector is real-valued.  
%
% Result has structure;  [DC(0), Cos(1:(nxcirc-1)/2), Sin(-(nxcirc-1)/2:-1)]';
%
% INPUTS:
%         x - vector or matrix 
%    nxcirc - # points in DFT (optional)
%           note: x is padded with zeros if nxcirc > length(x), 
%
% OUTPUTs:
%      xhat - DFT of x
%      wvec - vector of Fourier frequencies, if desired


% convert x to column if a row vector
if (size(x,1) == 1) 
    x = x';
end
if nargin == 1
    nxcirc = size(x,1);
end

% Take fourier transform
xfft = fft(x,nxcirc)/sqrt(nxcirc/2); 

% Fix DC term
xfft(1,:) = xfft(1,:)/sqrt(2);

% If nx is even, fix highest cosine term
if mod(size(x,1),2) == 0 
    imx = ceil((nxcirc+1)/2);
    xfft(imx,:) = xfft(imx,:)/sqrt(2);
end

xhat = real(xfft);
isin = ceil((nxcirc+3)/2); % index where sin terms start
xhat(isin:end,:) = -imag(xfft(isin:end,:));


% Compute freq vector, if desired
if nargout > 1
    ncos = ceil((nxcirc-1)/2); % number of negative frequencies;
    nsin = floor((nxcirc-1)/2); % number of positive frequencies;
    wvec = [0:ncos, -nsin:-1]'; % vector of frequencies
end