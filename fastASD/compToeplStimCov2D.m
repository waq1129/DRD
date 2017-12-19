function xcov = compToeplStimCov2D(x,nn)
% Compute stimulus autocovariance
%
% xacov = compToeplStimCov2(x,nn)
%
% NOTE THIS CURRENTLY ONLY WORKS WITH NN(1) = NN(2) (square stimuli)

nk1 = nn(1); % size of 1st dimension
nk2 = nn(2);  % size of 2nd dimension
nsamps = size(x,1);

% Compute number of terms needed in autocovariance
ncos = ceil((nk1+1)/2); % number of cosine terms
nsin = floor((nk2-1)/2); % number of sine terms

% % Compute autocovariance (using FFT)
% xh = ifft(mean(abs(fft(x',nx+ncos).^2),2));
% nnrm = min(ncos,nx); % number of coeffs to worry about normalizing
% xh(1:nnrm) = xh(1:nnrm)./(nx:-1:(nx-nnrm+1))'; % normalize some coeffs so 'unbiased'
% xcov = [xh(1:ncos);flipud(xh(2:nsin+1))];

% reshape x into tensor
xtens = reshape(x',nk1,nk2,nsamps);  % tensor x

% compute xcov using fft
xh = ifft2(mean(abs(fft2(xtens,nk1+ncos,nk2+ncos).^2),3));

% normalize by number of observations for each shift position
nnrm = min(ncos,nk1); % number of coeffs to worry about normalizing
knrm = (nk1:-1:(nk1-ncos+1))'*(nk2:-1:(nk2-ncos+1));  % normalizer (for unbiased estimate of cov)
xh(1:nnrm,1:nnrm) = xh(1:nnrm,1:nnrm)./knrm;

% reflect around midline to make matrix symmetric
xcov = ([xh(1:ncos,:);flipud(xh(2:nsin+1,:))]); % symmetrize vertically
xcov = ([xcov(:,1:ncos),fliplr(xcov(:,2:nsin+1))]); % symmetrize horizontally
