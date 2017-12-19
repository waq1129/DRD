% function [logev,khat,LL] = compLogEv(CpriorInv,nsevar,XX,XY,YY,ny)
% % function [logev,khat,LL] = compLogEv(Cprior,nsevar,XX,XY,YY,ny)
% % [logev,khat,LL] = compLogEv(CpriorInv,nsevar,X,Y)
% % [logev,khat,LL] = compLogEv(CpriorInv,nsevar,XX,XY,YY,ny)
% %  
% % Compute the log-evidence under a Gaussian likelihood and prior
% %
% % Inputs: 
% %   CpriorInv = inverse covariance matrix for prior
% %   nsevar = variance of likelihood
% %   X = design matrix
% %   Y = dependent variable
% %   XX = X'*X;  (alternative to passing X & Y)
% %   YY = Y'*Y;  (alternative to passing X & Y)
% %   ny = length(Y);  
% %
% % Outputs: 
% %   logev = log-evidence
% %   khat = MAP estimate (posterior mean)
% %   LL = posterior covariance matrix
% 
% if nargin == 4   % Compute these if necessary
%     ny = size(XY,1);
%     YY = XY'*XY;
%     XY = XX'*XY;
%     XX = XX'*XX;
% end
% 
% % 1. Compute MAP estimate of filter
% LL = inv(XX/nsevar + CpriorInv);  % Posterior Covariance
% LL = real(LL);
% % to avoid numerical error
% LL = (LL+LL')/2; % it should be always symmetric
% CpriorInv = (CpriorInv+CpriorInv')/2;
% 
% % [cx, cy] = size(Cprior);
% % Imat = eye(cx, cy);
% % LL = Cprior/(XX*Cprior/nsevar + Imat); 
% khat = LL*XY/nsevar;  % Posterior Mean (MAP estimate)
% 
% try
% 
%     % 1st term, from sqrt of log-determinants
% %     trm1 = .5*(logdet(LL) + logdet(CpriorInv) - (ny)*log(2*pi*nsevar));
%     trm1 = .5*(logdet(LL) + logdet(CpriorInv) - (ny)*log(2*pi*nsevar));
% %     trm1 = .5*(logdet(real(LL)+imag(LL)) + logdet(real(CpriorInv)) - (ny)*log(2*pi*nsevar));
% %      trm1 = .5*(- log(trace(XX*Cprior)/nsevar + 1) - (ny)*log(2*pi*nsevar));
% %     trm1 = .5*(- log(det((XX*Cprior)/nsevar + Imat)) - (ny)*log(2*pi*nsevar));
%     
% 
%     % 2nd term, from exponent
% %     trm2 = -.5*(YY/nsevar - (XY'*LL*XY)/nsevar.^2);
%     trm2 = -.5*(YY/nsevar - real(XY'*LL*XY)/nsevar.^2);
% 
%     logev = trm1+trm2;
% 
% catch
% 
%     logev= -1e20;
%     
% end
    

%%

function [logev,khat,LL] = compLogEv(CpriorInv,nsevar,XX,XY,YY,ny)
% [logev,khat,LL] = compLogEv(CpriorInv,nsevar,X,Y)
% [logev,khat,LL] = compLogEv(CpriorInv,nsevar,XX,XY,YY,ny)
%  
% Compute the log-evidence under a Gaussian likelihood and prior
%
% Inputs: 
%   CpriorInv = inverse covariance matrix for prior
%   nsevar = variance of likelihood
%   X = design matrix
%   Y = dependent variable
%   XX = X'*X;  (alternative to passing X & Y)
%   YY = Y'*Y;  (alternative to passing X & Y)
%   ny = length(Y);  
%
% Outputs: 
%   logev = log-evidence
%   khat = MAP estimate (posterior mean)
%   LL = posterior covariance matrix

if nargin == 4   % Compute these if necessary
    ny = size(XY,1);
    YY = XY'*XY;
    XY = XX'*XY;
    XX = XX'*XX;
end

% 1. Compute MAP estimate of filter
LL = inv(XX/nsevar+CpriorInv);
khat = LL*XY/nsevar;
% LL = inv((real(XX)+imag(XX))/nsevar + real(CpriorInv));  % Posterior Covariance
% khat = LL*(real(XY)+imag(XY))/nsevar;  % Posterior Mean (MAP estimate)

try

    % 1st term, from sqrt of log-determinants
%     trm1 = .5*(logdet(LL) + logdet(CpriorInv) - (ny)*log(2*pi*nsevar));
    trm1 = .5*(logdet(real(LL)+imag(LL)) + logdet(real(CpriorInv)) - (ny)*log(2*pi*nsevar));

    % 2nd term, from exponent
%     trm2 = -.5*(YY/nsevar - (XY'*LL*XY)/nsevar.^2);
    trm2 = -.5*(YY/nsevar - real(XY'*LL*XY)/nsevar.^2);

    logev = trm1+trm2;

catch

    logev= -1e20;
    
end
    




