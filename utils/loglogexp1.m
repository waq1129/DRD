function [logef, exp_neg_logef] = loglogexp1(x,m)
%  [f,df,ddf] = logexp1(x);
%
%  Computes the function:
%     f(x) = log(1+exp(x))
%     f(x) = exp(-log(1+exp(x))) = 1/(1+exp(x))
%  and returns first and second derivatives
if nargin<2
    m = inf;
end
ef = exp(x);
logef = log(1+ef-exp(-m));
exp_neg_logef = exp(-logef)*(1-exp(-m));

iix1 = logical(zeros(size(x)));
iix2 = logical(zeros(size(x)));

% Check for small values
if any(x(:)<-20)
    iix1 = (x(:)<-20);
    logef(iix1) = exp(x(iix1));
    exp_neg_logef(iix1) = exp(-exp(x(iix1)));
    
end

% Check for large values
if any(x(:)>500)
    iix2 = (x(:)>500);
    logef(iix2) = x(iix2);
    exp_neg_logef(iix2) = exp(-x(iix2));
end