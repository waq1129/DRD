function [logdprior] = gen_gammaprior(x,dmean,dstd,lb,ub,plotfig)
% Set prior for d
% dmean = 10; % mean
% dstd = 5;  % stdev
ad = (dmean/dstd)^2;
bd = (dstd^2/dmean);
% make d grid
% dd = 0.5;
% mind = 0.01; maxd = 20;
% dgrid = (mind:dd:maxd)';
% compute prior & visualize
dprior = gampdf(x,ad,bd)';
logdprior = log(dprior);

lbx = logical(sign(x-lb)+1);
ubx = logical(sign(ub-x)+1);
xx = ~(lbx.*ubx);
logdprior(xx) = -inf;

if plotfig
    plot(x,dprior);
    xlabel('d'); ylabel('P(d)'); title('prior over d');
end