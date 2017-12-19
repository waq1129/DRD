function [theta, U] = update_theta_sdrd(theta, ff, Lfn, Ufn, theta_Lprior, slice_width, Lpstar_min, randid)
% UPDATE_THETA_SIMPLE Standard slice-sampling MCMC update to GP hyper-param
% variant for sdrd update
% Iain Murray, January 2010

% Ufn = @(th) chol(Kfn(th));
% DEFAULT('theta_Lprior', @(l) log(double((l>log(0.1)) && (l<log(10)))));
% DEFAULT('U', Ufn(theta));

% Slice sample theta|ff
particle = struct('pos', theta, 'ff', ff, 'Ufn', Ufn, 'change_theta', 0, 'theta_Lprior', theta_Lprior, 'id', 0, 'pid', 0);
particle = eval_particle(particle, Lpstar_min, Lfn, theta_Lprior, Ufn);
step_out = mean(slice_width > 0);
slice_width = abs(slice_width);
slice_fn = @(pp, Lpstar_min) eval_particle(pp, Lpstar_min, Lfn, theta_Lprior, Ufn);
particle = slice_sweep(particle, slice_fn, slice_width, step_out, randid);
theta = particle.pos;
U = particle.U;

function pp = eval_particle(pp, Lpstar_min, Lfn, theta_Lprior, Ufn)

% Prior
theta = pp.pos;
Ltprior = theta_Lprior(theta);
if Ltprior == -Inf
    pp.Lpstar   = -Inf;
    pp.on_slice = false;
    return;
end

xsamp = pp.ff;
logkdiag = Ufn(theta(1),theta(2));
Lfprior = Lfn(xsamp, exp(logkdiag), theta(3), theta(4), theta(5));

pp.Lpstar = Ltprior + Lfprior;
pp.on_slice = (pp.Lpstar >= Lpstar_min);
pp.U = sqrt(exp(logkdiag));
pp.Lfprior = Lfprior;

