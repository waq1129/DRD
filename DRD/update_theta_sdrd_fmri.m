function [theta, U] = update_theta_sdrd_fmri(theta, ff, Gf, logcfdiag_fun, Lfn, Ufn, theta_Lprior, slice_width, Lpstar_min, opt, randid)
% UPDATE_THETA_SIMPLE Standard slice-sampling MCMC update to GP hyper-param
% variant for sdrd update
% Iain Murray, January 2010

% Ufn = @(th) chol(Kfn(th));
% DEFAULT('theta_Lprior', @(l) log(double((l>log(0.1)) && (l<log(10)))));
% DEFAULT('U', Ufn(theta));

% input theta: vv, kdiag, b, log_nsevar, logcfdiag, Gf

% Slice sample theta|ff
particle = struct('pos', theta, 'ff', ff, 'Ufn', Ufn, 'change_theta', 0, 'theta_Lprior', theta_Lprior, ...
    'id', 0, 'pid', 0, 'Gf', {Gf}, 'change_len_flag', 0);
particle = eval_particle(particle, Lpstar_min, Lfn, theta_Lprior, Ufn, logcfdiag_fun, opt);
step_out = mean(slice_width > 0);
slice_width = abs(slice_width);
slice_fn = @(pp, Lpstar_min) eval_particle(pp, Lpstar_min, Lfn, theta_Lprior, Ufn, logcfdiag_fun, opt);
particle = slice_sweep(particle, slice_fn, slice_width, step_out, randid);
theta = particle.pos;
U = particle.U;

function pp = eval_particle(pp, Lpstar_min, Lfn, theta_Lprior, Ufn, logcfdiag_fun, opt)

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
if pp.id == 5 % slice sampling for len
    [~, ~, Gf, cdiagcells_half] = logcfdiag_fun(theta(5));
    pp.Gf = kron_prod_mult_cell(cdiagcells_half, Gf);
end

if pp.id~=5 && pp.pid==5 && pp.change_len_flag==0
    Gftmp = expand_kron(pp.Gf);
    pp.Gf = Gftmp(:,opt.iikeep);
    pp.change_len_flag = 1;
end
Lfprior = Lfn(xsamp, exp(logkdiag), theta(3), theta(4), pp.Gf, pp.id);

pp.Lpstar = Ltprior + Lfprior;
pp.on_slice = (pp.Lpstar >= Lpstar_min);
pp.U = sqrt(exp(logkdiag));
pp.Lfprior = Lfprior;

% display(['slice id ', num2str(pp.id), ' previous slice id ', num2str(pp.pid), ' change_len_flag ', num2str(pp.change_len_flag)])
% display(theta)
