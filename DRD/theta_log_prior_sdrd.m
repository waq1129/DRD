function Ltprior = theta_log_prior_sdrd(theta, priorfun_r, priorfun_d, priorfun_b, priorfun_n, priorfun_l)

rho = theta(1);
delta = theta(2);
b = theta(3);
lognsevar = theta(4);
len = theta(5);

Ltprior = priorfun_r(rho,0)+priorfun_d(delta,0)+priorfun_b(b,0)+priorfun_n(lognsevar,0)+priorfun_l(len,0);