function Ltprior = theta_log_prior(theta, priorfun_r, priorfun_d, priorfun_b, priorfun_n)
rho = theta(1);
delta = theta(2);
b = theta(3);
lognsevar = theta(4);

Ltprior = priorfun_r(rho,0)+priorfun_d(delta,0)+priorfun_b(b,0)+priorfun_n(lognsevar,0);