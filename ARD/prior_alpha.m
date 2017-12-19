function L = prior_alpha(gam_a, alpha, priortype)

switch priortype
    case 'none'
        % improper prior
        L = 0;
    case 'gamma'
        % gamma prior
        L = gam_a*log(alpha);
    case 'exp'
        % exp prior for 1/alpha
        L = -gam_a./alpha;
end