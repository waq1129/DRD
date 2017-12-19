function plot_post(hypid,hypers_estimation,logdprior,dgrid,hypname,iter)
dgridhist = min(hypers_estimation(1:iter,hypid)):.1:max(hypers_estimation(1:iter,hypid));
smpInt = 1;
h = histc(hypers_estimation(1:smpInt:iter,hypid),dgridhist)/iter/.1*smpInt;
h1 = h;
h1 = h1/max(h1);
plot(dgrid, exp(logdprior-max(logdprior)),'g+-', dgridhist, h1,'r*-')
xlabel(hypname); ylabel(['P(',hypname,')']);
title(['P(',hypname,'|u) (iter=',num2str(iter),')']);