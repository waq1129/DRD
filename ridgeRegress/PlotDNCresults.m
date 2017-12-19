% Plot output 
rho = 2; % prior variance
alpha = 1/rho;  % prior precision

%  Set stimulus and sample parameters
signse = 5;   % stdev of added noise

load('simout/Nkernels');%,'kridgep','kridged','hprsD','hprsP')

figure;boxplot([hprsD(1,:) hprsP(1,:)],['primal' 'DNC dual'])
title('\rho')

figure;boxplot([hprsD(2,:) hprsP(2,:)],['primal' 'DNC dual'])
title('\sigma^2')

figure;plot(hprsD(1,:), hprsP(1,:),'.')
title('\rho');xlabel('primal');ylabel('DNC dual')

figure;plot(hprsD(2,:), hprsP(2,:),'.')
title('\sigma^2');xlabel('primal');ylabel('DNC dual')
