% Plot simulation results
clear all;close all
load('simout/Hyperpar');%'hprsD1','hprsD3','hprsP','tprimal','tdual1','tdual3','alpha','signse'

figure;
subplot(311)
boxplot(1./[hprsP(1,:)' hprsD1(1,:)' hprsD3(1,:)'])%,'labels' ,{'primal' 'DNC1' 'DNC2'}')
hold on 
plot(1,alpha,'x',2,alpha,'x',3,alpha,'x','markersize',12)
hold off
title('\alpha')
legend('true parameter','location','NW')
set(gca,'XTickLabel',{''})

subplot(312)
boxplot(sqrt([hprsP(2,:)' hprsD1(2,:)' hprsD3(2,:)']))%,'labels' ,{'primal' 'DNC1' 'DNC2'})
hold on 
plot(1,signse,'x',2,signse,'x',3,signse,'x','markersize',12)
hold off
title('\sigma')
set(gca,'XTickLabel',{''})


subplot(313);boxplot([tprimal tdual1' tdual3],'labels' ,{'primal' 'DNC1' 'DNC2'})
title('time')
ylabel('seconds')