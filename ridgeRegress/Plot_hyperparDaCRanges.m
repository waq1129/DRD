%% plot hyperparameter results

clear all;close all;
load('simout/Hyperpar_range');%'tprimal','tdual','hprsP','hprsD','sampszs','subsets','rho','signse','flagD','nk'

%% Plot conditions that do not meet subset sampling criterion of
figure;imagesc(squeeze(flagD(1,:,:)))
xticklabels   = cell(1,numel(subsets));
yticklabels = cell(1,numel(sampszs));
ylabel('sample size');xlabel('subsets')
title('Dubious (K<N^{1/3})=true')
for i = 1:numel(subsets);xticklabels{i} = num2str(subsets(i));end
for i = 1:numel(sampszs);yticklabels{i} = num2str(sampszs(i));end

set(gca,'XTick',1:numel(subsets))
set(gca,'YTick',1:numel(sampszs))
set(gca,'Xticklabel',xticklabels)
set(gca,'Yticklabel',yticklabels)
set(gca,'ydir','reverse')
colormap gray
%% plot time for evaluation
mtdual = squeeze(mean(tdual,1));
mtprimal = mean(tprimal,1);
figure;imagesc(log10(mtdual));set(gca,'YDir','reverse');
title('DaC eval logtime')
ylabel('sample size');xlabel('subsets')
colorbar

set(gca,'XTick',1:numel(subsets))
set(gca,'YTick',1:numel(sampszs))
set(gca,'Xticklabel',xticklabels)
set(gca,'Yticklabel',yticklabels)


figure;errorbar(sampszs,mtprimal,2*std(tprimal,0,1)./sqrt(numel(sampszs)))
title('Primal eval time')
ylabel('time (sec)');xlabel('sample size')


%% plot MSE

% For primal
rhohats = squeeze(hprsP(1,:,:));
signsehats = squeeze(hprsP(2,:,:));
rhoMSEP = mean((rhohats-ones(size(rhohats))*rho).^2,1);
signseMSe = mean((signsehats-ones(size(signsehats))*signse).^2,1);

figure;semilogy(sampszs,rhoMSEP,'o')
xlabel('sample size');ylabel('MSE')
title('\rho MSE primal')
figure;semilogy(sampszs,signseMSe,'o')
xlabel('sample size');ylabel('MSE')
title('\sigma^2 MSE primal')


% For Dual
rhohats = squeeze(hprsD(1,:,:,:));
signsehats = squeeze(hprsD(2,:,:,:));
rhoMSEP = squeeze(mean((rhohats-ones(size(rhohats))*rho).^2,1));
signseMSe = squeeze(mean((signsehats-ones(size(signsehats))*signse).^2,1));

figure;imagesc(log10(rhoMSEP))
title('log_{10}MSE(\rho)  dual')
ylabel('sample size');xlabel('subsets')
set(gca,'XTick',1:numel(subsets))
set(gca,'YTick',1:numel(sampszs))
set(gca,'Xticklabel',xticklabels)
set(gca,'Yticklabel',yticklabels)
colorbar

figure;imagesc(log10(signseMSe))
title('log_{10}MSE(\sigma^2) dual')
ylabel('sample size');xlabel('subsets')
set(gca,'XTick',1:numel(subsets))
set(gca,'YTick',1:numel(sampszs))
set(gca,'Xticklabel',xticklabels)
set(gca,'Yticklabel',yticklabels)
colorbar

%% plot bias

% For primal
rhohats = squeeze(hprsP(1,:,:));
rhoSD = std(rhohats,0,1);
% signseVar = sqrt(mean((signsehats-mean(signsehats).^2,1));

figure;errorbar(sampszs,mean(rhohats,1),rhoSD)
hold on
% for dual
rhohats = squeeze(hprsD(1,:,:,:));
rhoSD = squeeze(std(rhohats,0,1));
mrho  = squeeze(mean(rhohats,1));
for k = 1:numel(subsets)-2
    errorbar(sampszs,mrho(:,k),rhoSD(:,k))
end
legend('primal','2','5','10','100','500')
title('\rho')
plot([sampszs(1) sampszs(end)],[rho rho],'k')
hold off


sighats = (squeeze(hprsP(2,:,:)));
sigSD = std(sighats,0,1);
figure;errorbar(sampszs,mean(sighats,1),sigSD)
hold on
% for dual
sighats = squeeze(hprsD(2,:,:,:));
sigSD = squeeze(std(sighats,0,1));
msig  = squeeze(mean(sighats,1));
for k = 1:numel(subsets)-2
    errorbar(sampszs,msig(:,k),sigSD(:,k))
end
legend('primal','2','5','10','100','500')
title('\sigma^2')
plot([sampszs(1) sampszs(end)],[signse signse].^2,'k')
hold off


rhohats = squeeze(hprsP(1,:,:));
rhoSD = std(rhohats,0,1);
sighats = (squeeze(hprsP(2,:,:)));
sigSD = std(sighats,0,1);
lambhats = rhohats./sighats;
lambSD = squeeze(std(lambhats,0,1));
mlamb = squeeze(mean(lambhats,1));
figure;errorbar(sampszs,mlamb,lambSD)
hold on
rhohats = squeeze(hprsD(1,:,:,:));
sighats = (squeeze(hprsD(2,:,:,:)));
lambhats = rhohats./sighats;
lambSD = squeeze(std(lambhats,0,1));
mlamb = squeeze(mean(lambhats,1));
for k = 1:numel(subsets)-2
    errorbar(sampszs,mlamb(:,k),lambSD(:,k))
end
legend('primal','2','5','10','100','500')
title('\lambda')
plot([sampszs(1) sampszs(end)],[rho/signse^2 rho/signse^2],'k')
hold off
ylim([-10 10])