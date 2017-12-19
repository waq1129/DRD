function [w_elnet, w_lasso, w_ridge, lambda_elnet, lambda_lasso, lambda_ridge, alpha_elnet] = run_lasso_elnet(x, y)

betas=[];
lambdas=[];
cvm=[];
alpha_range=[0:0.1:1];
for alpha=alpha_range
    alpha
    options.alpha=alpha;
    options.lambda_min = 0.00001;
    options.nlambda=500;
    cvob = cvglmnet(x, y,[],options);
    [minloc minval]=find(cvob.cvm==min(cvob.cvm));
    minloc = minloc(1);
    betas = [betas cvob.glmnet_fit.beta(:,minloc)];
    lambdas = [lambdas cvob.glmnet_fit.lambda(minloc,:)];
    cvm=[cvm;min(cvob.cvm)];
end

[minloc minval]=find(cvm==min(cvm));
w_elnet = betas(:,minloc);
w_lasso = betas(:,end);
w_ridge = betas(:,1);
lambda_elnet = lambdas(minloc);
lambda_lasso = lambdas(end);
lambda_ridge = lambdas(1);
alpha_elnet = alpha_range(minloc);

% function [w_elnet] = run_lasso_elnet(x, y)
% 
% cc = 1;
% alpha_range=[0.000001:0.1:1];
% for alpha=alpha_range
%     k = 5;
%     c = cvpartition(numel(y),'kfold',k);
%     ind = randperm(numel(y));
%     x = x(ind,:);
%     y = y(ind);
%     err = [];
%     for i=1:k
%         x_train = x(c.training(i),:);
%         y_train = y(c.training(i));
%         x_test = x(c.test(i),:);
%         y_test = y(c.test(i));
%         ind = randperm(numel(y_train));
%         x_train = x_train(ind,:);
%         y_train = y_train(ind);
%         ind = randperm(numel(y_test));
%         x_test = x_test(ind,:);
%         y_test = y_test(ind);
%         
%         [w, fitinfo] = lasso(x_train,y_train,'alpha',alpha);
%         y_pred = x_test*w;
%         
%         y_pred1 = y_pred./repmat(sqrt(sum(y_pred.^2,2)),1,size(y_pred,2));
%         mse_te = sqrt(sum((repmat(y_test,1,size(y_pred,2))-y_pred1).^2,1)); 
%         minloc = find(mse_te==min(mse_te));
%         
%         err = [err mse_te(minloc(1))];
%     end
%     errs(cc)=mean(err);
%     cc = cc+1;
% end
% 
% minloc = find(errs==min(errs));
% w_elnet = (x'*x+alpha_range(minloc(1))*eye(size(x,2)))\(x'*y);
% [w_lasso, fitinfo] = lasso(x_train,y_train,'alpha',1,'CV',5);
