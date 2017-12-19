function [klasso, lambda_opt] = runLASSO(x, y)
% 5-fold cross validation for lasso
k = 10;
c = cvpartition(numel(y),'kfold',k);
ind = randperm(numel(y));
x = x(ind,:);
y = y(ind);
list = exp([-6:15]);
err = zeros(k,length(list));

for i=1:k
    x_train = x(c.training(i),:);
    y_train = y(c.training(i));
    x_test = x(c.test(i),:);
    y_test = y(c.test(i));
    ind = randperm(numel(y_train));
    x_train = x_train(ind,:);
    y_train = y_train(ind);
    ind = randperm(numel(y_test));
    x_test = x_test(ind,:);
    y_test = y_test(ind);
    for ll=1:length(list)
        lambda = list(ll);
        options.alpha = 1;
        options.lambda = lambda/size(x_train,1);
        optioins.standardize = false;
        fit = glmnet(x_train, y_train, 'gaussian', options);
        w = fit.beta;
        y_pred = x_test*w;
        mse_te = @(a) sum((y_test-a).^2)/sum((y_test-mean(y_test)).^2)-1;
        err(i,ll) = mse_te(y_pred);
    end
end

errs = mean(err,1);
ii = index_min(errs, 2);
minloc = ii(2);
options.alpha = 1;
options.lambda = list(minloc(1))/size(x,1);
optioins.standardize = false;
fit = glmnet(x, y, 'gaussian', options);
klasso = fit.beta;
lambda_opt = options.lambda;



