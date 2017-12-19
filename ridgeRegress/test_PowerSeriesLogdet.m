% The MATLAB code for Example 1 (and Example 2)
for nn=1:100, % the trivial initialization code is omitted for saving space
epsilon=1e-3;
C=5*(rand(N, N)-0.5)*2; % use C=5*rand(N, N) in Example 2 for numerical tests
C=C+C'; % make C symmetric
eigC=eig(C); eigCmin=min(eigC);
if eigCmin<0, % to make the minimal eigenvalue of C equal to epsilon if it is not greater than 0
C=C+(1+epsilon)*(-eigCmin)*eye(N);
elseif eigCmin==0,
C=C+epsilon*eye(N);
end; % if the minimal eigenvalue of original C is greater than 0, no action is taken
[cholC,pp]=chol(C); % Cholesky decomposition for true log-det(C)
if pp==0,
logdet2=2*sum(log(diag(cholC)));
else % turn to SVD for true log-det(C) if Cholesky decomposition fails
logdet2=sum(log(svd(C)));
end
clear cholC;
logdet1=PowerSeriesLogdet(C, N, 10, 30, 1.0);
clear C;
logdet=[logdet1 logdet2]; abserr=logdet1-logdet2;
relae=abserr/max(abs(logdet)); relaerr(nn,1)=relae;
if abs(relae)>0.1,
disp('error larger than 0.1:'); logdet
end
end; % the vector relaerr is saved with its mean and variance values calculated

% The MATLAB code for Example 3 (and Examples 4 ? 6)
BOUND1=7; BOUND2=8; SQRTN=53; STEP1=BOUND1/SQRTN; STEP2=BOUND2/SQRTN;
[X1,X2]=meshgrid([0:STEP1:BOUND1],[0:STEP2:BOUND2]);
M=[reshape(X1,numel(X1),1) reshape(X2,numel(X2),1)]; % Produce input data M
% In Example 4, M is a time-sequence vector
% Example 5 also uses this kind of input data M
% Example 6 use the input data M generated via the Wiener-Hammerstein system
clear X1 X2 BOUND1 BOUND2 SQRTQ STEP1 STEP2; % remove the variables from thememory space
N=size(M,1);
% inside the testing loop (from 1 to 100 in Examples 3 and 4)
v=0.5*abs(rand(1,1)); a=3*abs(rand(1,1)); g=abs(rand(2,1));
% In Example 4, g is only one dimensional
% In Examples 5 and 6, the hyperparameters are generated online via MLE optimization routines
expA=exp(-0.5*permute(sum((repmat(M,[1,1,N])-...
permute(repmat(M,[1,1,N]),[3,2,1])).^2.*repmat(g',[N,1,N]),2),[1,3,2]));
C=a*expA+a*v*eye(N); clear expA v a g;
logdet1=PowerSeriesLogdet(C,N,10,30,1.0);
