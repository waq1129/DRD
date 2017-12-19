
function result = lambda_interp(lambda,s)
% lambda is the index sequence that is produced by the model
% s is the new vector at which evaluations are required.
% the value is a vector of left and right indices, and a vector of fractions.
% the new values are interpolated bewteen the two using the fraction
% Note: lambda decreases. you take:
% sfrac*left+(1-sfrac*right)

if length(lambda)==1 % degenerate case of only one lambda
    nums=length(s);
    left=ones(nums,1);
    right=left;
    sfrac=ones(nums,1);
else
    s(s > max(lambda)) = max(lambda);
    s(s < min(lambda)) = min(lambda);
    k=length(lambda);
    sfrac =(lambda(1)-s)/(lambda(1) - lambda(k));
    lambda = (lambda(1) - lambda)/(lambda(1) - lambda(k));
    coord = interp1(lambda, 1:length(lambda), sfrac);
    left = floor(coord);
    right = ceil(coord);
    sfrac=(sfrac-lambda(right))./(lambda(left) - lambda(right));
    sfrac(left==right)=1;
end
result.left = left;
result.right = right;
result.frac = sfrac;
