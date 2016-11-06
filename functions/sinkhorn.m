function [alpha, beta, T, obj_primal, obj_dual] = sinkhorn(M, a, b, lambda, max_iter, tol)

epsilon = 1e-10;

l = length(a);
K = exp( - lambda * M);
Kt = bsxfun(@rdivide, K, a);
u = ones(l, 1)/l;
iter = 0;
change = Inf;

while change > tol && iter <= max_iter
    iter = iter + 1;
    u0 = u;
    u = 1./(Kt*(b./(K' * u)));
    change = norm(u-u0)/norm(u);
end

if min(u) <= 0
    u = u - min(u) + epsilon;
end
v = b./(K' * u);
if min(v) <= 0
    v = v - min(v) + epsilon;
end

alpha = log(u) ;
alpha = 1/lambda * (alpha - mean(alpha));
beta = log(v) ;
beta = 1/lambda * (beta - mean(beta));
T = bsxfun(@times, v', (bsxfun(@times, K, u)));
obj_primal = sum(sum(T.*M));
obj_dual = a' * alpha + b' * beta;
