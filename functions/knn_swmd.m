function [err] = knn_swmd(xtr, ytr, xte, yte,BOW_xtr, BOW_xte, indices_tr, indices_te, w, lambda, A)
ntr = length(ytr);
nte= length(yte);

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1e-4;
end

if ~exist('A', 'var') || isempty(A)
    A = eye(size(xtr{1},1));
end

WMD = zeros(ntr,nte);
parfor i = 1:ntr
    disp([num2str(i) ' done']);
    Wi = zeros(1,nte);
    xi    = xtr{i};
    bow_i = BOW_xtr{i}';
    a = bow_i .*w(indices_tr{i});
    a = a / sum(a);
    for j = 1:nte
        xj    = xte{j};
        bow_j = BOW_xte{j}';
        b =bow_j.*w(indices_te{j});
        b = b / sum(b);
        D  = distance(A*xi, A*xj);
        D(D < 0) = 0;
        D = full(D); 
        [alpha, beta, T, dprimal, ddual] = sinkhorn(D, a, b, lambda, 200, 1e-3);
        Wi(j) = dprimal;
    end
    WMD(i,:) = Wi;
end
err = knn_fall_back(WMD,ytr,yte,1:19);