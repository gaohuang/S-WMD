function [dw, dA] = grad_swmd(xtr, ytr, BOW_xtr, indices_tr, xtr_center, w, A, lambda, batch, range)

%disp(['Computing gradient with mini-batch ', num2str(batch), ' NN range: ', num2str(range)])

epsilon = 1e-8;
huge = 1e8;

dim = size(xtr{1},1);
ntr = length(ytr);

dw = zeros(size(w));
dA = zeros(dim,dim);

% Sample documents
sample_idx = randperm(ntr, batch);

Dc = distance(A * xtr_center, A * xtr_center);
tr_loss = 0;
n_nan = 0;

parfor ii = 1 : batch
% for ii = 1 : batch

    i = sample_idx(ii);
    xi = xtr{i};
    yi = ytr(i);
    idx_i = indices_tr{i};
    bow_i = BOW_xtr{i}';
    a = bow_i .*w(idx_i);
    a = a / sum(a);

    [~, nn_set] = sort(Dc(:,i));

    % Compute WMD from xi to the rest documents
    nn_set = nn_set(2:range+1);
    dd_dA_all = cell(1,range);
    alpha_all = cell(1,range);
    beta_all = cell(1,range);
    Di = zeros(range,1);

    xtr_nn = xtr(nn_set);
    ytr_nn = ytr(nn_set);
    BOW_xtr_nn = BOW_xtr(nn_set);
    indices_tr_nn = indices_tr(nn_set);

    % keyboard
    for j = 1 : range
        % disp(['Computing smoothed WMD ', num2str(j)])
        xj = xtr_nn{j};
        yj = ytr_nn(j);
        M = distance(A*xi, A*xj);
        idx_j = indices_tr_nn{j};
        bow_j = BOW_xtr_nn{j}';
        b = bow_j.*w(idx_j);
        b = b / sum(b);
        [alpha, beta, T, dprimal, ddual] = sinkhorn(M, a, b, lambda, 200, 1e-3);
        Di(j) = dprimal;
        if Di(j) ~= Di(j)
            Di(j) = huge;       
        end
        alpha_all{j} = alpha;
        beta_all{j} = beta;

        % Gradient for metric
        sumA = bsxfun(@times,xi,a')*xi' + bsxfun(@times,xj,b')*xj'- xi*T*xj'-xj*T'*xi';
        dd_dA_all{j} = sumA;
        %keyboard
    end

    % Compute NCA probabilities
    % keyboard
    Di(Di < 0) = 0;
    dmin = min(Di);
    Pi = exp(-Di+dmin) + epsilon;
    Pi(ytr_nn==i) = 0;
    Pi = Pi / sum(Pi);
    pa = sum(Pi(ytr_nn==yi)) + epsilon;   % to avoid division by 0

    % Compute gradient wrt w and A
    dw_ii = zeros(size(w));
    dA_ii = zeros(dim,dim);
    for j = 1 : range
        cij = Pi(j)/pa * (ytr_nn(j)==yi) - Pi(j);
        idx_j = indices_tr_nn{j};
        bow_j = BOW_xtr_nn{j}';
        b = bow_j.*w(idx_j);
        b = b / sum(b);
        % dw_ii(idx_i) = dw_ii(idx_i) + cij * (alpha_all{j} .* bow_i);
        % dw_ii(idx_j) = dw_ii(idx_j) + cij * (beta_all{j} .* bow_j);
        a_sum = w(idx_i)'*bow_i;
        b_sum = w(idx_j)'*bow_j;
        dwmd_dwi = bow_i.*alpha_all{j} / a_sum - bow_i * (alpha_all{j}'*a / a_sum);
        dwmd_dwj = bow_j.*beta_all{j} / b_sum - bow_j * (beta_all{j}'*b / b_sum);
        dw_ii(idx_i) = dw_ii(idx_i) + cij * dwmd_dwi;
        dw_ii(idx_j) = dw_ii(idx_j) + cij * dwmd_dwj;
        dA_ii = dA_ii + cij * dd_dA_all{j};
    end
    if sum(isnan(dw_ii)) == 0 && sum(isnan(dA_ii(:))) == 0
        dw = dw + dw_ii;
        dA = dA + dA_ii;
        tr_loss = tr_loss - log(pa);
    else
        n_nan = n_nan + 1;
    end
end
batch = batch - n_nan;
if n_nan > 0
    disp(['number of bad samples: ', num2str(n_nan)])
end

tr_loss = tr_loss / batch;
dA = A * dA;
dw = dw / batch;
dA = dA / batch;
%disp(['training loss: ', num2str(tr_loss)])
