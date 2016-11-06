clear
clc
format compact

RAND_SEED = 1;
rng(RAND_SEED,'twister')

addpath(genpath('functions'))

save_path = 'results/'

dataset = 'bbcsport'; 
MAX_DICT_SIZE = 50000; 

% Optimization parameters
max_iter = 200 % number of iterations
save_frequency = max_iter % frequency of saving results
batch = 32   % batch size in batch gradient descent (B in the paper)
range = 200  % neighborhood size (N in the paper)
lr_w = 1e+1  % learning rate for w
lr_A = 1e+0  % learing rate for A
lambda = 10  % parameter in regularized transport problem (lambda in the paper)

cv_folds = 5 % number of folds for cross-validation
for split = 1:cv_folds
    save_couter = 0;
    Err_v = [];
    Err_t = [];
    w_all = [];
    A_all = [];
    [xtr,xte,ytr,yte, BOW_xtr,BOW_xte, indices_tr, indices_te] = load_data(dataset, split);
    [idx_tr, idx_val] = makesplits(ytr, 1-1/cv_folds, 1, 1);

    xv = xtr(idx_val);
    yv = ytr(idx_val);
    BOW_xv = BOW_xtr(idx_val);
    indices_v = indices_tr(idx_val);
    xtr = xtr(idx_tr);
    ytr = ytr(idx_tr);
    BOW_xtr = BOW_xtr(idx_tr);
    indices_tr = indices_tr(idx_tr);
    
    ntr = length(ytr);
    nv = length(yv);
    nte= length(yte);
    dim = size(xtr{1},1);

    % Compute document center
    xtr_center = zeros(dim, ntr);
    for i  = 1:ntr
        xtr_center(:,i) = xtr{i} * BOW_xtr{i}' / sum(BOW_xtr{i});
    end
    xv_center = zeros(dim, nv);
    for i  = 1:nv
        xv_center(:,i) = xv{i} * BOW_xv{i}' / sum(BOW_xv{i});
    end
    xte_center = zeros(dim, nte);
    for i  = 1:nte
        xte_center(:,i) = xte{i} * BOW_xte{i}' / sum(BOW_xte{i});
    end

    % Load intialize A (trained with WCD)
    load(['metric_init/', dataset, '_seed', num2str(split), '.mat'])
    A = Ascaled;
    % Define optimization parameters
    w = ones(MAX_DICT_SIZE,1);  % weights over all words in the dictionary

    % Test learned metric for WCD
    Dc = distance(xtr_center, xte_center);
    err_wcd = knn_fall_back(Dc,ytr,yte,1:19)
    Dc = distance(A * xtr_center, A * xte_center);
    err_swcd = knn_fall_back(Dc,ytr,yte,1:19)

    tStart = tic;

    % Main loop
    for iter = 1 : max_iter

        fprintf('Dataset: %s  Split: %d  Iteration: %d \n',dataset,split,iter)

        tIterStart = tic;        
        [dw, dA] = grad_swmd(xtr,ytr,BOW_xtr,indices_tr,xtr_center,w,A,lambda,batch,range);

        % Update w and A
        w = w - lr_w * dw;
        lower_bound = 0.01;
        upper_bound = 10;
        w(w<lower_bound) = lower_bound;
        w(w>upper_bound) = upper_bound;
        A = A - lr_A * dA;

        % Compute loss
        filename = [save_path, dataset,'_', num2str(lambda), '_', num2str(lr_w), '_', num2str(lr_A), '_', num2str(max_iter), '_', num2str(batch), '_', num2str(range), '_', num2str(split), '.mat'];
        if mod(iter, save_frequency) == 0
            save_couter = save_couter + 1;
            err_v = knn_swmd(xtr, ytr, xv, yv, BOW_xtr, BOW_xv, indices_tr, indices_v, w, lambda, A);
            err_t = knn_swmd([xtr, xv], [ytr, yv], xte, yte, [BOW_xtr, BOW_xv], BOW_xte, [indices_tr, indices_v], indices_te, w, lambda, A);
            Err_v(save_couter,:) = err_v;
            Err_t(save_couter,:) = err_t;
            w_all{save_couter} = w;
            A_all{save_couter} = A;
            save(filename,'Err_v', 'Err_t', 'w_all', 'A_all', 'Ascaled')
        end
        tIterEnd = toc(tIterStart)
    end
    
    tt = toc(tStart);

    err_t_cv = Err_t(Err_v == min(min(Err_v)));
    T(split) = tt;
    results_cv(split) = err_t_cv(1)
    save([save_path, dataset, '_results'], 'results_cv', 'T')
end