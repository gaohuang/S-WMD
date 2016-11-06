clear
clc
format compact

RAND_SEED = 1;
rng(RAND_SEED,'twister')

data_set = 'bbcsport' 

MAX_DICT_SIZE = 50000; % dictionary size
scale_set = [1/4, 1/2, 1, 2, 4]; % canditate scale factors
lambda = 10 % default parameter for Sinkhorn algorithm
cv_folds = 5 % number of folds for cross-validation

for split = 1:cv_folds

    results=[];
    best_v = Inf;

    % Load data
    [xtr,xte,ytr,yte, BOW_xtr,BOW_xte, indices_tr, indices_te] = load_data(data_set, split);
    [idx_tr, idx_val] = makesplits_seed(1, ytr, 0.8, 1, 1, 1);
    xv = xtr(idx_val);
    yv = ytr(idx_val);
    BOW_xv = BOW_xtr(idx_val);
    indices_v = indices_tr(idx_val);
    xtr = xtr(idx_tr);
    ytr = ytr(idx_tr);
    BOW_xtr = BOW_xtr(idx_tr);
    indices_tr = indices_tr(idx_tr);

    % Load the metric (A_SWCD) learned from SWCD
    load(['Your_Folder_To_A_SWCD/A_init_', data_set, '_seed_', num2str(split)])


    % First heuristically find a proper scale to avoid numeric problems
    for ii = randperm(length(ytr),500)
        M = distance(A_SWCD*xtr{ii}, A_SWCD*xv{ii});
        dis_max(ii) = max(max(M));
    end
    A_SWCD = A_SWCD/sqrt(mean(dis_max));

    w = ones(MAX_DICT_SIZE,1);

    % Fine search for a proper scale
    for trial = 1:length(scale_set)
        scale_set(trial)
        A = A_SWCD*scale_set(trial);
        err_swmd_v = knn_swmd(xtr, ytr, xv, yv, BOW_xtr, BOW_xv, indices_tr, indices_v, w, lambda, A);
        results(trial, :) = err_swmd_v;
        scale_tried = scale_set(1:trial);

        if min(err_swmd_v) < best_v
            best_v = min(err_swmd_v);
            Ascaled = A;
        end

        save(['metric_init/',data_set,'_seed',num2str(split)], 'Ascaled', 'scale_tried','results')
        if min(err_swmd_v) > 1
            disp('Scale search terminated')
            break
        end
    end
end
