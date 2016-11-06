function [xtr,xte,ytr,yte, BOW_xtr,BOW_xte, indices_tr, indices_te] = load_data(dataset,seed)

if strcmp(dataset,'ohsumed') || strcmp(dataset,'r83') || strcmp(dataset,'20ng2') || strcmp(dataset,'20ng2_500')
	load(['dataset/', dataset,'_tr_te.mat']);
else
	load(['dataset/', dataset,'_tr_te_split.mat']);
	xtr = X(TR(seed,:));
	xte = X(TE(seed,:));
	BOW_xtr = BOW_X(TR(seed,:));
	BOW_xte = BOW_X(TE(seed,:));
	indices_tr = indices(TR(seed,:));
	indices_te = indices(TE(seed,:));
	ytr = Y(TR(seed,:));
	yte = Y(TE(seed,:));
end
