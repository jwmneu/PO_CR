function [myShapeLM3D, myAppearanceLM, fd_stat_LM_3D, P, A0P, N, m, KNonrigid, KRigid, K] = load_models(VERSIONCHECK)
% this function should be run in PO_CR_code_v1 folder
% load models: shape model, myShape model, myAppearance model, fd_stat model, and compute static variables.
	modelDir = 'matfiles/';
	myAppearanceLM = load([modelDir 'myAppearanceLM.mat']);
	fd_stat_LM_3D = load([modelDir 'fd_stat_LM_3D.mat']);
	if fd_stat_LM_3D.version ~= VERSIONCHECK
		disp('fd_stat_LM_3D model is stale');
	end
	myShapeLM3D = load([modelDir 'myShapeLM3D.mat']);
	if myShapeLM3D.version ~= VERSIONCHECK
		disp('myShapeLM3D model is stale');
	end
	P = eye(size(myAppearanceLM.A,1)) - myAppearanceLM.A * myAppearanceLM.A'; 
	A0P = myAppearanceLM.A0' * P;  
	N = size(myAppearanceLM.A, 1);				 % number of SIFT features
	m = size(myAppearanceLM.A, 2);                            % number of eigenvectors of myAppearance.A
	KNonrigid = size(myShapeLM3D.V, 2);		     % number of eigenvectors of myShape.Q
	KRigid = 6;
	K = KNonrigid + KRigid;
	
end
	
	