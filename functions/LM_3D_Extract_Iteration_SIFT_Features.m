function [] = LM_3D_Extract_Iteration_SIFT_Features(dataset, KTorresani, energy, KNonrigid, t, choice)
	% this function should run in foler PO_CR_code_v1
	% this function is used to extract SIFT features from learned p_mat after iteration iter.
	% input: p_mat
	% output :  features : size(n * Kpi, N);  b_mat : size(n * Kpi, N)
	% chosed shape model of KTorresani = 25, energy = 0.9, KNonrigid = 13, K = 19
	
	global VERSIONCHECK; 
	VERSIONCHECK = 'LM_3D_1';
	% default input values
	if nargin == 0
		t = 1; 
		choice = 'all';
		dataset = 'training';
		KTorresani = 25; 
		energy = 0.95;  
		KNonrigid = 17;
	end
	Kpi = 10; 
	s = rng;
	cd('vlfeat-0.9.20/toolbox');
 	vl_setup
	cd('../../');	
	% load models
	modelDir = 'matfiles/';
	myAppearance = load([modelDir 'myAppearanceLM.mat']);
	fd_stat_LM_3D = load([modelDir 'fd_stat_LM_3D.mat']);
	if fd_stat_LM_3D.version ~= VERSIONCHECK
		disp('fd_stat_LM_3D model is stale');
	end
	myShapeLM3D = load([modelDir 'myShapeLM3D.mat']);
	if myShapeLM3D.version ~= VERSIONCHECK
		disp('myShapeLM3D model is stale');
	end
	
	
	
	
	num_of_pts = 68;                                               % num of landmarks in the annotations
	P = eye(size(myAppearance.A,1)) - myAppearance.A * myAppearance.A'; 
	N = size(myAppearance.A, 1);				 % number of SIFT features
	KNonrigid = size(myShapeLM3D.V, 2);           % number of eigenvectors of myShape.Q
	KRigid = 6;
	K = KNonrigid + KRigid;
	
	A0P = myAppearance.A0' * P;  
	if strcmp(dataset, 'training')
		n1 = 2000;					
		n2 = 811;
	else
		n1 = 330;
		n2 = 223; 
	end
	n = n1 + n2; 

	% initialize learning parameters
	p_mat_gt = zeros(n * Kpi, K);
	p_mat_initialization = zeros(n * Kpi, K);
	delta_p_initialization = zeros(n * Kpi, K);
	
	feat = zeros(Kpi, n, N);
	b_mat_temp = zeros(Kpi, n, N);
	pt_pt_err0_temp = zeros(n , Kpi);
	pt_pt_err0_image = zeros(n, 1);
	rng(s);
	
	%% initialize p_mat and add noise to pertubations 
	disp( 'get learned shape parameters');
	
	LearningResultsDir = '../Learning_Results/'; 
	inputDir = [LearningResultsDir 'Iteration_' num2str(t) '/'] ; 
	Iteration_params = load([inputDir VERSIONCHECK '_' choice '_iteration' num2str(t) '.mat']);
	p_mat_initialization = Iteration_params.p_mat; 
	p_mat_initialization_validate = Iteration_params.p_mat_validate; 
	
	% extract features from p_mat and p_mat_validate, and compute delta_p, delta_p_validate
% 	'p_mat', 'delta_p', 'reconstruct_lms', 'b_mat', 'Risk', 'Jp', 'choice', 'debug_param',
% 		'pt_pt_err_image', 'pt_pt_err_allimages', 'cum_err', 'p_mat_validate', 'delta_p_validate', 'reconstruct_lms_validate', ...
% 		'pt_pt_err_image_validate', 'pt_pt_err_allimages_validate', 'cum_err_validate', 't', 'SIFT_scale', 'Kpi', 'ridge_param', 'learning_rate');  
 	
	gtParamDir = 'Inputs/';
	load([gtParamDir 'TR_Training_Params.mat']); 
	if(exist([gtParamDir 'TR_images_train.mat'], 'file'))
		load([gtParamDir 'TR_images_train.mat']);
	end
	
	for gg = 1:n
		face_size = TR_face_size{gg};
		for k = 1 : Kpi
			% record gt and initialize p parameters
			p_mat_gt( (gg - 1) * Kpi + k, :) = TR_myShape_3D_p{gg};
			
		end
	
	end	
% 	% save results
	OutputDir =[LearningResultsDir 'PertInit_' dataset '_LM_3D_iteration_' num2str(t) '_' choice '/'];
	if(exist(OutputDir, 'dir') == 0)
		mkdir(OutputDir);
	end
	
	save([OutputDir 'Perturbed_Shape_Param_Initialization_LM_3D.mat'], 'p_mat_gt', 'p_mat_initialization', 'delta_p_initialization', 'pt_pt_err0_allimages', 'pt_pt_err0_image', ...
		'cum_err0', 'Kpi', 'noise_scale', 'VERSIONCHECK');

	%% extract SIFT features
	plotDir = 'SIFT_Plots/'; 
	if(exist(plotDir, 'dir') == 0)
		mkdir(plotDir);
	end
	myShapeLM3D_M_2D = myShapeLM3D.M_2D; 
	myShapeLM3D_M = myShapeLM3D.M; 
	myShapeLM3D_V = myShapeLM3D.V; 
	
	for SIFT_scale = 2   % 1.5 : 0.5 : 2.5
 		disp([ 'extracting features... SIFT scale is ',num2str(SIFT_scale)]);
		% debug
		parfor gg = 1 : n
			gg
			face_size = TR_face_size{gg};
			input_image = TR_images{gg};

			for k = 1 : Kpi
				lm = getShapeFrom3DParam(myShapeLM3D_M, myShapeLM3D_V, p_mat_initialization((gg-1) * Kpi + k, :));
				if mod(gg, 100) == 0 && k == 1
					makeplot = 1; 
				else
					makeplot = 0;
				end
				Sfeat = my_SIFT_features(input_image, lm, SIFT_scale, face_size, myShapeLM3D_M_2D, makeplot, plotDir, gg);
				feat(k, gg, :) = reshape(Sfeat, 1, []); 
				b_mat_temp(k, gg, :) =  reshape(feat(k,gg, :), 1, []) * P - A0P;
			end
		end
		features = reshape(feat, Kpi * n, []); 
		b_mat = reshape(b_mat_temp, Kpi * n, []);
		save([OutputDir 'Perturbed_SIFT_Feature_labels_LM_3D_S-'  num2str(SIFT_scale) '_Kpi-'  num2str(Kpi) '.mat'], 'b_mat');
		save([OutputDir 'Perturbed_SIFT_Features_LM_3D_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat'], 'features');
	end
	disp('finished this function');
end











