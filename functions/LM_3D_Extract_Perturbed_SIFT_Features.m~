function [] = LM_3D_Extract_Perturbed_SIFT_Features(dataset, KTorresani, energy, KNonrigid, t, choice, perturbed_param, indexes_train, indexes_validate, list_train, list_validate, n_train, n_validate)
	% this function should run in foler PO_CR_code_v1
	% this function is used to initialize p_mat by perturbation and extract SIFT features from initialized p_mat. 
	% output : p_mat : size( n * Kpi, K);  delta_p : size(n * Kpi, K);
	% features : size(n * Kpi, N);  b_mat : size(n * Kpi, N)
	% chosed shape model of KTorresani = 25, energy = 0.9, KNonrigid = 13, K = 19
	
	global VERSIONCHECK; 
	VERSIONCHECK = 'LM_3D_1';
	% default input values
	if nargin == 0
		t = 1; 
		dataset = 'training';
		KTorresani = 25; 
		energy = 0.95;  
		KNonrigid = 17;
		choice = 'all';
		perturbed_param = 1:(KNonrigid + 6); 
	end
	Kpi = 10; 
	s = rng;
	cd('vlfeat-0.9.20/toolbox');
 	vl_setup
	cd('../../');	
	% load models
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
	num_of_pts = 68;                                               % num of landmarks in the annotations
	P = eye(size(myAppearanceLM.A,1)) - myAppearanceLM.A * myAppearanceLM.A'; 
	N = size(myAppearanceLM.A, 1);				 % number of SIFT features
	KNonrigid = size(myShapeLM3D.V, 2);           % number of eigenvectors of myShape.Q
	KRigid = 6;
	K = KNonrigid + KRigid;
	A0P = myAppearanceLM.A0' * P;  
	if strcmp(dataset, 'training')
		n1 = 2000;					
		n2 = 811;
	else
		n1 = 330;
		n2 = 223; 
	end
	n = n1 + n2; 
	
	feat = zeros(Kpi, n, N);
	b_mat_temp = zeros(Kpi, n, N);
	pt_pt_err0_temp = zeros(n , Kpi);
	pt_pt_err0_image = zeros(n, 1);
	rng(s);
	
	% get splited ground truth training and validation parameters
	gtParamDir = '../Learning_Inputs/'; 
	LearningResultsDir = '../Learning_Results/'; 
	if t == 0
		OutputDir =[LearningResultsDir 'PertInit_' dataset '_LM_3D_iteration_' num2str(t) '_' choice '/'];
	else
		OutputDir =[LearningResultsDir 'SIFT_Features_' dataset '_LM_3D_iteration_' num2str(t) '_' choice '/'];
	end
	if(exist(OutputDir, 'dir') == 0)
		mkdir(OutputDir);
	end
	
	%% initialize shape parameters by perturbation and compute delta_p
	if t == 0
		disp( 'initializing perturbed shape parameters');
		TR_Training_Validation_Params = load([gtParamDir 'TR_Training_Validation_Params.mat']);
		noise_scale = 0.35;					% scale the noise level to match the cum_error curve of initialization to that in the paper
		makeplot = 1; 
		
		[p_mat_gt, p_mat_initialization, delta_p_initialization, noise, pt_pt_err0_image, pt_pt_err0_allimages, cum_err0] = InitializePerturbedParams( ...
			TR_Training_Validation_Params.TR_face_size, TR_Training_Validation_Params.TR_myShape_3D_p, TR_Training_Validation_Params.TR_gt_landmarks, ...
			myShapeLM3D.M, myShapeLM3D.V, fd_stat_LM_3D, n, Kpi, noise_scale, K, KNonrigid, ...
			OutputDir, choice, perturb_param, makeplot);
		
		% split initialized parameters into train and validate set
		[p_mat_gt_train, p_mat_gt_validate, p_mat_initialization_train, p_mat_initialization_validate, delta_p_initialization_train, delta_p_initialization_validate, ...
			 pt_pt_err0_image_train, pt_pt_err0_allimages_train, cum_err0_train,  pt_pt_err0_image_validate, pt_pt_err0_allimages_validate, cum_err0_validate] = ...
			split_train_validate_params(p_mat_gt, p_mat_inialization, delta_p_initialization, pt_pt_err0_image, pt_pt_err0_allimages, cum_err0);
		
	else  
		TR_Training_Params = load([gtParamDir 'TR_Training_Params.mat']); 
		if(exist([gtParamDir 'TR_images_train.mat'], 'file'))
			TR_images_train = load([gtParamDir 'TR_images_train.mat']);
		end
		TR_Validation_Params = load([gtParamDir 'TR_Validation_Params.mat']); 
		if(exist([gtParamDir 'TR_images_validate.mat'], 'file'))
			TR_images_validate = load([gtParamDir 'TR_images_validate.mat']);
		end
	
		% load p_mat of iteration t
		InputDir = [LearningResultsDir 'Iteration_' num2str(t) '_' choice '/'] ; 
		Iteration_params = load([InputDir VERSIONCHECK '_' choice '_iteration' num2str(t) '.mat']);
		p_mat_train = Iteration_params.p_mat; 
		p_mat_validate = Iteration_params.p_mat_validate; 
		
% 		save([outputDir VERSIONCHECK '_' choice '_iteration' num2str(t) '.mat'], 'p_mat', 'delta_p', 'reconstruct_lms', 'b_mat', 'Risk', 'Jp', ...
% 		'pt_pt_err_image', 'pt_pt_err_allimages', 'cum_err', 'p_mat_validate', 'delta_p_validate', 'reconstruct_lms_validate', 'choice', 'debug_param', ...
% 		'pt_pt_err_image_validate', 'pt_pt_err_allimages_validate', 'cum_err_validate', 't', 'SIFT_scale', 'Kpi', 'ridge_param', 'learning_rate');  
  	
		% compute delta_p
		[delta_p, delta_p_validate] = ComputeDeltaP(p_mat_train, p_mat_validate);
	end

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


function [p_mat_gt, p_mat_initialization, delta_p_initialization, noise, pt_pt_err0_image, pt_pt_err0_allimages, cum_err0] = InitializePerturbedParams( ...
	TR_face_size, TR_myShape_3D_p, TR_gt_landmarks, myShapeLM3D_M, myShapeLM3D_V,  fd_stat_LM_3D, n, Kpi, noise_scale, K, KNonrigid, OutputDir, ...
	choice, perturb_param, makeplot)

	for gg = 1:n
		face_size = TR_face_size{gg};
		for k = 1 : Kpi
			% record gt and initialize p parameters
			p_mat_gt( (gg - 1) * Kpi + k, :) = TR_myShape_3D_p{gg};
			p_mat_initialization((gg-1) * Kpi + k, :) = TR_myShape_3D_p{gg};

			% construct noise
			noise = zeros(1, K);
			noise(1, 1) =  fd_stat_LM_3D.mean_delta_scale(1,1) + noise_scale * fd_stat_LM_3D.std_delta_scale(1,1) * randn(1); 
			noise(1, 2:3) = (fd_stat_LM_3D.mean_delta_translation(1, :) + fd_stat_LM_3D.std_delta_translation(1, :) .* randn(1,2)) * face_size; 
			noise(1, 4:6) =  fd_stat_LM_3D.std_rotation_euler(1,:) .* randn(1, 3);   %fd_stat_LM_3D.mean_rotation_euler(1,:) +
			noise(1, 7:end) = fd_stat_LM_3D.mean_nonrigid_params(1,:) + fd_stat_LM_3D.std_nonrigid_params(1, :) .* randn(1, KNonrigid); 

			% add noise to scale
			p_mat_initialization((gg-1) * Kpi + k, 1) = p_mat_initialization((gg-1) * Kpi + k, 1) * noise(1,1);

			% add noise to choisen parameters
			p_mat_initialization((gg-1) * Kpi + k, perturb_param) = p_mat_initialization((gg-1) * Kpi + k, perturb_param) + noise_scale * noise(1, perturb_param);

			% compute delta p
			delta_p_initialization((gg-1) * Kpi + k, :) = p_mat_gt( (gg - 1) * Kpi + k, :) - p_mat_initialization((gg-1) * Kpi + k, :); 

			% reconstruct landmarks
			[lm] = getShapeFrom3DParam(myShapeLM3D_M, myShapeLM3D_V, p_mat_initialization((gg-1) * Kpi + k, :) );	%(p_mat_initialization((gg-1) * Kpi + k, :)); 
			pt_pt_err0_temp(gg, k) = my_compute_error(TR_gt_landmarks{gg}, lm, face_size);

% 			if gg < 10 && k == 1
% 				figure; hold on;
% 				plot(lm(:, 1), -lm(:, 2));
% 				plot(TR_gt_landmarks{gg}(:, 1), -(TR_gt_landmarks{gg}(:, 2)));
% 				hold off; 
% 			end

		end
		% compute error and cumulative curve
		 pt_pt_err0_image(gg) = sum(pt_pt_err0_temp(gg, :)) / Kpi;
	end
	[pt_pt_err0_allimages, cum_err0] = Compute_cum_error(pt_pt_err0_image, n, OutputDir, ['cum_error_of_initialization_noice_scale_' num2str(noise_scale) '_' choice], makeplot); 

	save([OutputDir 'Perturbed_Shape_Param_Initialization_LM_3D.mat'], 'p_mat_gt', 'p_mat_initialization', 'delta_p_initialization', 'pt_pt_err0_allimages', 'pt_pt_err0_image', ...
		'cum_err0', 'Kpi', 'noise_scale', 'VERSIONCHECK');
	
end

	









