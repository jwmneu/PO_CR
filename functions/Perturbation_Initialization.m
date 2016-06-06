function [] = Perturbation_Initialization(dataset, choice_perturb, perturb_param)
	% this function should run in foler PO_CR_code_v1
	% this function is used to initialize p_mat by perturbation and extract SIFT features from initialized p_mat. 
	% output : p_mat : size( n * Kpi, K);  delta_p : size(n * Kpi, K);
	% features : size(n * Kpi, N);  b_mat : size(n * Kpi, N)
	% chosed shape model of KTorresani = 25, energy = 0.9, KNonrigid = 13, K = 19
	
	global VERSIONCHECK; 
	VERSIONCHECK = 'LM_3D_1';
	[myShapeLM3D, myAppearanceLM, fd_stat_LM_3D, P, A0P, N, m, KNonrigid, KRigid, K] = load_models(VERSIONCHECK);
	% default input values
	if nargin == 0
		dataset = 'training';
		choice_perturb = 'all';
		perturb_param = 2:(KNonrigid + 6); 
	end
	t = 0; 
	Kpi = 10; 
	cd('vlfeat-0.9.20/toolbox');
 	vl_setup
	cd('../../');	
	rng(0);
	if strcmp(dataset, 'training')
		n1 = 2000;					
		n2 = 811;
	else
		n1 = 330;
		n2 = 223; 
	end
	n = n1 + n2; 
	LearningResultsDir = '../Learning_Results/'; 
	OutputDir = [LearningResultsDir 'Iteration_' num2str(t) '_' choice_perturb '/'];
	if(exist(OutputDir, 'dir') == 0)
		mkdir(OutputDir);
	end
	
	%% initialize shape parameters by perturbation and compute delta_p
	disp( 'initializing perturbed shape parameters');
	
	% get splited ground truth training and validation parameters
	gtParamDir = '../Learning_Inputs/';			% the Learning Inputs have the splited train and validate sets
	load([gtParamDir 'TR_Training_Params.mat']);
	load([gtParamDir 'TR_Validation_Params.mat']);
	
	noise_scale = 0.2;					% scale the noise level to match the cum_error curve of initialization to that in the paper
	makeplot = 1; 

	% initialize training set
	[p_mat_gt_train, p_mat_train, labeled_delta_p_train, noise, pt_pt_err0_image_train, pt_pt_err0_allimages_train, cum_err0_train] = InitializePerturbedParams( ...
		TR_face_size_train, TR_myShape_3D_p_train, TR_gt_landmarks_train, ...
		myShapeLM3D.M, myShapeLM3D.V, fd_stat_LM_3D, n_train, Kpi, noise_scale, K, KNonrigid, ...
		OutputDir, choice_perturb, perturb_param, makeplot);
	
	% initialize validation set
	[p_mat_gt_validate, p_mat_validate, labeled_delta_p_validate, noise, pt_pt_err0_image_validate, pt_pt_err0_allimages_validate, cum_err0_validate] = InitializePerturbedParams( ...
		TR_face_size_validate, TR_myShape_3D_p_validate, TR_gt_landmarks_validate, ...
		myShapeLM3D.M, myShapeLM3D.V, fd_stat_LM_3D, n_validate, Kpi, noise_scale, K, KNonrigid, ...
		OutputDir, choice_perturb, perturb_param, makeplot);
	
	save([OutputDir 'Shape_Parameters_LM_3D.mat'], 'p_mat_gt_train', 'p_mat_train', 'labeled_delta_p_train', 'pt_pt_err0_allimages_train', 'pt_pt_err0_image_train', ...
		'cum_err0_train', 'p_mat_gt_validate', 'p_mat_validate', 'labeled_delta_p_validate', 'pt_pt_err0_allimages_validate', 'pt_pt_err0_image_validate', 'cum_err0_validate', ...
		'Kpi', 'noise_scale', 'VERSIONCHECK', 'choice_perturb', 'perturb_param');
	
end


function [p_mat_gt, p_mat_initialization, labeled_delta_p_initialization, noise, pt_pt_err0_image, pt_pt_err0_allimages, cum_err0] = InitializePerturbedParams( ...
	TR_face_size, TR_myShape_3D_p, TR_gt_landmarks, myShapeLM3D_M, myShapeLM3D_V,  fd_stat_LM_3D, n, Kpi, noise_scale, K, KNonrigid, OutputDir, ...
	choice_perturb, perturb_param, makeplot)

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
			labeled_delta_p_initialization((gg-1) * Kpi + k, :) = p_mat_gt( (gg - 1) * Kpi + k, :) - p_mat_initialization((gg-1) * Kpi + k, :); 

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
	[pt_pt_err0_allimages, cum_err0] = Compute_cum_error(pt_pt_err0_image, n, OutputDir, ['cum_error_of_initialization_' choice_perturb], makeplot); 

end

	









