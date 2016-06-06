function [] = Evaluate_3D_validation(InitializationTestingDir, SIFT_scale, Kpi, smallsize, use_5_lms, outputDir, plot)
% 	InitializationTestingDir = ['../PertInit_testing_SM_3D_0.35_all_param/'];
	n1 = 330; 
	n2 = 223; 
	
	gtParamDir = 'TR_params/';
	load([gtParamDir 'TR_testing.mat']); 
	
	
	% update testing p and evaluate on testing dataset
	[Testing_p_mat_initialization, Testing_delta_p_initialization, Testing_p_mat_gt, Testing_Perturbed_Shape_Param_Initialization, Testing_Perturbed_SIFT_Feature_labels, ...
		Testing_Perturbed_SIFT_Features] = loadInitialPerturbations(InitializationTestingDir, n1, n2, SIFT_scale, Kpi, smallsize, use_5_lms);

	% compute error of testing dataset initialization
	[pt_pt_err_allimages0, cum_err0] = Compute_cum_error(Testing_Perturbed_Shape_Param_Initialization.pt_pt_err0_image, n, outputDir, 'cum error after initilaization', plot);
	
	% compute error after 1st iteration
	[Testing_p_mat, Testing_delta_p,Testing_reconstruct_lms, Testing_pt_pt_err_image, Testing_pt_pt_err_allimages, Testing_cum_err] = Update_and_Evaluate(...
		Testing_p_mat_initialization, Testing_Perturbed_SIFT_Features, TR_gt_landmarks, TR_face_size, ...
		Risk, myShapeSM3D.M, myShapeSM3D.V, myAppearance.A0, n, Kpi, learning_rate, N, K, KNonrigid,  lm_pts, lm_ind1, t, outputDir, choice, std_params, ...
		NORMALIZE, debug_param, 'training'); 
			
end

function [p_mat_initialization, delta_p_initialization, p_mat_gt, Perturbed_Shape_Param_Initialization, Perturbed_SIFT_Feature_labels, Perturbed_SIFT_Features] = loadInitialPerturbations( ...
	InitializationDir, n1, n2, SIFT_scale, Kpi, smallset, use_5_lms)

	% load p initial perturbations according to image number n1, n2 (or full training set) , SIFT scale and Kpi
	global VERSIONCHECK; 
	if smallset == 0 && n1 == 2000 && n2 == 811
		if use_5_lms == 1
			Perturbed_SIFT_Feature_labels = load([InitializationDir 'Perturbed_SIFT_Feature_labels_SM_3D_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			Perturbed_SIFT_Features = load([InitializationDir 'Perturbed_SIFT_Features_SM_3D_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			Perturbed_Shape_Param_Initialization = load([InitializationDir 'Perturbed_Shape_Param_Initialization_SM_3D.mat']);
			Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels.b_mat; 
			Perturbed_SIFT_Features = Perturbed_SIFT_Features.features; 
			
			delta_p_initialization = Perturbed_Shape_Param_Initialization.delta_p_initialization;
			p_mat_initialization = Perturbed_Shape_Param_Initialization.p_mat_initialization;
			p_mat_gt = Perturbed_Shape_Param_Initialization.p_mat_gt; 
			version = Perturbed_Shape_Param_Initialization.VERSIONCHECK; 
			if version ~= VERSIONCHECK
				disp('the PerturbationInitialization_SM_3D is stale'); return;
			end
		end
	end
end