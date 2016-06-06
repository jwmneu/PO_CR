function [] = Extract_Features(t, SIFT_scale, choice_perturb)
	global VERSIONCHECK; 
	VERSIONCHECK = 'LM_3D_1'; 
	iter = t; 
	cd('vlfeat-0.9.20/toolbox');
 	vl_setup
	cd('../../');	
	plotDir = 'SIFT_Plots/'; 
	if(exist(plotDir, 'dir') == 0)
		mkdir(plotDir);
	end
	if nargin == 1
		SIFT_scale = 2; 
		choice_perturb = 'all';
	end
	n = 2811; 
	% load models
	[myShapeLM3D, myAppearanceLM, fd_stat_LM_3D, P, A0P, N, m, KNonrigid, KRigid, K] = load_models(VERSIONCHECK);
	
	% load ground truth parameters
	gtParamDir = '../Learning_Inputs/';			% the Learning Inputs have the splited train and validate sets
	load([gtParamDir 'TR_Training_Params.mat']);
	load([gtParamDir 'TR_Validation_Params.mat']);
	load([gtParamDir 'TR_images_train.mat']);
	load([gtParamDir 'TR_images_validate.mat']);
	
	% load shape parameters p_mat
	LearningResultsDir = '../Learning_Results/'; 
	shapeParamDir = [LearningResultsDir 'Iteration_' num2str(iter) '_' choice_perturb '/']
	load([shapeParamDir 'Shape_Parameters_LM_3D.mat']);
	
	outputDir = [LearningResultsDir 'SIFT_Features_' num2str(iter) '_' choice_perturb '/']
	if(exist(outputDir, 'dir') == 0)
		mkdir(outputDir); 
	end
	% extract features from training set
	[features_train, b_mat_train] = extract_features_helper(SIFT_scale, p_mat_train, TR_face_size_train, TR_images_train, myShapeLM3D.M, myShapeLM3D.V, ...
		myShapeLM3D.M_2D, Kpi, n_train, N, outputDir, P, A0P);
	
	% extract features from validation set
	[features_validate, b_mat_validate] = extract_features_helper(SIFT_scale, p_mat_validate, TR_face_size_validate, TR_images_validate, myShapeLM3D.M, ...
		myShapeLM3D.V, myShapeLM3D.M_2D, Kpi, n_validate, N, outputDir, P, A0P);
	
	% save results
	save([outputDir 'SIFT_Features.mat'], 'features_train', 'b_mat_train', 'features_validate', 'b_mat_validate');	
end

function [features, b_mat] = extract_features_helper(SIFT_scale, p_mat, TR_face_size, TR_images, myShapeLM3D_M, myShapeLM3D_V, myShapeLM3D_M_2D, Kpi, n, N, plotDir, P, A0P)
	disp([ 'extracting features... SIFT scale is ',num2str(SIFT_scale)]);
	feat = zeros(Kpi, n, N);
	b_mat_temp = zeros(Kpi, n, N);
	
	% debug
	parfor gg = 1 : n
		gg
		face_size = TR_face_size{gg};
		input_image = TR_images{gg};

		for k = 1 : Kpi
			lm = getShapeFrom3DParam(myShapeLM3D_M, myShapeLM3D_V, p_mat((gg-1) * Kpi + k, :));
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
end