function [] = regressorsep3D()
% chosed shape model of KTorresani = 10, energy = 0.8, KNonrigid = 5
	clear;
	global VERSIONCHECK; 
	VERSIONCHECK = 'SM_3D_1'; 
	Kpi = 10;
	T = 1;
	ridge_param = 0;
	smallsize = 0;
	use_5_lms = 1;
	outputDir = 'Results/'; 
% 	Setup_createDir(outputDir, 0);
	datasetDir = '../dataset/'; 
	testsetDir = '../test_data/';
	gtParamDir = 'TR_params/';
	CLMDir = './';
	num_of_pts = 68;                                              % num of landmarks in the annotations
	% initialize parameters
	if smallsize == 0
		n1 = 2000;					   
		n2 = 811;	
	else
		n1 = 30;
		n2 = 30;
	end
	n = n1 + n2; 
	lm_pts = 5;
	lm_ind1 = [34, 37, 46, 61, 65]; 
	lm_ind2 = [34, 37, 46, 61, 65, 102, 105, 114, 129, 133]; 
	% load models
	KTorresani = 4;
	energy = 0.999;
	KNonrigid = 7;
	[myShapeSM3D, myAppearance, fd_stat_SM_3D, std_params, P, A0P, N, m, KNonrigid, KRigid, K] = load_models(KTorresani, energy, KNonrigid);

	% collect training data
% 	[TR_images, TR_face_size, TR_gt_landmarks, TR_myShape_3D_p, TR_detections] = Collect_training_images_3D(n1, n2);

	gtParamDir = 'TR_params/';
	load([gtParamDir 'TR_detections.mat']); 
	load([gtParamDir 'TR_face_size.mat']);
	load([gtParamDir 'TR_gt_landmarks.mat']);

	if T > 1
		load([gtParamDir 'TR_images.mat']);
	end
	InitializationDir = '../PerturbationInitialization_SM_3D_0.7_nonrigid/';

	for SIFT_scale = 2
		% collect initial p perturbations according to image numbers n1, n2, SIFT_scale and Kpi
		[p_mat_initialization, delta_p_initialization, p_mat_gt, Perturbed_Shape_Param_Initialization, Perturbed_SIFT_Feature_labels, Perturbed_SIFT_Features] = ...
			loadInitialPerturbations(InitializationDir, n1, n2, SIFT_scale, Kpi, smallsize, use_5_lms);

		for gg = 1 : n
			[pt_pt_err_image_gt(gg)] = Compute_pt_pt_error_image1(myShapeSM3D.M, myShapeSM3D.V, p_mat_gt((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :) , TR_gt_landmarks{gg}, TR_face_size{gg}, gg, lm_ind1);
		end
		[pt_pt_err_allimages_gt, cum_err_gt] = Compute_cum_error(pt_pt_err_image_gt, n, outputDir, 'ground truth cum_error', 1);

		% initialize learning parameters
		p_mat = zeros(n * Kpi, K); 
		delta_p = zeros(n*Kpi, N);
		reconstruct_lms = zeros(n*Kpi, lm_pts * 2);
		pt_pt_err_image	= zeros(T, n);
		pt_pt_err_allimages = zeros(T);

%% training
		for t = 1 : T  
			disp(['iteration is ' num2str(t)]);
%% extract features using perturbed p and compute b_mat (labels)
			if t == 1
				p_mat = p_mat_initialization; 
				delta_p = delta_p_initialization; 
				b_mat = Perturbed_SIFT_Feature_labels; 
				features = Perturbed_SIFT_Features; 
				% plot initialization cum error curve
				[pt_pt_err_allimages0, cum_err0] = Compute_cum_error(Perturbed_Shape_Param_Initialization.pt_pt_err0_image, n, outputDir, 'cum error after initilaization', 1);
			else
				[features, b_mat] = extract_SIFT_features(myShapeSM3D.M, myShapeSM3D.V,  myShapeSM3D.M_2D, p_mat, TR_images, TR_face_size, SIFT_scale, P, A0P, Kpi, n, n1, n2, lm_ind1, lm_pts);
			end
% 			save('debug/groundtruthParam.mat', 'b_mat', 'delta_p');
%% ridge regression to compute Jp
			sep_joint = 1;
			NORMALIZE = 0;

			% debug, choise debug parameters
			learning_rate = [0.2,  0.2* ones(1,2), 0.2 * ones(1,3), 0.2 * ones(1, KNonrigid)];	% learning rate for scale, translation, rotation, nonrigid
			cho = {'scale', 'translation', 'rotation', 'scale_translation', 'translation_rotation', ...
				'rigid', 'nonrigid', 'translation_nonrigid', 'nonrigid1', 'nonrigid2', ...
				'nonrigid3', 'nonrigid12', 'nonrigid123', 'nonrigid1234', 'nonrigid12345', 'all'}; 
			debug_param = { [1], [2:3], [4:6], [1:3], [2:6], [1:6], [7:K], [2:3, 7:K], [7], [8], [9], [7:8], [7:9], [7:10], [7:11], [1:K]}; 
			dict = containers.Map(cho, debug_param);
			choice = cho{7}
			debug_param = dict(choice)
			
			if sep_joint == 1
				 [Jp, Hessian, Risk, reconstruct_b_mat] = ridge_seperate(b_mat, delta_p, N, ridge_param, choice, std_params, NORMALIZE, debug_param); 
			else
				[Jp, Hessian, Risk, reconstruct_b_mat] = ridge_joint(b_mat, delta_p, N, ridge_param);
			end

%% update p and compute pt-pt error
			disp('updating shape parameters for Helen train set and computing pt-pt error');
			for gg = 1:n
				% update p_mat by previous p_mat + delta_p
				[p_mat((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), delta_p((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), reconstruct_lms((gg-1)*Kpi + 1 : (gg-1)*Kpi + Kpi, :)] = ...
					Update_p_mat(p_mat((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), Risk, features((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi,:), ...
					myShapeSM3D.M, myShapeSM3D.V, myAppearance.A0,  Kpi, learning_rate, N, K, KNonrigid, choice, std_params, NORMALIZE, debug_param);

				% update rotation euler parameters
% 					[p_mat((gg-1) *Kpi + 1 : (gg-1) *Kpi + Kpi, 4:6)] = Udate_p_mat_rotation_euler(reconstruct_lms((gg-1)*Kpi + 1 : (gg-1)*Kpi + Kpi, :), TR_gt_landmarks{gg});	% for every image

				% compute pt-pt error of this image
				[pt_pt_err_image(t, gg)] = Compute_pt_pt_error_image(myShapeSM3D.M, myShapeSM3D.V, p_mat((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :),  TR_gt_landmarks{gg}, TR_face_size{gg}, Kpi, gg, lm_ind1);
			end 

			% compute error and cumulative curve
			[pt_pt_err_allimages(t), cum_err(t, :) ] = Compute_cum_error(pt_pt_err_image(t, :), n, outputDir, ['cum error after ' num2str(t) ' iteration ' choice], 1);

			% save intermediate results per iteration
			save([outputDir VERSIONCHECK '_' choice '_iteration' num2str(t) '.mat'], 'sep_joint', 'b_mat', 'Risk', 'Jp', 'p_mat', 'delta_p', 't', 'SIFT_scale', 'Kpi', 'ridge_param', 'learning_rate'); 
		end 

		% save result
		save([outputDir VERSIONCHECK '_' choice '_finalresults.mat'], 'sep_joint', 'pt_pt_err_image', 'pt_pt_err_allimages', 'cum_err', 'SIFT_scale', 'Kpi', 'ridge_param', 'learning_rate');
	end
end


%% ################################################     helper functions    ##################################################


function [Jp, Hessian, Risk, reconstruct_b_mat] = ridge_seperate(b_mat, delta_p, N, ridge_param, choice, std_params, NORMALIZE, debug_param)
	disp(['doing ridge regression seperately for ' choice]);
	K = size(delta_p, 2);
	KNonrigid = K - 6;

	% debug, seperately
% 	Jp = zeros(N, size(debug_param, 2));
% 	for reg = 1:N
% 		Jp(reg,:) = ridge( b_mat(:,reg),  delta_p(:, debug_param), 0);
% 	end
	
	% debug, jointly
% 	b_mat_joint = sum(b_mat, 2); 
% 	delta_p_joint = repmat(delta_p(:, debug_param), 1, N); 
% 	Jp_joint = ridge(b_mat_joint, delta_p_joint, 0) ; 
% 	Jp = reshape(Jp_joint', 2, [])';
	

	% debug, direct compute
% 	Jp_transpose = delta_p(:, debug_param)' * delta_p(:, debug_param) \ (delta_p(:, debug_param)' * b_mat); 
% 	Jp = Jp_transpose'; 

	%debug, use lsqr to compute Jp
	Jp = zeros(N, size(delta_p, 2));
	for reg = 1 : N
		[Jp(reg,:) , flag] = lsqr(delta_p, b_mat(:,reg));
	end
	Jp = Jp(:, debug_param); 
	
	% -----------------------------------------------------------------%
	reconstruct_b_mat = delta_p(:, debug_param) * Jp'; 
	Hessian = Jp' * Jp; 
	Risk = Hessian \ Jp';		 
	
% 	save('debug/ridge_seperate.mat', 'Jp', 'Hessian', 'Risk', 'reconstruct_b_mat');
end

function [p_mat_gg_new, delta_p_image, reconstruct_lm] = Update_p_mat(p_mat_gg, Risk, features_image, M, V, A0, Kpi, learning_rate, N, K, KNonrigid, choice, std_params, NORMALIZE, debug_param)
	num_lm = 5; 
	delta_p_image = zeros(Kpi, K);
	p_mat_gg_new = zeros(Kpi, K);
	reconstruct_lm = zeros(Kpi, num_lm * 2); 
	for k = 1 : Kpi
		if isContinuous(debug_param) == 1
			delta_p_image(k, :) = [zeros(debug_param(1)-1, 1); learning_rate(1, debug_param)' .* ( Risk * (reshape(features_image(k, :), [], 1) - A0) ); zeros(K - debug_param(end), 1)]';
		else
			if strcmp(choice, 'translationandnonrigid123')
				delta_p_temp =  learning_rate(1, debug_param)' .* ( Risk * (reshape(features_image(k, :), [], 1) - A0) );
				delta_p_image(k, :) =[zeros(1,1); delta_p_temp(1:2, 1); zeros(3,1); delta_p_temp(3:end, 1); zeros(KNonrigid-3, 1)]';   
			elseif strcmp(choice , 'translationandnonrigid')
				delta_p_temp =  learning_rate(1, debug_param)' .* ( Risk * (reshape(features_image(k, :), [], 1) - A0) );
				delta_p_image(k, :) =[zeros(1,1); delta_p_temp(1:2, 1); zeros(3, 1); delta_p_temp(3:end, 1)]'; 
			end
		end
			
		if NORMALIZE == 1
			delta_p_image(k, debug_param)  = delta_p_image(k, debug_param) .* std_params(1, debug_param);
		end
		
		% update p
		[p_mat_gg_new(k, :)] = myCalcReferenceUpdate(delta_p_image(k, :), p_mat_gg(k, :));

		reconstruct_lm(k, :) = reshape(getShapeFrom3DParam(M, V, p_mat_gg_new(k, :) ), 1, []);
	end
end		
				
function [eulers] = Udate_p_mat_rotation_euler(alignFrom, alignTo)
	for k = 1:Kpi
		[ R, T ] = myAlignShapesKabsch (alignFrom(k, :) , alignTo ); 
		eulers = Rot2Euler(R);
	end
end

function [Jp, reconstruct_b_mat] = ridge_joint(b_mat, delta_p, N, ridge_param)
	disp('doing ridge regression jointly');
	Jp_joint = zeros(N * KRigid,1);
	delta_p_new_joint = repmat(delta_p, 1, N);
	b_mat_new_joint = sum(b_mat, 2);
	Jp_joint = ridge(b_mat_new_joint, delta_p_new_joint, ridge_param);
	reconstruct_b_mat = delta_p_new_joint * Jp_joint; 
	start = 1;
	nn = size(Jp_joint,1) / N; 
	for reg = 1:N
		Jp(reg,:) = Jp_joint(start : start + nn -1);
		start = start + nn; 
	end
end

function [features, b_mat] =  extract_SIFT_features(M, V, M_2D, p_mat, TR_images, TR_face_size, SIFT_scale, P, A0P, Kpi, n, n1, n2, lm_ind1, lm_pts )
	disp( 'extracting features from training dataset');
	num_feat_per_lm = 128;
	feat = zeros(Kpi, n,  num_feat_per_lm * lm_pts);
	b_mat_temp = zeros(Kpi, n,  num_feat_per_lm * lm_pts);
	disp('finished reading all images');
	parfor gg = 1 : n
		face_size = TR_face_size{gg};
		input_image = TR_images{gg};

		for k = 1 : Kpi
			[lm] = getShapeFrom3DParam(M, V, p_mat((gg-1) * Kpi + k, :) );
			Sfeat = SIFT_features(input_image, lm, SIFT_scale, face_size, M_2D);
			feat(k, gg, :) = reshape(Sfeat, 1, []); 
			b_mat_temp(k, gg, :) =  reshape(feat(k , gg, :), 1, []) * P - A0P;
		end
	end   
	features = reshape(feat, n * Kpi, []);
	b_mat = reshape(b_mat_temp, n * Kpi, []);
end
			
function [pt_pt_err_image] = Compute_pt_pt_error_image1(M, V, p_mat_gg, gt_landmark, face_size, gg, lm_ind1)
	lm = getShapeFrom3DParam(M, V, p_mat_gg(1, :));   
	pt_pt_err_image = my_compute_error(gt_landmark(lm_ind1,:), lm, face_size );
end

function [pt_pt_err_image] = Compute_pt_pt_error_image(M, V, p_mat_gg, gt_landmark, face_size, Kpi, gg, lm_ind1)
	pt_pt_err_k = zeros(1, Kpi);
	for k = 1 : Kpi
		lm =  getShapeFrom3DParam(M, V, p_mat_gg(k, :));   
		pt_pt_err_k(1, k) = my_compute_error(gt_landmark(lm_ind1,:), lm, face_size );
	end
	pt_pt_err_image = sum(pt_pt_err_k) / Kpi;
end

function [myShapeSM3D, myAppearance, fd_stat_SM_3D, std_params, P, A0P, N, m, KNonrigid, KRigid, K] = load_models(KTorresani, energy, KNonrigid)
% load models: shape model, myShape model, myAppearance model, fd_stat model, and compute static variables.
	modelDir = 'matfiles/';
	myAppearance = load([modelDir 'myAppearanceSM.mat']);
	fd_stat_SM_3D = load([modelDir 'fd_stat_SM_3D.mat']);
	if fd_stat_SM_3D.version ~= 'SM_3D_1'
		disp('fd_stat_SM_3D model is stale');
	end
	std_params = [fd_stat_SM_3D.std_delta_scale, fd_stat_SM_3D.std_delta_translation, fd_stat_SM_3D.std_rotation_euler, fd_stat_SM_3D.std_nonrigid_params];
	myShapeSM3D = load([modelDir 'myShapeSM3D.mat']);
	if myShapeSM3D.version ~= 'SM_3D_1'
		disp('myShapeSM3D model is stale');
	end
	P = eye(size(myAppearance.A,1)) - myAppearance.A * myAppearance.A'; 
	A0P = myAppearance.A0' * P;  
	N = size(myAppearance.A, 1);				% number of SIFT features
	m = size(myAppearance.A, 2);                            % number of eigenvectors of myAppearance.A
	KNonrigid = size(myShapeSM3D.V, 2);                                     
	KRigid = 6;
	K = KNonrigid + KRigid;
end

function [p_mat_initialization, delta_p_initialization, p_mat_gt, Perturbed_Shape_Param_Initialization, Perturbed_SIFT_Feature_labels, Perturbed_SIFT_Features] = loadInitialPerturbations(InitializationDir, n1, n2, SIFT_scale, Kpi, smallset, use_5_lms)
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

function [] = saveCollectedData(TR_images, TR_face_size, TR_gt_landmarks, TR_myShape_pRigid, TR_myShape_pNonRigid)
	if(exist('CollectedTrainingDataset/', 'dir') == 0)
		mkdir('CollectedTrainingDataset');
	end
	save('CollectedTrainingDataset/TR_images.mat', 'TR_images'); 
	save('CollectedTrainingDataset/TR_face_size.mat', 'TR_face_size'); 
	save('CollectedTrainingDataset/TR_gt_landmarks.mat', 'TR_gt_landmarks'); 
	save('CollectedTrainingDataset/TR_myShape_pRigid.mat', 'TR_myShape_pRigid'); 
	save('CollectedTrainingDataset/TR_myShape_pNonRigid.mat', 'TR_myShape_pNonRigid');  
end

function [isContinu] = isContinuous(a)	% a is a row vector
	isContinu = 0;
	if size(a, 2) == 1
		isContinu = 1; 
	end
	b = [0 a];
	d = a(2:end) - b(2:end-1);
	n = size(a, 2);
	if sum(d - ones(1,n-1)) == 0
		isContinu = 1; 
	end
end