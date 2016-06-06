function [] = RegressorSM3D()
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
	KTorresani = 10;
	energy = 0.8;
	KNonrigid = 5;
	[myShapeSM3D, myAppearance, fd_stat_SM_3D, std_params, P, A0P, N, m, KNonrigid, KRigid, K] = load_models(KTorresani, energy, KNonrigid);

	% collect training data
% 	[TR_images, TR_face_size, TR_gt_landmarks, TR_myShape_3D_p, TR_detections] = Collect_training_images_3D(n1, n2);

	plot = 1; 
	
	for SIFT_scale = 2
		% collect initial p perturbations according to image numbers n1, n2, SIFT_scale and Kpi
		
		InputDir = 'Inputs/'; 
% 		Split_training_validation_set(n, SIFT_scale, Kpi, InputDir);			% run this if not yet split dataset
		load([InputDir 'TR_PertInit_Training_Validation_Params.mat']);
		
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
				p_mat = p_mat_initialization_train; 
				delta_p = delta_p_initialization_train; 
				b_mat = b_mat_train;
				features = features_train; 
			else
				[features, b_mat] = extract_SIFT_features(myShapeSM3D.M, myShapeSM3D.V,  myShapeSM3D.M_2D, p_mat, TR_images, TR_face_size, SIFT_scale, P, A0P, Kpi, n, n1, n2, lm_ind1, lm_pts);
			end
			
%% ridge regression to compute Jp

			% debug, choise debug parameters
% 			sc_rigid = 0.15; 
% 			sc_nonrigid = 0.4;

			sc_rigid = 0.2; 
			sc_nonrigid = 0.08;
			
			learning_rate = [sc_rigid,  sc_rigid* ones(1,2), sc_rigid * ones(1,3), sc_nonrigid * ones(1, KNonrigid)];		% learning rate for scale, translation, rotation, nonrigid
			cho = {'scale', 'translation', 'rotation', 'scale_translation', 'translation_rotation', ...
				'rigid', 'nonrigid', 'translation_nonrigid', 'all', 'nonrigid1', ...
				'nonrigid2', 'nonrigid3', 'nonrigid12', 'nonrigid123', 'nonrigid1234', 'nonrigid12345'}; 
			debug_param = { [1], [2:3], [4:6], [1:3], [2:6], [1:6], [7:K], [2:3, 7:K], [1:K], [7], [8], [9], [7:8], [7:9], [7:10], [7:11]}; 
			dict = containers.Map(cho, debug_param);
			choice = cho{9}
			debug_param = dict(choice)
			
			% compute Jacobian
			[Jp, Hessian, Risk, reconstruct_b_mat, scaled, std_delta_p] = Compute_Jacobian(b_mat, delta_p, N, ridge_param, choice, std_params, debug_param); 
			

%% update p and compute pt-pt error
			
			% update training p and evaluate on training dataset
			[p_mat, delta_p, reconstruct_lms, pt_pt_err_image, pt_pt_err_allimages, cum_err] = Update_and_Evaluate(p_mat, features, ...
				TR_gt_landmarks_train, TR_face_size_train, Risk, myShapeSM3D.M, myShapeSM3D.V, myAppearance.A0, n_train, Kpi, learning_rate, ...
				N, K, KNonrigid, lm_pts, lm_ind1, t, outputDir, choice, std_params, debug_param, scaled, std_delta_p, plot, 'training'); 
			
			% evaluate on validation dataset
			[p_mat_validate, delta_p_validate, reconstruct_lms_validate, pt_pt_err_image_validate, pt_pt_err_allimages_validate, cum_err_validate] = ...
				Update_and_Evaluate(p_mat_initialization_validate, features_validate, TR_gt_landmarks_validate, TR_face_size_validate, Risk, myShapeSM3D.M, ...
				myShapeSM3D.V, myAppearance.A0, n_validate, Kpi, learning_rate, N, K, KNonrigid, lm_pts, lm_ind1, t, outputDir, choice, std_params, ...
				debug_param, scaled, std_delta_p, plot, 'validation'); 
			
			% save intermediate results per iteration
			save([outputDir VERSIONCHECK '_' choice '_iteration' num2str(t) '.mat'], 'p_mat', 'delta_p', 'reconstruct_lms', 'b_mat', 'Risk', 'Jp', ...
				'pt_pt_err_image', 'pt_pt_err_allimages', 'cum_err', 'p_mat_validate', 'delta_p_validate', 'reconstruct_lms_validate', ...
				'pt_pt_err_image_validate', 'pt_pt_err_allimages_validate', 'cum_err_validate', 't', 'SIFT_scale', 'Kpi', 'ridge_param', 'learning_rate');  
		
		end
	end
end


%% ################################################     helper functions    ##################################################

function [Jp, Hessian, Risk, reconstruct_b_mat, scaled, std_delta_p] = Compute_Jacobian(b_mat, delta_p, N, ridge_param, choice, std_params, debug_param)
	disp(['doing ridge regression seperately for ' choice]);
	K = size(delta_p, 2);
	KNonrigid = K - 6;
	scaled = false; 
	std_delta_p = 0; 
	
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

	%debug, lsqr to compute Jp
	Jp = zeros(N, size(delta_p, 2));
	for reg = 1 : N
		[Jp(reg,:) , flag] = lsqr(delta_p, b_mat(:,reg));
	end
	Jp = Jp(:, debug_param); 
	
	%debug, lsqr with regularization
% 	Jp = zeros(N, size(delta_p, 2));
% 	lambda = 1; 
% 	X = [delta_p; lambda * eye(K)];
% 	
% 	for reg = 1 : N
% 		y = [b_mat(:, reg); zeros(K, 1)]; 
% 		[Jp(reg,:) , flag] = lsqr(X, y);
% 	end
% 	Jp = Jp(:, debug_param); 
	
	% debug, lsqr with scaling
% 	std_delta_p = std(delta_p, 1);
% 	delta_p = delta_p ./ repmat(std_delta_p, size(delta_p, 1), 1); 
% 	Jp = zeros(N, size(delta_p, 2));
% 	for reg = 1 : N
% 		[Jp(reg,:) , flag] = lsqr(delta_p, b_mat(:,reg));
% 	end
% 	Jp = Jp(:, debug_param); 
% 	scaled = true; 
	
	% ----------------------------------------------------------------- %
	reconstruct_b_mat = delta_p(:, debug_param) * Jp'; 
	Hessian = Jp' * Jp; 
	Risk = Hessian \ Jp';		 
	
% 	save('debug/ridge_seperate.mat', 'Jp', 'Hessian', 'Risk', 'reconstruct_b_mat');
end


function [p_mat_new, delta_p, reconstruct_lms, pt_pt_err_image, pt_pt_err_allimages, cum_err] = Update_and_Evaluate(p_mat, features, TR_gt_landmarks, TR_face_size, Risk, M, V,...
	A0, n,  Kpi, learning_rate, N, K, KNonrigid, lm_pts, lm_ind1, t, outputDir, choice, std_params, debug_param, scaled, std_delta_p, plot, dataset)
	
	disp(['updating shape parameters for ' dataset ' dataset']);
			
	p_mat_new = zeros(n * Kpi, K);
	delta_p = zeros(n * Kpi, K);
	reconstruct_lms = zeros(n * Kpi, lm_pts * 2); 
	pt_pt_err_image = zeros(1, n);
	
	for gg = 1:n
		% update p_mat by previous p_mat + delta_p
		[p_mat_new((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), delta_p((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), reconstruct_lms((gg-1)*Kpi + 1 : (gg-1)*Kpi + Kpi, :)] = ...
			Update_p_mat(p_mat((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), Risk, features((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi,:), ...
			M, V, A0,  Kpi, learning_rate, N, K, KNonrigid, choice, std_params, debug_param, scaled, std_delta_p);

		% compute pt-pt error of this image
		[pt_pt_err_image(1, gg)] = Compute_pt_pt_error_image(M, V, p_mat_new((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :),  TR_gt_landmarks{gg}, TR_face_size{gg}, Kpi, gg, lm_ind1);
	end 

	% compute error and cumulative curve
	[pt_pt_err_allimages, cum_err(1, :) ] = Compute_cum_error(pt_pt_err_image(1, :), n, outputDir, ['cum error after ' num2str(t) ' iteration of ' dataset ' dataset, ' choice], plot);

end

function [p_mat_gg_new, delta_p_image, reconstruct_lm] = Update_p_mat(p_mat_gg, Risk, features_image, M, V, A0, Kpi, learning_rate, N, K, KNonrigid, choice, ... 
	std_params, debug_param, scaled, std_delta_p)
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
			
		if scaled
			delta_p_image(k, debug_param)  = delta_p_image(k, debug_param) .* std_delta_p(1, debug_param);
		end
		
		% update p
		[p_mat_gg_new(k, :)] = myCalcReferenceUpdate(delta_p_image(k, :), p_mat_gg(k, :));

		reconstruct_lm(k, :) = reshape(getShapeFrom3DParam(M, V, p_mat_gg_new(k, :) ), 1, []);
	end
end		
				
function [eulers] = Udate_p_mat_rotation_euler(alignFrom, alignTo)
	for k = 1:Kpi
		[ R, T ] = myAlignShapesKabsch(alignFrom(k, :) , alignTo ); 
		eulers = Rot2Euler(R);
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