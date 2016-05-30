function [] = regressorsep()
	% clear;
	global VERSIONCHECK; 
	VERSIONCHECK = 'SM_1'; 
	Kpi = 10;
	T = 1;
	ridge_param = 0;
	smallsize = 0;
	use_5_lms = 1;
	outputDir = 'Results/'; 
	Setup_createDir(outputDir, 0);
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
	[shapemodel, myShape, myAppearance, fd_stat, P, A0P, N, m, KNonrigid, KRigid, K] = load_models();

	% collect training data
	% [TR_images, TR_face_size, TR_gt_landmarks, TR_myShape_pRigid, TR_myShape_pNonRigid, ~] = Collect_training_images(n1, n2) ; 
% 	load([gtParamDir 'TR_images.mat']);
	load([gtParamDir 'TR_detections.mat']); 
	load([gtParamDir 'TR_face_size.mat']);
	load([gtParamDir 'TR_gt_landmarks.mat']);
	load([gtParamDir 'TR_myShape_p.mat']);
	InitializationDir = '../PerturbationInitialization_SM_0.3_translation_1_nonrigid/';

	for SIFT_scale = 2
		for learning_rate = 1
			% collect initial p perturbations according to image numbers n1, n2, SIFT_scale and Kpi
			[p_mat_initialization, delta_p_initialization, p_mat_gt,  Perturbed_SIFT_Feature_labels, Perturbed_SIFT_Features, Perturbed_Shape_Param_Initializatoin] = loadInitialPerturbations(InitializationDir, n1, n2, SIFT_scale, Kpi, smallsize, use_5_lms);

			for gg = 1 : n
				[pt_pt_err_image_gt(gg)] = Compute_pt_pt_error_image1(myShape.s0, myShape.QNonrigid, p_mat_gt((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :) , TR_gt_landmarks{gg}, TR_face_size{gg}, gg, lm_ind1);
			end
			[pt_pt_err_allimages_gt, cum_err_gt] = Compute_cum_error(pt_pt_err_image_gt, n, outputDir, 'ground truth cum_error', 1);

			% initialize learning parameters
			p_mat = zeros(n * Kpi, K); 
			delta_p = zeros(n*Kpi, N);
			b_mat = zeros(n*Kpi, N);
			pt_pt_err_image	= zeros(T, n);
			pt_pt_err_allimages = zeros(T);
			features = zeros(n*Kpi, N); 

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
					[pt_pt_err_allimages0, cum_err0] = Compute_cum_error(Perturbed_Shape_Param_Initializatoin.pt_pt_err0_image, n, outputDir, 'cum error after initilaization', 1);
				else
					[features, b_mat] = extract_SIFT_features(p_mat, TR_face_size, SIFT_scale, P, A0P, Kpi, n, n1, n2, lm_ind1, lm_pts, myShape_s0 )
				end

%% ridge regression to compute Jp
				sep_joint = 1;
				cho = {'translation', 'scale', 'scaleandtranslation', 'nonrigid', 'translationandnonrigid', 'all', 'nonrigid1', 'nonrigid2', 'nonrigid1234'};
				choice = cho{5}
				if sep_joint == 1
					 [Jp, reconstruct_b_mat] = ridge_seperate(b_mat, delta_p, N, ridge_param, choice, fd_stat.std_nonrigid_params); 
				else
					[Jp, reconstruct_b_mat] = ridge_joint(b_mat, delta_p, N, ridge_param);
				end

%% update p and compute pt-pt error
				disp('updating shape parameters for Helen train set and computing pt-pt error');
				Hessian = Jp' * Jp; 
				Risk = Hessian \ Jp';		 

				for gg = 1:n
					% update p_mat by previous p_mat + delta_p
					[p_mat((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), delta_p((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :)] = ...
						Update_p_mat(p_mat((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), Risk, features((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi,:), myAppearance.A0,  Kpi, learning_rate, N, K, KNonrigid, choice);

					% compute pt-pt error of this image
					[pt_pt_err_image(t, gg)] = Compute_pt_pt_error_image(myShape.s0, myShape.QNonrigid, p_mat((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :),  TR_gt_landmarks{gg}, TR_face_size{gg}, Kpi, gg, lm_ind1);
				end 

				% compute error and cumulative curve
				[pt_pt_err_allimages(t), cum_err(t, :) ] = Compute_cum_error(pt_pt_err_image(t, :), n, outputDir, ['cum error after ' num2str(t) ' iteration ' choice], 1);

				% save intermediate results per iteration
				save([outputDir VERSIONCHECK '_iteration' num2str(t) '.mat'], 'sep_joint', 'b_mat', 'Risk', 'Jp', 'p_mat', 'delta_p', 't', 'SIFT_scale', 'Kpi', 'ridge_param', 'learning_rate'); 

			end 

			% save result
			save([outputDir VERSIONCHECK '_finalresults.mat'], 'sep_joint', 'pt_pt_err_image', 'pt_pt_err_allimages', 'cum_err', 'SIFT_scale', 'Kpi', 'ridge_param', 'learning_rate');
		end
	end
end



%% ################################################     helper functions    ##################################################

function [Jp, reconstruct_b_mat] = ridge_seperate(b_mat, delta_p, N, ridge_param, choice, std_nonrigid_params)
	disp(['doing ridge regression seperately for ' choice]);
	% debug
	if strcmp(choice, 'translation')
		debug_param = 2:3;
	elseif strcmp(choice, 'nonrigid1')
		debug_param = 5;
	elseif strcmp(choice, 'nonrigid2')
		debug_param = 6; 
	elseif strcmp(choice, 'nonrigid1234')
		debug_param = [5 6 7 8];
	elseif strcmp(choice,'scale')
		debug_param = 1;
	elseif strcmp(choice , 'scaleandtranslation')
		debug_param = 1:3;
	elseif strcmp(choice , 'nonrigid')
		debug_param = 5:11;
	elseif strcmp(choice , 'translationandnonrigid')
		K = size(delta_p, 2);
		debug_param = [2:3, 5:K]; 
	elseif strcmp(choice , 'all')
		debug_param = 1:11;
	end
	Jp = zeros(N, size(debug_param, 2));
% 	delta_p(:, debug_param)  = delta_p(:, debug_param) ./ repmat(std_nonrigid_params, size(delta_p, 1), 1); 
% 	for reg = 1:N
% 		Jp(reg,:) = ridge( b_mat(:,reg), delta_p(:, debug_param), ridge_param);
% 	end
	inverse = delta_p(:, debug_param)' * delta_p(:, debug_param) \ delta_p(:, debug_param)' ; 
	Jp_transpose = inverse * b_mat; 
	Jp = Jp_transpose'; 
	reconstruct_b_mat =  delta_p(:, debug_param) * Jp';
end

function [p_mat_gg_new, delta_p_image] = Update_p_mat(p_mat_gg, Risk, features_image, A0, Kpi, learning_rate, N, K, KNonrigid, choice)
	for k = 1 : Kpi
		if strcmp(choice, 'translation')
			delta_p_image(k, :) =[zeros(1,1); learning_rate * Risk * (reshape(features_image(k, :), [], 1) - A0); zeros(1 + KNonrigid, 1)]';   % ; zeros(5 + KNonrigid, 1)
		elseif strcmp(choice, 'nonrigid1')
			delta_p_image(k, :) =[zeros(4,1); learning_rate * Risk * (reshape(features_image(k, :), [], 1) - A0); zeros(KNonrigid-1, 1)]';
		elseif strcmp(choice, 'nonrigid2')
			delta_p_image(k, :) =[zeros(5,1); learning_rate * Risk * (reshape(features_image(k, :), [], 1) - A0); zeros(KNonrigid-2, 1)]';
		elseif strcmp(choice, 'nonrigid1234')
			delta_p_image(k, :) =[zeros(4,1); learning_rate * Risk * (reshape(features_image(k, :), [], 1) - A0); zeros(KNonrigid-4, 1)]';
		elseif strcmp(choice,'scale')
			delta_p_image(k, :) =[learning_rate * Risk * (reshape(features_image(k, :), [], 1) - A0); zeros(3 + KNonrigid, 1)]'; 
		elseif strcmp(choice , 'scaleandtranslation')
			delta_p_image(k, :) =[learning_rate * Risk * (reshape(features_image(k, :), [], 1) - A0); zeros(1 + KNonrigid, 1)]'; 
		elseif strcmp(choice , 'nonrigid')
			delta_p_image(k, :) =[zeros(4,1); learning_rate * Risk * (reshape(features_image(k, :), [], 1) - A0)]'; 
		elseif strcmp(choice , 'translationandnonrigid')
			delta_p_temp = ( learning_rate * Risk * (reshape(features_image(k, :), [], 1) - A0) );
			delta_p_image(k, :) =[zeros(1,1); delta_p_temp(1:2, 1); zeros(1, 1); delta_p_temp(3:end, 1)]'; 
		elseif strcmp(choice , 'all')
			delta_p_image(k, :) =(learning_rate * Risk * (reshape(features_image(k, :), [], 1) - A0))'; 
		end
		p_mat_gg_new(k, :) = reshape(p_mat_gg(k, :), 1, K ) + delta_p_image(k, :);
	end
end		

% function [Jp, reconstruct_b_mat] = ridge_seperate(b_mat, delta_p, N, ridge_param)
% 	disp('doing ridge regression seperately');
% 	debug_params = [2,3]; 
% 	Jp = zeros(N, size(debug_params, 2));
% 	for reg = 1:N
% 		Jp(reg,:) = ridge( b_mat(:,reg), delta_p(:, debug_params), ridge_param);
% 	end
% 	reconstruct_b_mat =  delta_p(:, debug_params) * Jp';
% end
				
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

% function [p_mat_gg_new, delta_p_image] = Update_p_mat(p_mat_gg, Risk, features_image, A0, Kpi, learning_rate, N, K, KNonrigid)
% 	for k = 1 : Kpi
% 		delta_p_image(k, :) =[zeros(1,1); learning_rate * Risk * (reshape(features_image(k, :), [], 1) - A0); zeros(1 + KNonrigid, 1)]'; 
% 		p_mat_gg_new(k, :) = reshape(p_mat_gg(k, :), 1, K ) + delta_p_image(k, :);
% 	end
% end		
	
function [features, b_mat] =  extract_SIFT_features(p_mat, TR_face_size, SIFT_scale, P, A0P, Kpi, n, n1, n2, lm_ind1, lm_pts, myShape_s0 )
	disp( 'extracting features from training dataset');
	num_feat_per_lm = 128;
	feat = zeros(Kpi, n,  num_feat_per_lm * lm_pts);
	b_mat_temp = zeros(Kpi, n,  num_feat_per_lm * lm_pts);
	[TR_images] = readAllImages();
	disp('finished reading all images');
	parfor gg = 1 : n
		p_mat_image = p_mat((gg-1) *Kpi + 1 : (gg - 1) * Kpi + Kpi , : );
		face_size = TR_face_size{gg};
		input_image = TR_images{gg};

		for k = 1 : Kpi
			lm = reconstruct_lm(p_mat_image(k, :));
			Sfeat = SIFT_features(input_image, lm, SIFT_scale, face_size, myShape_s0);
			feat(k, gg, :) = reshape(Sfeat, 1, []); 
			b_mat_temp(k, gg, :) =  reshape(feat(k , gg, :), 1, []) * P - A0P;
		end
	end   
	features = reshape(feat, n * Kpi, []);
	b_mat = reshape(b_mat_temp, n * Kpi, []);
end
		
function [pt_pt_err_image] = Compute_pt_pt_error_image1(myShape_s0, myShape_QNonrigid, p_mat_gg, gt_landmark, face_size, gg, lm_ind1)
	lm = reconstruct_lm(myShape_s0, myShape_QNonrigid, p_mat_gg(1, :));
	pt_pt_err_image = my_compute_error(gt_landmark(lm_ind1,:), lm, face_size );
end

function [pt_pt_err_image] = Compute_pt_pt_error_image(myShape_s0, myShape_QNonrigid, p_mat_gg, gt_landmark, face_size, Kpi, gg, lm_ind1)
	pt_pt_err_k = zeros(1, Kpi);
	for k = 1 : Kpi
		lm = reconstruct_lm(myShape_s0, myShape_QNonrigid, p_mat_gg(k, :));
		pt_pt_err_k(1, k) = my_compute_error(gt_landmark(lm_ind1,:), lm, face_size );
	end
	pt_pt_err_image = sum(pt_pt_err_k) / Kpi;
end
				


function [shapemodel, myShape, myAppearance, fd_stat, P, A0P, N, m, KNonrigid, KRigid, K] = load_models()
% load models: shape model, myShape model, myAppearance model, fd_stat model, and compute static variables.
	modelDir = 'matfiles/';
	shapemodel = load([modelDir 'shape_model.mat']);
	myShape = load([modelDir 'myShape.mat']); 
	myAppearance = load([modelDir 'myAppearanceSM.mat']);
	fd_stat = load([modelDir 'fd_stat_SM.mat']);
	shapemodel = shapemodel.shape;
	myShape = myShape.myShape;
	P = eye(size(myAppearance.A,1)) - myAppearance.A * myAppearance.A'; 
	A0P = myAppearance.A0' * P;  
	N = size(myAppearance.A, 1);				% number of SIFT features
	m = size(myAppearance.A, 2);                            % number of eigenvectors of myAppearance.A
	KNonrigid = size(myShape.pNonrigid, 2);                                      % number of eigenvectors of myShape.Q
	KRigid = size(myShape.pRigid, 2);
	K = KNonrigid + KRigid;
end

% function [] = saveFinalResults(sep_joint, pt_pt_err_image, pt_pt_err_allimages, cum_err, outputDir, SIFT_scale, Kpi, ridge_param, learning_rate)
% 	disp( 'saving final results to output directory');
% 	if sep_joint == 1
% 		save([outputDir 'cum_err/cum_err_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'cum_err');		
% 		save([outputDir 'pt_pt_err_allimages/pt_pt_err_allimages_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'pt_pt_err_allimages');
% 		save([outputDir 'pt_pt_err_image/pt_pt_err_image_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'pt_pt_err_image');
% 	else
% 		save([outputDir 'cum_err/cum_err_joint_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'cum_err');		
% 		save([outputDir 'pt_pt_err_allimages/pt_pt_err_allimages_joint_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'pt_pt_err_allimages');
% 		save([outputDir 'pt_pt_err_image/pt_pt_err_image_joint_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'pt_pt_err_image');
% 	end
% end
	
% function [] = saveCurrentIterationResults(sep_joint, outputDir, b_mat, Risk, Jp, p_mat, delta_p, t, SIFT_scale, Kpi, ridge_param, learning_rate)
% 	disp( 'saving results of current itertion to output directory');
% 	if sep_joint == 1
% 		save([outputDir 'b_mat/b_mat_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'b_mat');
% 		save([outputDir 'Risks/Risk_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'Risk');
% 		save([outputDir 'JPs/Jp_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'Jp');
% 		save([outputDir 'ppp/p_mat_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'p_mat');
% 		save([outputDir 'delta_p/delta_p_i_' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'delta_p');
% 	else
% 		save([outputDir 'b_mat/b_mat_joint_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'b_mat');
% 		save([outputDir 'Risks/Risk_joint_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'Risk');
% 		save([outputDir 'JPs/Jp_joint_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'Jp');
% 		save([outputDir 'ppp/p_mat_nonrigid_joint_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'p_mat_nonrigid');
% 		save([outputDir 'ppp/p_mat_rigid_joint_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'p_mat_rigid');
% 		save([outputDir 'delta_p/delta_p_joint_i_' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'delta_p');
% 	end
% end

function [TR_images] = readAllImages()
	[images_helen] = readAllImagesFromHelen();
	[images_lfpw] = readAllImagesFromLFPW();
	TR_images = {};
	TR_images = cat(1, TR_images, images_helen'); 
	TR_images = cat(1, TR_images, images_lfpw');
end

function [images] = readAllImagesFromHelen()
	datasetDir = [pwd '/../dataset/'];
	folder = [datasetDir 'helen/trainset/'];
	what = 'jpg';
	names_img = dir([folder '*.' what]);
	for gg = 1 : 2000
		images{gg} = imread([folder names_img(gg).name]); 
	end
end

function [images] = readAllImagesFromLFPW()
	datasetDir = [pwd '/../dataset/'];
	folder = [datasetDir 'lfpw/trainset/'];
	what = 'png';
	names_img = dir([folder '*.' what]);
	for gg = 1 : 811
		images{gg} = imread([folder names_img(gg).name]); 
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

function [p_mat_initialization, delta_p_initialization, p_mat_gt, Perturbed_SIFT_Feature_labels, Perturbed_SIFT_Features, Perturbed_Shape_Param_Initializatoin] = loadInitialPerturbations(FileDir, n1, n2, SIFT_scale, Kpi, smallset, use_5_lms)
% load p initial perturbations according to image number n1, n2 (or full training set) , SIFT scale and Kpi
	global VERSIONCHECK; 
	if smallset == 0
		if use_5_lms == 1
			Perturbed_SIFT_Feature_labels = load([FileDir 'Perturbed_SIFT_Feature_labels_5_lms_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			Perturbed_SIFT_Features = load([FileDir 'Perturbed_SIFT_Features_5_lms_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			Perturbed_Shape_Param_Initializatoin = load([FileDir 'Perturbed_Shape_Param_Initialization_SM.mat']);
			
			delta_p_initialization = Perturbed_Shape_Param_Initializatoin.delta_p_initialization; 
			p_mat_initialization = Perturbed_Shape_Param_Initializatoin.p_mat_initialization;
			p_mat_gt = Perturbed_Shape_Param_Initializatoin.p_mat_gt; 

			if Perturbed_Shape_Param_Initializatoin.VERSIONCHECK ~= VERSIONCHECK
				disp('the PerturbationInitialization_5_lms is stale'); return;
			end
			
			Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels.b_mat;
			Perturbed_SIFT_Features = Perturbed_SIFT_Features.features; 
		end
	end
end



% function [p_mat_initialization, delta_p_initialization, p_mat_gt, Perturbed_SIFT_Feature_labels, Perturbed_SIFT_Features] = loadInitialPerturbations(n1, n2, SIFT_scale, Kpi, smallset, use_5_lms)
% % load p initial perturbations according to image number n1, n2 (or full training set) , SIFT scale and Kpi
% 	global VERSIONCHECK; 
% 	if smallset == 1
% 		if use_5_lms == 1
% 			FileDir = '../PerturbationInitialization_5_lms_Smalldataset/';
% 			Perturbed_SIFT_Feature_labels_n = load([FileDir 'Perturbed_SIFT_Feature_labels_5_lms_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
% 			Perturbed_SIFT_Features = load([FileDir 'Perturbed_SIFT_Features_5_lms_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_rigid_initialization_n = load([FileDir 'p_mat_rigid_initialization_5_lms_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_nonrigid_initialization_n = load([FileDir 'p_mat_nonrigid_initialization_5_lms_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_nonrigid_gtperturbed_n = load([FileDir 'p_mat_nonrigid_gtperturbed_5_lms_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_rigid_gtperturbed_n = load([FileDir 'p_mat_rigid_gtperturbed_5_lms_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat']);
% 
% 			Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels_n.Perturbed_SIFT_Feature_labels_5_lms;
% 			Perturbed_SIFT_Features = Perturbed_SIFT_Features.Perturbed_SIFT_Feature_5_lms; 
% 			p_mat_rigid_initialization = p_mat_rigid_initialization_n.p_mat_rigid_initialization_5_lms;
% 			p_mat_nonrigid_initialization = p_mat_nonrigid_initialization_n.p_mat_nonrigid_initialization_5_lms;
% 			p_mat_nonrigid_gtperturbed = p_mat_nonrigid_gtperturbed_n.p_mat_nonrigid_gtperturbed_5_lms;
% 			p_mat_rigid_gtperturbed = p_mat_rigid_gtperturbed_n.p_mat_rigid_gtperturbed_5_lms;
% 		else
% 			FileDir = '../PerturbationInitialization_Smalldataset/';
% 			Perturbed_SIFT_Feature_labels_n = load([FileDir 'Perturbed_SIFT_Feature_labels_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
% 			Perturbed_SIFT_Features = load([FileDir 'Perturbed_SIFT_Features_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_rigid_initialization_n = load([FileDir 'p_mat_rigid_initialization_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_nonrigid_initialization_n = load([FileDir 'p_mat_nonrigid_initialization_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_nonrigid_gtperturbed_n = load([FileDir 'p_mat_nonrigid_gtperturbed_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_rigid_gtperturbed_n = load([FileDir 'p_mat_rigid_gtperturbed_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
% 			
% 			Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels_n.Perturbed_SIFT_Feature_labels_n;
% 			Perturbed_SIFT_Features = Perturbed_SIFT_Features.Perturbed_SIFT_Feature_n;
% 			p_mat_rigid_initialization = p_mat_rigid_initialization_n.p_mat_rigid_initialization_n;
% 			p_mat_nonrigid_initialization = p_mat_nonrigid_initialization_n.p_mat_nonrigid_initialization_n;
% 			p_mat_nonrigid_gtperturbed = p_mat_nonrigid_gtperturbed_n.p_mat_nonrigid_gtperturbed_n;
% 			p_mat_rigid_gtperturbed = p_mat_rigid_gtperturbed_n.p_mat_rigid_gtperturbed_n;
% 		end
% 	else
% 		if use_5_lms == 1
% 			FileDir = '../PerturbationInitialization_5_lms/';
% 			Perturbed_SIFT_Feature_labels = load([FileDir 'Perturbed_SIFT_Feature_labels_5_lms_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
% 			Perturbed_SIFT_Features = load([FileDir 'Perturbed_SIFT_Features_5_lms_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
% 			delta_p_initialization = load([FileDir 'delta_p_initialization_5_lms_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_initialization = load([FileDir 'p_mat_initialization_5_lms_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_gt = load([FileDir 'p_mat_gt_5_lms_Kpi-' num2str(Kpi) '.mat']);
% 			version = load([FileDir 'version.mat']); 
% 			if version ~= VERSIONCHECK
% 				disp('the PerturbationInitialization_5_lms is stale'); return;
% 			end
% 			
% 			Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels.b_mat;
% 			Perturbed_SIFT_Features = Perturbed_SIFT_Features.features; 
% 			delta_p_initialization = delta_p_initialization.delta_p_initialization; 
% 			p_mat_initialization = p_mat_initialization.p_mat_initialization;
% 			p_mat_gt = p_mat_gt.p_mat_gt;
% 		else
% 			FileDir = '../PerturbationInitialization/'; 
% 			Perturbed_SIFT_Feature_labels = load([FileDir 'Perturbed_SIFT_Feature_labels_S-'  num2str(SIFT_scale) '_Kpi-'  num2str(Kpi) '.mat']);
% 			Perturbed_SIFT_Feature = load([FileDir 'Perturbed_SIFT_Features_S-'  num2str(SIFT_scale) '_Kpi-'  num2str(Kpi) '.mat']);
% 			p_mat_rigid_initialization = load([FileDir 'p_mat_rigid_initialization_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_nonrigid_initialization = load([FileDir 'p_mat_nonrigid_initialization_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_rigid_gtperturbed = load([FileDir 'p_mat_rigid_gtperturbed_Kpi-' num2str(Kpi) '.mat']);
% 			p_mat_nonrigid_gtperturbed = load([FileDir 'p_mat_nonrigid_gtperturbed_Kpi-' num2str(Kpi) '.mat']);
% 
% 			Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels.b_mat;
% 			Perturbed_SIFT_Feature = Perturbed_SIFT_Feature.feat;
% 			p_mat_rigid_initialization = p_mat_rigid_initialization.p_mat_rigid;
% 			p_mat_nonrigid_initialization = p_mat_nonrigid_initialization.p_mat_nonrigid;
% 			p_mat_rigid_gtperturbed = p_mat_rigid_gtperturbed.p_mat_rigid_gtperturbed;
% 			p_mat_nonrigid_gtperturbed = p_mat_nonrigid_gtperturbed.p_mat_nonrigid_gtperturbed;
% 		end
% 	end
% 	
% end

% function [] = plot_cumulative_error_curve()
% 	figure; hold on;
% 	
% 	color = [ 0, 0, 0; 1, 0, 0; 1, 1, 0; 0, 1, 0; 0, 0, 1];
% 	for t = 1 : T
% 		plot(var, cum_err_full, 'Color', color(t,:), 'linewidth', 2); grid on;
% 	end
% 	xtick = 5*var;
% 	ytick = 0:0.05:1;
% 	set(gca, 'xtick', xtick);
% 	set(gca, 'ytick', ytick);
% 	ylabel('Percentage of Images', 'Interpreter','tex', 'fontsize', 15)
% 	xlabel('Pt-Pt error normalized by face size', 'Interpreter','tex', 'fontsize', 13)
% 	legend(['iteration' num2str(t)]);
% end
% 
% function [] = visualize_iterations()
% 		pp = load([outputDir 'ppp/ppp_initial_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat']);
% 		pp = pp.pp;
% 		ppp = load([outputDir 'ppp/ppp_i-' num2str(T) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat']);
% 		ppp = ppp.ppp;
% 		figure; hold on;
% 		for gg = 1:15
% 			lm0 = myShape.s0 + myShape.Q(:, 2:end) * reshape(pp(1, gg, 2:end), 1, [])';
% 			lm1 = myShape.s0 + myShape.Q(:, 2:end) * reshape(ppp(1, gg, 2:end), 1, [])';
% % 			lm2 = myShape.s0 + myShape.Q(:, 2:end) * reshape(ppp(2, gg, 2:end), 1, [])';
% % 			lm3 = myShape.s0 + myShape.Q(:, 2:end) * reshape(ppp(3, gg,  2:end), 1, [])';
% % 			lm4 = myShape.s0 + myShape.Q(:, 2:end) * reshape(ppp(4, gg,  2:end), 1, [])';
% % 			lm5 = myShape.s0 + myShape.Q(:, 2:end) * reshape(ppp(5, gg,  2:end), 1, [])';
% 			lm0 = reshape(lm0, [],2);
% 			lm1 = reshape(lm1, [],2);
% % 			lm2 = reshape(lm2, [],2);
% % 			lm3 = reshape(lm3, [],2);
% % 			lm4 = reshape(lm4, [],2);
% % 			lm5 = reshape(lm5, [],2);
% 			pts = read_shape([folder1 names2(gg).name], num_of_pts);                         
% 			gt_landmark = (pts-1);
% 			gt_landmark = reshape(gt_landmark, 68, 2);
% 			input_image = imread([folder1 names1(gg).name]); 
% 			[~,~,Tt] = procrustes(shapemodelS0, gt_landmark);        
% 			scl = 1/Tt.b;
% 			gt_landmark = gt_landmark*(1/scl); 
% 			input_image = imresize(input_image, (1/scl));
% 			subplot(5,6,gg);
% 			imagesc(input_image); colormap(gray); hold on;
% 			plot(lm0(:,1), lm0(:,2), 'Color', 'green');
% 			plot(lm1(:,1), lm1(:,2), 'Color', 'blue');
% % 			plot(lm2(:,1), lm2(:,2), 'Color', 'red');
% % 			plot(lm3(:,1), lm3(:,2), 'Color', 'blue');
% % 			plot(lm4(:,1), lm4(:,2), 'Color', 'yellow');
% % 			plot(lm5(:,1), lm5(:,2), 'Color', 'grey');
% 		end
% end

% check landmarks by shape model and ground truth landmarksS 
% a = 347;
% p_gt = p_mat_gt((a - 1) * 10 + 1, :); 
% gt_lm = reconstruct_lm(p_gt);
% lm_gt = TR_gt_landmarks{a};
% lm_ind1 = [34, 37, 46, 61, 65]; 
% lm_gtt = lm_gt(lm_ind1, :);
% figure; hold on; 
% plot(lm_gtt(:,1), -1 * lm_gtt(:, 2));
% plot(gt_lm(:,1), -1 * gt_lm(:,2));




