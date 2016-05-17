function [] = regressorsep()
	% clear;
	global VERSIONCHECK; 
	VERSIONCHECK = 'SM_1'; 
	Kpi = 10;
	T = 1;
	ridge_param = 0;
	smallsize = 0;
	use_5_lms = 1;
	addpath('functions/');
	outputDir = 'Results/'; 
	Setup_createDir(outputDir, 0);
	datasetDir = '../dataset/'; 
	testsetDir = '../test_data/';
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
	lm_ind2 = [34, 37, 46, 61, 65]; 
	lm_ind = [34, 37, 46, 61, 65, 102, 105, 114, 129, 133]; 
	% load models
	[shapemodel, myShape, myAppearance, fd_stat, P, A0P, N, m, KNonrigid, KRigid, K] = load_models();

	% collect training data
	% [TR_images, TR_face_size, TR_gt_landmarks, TR_myShape_pRigid, TR_myShape_pNonRigid, ~] = Collect_training_images(n1, n2) ; 
	load('CollectedTrainingDataset/TR_detections.mat'); 
	load('CollectedTrainingDataset/TR_face_size.mat');
	load('CollectedTrainingDataset/TR_gt_landmarks.mat');
	load('CollectedTrainingDataset/TR_myShape_pNonRigid.mat');
	load('CollectedTrainingDataset/TR_myShape_pRigid.mat');

	for SIFT_scale = 2
		for learning_rate = 1
			% collect initial p perturbations according to image numbers n1, n2, SIFT_scale and Kpi
			[p_mat_initialization, delta_p_initialization, p_mat_gt,  Perturbed_SIFT_Feature_labels, Perturbed_SIFT_Features] = loadInitialPerturbations(n1, n2, SIFT_scale, Kpi, smallsize, use_5_lms);

			for gg = 1 : n
				[pt_pt_err_image_gt(gg)] = Compute_pt_pt_error_image1(p_mat_gt((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :) , TR_gt_landmarks{gg}, gg, lm_ind2);
			end
			[pt_pt_err_allimages_gt, cum_err_gt] = Compute_cum_error(pt_pt_err_image_gt, n, 'ground truth cum_error');

	% 		if n1, n2 are not 30: (run this on server since the files are on server)
	%  		[p_mat_rigid_initialization, p_mat_nonrigid_initialization, Perturbed_SIFT_Feature_labels] = Collect_Initial_Perturbations(n1, n2, SIFT_scale, Kpi);

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
					load('../PerturbationInitialization_5_lms/pt_pt_err0_image_5_lms_Kpi-10.mat');
					[pt_pt_err_allimages0, cum_err0] = Compute_cum_error(pt_pt_err0_image, n, 'cum error after initilaization');
				else
					[features, b_mat] =  extract_SIFT_features(p_mat, TR_face_size, SIFT_scale, P, A0P, Kpi, n, n1, n2, lm_ind2, lm_pts )
				end

%% ridge regression to compute Jp
				sep_joint = 1;
				if sep_joint == 1
					 [Jp, reconstruct_b_mat] = ridge_seperate(b_mat, delta_p, N, ridge_param); 
				else
					[Jp, reconstruct_b_mat] = ridge_joint(b_mat, delta_p, N, ridge_param);
				end

%% update p and compute pt-pt error
				disp('updating shape parameters for Helen train set and computing pt-pt error');
				Hessian = Jp' * Jp; 
				Risk = Hessian \ Jp';		 

				for gg = 1:n
					% update p_mat by previous p_mat + delta_p
					[p_mat((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), delta_p((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :)] = Update_p_mat(p_mat((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), Risk, features((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi,:), myAppearance.A0,  Kpi, learning_rate, N, K, KNonrigid);

					% compute pt-pt error of this image
					[pt_pt_err_image(t, gg)] = Compute_pt_pt_error_image(p_mat((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :),  TR_gt_landmarks{gg}, Kpi, gg, lm_ind2);
				end 

				% compute error and cumulative curve
				[pt_pt_err_allimages(t), cum_err(t, :) ] = Compute_cum_error(pt_pt_err_image(t, :), n, ['cum error after' num2str(t) ' iteration']);

				% save intermediate results per iteration
				saveCurrentIterationResults(sep_joint, outputDir, b_mat, Risk, Jp, p_mat, delta_p, t, SIFT_scale, Kpi, ridge_param, learning_rate);

			end 

			% save result
			saveFinalResults(sep_joint, pt_pt_err_image, pt_pt_err_allimages, cum_err, outputDir, SIFT_scale, Kpi, ridge_param, learning_rate);
		end
	end
end



%% ################################################     helper functions    ##################################################

function [Jp, reconstruct_b_mat] = ridge_seperate(b_mat, delta_p, N, ridge_param)
	disp('doing ridge regression seperately');
	debug_params = [2,3]; 
	Jp = zeros(N, size(debug_params, 2));
	for reg = 1:N
		Jp(reg,:) = ridge( b_mat(:,reg), delta_p(:, debug_params), ridge_param);
	end
	reconstruct_b_mat =  delta_p(:, debug_params) * Jp';
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

function [features, b_mat] =  extract_SIFT_features(p_mat, TR_face_size, SIFT_scale, P, A0P, Kpi, n, n1, n2, lm_ind2, lm_pts )
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
			Sfeat = SIFT_features(input_image, lm, SIFT_scale, face_size);
			feat(k, gg, :) = reshape(Sfeat, 1, []); 
			b_mat_temp(k, gg, :) =  reshape(feat(k , gg, :), 1, []) * P - A0P;
		end
	end   
	features = reshape(feat, n * Kpi, []);
	b_mat = reshape(b_mat_temp, n * Kpi, []);
end

function [p_mat_gg_new, delta_p_image] = Update_p_mat(p_mat_gg, Risk, features_image, A0, Kpi, learning_rate, N, K, KNonrigid)
	for k = 1 : Kpi
		delta_p_image(k, :) =[zeros(1,1); learning_rate * Risk * (reshape(features_image(k, :), [], 1) - A0); zeros(1 + KNonrigid, 1)]'; 
		p_mat_gg_new(k, :) = reshape(p_mat_gg(k, :), 1, K ) + delta_p_image(k, :);
	end
end		
				
function [pt_pt_err_image] = Compute_pt_pt_error_image1(p_mat_gg, gt_landmark, gg, lm_ind2)
	lm = reconstruct_lm(reshape(p_mat_gg(1,5:end), 1, []), reshape(p_mat_gg(1,1:4), 1, []));
	pt_pt_err_image = compute_error(gt_landmark(lm_ind2,:), lm );
end

function [pt_pt_err_image] = Compute_pt_pt_error_image(p_mat_gg, gt_landmark, Kpi, gg, lm_ind2)
	pt_pt_err_k = zeros(1, Kpi);
	for k = 1 : Kpi
		lm = reconstruct_lm(reshape(p_mat_gg(k, 5:end), 1, []), reshape(p_mat_gg(k,1:4), 1, []));
		pt_pt_err_k(1, k) = compute_error(gt_landmark(lm_ind2,:), lm );
	end
	pt_pt_err_image = sum(pt_pt_err_k) / Kpi;
end
				
function [ error_per_image ] = compute_error( ground_truth_all, detected_points_all )
	%compute_error
	%   compute the average point-to-point Euclidean error normalized by the
	%   inter-ocular distance (measured as the Euclidean distance between the
	%   outer corners of the eyes)
	%
	%   Inputs:
	%          grounth_truth_all, size: num_of_points x 2 x num_of_images
	%          detected_points_all, size: num_of_points x 2 x num_of_images
	%   Output:
	%          error_per_image, size: num_of_images x 1

	num_of_images = size(ground_truth_all,3);
	num_of_points = size(ground_truth_all,1);
	error_per_image = zeros(num_of_images,1);

	for i =1:num_of_images
		detected_points = detected_points_all(:,:,i);
		ground_truth_points = ground_truth_all(:,:,i);
		if(num_of_points == 66 || num_of_points == 68)
			interocular_distance = norm(ground_truth_points(37,:)-ground_truth_points(46,:));
		elseif (num_of_points == 5)				  % small model
			interocular_distance = norm(ground_truth_points(2,:)-ground_truth_points(3,:));
		else
			interocular_distance = norm(ground_truth_points(37-17,:)-ground_truth_points(46-17,:));
		end
		sum=0;
		for j=1:num_of_points
			sum = sum+norm(detected_points(j,:)-ground_truth_points(j,:));
		end
		error_per_image(i) = sum/(num_of_points*interocular_distance);
	end
end

function [shapemodel, myShape, myAppearance, fd_stat, P, A0P, N, m, KNonrigid, KRigid, K] = load_models()
% load models: shape model, myShape model, myAppearance model, fd_stat model, and compute static variables.
	modelDir = 'matfiles/';
	shapemodel = load([modelDir 'shape_model.mat']);
	myShape = load([modelDir 'myShape.mat']); 
	myAppearance = load([modelDir 'myAppearanceSM.mat']);
	fd_stat = load([modelDir 'fd_stat.mat']);
	shapemodel = shapemodel.shape;
	myShape = myShape.myShape;
	myAppearance = myAppearance.myAppearance;
	fd_stat = fd_stat.fd_stat;
	P = eye(size(myAppearance.A,1)) - myAppearance.A * myAppearance.A'; 
	A0P = myAppearance.A0' * P;  
	N = size(myAppearance.A, 1);				% number of SIFT features
	m = size(myAppearance.A, 2);                            % number of eigenvectors of myAppearance.A
	KNonrigid = size(myShape.pNonrigid, 2);                                      % number of eigenvectors of myShape.Q
	KRigid = size(myShape.pRigid, 2);
	K = KNonrigid + KRigid;
end

function [] = saveFinalResults(sep_joint, pt_pt_err_image, pt_pt_err_allimages, cum_err, outputDir, SIFT_scale, Kpi, ridge_param, learning_rate)
	disp( 'saving final results to output directory');
	if sep_joint == 1
		save([outputDir 'cum_err/cum_err_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'cum_err');		
		save([outputDir 'pt_pt_err_allimages/pt_pt_err_allimages_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'pt_pt_err_allimages');
		save([outputDir 'pt_pt_err_image/pt_pt_err_image_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'pt_pt_err_image');
	else
		save([outputDir 'cum_err/cum_err_joint_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'cum_err');		
		save([outputDir 'pt_pt_err_allimages/pt_pt_err_allimages_joint_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'pt_pt_err_allimages');
		save([outputDir 'pt_pt_err_image/pt_pt_err_image_joint_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'pt_pt_err_image');
	end
end
	
function [] = saveCurrentIterationResults(sep_joint, outputDir, b_mat, Risk, Jp, p_mat, delta_p, t, SIFT_scale, Kpi, ridge_param, learning_rate)
	disp( 'saving results of current itertion to output directory');
	if sep_joint == 1
		save([outputDir 'b_mat/b_mat_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'b_mat');
		save([outputDir 'Risks/Risk_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'Risk');
		save([outputDir 'JPs/Jp_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'Jp');
		save([outputDir 'ppp/p_mat_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'p_mat');
		save([outputDir 'delta_p/delta_p_i_' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'delta_p');
	else
		save([outputDir 'b_mat/b_mat_joint_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'b_mat');
		save([outputDir 'Risks/Risk_joint_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'Risk');
		save([outputDir 'JPs/Jp_joint_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'Jp');
		save([outputDir 'ppp/p_mat_nonrigid_joint_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'p_mat_nonrigid');
		save([outputDir 'ppp/p_mat_rigid_joint_i-' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'p_mat_rigid');
		save([outputDir 'delta_p/delta_p_joint_i_' num2str(t) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat'], 'delta_p');
	end
end

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

function [p_mat_initialization, delta_p_initialization, p_mat_gt, Perturbed_SIFT_Feature_labels, Perturbed_SIFT_Features] = loadInitialPerturbations(n1, n2, SIFT_scale, Kpi, smallset, use_5_lms)
% load p initial perturbations according to image number n1, n2 (or full training set) , SIFT scale and Kpi
	global VERSIONCHECK; 
	if smallset == 1
		if use_5_lms == 1
			FileDir = '../PerturbationInitialization_5_lms_Smalldataset/';
			Perturbed_SIFT_Feature_labels_n = load([FileDir 'Perturbed_SIFT_Feature_labels_5_lms_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			Perturbed_SIFT_Features = load([FileDir 'Perturbed_SIFT_Features_5_lms_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			p_mat_rigid_initialization_n = load([FileDir 'p_mat_rigid_initialization_5_lms_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat']);
			p_mat_nonrigid_initialization_n = load([FileDir 'p_mat_nonrigid_initialization_5_lms_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat']);
			p_mat_nonrigid_gtperturbed_n = load([FileDir 'p_mat_nonrigid_gtperturbed_5_lms_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat']);
			p_mat_rigid_gtperturbed_n = load([FileDir 'p_mat_rigid_gtperturbed_5_lms_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat']);

			Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels_n.Perturbed_SIFT_Feature_labels_5_lms;
			Perturbed_SIFT_Features = Perturbed_SIFT_Features.Perturbed_SIFT_Feature_5_lms; 
			p_mat_rigid_initialization = p_mat_rigid_initialization_n.p_mat_rigid_initialization_5_lms;
			p_mat_nonrigid_initialization = p_mat_nonrigid_initialization_n.p_mat_nonrigid_initialization_5_lms;
			p_mat_nonrigid_gtperturbed = p_mat_nonrigid_gtperturbed_n.p_mat_nonrigid_gtperturbed_5_lms;
			p_mat_rigid_gtperturbed = p_mat_rigid_gtperturbed_n.p_mat_rigid_gtperturbed_5_lms;
		else
			FileDir = '../PerturbationInitialization_Smalldataset/';
			Perturbed_SIFT_Feature_labels_n = load([FileDir 'Perturbed_SIFT_Feature_labels_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			Perturbed_SIFT_Features = load([FileDir 'Perturbed_SIFT_Features_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			p_mat_rigid_initialization_n = load([FileDir 'p_mat_rigid_initialization_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			p_mat_nonrigid_initialization_n = load([FileDir 'p_mat_nonrigid_initialization_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			p_mat_nonrigid_gtperturbed_n = load([FileDir 'p_mat_nonrigid_gtperturbed_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			p_mat_rigid_gtperturbed_n = load([FileDir 'p_mat_rigid_gtperturbed_' num2str(n1) '-' num2str(n2) '_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			
			Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels_n.Perturbed_SIFT_Feature_labels_n;
			Perturbed_SIFT_Features = Perturbed_SIFT_Features.Perturbed_SIFT_Feature_n;
			p_mat_rigid_initialization = p_mat_rigid_initialization_n.p_mat_rigid_initialization_n;
			p_mat_nonrigid_initialization = p_mat_nonrigid_initialization_n.p_mat_nonrigid_initialization_n;
			p_mat_nonrigid_gtperturbed = p_mat_nonrigid_gtperturbed_n.p_mat_nonrigid_gtperturbed_n;
			p_mat_rigid_gtperturbed = p_mat_rigid_gtperturbed_n.p_mat_rigid_gtperturbed_n;
		end
	else
		if use_5_lms == 1
			FileDir = '../PerturbationInitialization_5_lms/';
			Perturbed_SIFT_Feature_labels = load([FileDir 'Perturbed_SIFT_Feature_labels_5_lms_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			Perturbed_SIFT_Features = load([FileDir 'Perturbed_SIFT_Features_5_lms_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat']);
			delta_p_initialization = load([FileDir 'delta_p_initialization_5_lms_Kpi-' num2str(Kpi) '.mat']);
			p_mat_initialization = load([FileDir 'p_mat_initialization_5_lms_Kpi-' num2str(Kpi) '.mat']);
			p_mat_gt = load([FileDir 'p_mat_gt_5_lms_Kpi-' num2str(Kpi) '.mat']);
			version = load([FileDir 'version.mat']); 
			if version.version ~= VERSIONCHECK
				disp('the PerturbationInitialization_5_lms is stale'); return;
			end
			
			Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels.b_mat;
			Perturbed_SIFT_Features = Perturbed_SIFT_Features.features; 
			delta_p_initialization = delta_p_initialization.delta_p_initialization; 
			p_mat_initialization = p_mat_initialization.p_mat_initialization;
			p_mat_gt = p_mat_gt.p_mat_gt;
		else
			FileDir = '../PerturbationInitialization/'; 
			Perturbed_SIFT_Feature_labels = load([FileDir 'Perturbed_SIFT_Feature_labels_S-'  num2str(SIFT_scale) '_Kpi-'  num2str(Kpi) '.mat']);
			Perturbed_SIFT_Feature = load([FileDir 'Perturbed_SIFT_Features_S-'  num2str(SIFT_scale) '_Kpi-'  num2str(Kpi) '.mat']);
			p_mat_rigid_initialization = load([FileDir 'p_mat_rigid_initialization_Kpi-' num2str(Kpi) '.mat']);
			p_mat_nonrigid_initialization = load([FileDir 'p_mat_nonrigid_initialization_Kpi-' num2str(Kpi) '.mat']);
			p_mat_rigid_gtperturbed = load([FileDir 'p_mat_rigid_gtperturbed_Kpi-' num2str(Kpi) '.mat']);
			p_mat_nonrigid_gtperturbed = load([FileDir 'p_mat_nonrigid_gtperturbed_Kpi-' num2str(Kpi) '.mat']);

			Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels.b_mat;
			Perturbed_SIFT_Feature = Perturbed_SIFT_Feature.feat;
			p_mat_rigid_initialization = p_mat_rigid_initialization.p_mat_rigid;
			p_mat_nonrigid_initialization = p_mat_nonrigid_initialization.p_mat_nonrigid;
			p_mat_rigid_gtperturbed = p_mat_rigid_gtperturbed.p_mat_rigid_gtperturbed;
			p_mat_nonrigid_gtperturbed = p_mat_nonrigid_gtperturbed.p_mat_nonrigid_gtperturbed;
		end
	end
	
end

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