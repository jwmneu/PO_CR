function [] = SM_Extract_Perturbed_SIFT_Features()
	% this function should run in foler PO_CR_code_v1
	% output : p_mat : size( n * Kpi, K);  delta_p : size(n * Kpi, K);
	% features : size(n * Kpi, N);  b_mat : size(n * Kpi, N)
	
	clear;
	global VERSIONCHECK; 
	VERSIONCHECK = 'SM_1';
	Kpi = 10;
	s = rng;

	% load models
	cd([pwd '/vlfeat-0.9.20/toolbox']);
	vl_setup
	cd '../../'
	OutputDir = '../PerturbationInitialization_5_lms/';
	% OutputDir = 'debug/';
	modelDir = 'matfiles/';
	shapemodel = load([modelDir 'shape_model.mat']);
	myShape = load([modelDir 'myShape.mat']); 
	myAppearance = load([modelDir 'myAppearanceSM.mat']);
	fd_stat = load([modelDir 'fd_stat_SM.mat']);
	shapemodel = shapemodel.shape;
	myShape = myShape.myShape;
	myAppearance = myAppearance.myAppearance;
	fd_stat = fd_stat.fd_stat;
	if myShape.version ~= VERSIONCHECK
		disp('myShape model is stale');
	end
	if  fd_stat.version ~= VERSIONCHECK
		disp('fd_stat model is stale');
	end
	num_of_pts = 68;                                              % num of landmarks in the annotations
	P = eye(size(myAppearance.A,1)) - myAppearance.A * myAppearance.A'; 
	N = size(myAppearance.A, 1);				% number of SIFT features
	m = size(myAppearance.A, 2);                            % number of eigenvectors of myAppearance.A
	KNonrigid = size(myShape.pNonrigid, 2);                                      % number of eigenvectors of myShape.Q
	KRigid = size(myShape.pRigid, 2);
	K = KNonrigid + KRigid;
	A0P = myAppearance.A0' * P;  
	n1 = 2000;					
	n2 = 811;
	n = n1 + n2; 
	lm_pts = 5;
	lm_ind2 = [34, 37,46, 61, 65]; 
	lm_ind = [34,  37,46,  61, 65, 102,  105 ,114 ,129, 133]; 
	% collect training images from two datasets
  	[TR_images, TR_face_size, TR_gt_landmarks, TR_myShape_pRigid, TR_myShape_pNonRigid, TR_detections] = Collect_training_images(n1, n2) ;
% 	load('CollectedTrainingDataset/TR_detections.mat'); 
% 	load('CollectedTrainingDataset/TR_face_size.mat');
% 	load('CollectedTrainingDataset/TR_gt_landmarks.mat');
% 	load('CollectedTrainingDataset/TR_myShape_pNonRigid.mat');
% 	load('CollectedTrainingDataset/TR_myShape_pRigid.mat');
	
	% initialize learning parameters
	fd_stat_std = fd_stat.std;
	fd_stat_mean = fd_stat.mean;
	
	p_mat_nonrigid = zeros(n * Kpi, KNonrigid);
	p_mat_rigid = zeros(n * Kpi,  KRigid);
	p_mat_nonrigid_gtperturbed = zeros(n * Kpi, KNonrigid);
	p_mat_rigid_gtperturbed = zeros( n * Kpi, KRigid);
	p_mat_gt = zeros(n * Kpi, K);
	p_mat_initialization = zeros(n * Kpi, K);
	delta_p_initialization = zeros(n * Kpi, K);
	
	feat = zeros(Kpi, n, N);
	b_mat_temp = zeros(Kpi, n, N);
	pt_pt_err0_temp = zeros(n , Kpi);
	pt_pt_err0_image = zeros(n, 1);
	norm_face_size_scale = ((max(shapemodel.s0(:,1)) - min(shapemodel.s0(:,1))) + (max(shapemodel.s0(:,2)) - min(shapemodel.s0(:,1)))) / 2; 
	rng(s);

	%% initialize p_mat and add noise to pertubations 
	disp( 'initializing perturbed shape parameters');
	for gg = 1:n
		gt_landmark = TR_gt_landmarks{gg};
		face_size = TR_face_size{gg};
		pt_pt_err1 = zeros(1, Kpi);
		for k = 1 : Kpi
			% record gt
			p_mat_rigid_gtperturbed((gg-1) * Kpi + k, :) = TR_myShape_pRigid(gg, :);
			p_mat_nonrigid_gtperturbed((gg-1) * Kpi + k, :) = TR_myShape_pNonRigid(gg, :);
			% initialize p parameters
			p_mat_nonrigid((gg-1) * Kpi + k, :) = TR_myShape_pNonRigid(gg, :);		% zeros(1, KNonrigid);	% for debugging: myShape_pNonrigid(gg, :)
			p_mat_rigid((gg-1) * Kpi + k, :)  = TR_myShape_pRigid(gg, :);
			% add noise to scale, x_transform, y_transform,  rotation -- debug: only perturb translation parameters, use ideal values for other paramenters
			ss = face_size / norm_face_size_scale;
			p_mat_rigid((gg-1) * Kpi + k, 2:3) = reshape(p_mat_rigid((gg-1) * Kpi + k, 2:3), 1, [])+ (fd_stat_mean(1, 2:3) + fd_stat_std(1, 2:3) .* randn(1,2)) * ss; 
% 			p_mat_rigid(gg,k,1) = p_mat_rigid(gg,k,1) * (fd_stat_mean(1, 1) + fd_stat_std(1,1) * randn(1)); 
% 			p_mat_rigid(gg, k,4) = p_mat_rigid(gg, k,4) + fd_stat_mean(1,4) + fd_stat_std(1,4) * randn(1); 
			% compute delta p
			p_mat_gt( (gg - 1) * Kpi + k, :) = [reshape( p_mat_rigid_gtperturbed((gg-1) * Kpi + k, :) , 1, []) , reshape( p_mat_nonrigid_gtperturbed((gg-1) * Kpi + k, :), 1, [])];
			p_mat_initialization((gg-1) * Kpi + k, :) = [reshape(p_mat_rigid((gg-1) * Kpi + k, :), 1, []) , reshape(p_mat_nonrigid((gg-1) * Kpi + k, :) , 1, [])]; 
			delta_p_initialization((gg-1) * Kpi + k, :) = [reshape(p_mat_rigid_gtperturbed((gg-1) * Kpi + k, :), 1, []), reshape(p_mat_nonrigid_gtperturbed((gg-1) * Kpi + k,:), 1, [])] - [reshape(p_mat_rigid((gg-1) * Kpi + k, :) , 1, []), reshape(p_mat_nonrigid((gg-1) * Kpi + k, :), 1, [])]; 
			% reconstruct landmarks
			lm = reconstruct_lm(reshape(p_mat_nonrigid((gg-1) * Kpi + k, :), 1, []), reshape(p_mat_rigid((gg-1) * Kpi + k, :), 1, []));  % plotgg, input_image, gt_landmark
			pt_pt_err0_temp(gg, k) = compute_error(TR_gt_landmarks{gg}(lm_ind2, :), lm);
		end
		% compute error and cumulative curve
		 pt_pt_err0_image(gg) = sum(pt_pt_err0_temp(gg, :)) / Kpi;
	end
	[pt_pt_err0_allimages, cum_err0] = Compute_cum_error(pt_pt_err0_image, n, 'cum error of initialization'); 
	
	% save results
	saveInitialResults(OutputDir, pt_pt_err0_allimages, pt_pt_err0_image, cum_err0, p_mat_rigid, p_mat_nonrigid, p_mat_rigid_gtperturbed, p_mat_nonrigid_gtperturbed, delta_p_initialization,  p_mat_initialization, p_mat_gt,  Kpi);	

	%% extract SIFT features
	for SIFT_scale = 1.5 : 0.5 : 2.5
		disp( 'extracting features from training dataset. SIFT scale is ' + SIFT_scale);
		parfor gg = 1 : n
			p_mat_gg_nonrigid = p_mat_nonrigid((gg-1) * Kpi + 1 : (gg-1) * Kpi + Kpi, :);		
			p_mat_gg_rigid = p_mat_rigid((gg-1) * Kpi + 1 : (gg-1) * Kpi + Kpi, :);
			gt_landmark = TR_gt_landmarks{gg};
			face_size = TR_face_size{gg};
			input_image = TR_images{gg};

			for k = 1 : Kpi
				lm = reconstruct_lm(reshape(p_mat_gg_nonrigid(k, :), 1, []), reshape(p_mat_gg_rigid(k,:),1,[]));
				Sfeat = SIFT_features(input_image, lm, SIFT_scale, k, face_size);
				feat(k, gg, :) = reshape(Sfeat, 1, []); 
				b_mat_temp(k, gg, :) =  reshape(feat(k,gg, :), 1, []) * P - A0P;
% 				b_mat_without_p(k, gg, :) = reshape(feat(k,gg, :), [], 1) - myAppearance.A0; 
			end
		end
		features = reshape(feat, Kpi * n, []); 
		b_mat = reshape(b_mat_temp, Kpi * n, []);
		save([OutputDir 'Perturbed_SIFT_Feature_labels_5_lms_S-'  num2str(SIFT_scale) '_Kpi-'  num2str(Kpi) '.mat'], 'b_mat');
		save([OutputDir 'Perturbed_SIFT_Features_5_lms_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat'], 'features');
	end
	disp('finished this function');
end

function [] = saveInitialResults(OutputDir, pt_pt_err0_allimages, pt_pt_err0_image, cum_err0, p_mat_rigid, p_mat_nonrigid, p_mat_rigid_gtperturbed, p_mat_nonrigid_gtperturbed, delta_p_initialization, p_mat_initialization, p_mat_gt, Kpi)
	global VERSIONCHECK; 
	save([OutputDir 'pt_pt_err0_allimages_5_lms_Kpi-' num2str(Kpi) '.mat'], 'pt_pt_err0_allimages');
	save([OutputDir 'pt_pt_err0_image_5_lms_Kpi-' num2str(Kpi) '.mat'], 'pt_pt_err0_image');
	save([OutputDir 'cum_err0_5_lms_Kpi-' num2str(Kpi)  '.mat'], 'cum_err0');
	save([OutputDir 'p_mat_rigid_initialization_5_lms_Kpi-' num2str(Kpi) '.mat'], 'p_mat_rigid');
	save([OutputDir 'p_mat_nonrigid_initialization_5_lms_Kpi-' num2str(Kpi) '.mat'], 'p_mat_nonrigid');
	save([OutputDir 'p_mat_rigid_gtperturbed_5_lms_Kpi-' num2str(Kpi) '.mat'], 'p_mat_rigid_gtperturbed');
	save([OutputDir 'p_mat_nonrigid_gtperturbed_5_lms_Kpi-' num2str(Kpi) '.mat'], 'p_mat_nonrigid_gtperturbed');
	save([OutputDir 'delta_p_initialization_5_lms_Kpi-' num2str(Kpi) '.mat'], 'delta_p_initialization');
	save([OutputDir 'p_mat_initialization_5_lms_Kpi-' num2str(Kpi) '.mat'], 'p_mat_initialization');
	save([OutputDir 'p_mat_gt_5_lms_Kpi-' num2str(Kpi) '.mat'], 'p_mat_gt');
	version = VERSIONCHECK;
	save([OutputDir 'version.mat'], 'version');
end











