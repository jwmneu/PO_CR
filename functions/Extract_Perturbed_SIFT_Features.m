function [] = Extract_Perturbed_SIFT_Features()
	% this function should run in foler PO_CR_code_v1
	clear;
	Kpi = 10;
	s = rng;

	% load models
	cd([pwd '/vlfeat-0.9.20/toolbox']);
	vl_setup
	cd '../../'
	OutputDir = '../PerturbationInitialization/';
	% OutputDir = 'debug/';
	modelDir = 'matfiles/';
	shapemodel = load([modelDir 'shape_model.mat']);
	myShape = load([modelDir 'myShape.mat']); 
	myAppearance = load([modelDir 'myAppearance.mat']);
	fd_stat = load([modelDir 'fd_stat.mat']);
	shapemodel = shapemodel.shape;
	myShape = myShape.myShape;
	myAppearance = myAppearance.myAppearance;
	fd_stat = fd_stat.fd_stat;
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
	var = 0:0.2:1;
	% collect training images from two datasets
	[TR_images, TR_face_sizes, TR_gt_landmarks, TR_myShape_pRigid, TR_myShape_pNonRigid, TR_detections] = Collect_training_images(n1, n2) ;

	% initialize learning parameters
	fd_stat_std = fd_stat.std;
	fd_stat_mean = fd_stat.mean;
	p_mat_nonrigid = zeros(n, Kpi, KNonrigid);
	p_mat_rigid = zeros(n, Kpi, KRigid);
	p_mat_nonrigid_gtperturbed = zeros(n, Kpi, KNonrigid);
	p_mat_rigid_gtperturbed = zeros(n, Kpi, KRigid);
	feat = zeros(n, Kpi, N);
	b_mat = zeros(n, Kpi, N);
	pt_pt_err0 = zeros(1, n);
	norm_face_size_scale = ((max(shapemodel.s0(:,1)) - min(shapemodel.s0(:,1))) + (max(shapemodel.s0(:,2)) - min(shapemodel.s0(:,1)))) / 2; 
	rng(s);

	%% initialize p_mat and add noise to pertubations 
	disp( 'initializing perturbed shape parameters');
	for gg = 1:n
		gg
		gt_landmark = TR_gt_landmarks{gg};
		face_size = TR_face_sizes{gg};
		pt_pt_err1 = zeros(1, Kpi);
		for k = 1 : Kpi
			% record gt
			p_mat_rigid_gtperturbed(gg, k, :) = TR_myShape_pRigid(gg, :);
			p_mat_nonrigid_gtperturbed(gg, k, :) = TR_myShape_pNonRigid(gg, :);
			% initialize p parameters
			p_mat_nonrigid(gg, k, :) = zeros(1, KNonrigid);				% for debugging: myShape_pNonrigid(gg, :)
			p_mat_rigid(gg,k, :)  = TR_myShape_pRigid(gg, :);
			% add noise to scale, x_transform, y_transform, rotation
			ss = face_size / norm_face_size_scale;
			p_mat_rigid(gg,k, 2:3) = reshape(p_mat_rigid(gg,k, 2:3), 1, [])+ (fd_stat_mean(1, 2:3) + fd_stat_std(1, 2:3) .* randn(1,2)) * ss; 
			p_mat_rigid(gg,k,1) = p_mat_rigid(gg,k,1) * (fd_stat_mean(1, 1) + fd_stat_std(1,1) * randn(1)); 
			p_mat_rigid(gg, k,4) = p_mat_rigid(gg, k,4) + fd_stat_mean(1,4) + fd_stat_std(1,4) * randn(1); 
			% reconstruct landmarks
			lm = reconstruct_lm(reshape(p_mat_nonrigid(gg,k, :), 1, []), reshape(p_mat_rigid(gg,k,:), 1, []));  % plotgg, input_image, gt_landmark
			pt_pt_err0_temp(gg, k) = compute_error(TR_gt_landmarks{gg}(lm_ind2, :), lm);
		end
		% compute error and cumulative curve
		 pt_pt_err0_image(gg) = sum(pt_pt_err0_temp(gg, :)) / Kpi;
	end
	[pt_pt_err0_allimages, cum_err0] = Compute_cum_error(pt_pt_err0_image, n); 
	saveInitialResults(OutputDir, pt_pt_err_all0,pt_pt_err0, cum_err0, p_mat_rigid, p_mat_nonrigid, p_mat_rigid_gtperturbed, p_mat_nonrigid_gtperturbed, Kpi);	

	%% extract SIFT features
	for SIFT_scale = 10 : 5 : 30
		disp( 'extracting features from training dataset. SIFT scale is ' + SIFT_scale);
		parfor gg = 1 : n
			gg
			p_mat_gg_nonrigid = p_mat_nonrigid(gg, :, :);		
			p_mat_gg_rigid = p_mat_rigid(gg, :, :);
			gt_landmark = TR_gt_landmarks{gg};
			face_size = TR_face_sizes{gg};
			input_image = TR_images{gg};

			for k = 1 : Kpi
				lm = reconstruct_lm(reshape(p_mat_gg_nonrigid(1,k, :), 1, []), reshape(p_mat_gg_rigid(1,k,:),1,[]));
				Sfeat = SIFT_features(input_image, lm, SIFT_scale);
				feat(gg, k, :) = reshape(Sfeat, 1, []); 
				b_mat(gg, k, :) =  reshape(feat(gg, k, :), 1, []) * P - A0P;
			end
		end   
		save([OutputDir 'Perturbed_SIFT_Feature_labels_S-'  num2str(SIFT_scale) '_Kpi-'  num2str(Kpi) '.mat'], 'b_mat');
		save([OutputDir 'Perturbed_SIFT_Features_S-' num2str(SIFT_scale) '_Kpi-' num2str(Kpi) '.mat'], 'feat');

	end
	disp('finished this function');
end


function [] = saveInitialResults(OutputDir, pt_pt_err_all0,pt_pt_err0, cum_err0, p_mat_rigid, p_mat_nonrigid,p_mat_rigid_gtperturbed, p_mat_nonrigid_gtperturbed,  Kpi)
	save([OutputDir 'pt_pt_err_all_initial_Kpi-' num2str(Kpi) '.mat'], 'pt_pt_err_all0');
	save([OutputDir 'pt_pt_err_initial_Kpi-' num2str(Kpi) '.mat'], 'pt_pt_err0');
	save([OutputDir 'cum_err_all_initial_Kpi-' num2str(Kpi)  '.mat'], 'cum_err0');
	save([OutputDir 'p_mat_rigid_initialization_Kpi-' num2str(Kpi) '.mat'], 'p_mat_rigid');
	save([OutputDir 'p_mat_nonrigid_initialization_Kpi-' num2str(Kpi) '.mat'], 'p_mat_nonrigid');
	save([OutputDir 'p_mat_rigid_gtperturbed_Kpi-' num2str(Kpi) '.mat'], 'p_mat_rigid_gtperturbed');
	save([OutputDir 'p_mat_nonrigid_gtperturbed_Kpi-' num2str(Kpi) '.mat'], 'p_mat_nonrigid_gtperturbed');
end



% parfor SIFT_scale = 10 : 30
% 	disp( 'extracting features from training dataset. SIFT scale is ' + SIFT_scale);
% 	for gg = 1 : n
% 		gg
% 		p_mat_gg_nonrigid = p_mat_nonrigid(gg, :, :);		
% 		p_mat_gg_rigid = p_mat_rigid(gg, :, :);
% 		gt_landmark = TR_gt_landmarks{gg};
% 		face_size = TR_face_sizes{gg};
% 		input_image = TR_images{gg};
% 
% 		for k = 1 : Kpi
% 			lm = reconstruct_lm(reshape(p_mat_gg_nonrigid(1,k, :), 1, []), reshape(p_mat_gg_rigid(1,k,:),1,[]),gg, k, [], input_image, gt_landmark);
% 			Sfeat = SIFT_features(input_image, lm, SIFT_scale);
% 			feat{SIFT_scale}(gg, k, :) = reshape(Sfeat, 1, []); 
% 			b_mat{SIFT_scale}(gg, k, :) =  reshape(feat{SIFT_scale}(gg, k, :), 1, []) * P - A0P;
% 		end
% 	end   
% end
% save(['matfiles/Perturbed_SIFT_Feature_labels_Kpi-'  num2str(Kpi) '.mat'], 'b_mat');
% save(['matfiles/Perturbed_SIFT_Features_Kpi-' num2str(Kpi) '.mat'], 'feat');














		