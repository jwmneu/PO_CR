function [myShape_testing] = SM_Compute_Shape_Params()
% this function computes shape parameters for testing images, using myShape model
	% clear;
	global VERSIONCHECK; 
	VERSIONCHECK = 'SM_1'; 
	gtParamDir = 'TR_testing_params/';
	% initialize parameters
	num_of_pts = 68;
	n1 = 330;					   
	n2 = 223;
	n = n1 + n2; 
	use_5_lms = 1;
	lm_pts = 5;
	lm_ind1 = [34, 37, 46, 61, 65]; 
	lm_ind2 = [34, 37, 46, 61, 65, 102, 105, 114, 129, 133]; 
	% load models
	modelDir = 'matfiles/';
	shape = load([modelDir 'shape_model.mat']);
	shape = shape.shape;
	myShape = load([modelDir 'myShape.mat']); 
	myShape = myShape.myShape;
	if myShape.version ~= VERSIONCHECK
		disp('myShape model is stale');
	end
	K = size(myShape.p, 2);
	
	% collect testing data
% 	[TR_testing_images, TR_testing_face_size, TR_testing_gt_landmarks, TR_testing_myShape_p, TR_testing_detections] = Collect_testing_images(n1, n2); 
	load([gtParamDir 'TR_testing_detections.mat']); 
	load([gtParamDir 'TR_testing_face_size.mat']);
	load([gtParamDir 'TR_testing_gt_landmarks.mat']);
	
	TR_testing_myShape_p = zeros(n, K);
	pt_pt_err_image = zeros(1, n);
	for gg = 1:n
		gt_landmark = TR_testing_gt_landmarks{gg};
		
		% rigid parameters
		[~, zi, Tti] = procrustes(gt_landmark, shape.s0);	% match shape.s0 to gt_landmarks
		TR_testing_myShape_p(gg, 1) = Tti.b;				     %scale
		TR_testing_myShape_p(gg, 2:3) = Tti.c(1, :);			   % translation
		TR_testing_myShape_p(gg, 4) = asin(Tti.T(1,2));			% rotation
		
		% use ridge regression to find nonrigid parameters
		[~,z,~] = procrustes(shape.s0, gt_landmark);            % mean landmark & groundtruth landmark. z is the transformed result. T captures translation, rotation and scale. 
		TR_testing_myShape_p(gg, 5:end) = ridge(reshape((z(lm_ind1, :) - shape.s0(lm_ind1, :)), [], 1), myShape.QNonrigid, 0);
	
		[pt_pt_err_image(1, gg)]  = Compute_pt_pt_error_image1(TR_testing_myShape_p(gg,:), gt_landmark, TR_testing_face_size{gg}, gg, lm_ind1); 
	end
	
	% compute cum error
	[pt_pt_err_allimages, cum_err] = Compute_cum_error(pt_pt_err_image, n, 'cum error of shape model for testing dataset', 1);

	save([gtParamDir 'TR_testing_myShape_p.mat'], 'TR_testing_myShape_p');
end


function [pt_pt_err_image] = Compute_pt_pt_error_image1(p_mat_gg, gt_landmark, face_size, gg, lm_ind1)
	lm = reconstruct_lm(p_mat_gg(1, :));
	pt_pt_err_image = my_compute_error(gt_landmark(lm_ind1,:), lm, face_size );
end
