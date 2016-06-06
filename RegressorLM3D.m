function [] = RegressorLM3D(t, debugchoice, sc_rigid, sc_nonrigid, regu_lambda)
% chosed shape model of KTorresani = 25, energy = 0.95, KNonrigid = 17
	t = 5; debugchoice = 'all'; sc_rigid = 0.05; sc_nonrigid = 0.05; regu_lambda = 0; 
	debug_ch = debugchoice
	disp(['iteration is ' num2str(t)]);
	iter = t; 
	global VERSIONCHECK; 
	VERSIONCHECK = 'LM_3D_1'; 
	Kpi = 10;
	ridge_param = 0;
	smallsize = 0;
	if smallsize == 0
		n1 = 2000;					   
		n2 = 811;	
	else
		n1 = 30;
		n2 = 30;
	end
	n = n1 + n2; 
	lm_pts = 68;
	lm_ind1 = 1:68; 
	lm_ind2 = 1:136; 
	makeplot = 1; 
	SIFT_scale = 2; 
	% load models
	[myShapeLM3D, myAppearanceLM, fd_stat_LM_3D, P, A0P, N, m, KNonrigid, KRigid, K] = load_models(VERSIONCHECK);
	gtParamDir = '../Learning_Inputs/';				% the Learning Inputs have the splited train and validate sets
	load([gtParamDir 'TR_Training_Params.mat']);
	load([gtParamDir 'TR_Validation_Params.mat']);
%% load shape parameters and SIFT features for current iteration
	perturb_choice = 'all';
	LearningResultsDir = '../Learning_Results/'; 
	shapeParamDir = [LearningResultsDir 'Iteration_' num2str(iter-1) '_' perturb_choice '/'];			% t must be an iterger larger than 0
	FeaturesDir = [LearningResultsDir 'SIFT_Features_' num2str(iter-1) '_' perturb_choice '/']; 
	load([shapeParamDir 'Shape_Parameters_LM_3D.mat']); % get 'p_mat_gt_train', 'p_mat_train', 'labeled_delta_p_train', 'pt_pt_err0_allimages_train', 'pt_pt_err0_image_train', 'cum_err0_train', 'choice_perturb', 'perturb_param');
	load([FeaturesDir 'SIFT_Features.mat']);			 % get  'features_train', 'b_mat_train', 'features_validate', 'b_mat_validate'
	
%% ridge regression to compute Jp

	% debug, choise debug parameters
% 	sc_rigid = 0.15; 
% 	sc_nonrigid = 0.4;
% 
% 	sc_rigid = 0.2; 
%  	sc_nonrigid = 0.2;

	debug_ch
	
	learning_rate = [sc_rigid,  sc_rigid* ones(1,2), sc_rigid * ones(1,3), sc_nonrigid * ones(1, KNonrigid)];		% learning rate for scale, translation, rotation, nonrigid
	cho = {'scale', 'translation', 'rotation', 'scale_translation', 'translation_rotation', ...
		'rigid', 'nonrigid', 'translation_nonrigid', 'all', 'nonrigid1', ...
		'nonrigid2', 'nonrigid3', 'nonrigid12', 'nonrigid123', 'nonrigid1234', 'nonrigid12345'}; 
	debug_param = { [1], [2:3], [4:6], [1:3], [2:6], [1:6], [7:K], [2:3, 7:K], [1:K], [7], [8], [9], [7:8], [7:9], [7:10], [7:11]}; 
	dict = containers.Map(cho, debug_param);
% 	debugchoice = cho{7}
	debug_param = dict(debug_ch)

	% compute Jacobian
	regu_lambda
	
	[Jp, Hessian, Risk, reconstruct_b_mat_train, scaled, std_delta_p] = Compute_Jacobian(b_mat_train, labeled_delta_p_train, N, ridge_param, debug_ch, ...
		debug_param, regu_lambda); 


%% update p and compute pt-pt error

	debugchoice
	debug_param
	
	LearningResultsDir = '../Learning_Results/'; 
	outputDir = [LearningResultsDir 'Iteration_' num2str(iter) '_' perturb_choice '_debugchoice_' debug_ch '_sc_rigid_' num2str(sc_rigid) '_sc_nonrigid_' num2str(sc_nonrigid) '_regu_lambda_' num2str(regu_lambda) '/'] ; 
	if(exist(outputDir, 'dir') == 0)
		mkdir(outputDir);
	end
		
	% update training p and evaluate on training dataset
	[p_mat_train, learned_delta_p_train, reconstruct_lms_train, pt_pt_err_image_train, pt_pt_err_allimages_train, cum_err_train] = Update_and_Evaluate(p_mat_train, features_train, ...
		TR_gt_landmarks_train, TR_face_size_train, Risk, myShapeLM3D.M, myShapeLM3D.V, myAppearanceLM.A0, n_train, Kpi, learning_rate, ...
		N, K, KNonrigid, lm_pts, lm_ind1, iter, outputDir, debug_ch, debug_param, scaled, std_delta_p, makeplot, 'training', sc_nonrigid); 
	
	% evaluate on validation dataset
	[p_mat_validate, learned_delta_p_validate, reconstruct_lms_validate, pt_pt_err_image_validate, pt_pt_err_allimages_validate, cum_err_validate] = ...
		Update_and_Evaluate(p_mat_validate, features_validate, TR_gt_landmarks_validate, TR_face_size_validate, Risk, myShapeLM3D.M, ...
		myShapeLM3D.V, myAppearanceLM.A0, n_validate, Kpi, learning_rate, N, K, KNonrigid, lm_pts, lm_ind1, iter, outputDir, debug_ch, ...
		debug_param, scaled, std_delta_p, makeplot, 'validation', sc_nonrigid); 
	
	[labeled_delta_p_train, labeled_delta_p_validate] = compute_labeled_delta_p(p_mat_train, p_mat_validate, p_mat_gt_train, p_mat_gt_validate);
	
	% save intermediate results per iteration
	t = iter; 
	debugchoice = debug_ch; 
	save([outputDir 'Shape_Parameters_LM_3D.mat'], 'p_mat_train', 'p_mat_gt_train', 'learned_delta_p_train', 'labeled_delta_p_train', 'reconstruct_lms_train', 'b_mat_train', 'pt_pt_err_image_train', ...
		'pt_pt_err_allimages_train', 'cum_err_train', 'p_mat_validate', 'p_mat_gt_validate', 'learned_delta_p_validate', 'labeled_delta_p_validate', 'reconstruct_lms_validate', 'pt_pt_err_image_validate', ...
		'pt_pt_err_allimages_validate', 'cum_err_validate', 'Risk', 'Jp', 't', 'SIFT_scale', 'Kpi', 'ridge_param', 'learning_rate', 'debugchoice', 'debug_param');  

end


%% ################################################     helper functions    ##################################################

function [Jp, Hessian, Risk, reconstruct_b_mat, scaled, std_delta_p] = Compute_Jacobian(b_mat, delta_p, N, ridge_param, debugchoice, debug_param, regu_lambda)
	disp(['doing ridge regression seperately for ' debugchoice]);
	K = size(delta_p, 2);
	KNonrigid = K - 6;
	scaled = false; 
	std_delta_p = 0; 
	
	% ------------- debug, seperately -------------
% 	Jp = zeros(N, size(debug_param, 2));
% 	for reg = 1:N
% 		Jp(reg,:) = ridge( b_mat(:,reg),  delta_p(:, debug_param), 0);
% 	end
	
	% ------------- debug, jointly -------------
% 	b_mat_joint = sum(b_mat, 2); 
% 	delta_p_joint = repmat(delta_p(:, debug_param), 1, N); 
% 	Jp_joint = ridge(b_mat_joint, delta_p_joint, 0) ; 
% 	Jp = reshape(Jp_joint', 2, [])';
	
	% debug, direct compute
% 	Jp_transpose = delta_p(:, debug_param)' * delta_p(:, debug_param) \ (delta_p(:, debug_param)' * b_mat); 
% 	Jp = Jp_transpose'; 

	% ------------- debug, lsqr to compute Jp -------------
% 	Jp = zeros(N, size(delta_p, 2));
% 	for reg = 1 : N
% 		[Jp(reg,:) , flag] = lsqr(delta_p, b_mat(:,reg));
% 	end
% 	Jp = Jp(:, debug_param); 
	
	% ------------- debug, lsqr with regularization -------------
% 	Jp = zeros(N, size(delta_p, 2));
% 	X = [delta_p; regu_lambda * eye(K)];
% 	
% 	for reg = 1 : N
% 		y = [b_mat(:, reg); zeros(K, 1)]; 
% 		[Jp(reg,:) , flag] = lsqr(X, y);
% 	end
% 	Jp = Jp(:, debug_param); 
	
	% ------------- debug, lsqr with scaling --------------
% 	std_delta_p = std(delta_p, 1);
% 	delta_p = delta_p ./ repmat(std_delta_p, size(delta_p, 1), 1); 
% 	Jp = zeros(N, size(delta_p, 2));
% 	for reg = 1 : N
% 		[Jp(reg,:) , flag] = lsqr(delta_p, b_mat(:,reg));
% 	end
% 	Jp = Jp(:, debug_param); 
% 	scaled = true; 
	
	% ------------- debug, lsqr with scaling and regularization -------------
	% scaling
	std_delta_p = std(delta_p, 1);
	delta_p = delta_p ./ repmat(std_delta_p, size(delta_p, 1), 1); 
	Jp = zeros(N, size(delta_p, 2));
	% regularization
	regu_lambda
	X = [delta_p; regu_lambda * eye(K)];
	% compute
	for reg = 1 : N
		y = [b_mat(:, reg); zeros(K, 1)]; 
		[Jp(reg,:) , flag] = lsqr(X, y);
	end
	Jp = Jp(:, debug_param); 
	scaled = true; 
	
	% ------------- debug, lsqr with scaling and different regularization for different parameters-------------
% 	% scaling
% 	std_delta_p = std(delta_p, 1);
% 	delta_p = delta_p ./ repmat(std_delta_p, size(delta_p, 1), 1); 
% 	Jp = zeros(N, size(delta_p, 2));
% 	% regularization
% 	param_lambda = []; 
% 	X = [delta_p; regu_lambda * eye(K)];
% 	% compute
% 	for reg = 1 : N
% 		y = [b_mat(:, reg); zeros(K, 1)]; 
% 		[Jp(reg,:) , flag] = lsqr(X, y);
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
	A0, n,  Kpi, learning_rate, N, K, KNonrigid, lm_pts, lm_ind1, t, outputDir, debugchoice, debug_param, scaled, std_delta_p, plot, dataset, sc_nonrigid)
	
	disp(['updating shape parameters for ' dataset ' dataset']);
			
	p_mat_new = zeros(n * Kpi, K);
	delta_p = zeros(n * Kpi, K);
	reconstruct_lms = zeros(n * Kpi, lm_pts * 2); 
	pt_pt_err_image = zeros(1, n);
	
	debugchoice
	debug_param
	
	for gg = 1:n
		% update p_mat by previous p_mat + delta_p
		[p_mat_new((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), delta_p((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), reconstruct_lms((gg-1)*Kpi + 1 : (gg-1)*Kpi + Kpi, :)] = ...
			Update_p_mat(p_mat((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :), Risk, features((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi,:), ...
			M, V, A0,  Kpi, learning_rate, N, K, KNonrigid, debugchoice, debug_param, scaled, std_delta_p, lm_pts);
		
		if gg == 1 || gg == 10 || gg == 100
			delta_p((gg-1)*Kpi + 1, :)
		end
		
		% compute pt-pt error of this image
		[pt_pt_err_image(1, gg)] = Compute_pt_pt_error_image(M, V, p_mat_new((gg-1) *Kpi + 1 : (gg-1) * Kpi + Kpi, :),  TR_gt_landmarks{gg}, TR_face_size{gg}, Kpi, gg, lm_ind1);
	end 
	
	% compute error and cumulative curve
	dataset
	debugchoice
	
	[pt_pt_err_allimages, cum_err(1, :) ] = Compute_cum_error(pt_pt_err_image(1, :), n, outputDir, ['cum_error_after_' num2str(t) '_iteration_of_' dataset '_dataset_' debugchoice], plot);

end

function [p_mat_gg_new, delta_p_image, reconstruct_lm] = Update_p_mat(p_mat_gg, Risk, features_image, M, V, A0, Kpi, learning_rate, N, K, KNonrigid, debugchoice, ... 
	debug_param, scaled, std_delta_p, lm_pts)
	delta_p_image = zeros(Kpi, K);
	p_mat_gg_new = zeros(Kpi, K);
	reconstruct_lm = zeros(Kpi, lm_pts * 2); 
		
	for k = 1 : Kpi
		if isContinuous(debug_param) == 1
			delta_p_image(k, :) = [zeros(debug_param(1)-1, 1); learning_rate(1, debug_param)' .* ( Risk * (reshape(features_image(k, :), [], 1) - A0) ); zeros(K - debug_param(end), 1)]';
		else
			if strcmp(debugchoice, 'translationandnonrigid123')
				delta_p_temp =  learning_rate(1, debug_param)' .* ( Risk * (reshape(features_image(k, :), [], 1) - A0) );
				delta_p_image(k, :) =[zeros(1,1); delta_p_temp(1:2, 1); zeros(3,1); delta_p_temp(3:end, 1); zeros(KNonrigid-3, 1)]';   
			elseif strcmp(debugchoice , 'translationandnonrigid')
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

function [labeled_delta_p_train, labeled_delta_p_validate] = compute_labeled_delta_p(p_mat_train, p_mat_validate, p_mat_gt_train, p_mat_gt_validate)
	labeled_delta_p_train = p_mat_gt_train - p_mat_train; 
	labeled_delta_p_validate = p_mat_gt_validate - p_mat_validate; 
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

function [isContinu] = isContinuous(a)			% a is a row vector
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