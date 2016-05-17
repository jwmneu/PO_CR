function [myShape] = SM_shape_model()
%% initialization
	addpath('functions/');
	modelDir = 'matfiles/';
	shape = load([modelDir 'shape_model.mat']);
	shape = shape.shape;
	datasetDir = '../dataset/'; 
	testsetDir = '../test_data/'; 
	CLMDir = './';
	folder1 = [datasetDir 'helen/trainset/'];
	what1 = 'jpg';
	folder2 = [datasetDir 'lfpw/trainset/'];
	what2 = 'png';
	names1 = dir([folder1 '*.' what1]);
	names2 = dir([folder1 '*.pts']);
	names3 = dir([folder2 '*.' what2]);
	names4 = dir([folder2 '*.pts']);
	num_of_pts = 68; % num of landmarks in the annotations
	lm_pts = 5;
	lm_ind2 = [34, 37,46, 61, 65]; 
	lm_ind = [34,  37,46,  61, 65, 102,  105 ,114 ,129, 133]; 
	Z = []; 
	trans = zeros(length(names1)+length(names3), 2);
	scales = zeros(length(names1)+length(names3),1);
	rotate = zeros(length(names1)+length(names3),1); 
	bb_gt_s0 = [min(shape.s0(:, 1)), min(shape.s0(:,2))];
	n1 = length(names1);
	n2 = length(names3);
	n = n1 + n2; 
% 	[TR_images, TR_face_sizes, TR_gt_landmarks, TR_myShape_pRigid, TR_myShape_pNonRigid, ~] = Collect_training_images(n1, n2) ; 
	
	load('CollectedTrainingDataset/TR_face_size.mat'); 
	load('CollectedTrainingDataset/TR_gt_landmarks.mat'); 
	load('CollectedTrainingDataset/TR_myShape_pRigid.mat'); 
	load('CollectedTrainingDataset/TR_myShape_pNonRigid.mat'); 

	for gg = 1 : n
		gt_landmark = TR_gt_landmarks{gg};
		
		% rigid parameters
		[~, zi, Tti] = procrustes(gt_landmark, shape.s0);
		lmrec = Tti.b *  shape.s0 * Tti.T + Tti.c; 
		scales(gg, 1) = Tti.b; 
		trans(gg, :) = Tti.c(1, :);
		rotate(gg, 1) = asin(Tti.T(1,2)); 
	
		% for nonrigid parameters
		[~,z,Tt] = procrustes(shape.s0, gt_landmark);            % mean landmark & groundtruth landmark. z is the transformed result. T captures translation, rotation and scale. 
		Z = [Z; reshape(z, 1, [])]; 
		
	end
	
%% build rigid shape model
	p_rigid = [scales'; trans(:,1)'; trans(:,2)'; rotate']';
	rotate_column = [-2.23944349742622e-17;-6.17484345920759e-18;-2.80099992007017e-17;-3.72685656486504e-17;-4.36993880933536e-17;0.121267812518166;0.121267812518166;0.121267812518166;0.121267812518166;0.121267812518166];
	Q_rigid = [ones(1, 2 * lm_pts); [ones(1, lm_pts), zeros(1, lm_pts)]; [zeros(1, lm_pts) , ones(1, lm_pts)]; rotate_column']';

%% build non-rigid shape model
	gt_lm = Z; 
	s0 = (sum(gt_lm, 1) / size(gt_lm, 1))';   
	SM_s0 = s0(lm_ind);
	SM_gt_lm = gt_lm(:, lm_ind);
	[Q_nonrigid, p_nonrigid, EiVal] = pca(SM_gt_lm);

	% choose top 95% of eigenvectors
	var_total = sum(EiVal);
	summ = 0;  
	for i = 1:size(EiVal)
		summ = summ + EiVal(i);
		if summ / var_total > 0.95
			PCA_dim = i; 
			break;
		end
	end

	Q_nonrigid = Q_nonrigid(:, 1:PCA_dim);
	p_nonrigid = p_nonrigid(:, 1:PCA_dim);
	EiVal = EiVal(1:PCA_dim, :);
	
	KRigid = 4; 
	KNonrigid = PCA_dim; 
	K = KRigid + KNonrigid;
	p(:, 1:4) = p_rigid; 
	p(:, 5:K) = p_nonrigid; 
	Q(:, 1:4) = Q_rigid; 
	Q(:, 5:K) = Q_nonrigid; 
%% save shape model to myShape
	myShape = struct('s0', SM_s0, 'Q', Q, 'p', p, 'QRigid', Q_rigid, 'QNonrigid', Q_nonrigid, 'pRigid', p_rigid, 'pNonrigid', p_nonrigid, 'version', 'SM_1');
	save('matfiles/myShape.mat', 'myShape');
    
%% test correctness of shape parameters
	for gg = 1 : n
		[pt_pt_err_image(gg)] = Compute_pt_pt_error_image1([p_rigid(gg, :), p_nonrigid(gg, :)], TR_gt_landmarks{gg}, lm_ind2);
	end
	var = 0 : 0.01 : 0.07; 
	[pt_pt_err_allimages, cum_err] = Compute_cum_error(pt_pt_err_image, var, n);
	
% 	% visualization
% 	for gg = 2007:2010
% 		image = imread([folder2 names3(gg-2000).name]); 
% 		pts = read_shape([folder2 names4(gg-2000).name], num_of_pts);     % read ground truth landmarks
% 		gt_landmark = (pts-1);
% 		gt_landmark = reshape(gt_landmark, 68, 2);
% 		[lm] = Computelm(reshape(p_nonrigid(gg, :), 1, []), reshape( p_rigid(gg,:), 1, []), gg, 1,2007:2015, image, gt_landmark);
% 	end
	
end

function [lmrect] = reconstruct_lm(p_nonrigid, p_rigid, lm_ind2)
	modelDir = 'matfiles/';
	myShape = load([modelDir 'myShape.mat']); 
	myShape = myShape.myShape;
	
	% nonrigid parameters
	lm_center = myShape.s0 + myShape.QNonrigid * p_nonrigid'; 
	lm_center = reshape(lm_center , [], 2);
	
	Rot = [ cos(p_rigid(1,4)) , sin(p_rigid(1,4)); -1 * sin(p_rigid(1,4)), cos(p_rigid(1,4))];
	lmrect = p_rigid(1, 1) * lm_center * Rot + repmat(p_rigid(1, 2:3), 5, 1); 
end

function [pt_pt_err_image] = Compute_pt_pt_error_image1(p_mat_gg, gt_landmark, lm_ind2)
	lm = reconstruct_lm(reshape(p_mat_gg(1,5:end), 1, []), reshape(p_mat_gg(1,1:4), 1, []));
	pt_pt_err_image = compute_error(gt_landmark(lm_ind2,:), lm );
end

function [pt_pt_err_allimages, cum_err] = Compute_cum_error(pt_pt_err_image, var, n)
	pt_pt_err_allimages = sum(pt_pt_err_image) / n;
	cum_err = zeros(size(var));
	for ii = 1:length(cum_err)
		cum_err(ii) = length(find(pt_pt_err_image<var(ii)))/length(pt_pt_err_image);
	end
	figure; hold on;
	plot(var, cum_err);
	grid;
end







