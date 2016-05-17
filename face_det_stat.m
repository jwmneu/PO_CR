% function that comptues statistics of face detection initialization
function [fd_stat] = face_det_stat()

%% initialization
	addpath('functions/');
	addpath('matfiles/');
	load shape_model;
	load myShape;
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
	num_of_pts = 68;
	if myShape.version ~= 'SM_1'
		disp('myShape model is stale');
		return;
	end
	
% 	[TR_images, TR_face_sizes, TR_gt_landmarks, TR_myShape_pRigid, TR_myShape_pNonRigid, ~] = Collect_training_images(n1, n2) ; 
	
	load('CollectedTrainingDataset/TR_face_sizes.mat'); 
	load('CollectedTrainingDataset/TR_gt_landmarks.mat'); 
	load('CollectedTrainingDataset/TR_myShape_pRigid.mat'); 
	load('CollectedTrainingDataset/TR_myShape_pNonRigid.mat'); 

	
%%  compute mean and std from folder1 and folder2 for non-rigid shape parameters
	% assume shape parameters from bounding box are all zero
	var_nonrigid = zeros(size(myShape.pNonrigid, 2), 1);
	mean_nonrigid = zeros(size(myShape.pNonrigid, 2), 1);											     
	for i = 1:size(myShape.pNonrigid, 2)
		var_nonrigid(i, 1) = sum(power((myShape.pNonrigid(:, i) - zeros(size(myShape.pNonrigid,1),1)) , 2)) / size(myShape.pNonrigid,1);
	end
	std_nonrigid = sqrt(var_nonrigid); 
     
%% compute mean and std for rigid shape parameters by train data of Helen
	load('../BoundingBoxes/bounding_boxes_helen_trainset.mat');									 % initializations produced for noise with sigma = 5; see [1] for more details
	bb_gt = zeros(length(bounding_boxes), 4);
	bb_var = zeros(length(bounding_boxes), 4);
	for gg = 1 : length(bounding_boxes)
		imgname  = strsplit(bounding_boxes{gg}.imgName, '.');
		s = strcat(folder1, strcat(imgname(1) ,'.pts'));
		s = s{1};
		pts = read_shape(s, num_of_pts);								    % read ground truth landmarks
		gt_landmark = (pts-1);
		gt_landmark = reshape(gt_landmark, 68, 2);
		% scale ground truth landmarks and bounding boxes to mean face size
		[~,~,T] = procrustes(shape.s0, gt_landmark);     
		scl = 1/T.b;
		gt_landmark = gt_landmark*(1/scl);
		% 	input_image = imread([folder1 bounding_boxes{gg}.imgName]); 
		% 	input_image = imresize(input_image, (1/scl));
		% 	figure;
		% 	imshow(input_image);
		% 	hold on;

		% bounding box format: x, y, width, height
		bb_gt(gg, :) = [min(gt_landmark(:,1)), min(gt_landmark(:, 2)), max(gt_landmark(:,1)) - min(gt_landmark(:,1)), max(gt_landmark(:,2)) - min(gt_landmark(:,2))]; 
		bbox(gg,:) = bounding_boxes{gg}.bb_detector * (1/scl);
		bbox(gg, 3) = bbox(gg, 3) - bbox(gg, 1);
		bbox(gg, 4) = bbox(gg, 4) - bbox(gg, 2);
		% 	rectangle('Position', bb_gt(gg, :),'edgecolor', 'red');
		% 	rectangle('Position', bbox(gg, :),'edgecolor', 'blue');

		% compute varance in mean size
		scale = ((bbox(gg, 3) / bb_gt(gg, 3)) + (bbox(gg, 4) / bb_gt(gg, 4)) )	/2;		% average scale of width and height
		bb_var(gg, :) = [scale, bb_gt(gg, 1) - bbox(gg,1), bb_gt(gg, 2) - bbox(gg,2), asin(T.T(2,1))];	    % scale, delta_x, delta_y, rotation
	end
    
	bb_mean1 = sum(bb_var, 1) / size(bb_var, 1);
	var_rigid1 = zeros(1, size(bb_var,2));
	for i = 1: size(bb_var, 2)
		var_rigid1(:, i)  =  sum(power(bb_var(:, i) - bb_mean1(:, i) * ones(size(bb_var,1), 1) , 2)) / size(bb_var, 1); 
	end
	std_rigid1 = sqrt(var_rigid1); 

%% compute mean and std for rigid shape parameters by train data of LFPW
	what  = 'png';
	load('../BoundingBoxes/bounding_boxes_lfpw_trainset.mat');		                                                                             % initializations produced for noise with sigma = 5; see [1] for more details
	bb_gt = zeros(length(bounding_boxes), 4);
	bb_var = zeros(length(bounding_boxes), 4);
	for gg = 1 : length(bounding_boxes)
		imgname  = strsplit(bounding_boxes{gg}.imgName, '.');
		s = strcat(folder2, strcat(imgname(1) ,'.pts'));
		s = s{1};
		pts = read_shape(s, num_of_pts);	
		gt_landmark = (pts-1);
		gt_landmark = reshape(gt_landmark, 68, 2);
		% scale ground truth landmarks and bounding boxes to mean face size
		[~,~,T] = procrustes(shape.s0, gt_landmark);     
		scl = 1/T.b;
		gt_landmark = gt_landmark*(1/scl);
		% 	input_image = imread([folder2 bounding_boxes{gg}.imgName]); 
		% 	input_image = imresize(input_image, (1/scl));
		% 	figure;
		% 	imshow(input_image);
		% 	hold on;
		bb_gt(gg, :) = [min(gt_landmark(:,1)), min(gt_landmark(:, 2)), max(gt_landmark(:,1)) - min(gt_landmark(:,1)),max(gt_landmark(:,2)) - min(gt_landmark(:,2)) ]; 
		bbox(gg,:) = bounding_boxes{gg}.bb_detector * (1/scl);
		bbox(gg, 3) = bbox(gg, 3) - bbox(gg, 1);
		bbox(gg, 4) = bbox(gg, 4) - bbox(gg, 2);
		% 	rectangle('Position', bb_gt(gg, :),'edgecolor', 'red');
		% 	rectangle('Position', bbox(gg, :),'edgecolor', 'blue');
		% compute varance in mean size
		scale = ((bbox(gg, 3) / bb_gt(gg, 3)) + (bbox(gg, 4) / bb_gt(gg, 4)) )	/2;				% ratio between areas
		bb_var(gg, :) = [scale, bb_gt(gg, 1) - bbox(gg,1), bb_gt(gg, 2) - bbox(gg,2), asin(T.T(2,1))];
	end

	bb_mean2 = sum(bb_var, 1) / size(bb_var, 1);
	var_rigid2 = zeros(1, size(bb_var,2));
	for i = 1: size(bb_var, 2)
		var_rigid2(:, i)  =  sum(power(bb_var(:, i) - bb_mean2(:, i) * ones(size(bb_var,1), 1) , 2)) / size(bb_var, 1); 
	end
	std_rigid2 = sqrt(var_rigid2); 
    
%% combine result from two datasets and save to fd_stat
	bb_mean = (bb_mean1 + bb_mean2) / 2; 
	std_rigid = (std_rigid1 + std_rigid2) / 2; 
	mean = [bb_mean, mean_nonrigid'];
	std = [std_rigid, std_nonrigid'];
	fd_stat = struct('mean', mean, 'std', std, 'version', myShape.version);
	save('matfiles/fd_stat_SM.mat', 'fd_stat');
end