function [myShape] = shape_model()
%% initialization
	addpath('functions/');
	load shape_model;
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
	Z = []; 
	trans = zeros(length(names1)+length(names3), 2);
	scales = zeros(length(names1)+length(names3),1);
	rotate = zeros(length(names1)+length(names3),1); 
	bb_gt_s0 = [min(shape.s0(:, 1)), min(shape.s0(:,2))];

%% input training images from folder1
	for gg = 1:length(names1)
		pts = read_shape([folder1 names2(gg).name], num_of_pts);     % read ground truth landmarks
 		gt_landmark = (pts-1);
		gt_landmark = reshape(gt_landmark, 68, 2);
		[~,z,Tt] = procrustes(shape.s0, gt_landmark);            % mean landmark & groundtruth landmark. z is the transformed result. T captures translation, rotation and scale. 
		Z = [Z; reshape(z, 1, [])]; 
		scales(gg, 1) = (max(gt_landmark(:,1)) - min(gt_landmark(:,1))) / (max(shape.s0(:,1)) - min(shape.s0(:,1)));
		scale_z = z * scales(gg,1);
		bb_gt(1,:) = [min(gt_landmark(:,1)), min(gt_landmark(:, 2)), max(gt_landmark(:,1)) - min(gt_landmark(:,1)), max(gt_landmark(:,2)) - min(gt_landmark(:,2))]; 
		trans(gg, :) = bb_gt(1, 1:2) - [min(scale_z(:,1)), min(scale_z(:,2))]; 
		rotate(gg, 1) = asin(Tt.T(2,1));    
		% test
% 		if gg == 2
% 			image = imread([folder1 names1(gg).name]); 
% 			figure;imagesc(image); colormap(gray); hold on;
% 			plot(gt_landmark(:,1), gt_landmark(:,2));
% 			plot(z(:,1), z(:,2));
% 			plot(scale_z(:,1), scale_z(:,2));
% 			rectangle('Position', bb_gt(1,:));
% 		end
	end

%% input training images from folder2
	for gg = 1:length(names3)
		pts = read_shape([folder2 names4(gg).name], num_of_pts);     % read ground truth landmarks
		gt_landmark = (pts-1);
		gt_landmark = reshape(gt_landmark, 68, 2);
		[~,z,Tt] = procrustes(shape.s0, gt_landmark);            % mean landmark & groundtruth landmark. procrustes:compute linear tranformation between two matrices. 
		Z = [Z; reshape(z, 1, [])]; 
		scales(gg+length(names1), 1) = (max(gt_landmark(:,1)) - min(gt_landmark(:,1))) / (max(shape.s0(:,1)) - min(shape.s0(:,1)));
		scale_z = z * scales(gg+length(names1),1);
		bb_gt(1,:) = [min(gt_landmark(:,1)), min(gt_landmark(:, 2)), max(gt_landmark(:,1)) - min(gt_landmark(:,1)), max(gt_landmark(:,2)) - min(gt_landmark(:,2))]; 
		trans(gg+length(names1), :) = bb_gt(1, 1:2) - [min(scale_z(:,1)), min(scale_z(:,2))]; 
		rotate(gg+length(names1), 1) = asin(Tt.T(2,1)); 
		% test
% 		if gg == 2
% 			image = imread([folder2 names3(gg).name]); 
% 			figure;imagesc(image); colormap(gray); hold on;
% 			plot(gt_landmark(:,1), gt_landmark(:,2));
% 			plot(z(:,1), z(:,2));
% 			plot(scale_z(:,1), scale_z(:,2));
% 			rectangle('Position', bb_gt(1,:));
% 		end

	end
	
%% build rigid shape model
	rigidp = [scales'; trans(:,1)'; trans(:,2)'; rotate']';
	rigidQ = [ones(1, 2 * num_of_pts); [ones(1, num_of_pts), zeros(1, num_of_pts)]; [zeros(1, num_of_pts) , ones(1, num_of_pts)]; shape.Q(:,2 )']';

%% build non-rigid shape model
	gt_lm = Z; 
	s0 = (sum(gt_lm, 1) / size(gt_lm, 1))';                            
	[Q, p, EiVal] = pca(gt_lm);

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

	Q = Q(:, 1:PCA_dim);
	p = p(:, 1:PCA_dim);
	EiVal = EiVal(1:PCA_dim, :);

%% save shape model to myShape
	myShape = struct('s0', s0, 'QRigid', rigidQ, 'QNonrigid', Q, 'pRigid', rigidp, 'pNonrigid', p);
	save('matfiles/myShape.mat', 'myShape');
    
%% test correctness of shape parameters
	for gg = 2007:2015
		image = imread([folder2 names3(gg-2000).name]); 
		pts = read_shape([folder2 names4(gg-2000).name], num_of_pts);     % read ground truth landmarks
		gt_landmark = (pts-1);
		gt_landmark = reshape(gt_landmark, 68, 2);
		[lm] = Computelm(reshape(p(gg, :), 1, []), reshape( rigidp(gg,:), 1, []), gg, 1,2007:2015, image, gt_landmark);
	end
	
%% plot shape model -- plot mean shape and first three eigenvectors added to the mean shape
% 	gg = 1;
% 
% 	p1 = zeros(size(Q,2), 1);
% 	p1(1, 1) = p(gg, 1);
% 	p2 = zeros(size(Q,2), 1);
% 	p2(2,1) = p(gg, 2);
% 	p3 = zeros(size(Q,2), 1);
% 	p3(2,1) = p(gg, 3);
% 
% 	landmarks = s0 + Q * p(gg, :)';
% 	landmarks1 = s0 + Q * p1; 
% 	landmarks2 = s0 + Q * p2; 
% 	landmarks3 = s0 + Q * p3; 
% 	landmarks = reshape(landmarks, [], 2);
% 	landmarks1 = reshape(landmarks1, [], 2);
% 	landmarks2 = reshape(landmarks2, [], 2);
% 	landmarks3 = reshape(landmarks3, [], 2);
% 
% 	figure;  hold on; 
% 	plot(landmarks(:,1), landmarks(:,2), 'o');     
% 	plot(landmarks1(:,1), landmarks1(:,2), '+');    
% 	plot(landmarks2(:,1), landmarks2(:,2), '*');    
% 	plot(landmarks3(:,1), landmarks3(:,2), '-');   
    
end












