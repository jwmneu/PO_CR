% function that comptues statistics of face detection initialization
function [fd_stat] = face_det_stat()

%% initialization
    load std_detector.mat
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
    num_of_pts = 68; % num of landmarks in the annotations
    
%%  compute mean and std from folder1 and folder2 for non-rigid shape parameters
    mean_nonrigid = zeros(size(myShape.p, 2), 1);											      % assume shape parameters from bounding box are all zero
    var_nonrigid = zeros(size(myShape.p, 2), 1);
     for i = 1:size(myShape.p, 2)
         var_nonrigid(i, 1) = sum(power((myShape.p(:, i) - zeros(size(myShape.p,1),1)) , 2)) / size(myShape.p,1);
     end
     std_nonrigid = sqrt(var_nonrigid); 
     mean_nonrigid = mean_nonrigid(5:end);
     std_nonrigid = std_nonrigid(5:end);
     
%% compute mean and std for rigid shape parameters by testing data of Helen
    folder   = [testsetDir 'Helen/'];
    what  = 'jpg';
    load('bounding_boxes_Helen');											 % initializations produced for noise with sigma = 5; see [1] for more details
    names_test1 = dir([folder '*.' what]);
    names_test2 = dir([folder '*.pts']);
    bb_gt = zeros(length(names_test1), 4);
    bb_var = zeros(length(names_test1), 3);
    for gg = 1 : length(names_test1)
        pts = read_shape([folder names_test2(gg).name], num_of_pts);								    % read ground truth landmarks
        gt_landmark = (pts-1);
        gt_landmark = reshape(gt_landmark, 68, 2);
	% scale ground truth landmarks and bounding boxes to mean face size
	[~,~,T] = procrustes(shape.s0, gt_landmark);     
	scl = 1/T.b;
	gt_landmark = gt_landmark*(1/scl);
	bb_gt(gg, :) = [min(gt_landmark(:,1)), max(gt_landmark(:, 2)), max(gt_landmark(:,2)) - min(gt_landmark(:,2)), max(gt_landmark(:,1)) - min(gt_landmark(:,1))]; 
	bounding_boxes(gg,:) = bounding_boxes(gg,:)*(1/scl);
	% compute varance in mean size
	scale = (bounding_boxes(gg,3) * bounding_boxes(gg,4)) / (bb_gt(gg, 3) * bb_gt(gg, 4));				% ratio between areas
        bb_var(gg, :) = [scale, bb_gt(gg, 1) - bounding_boxes(gg,1), bb_gt(gg, 2) - bounding_boxes(gg,2)];
    end
    
    bb_mean1 = sum(bb_var, 1) / size(bb_var, 1);
    var_rigid1 = zeros(1, size(bb_var,2));
    for i = 1: size(bb_var, 2)
        var_rigid1(:, i)  =  sum(power(bb_var(:, i) - bb_mean1(:, i) * ones(size(bb_var,1), 1) , 2)) / size(bb_var, 1); 
    end
    std_rigid1 = sqrt(var_rigid1); 

%% compute mean and std for rigid shape parameters by testing data of LFPW
    folder   = [testsetDir 'LFPW/'];
    what  = 'png';
    load('bounding_boxes_LFPW');                                                                               % initializations produced for noise with sigma = 5; see [1] for more details
    names_test1 = dir([folder '*.' what]);
    names_test2 = dir([folder '*.pts']);
    bb_gt = zeros(length(names_test1), 4);
    bb_var = zeros(length(names_test1), 3);
    for gg = 1 : length(names_test1)
        pts = read_shape([folder names_test2(gg).name], num_of_pts);								  % read ground truth landmarks
        gt_landmark = (pts-1);
        gt_landmark = reshape(gt_landmark, 68, 2);
	% scale ground truth landmarks and bounding boxes to mean face size
	[~,~,T] = procrustes(shape.s0, gt_landmark);     
	scl = 1/T.b;
	gt_landmark = gt_landmark*(1/scl);
	bb_gt(gg, :) = [min(gt_landmark(:,1)), max(gt_landmark(:, 2)), max(gt_landmark(:,2)) - min(gt_landmark(:,2)), max(gt_landmark(:,1)) - min(gt_landmark(:,1))]; 
	bounding_boxes(gg,:) = bounding_boxes(gg,:)*(1/scl);
	% compute varance in mean size
	scale = (bounding_boxes(gg,3) * bounding_boxes(gg,4)) / (bb_gt(gg, 3) * bb_gt(gg, 4));				% ratio between areas
        bb_var(gg, :) = [scale, bb_gt(gg, 1) - bounding_boxes(gg,1), bb_gt(gg, 2) - bounding_boxes(gg,2)];
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
    bb_mean = [bb_mean, 0];
    std_rigid = [std_rigid, 0];                                % rotation is zero
    mean = [bb_mean, mean_nonrigid'];
    std = [std_rigid, std_nonrigid'];
    fd_stat = struct('mean', mean, 'std', std);
    save('fd_stat.mat', 'fd_stat');
end