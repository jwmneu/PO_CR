function [images, TR_face_size, TR_gt_landmarks, TR_myShape_pRigid, TR_myShape_pNonRigid, TR_detections] = Collect_training_images(Helen_numimages, LFPW_numimages) 
	% expect input is  ( '../dataset/', ['helen'; 'lfpw'], [2000, 200])
	global VERSIONCHECK; 
	VERSIONCHECK = 'SM_1';
	images = {};
	TR_face_size = {};
	TR_gt_landmarks = {};
	TR_myShape_pRigid = [];
	TR_myShape_pNonRigid =[];
	TR_detections = {};
		
	[img, gt_lm, facesize, pRigid, pNonRigid, detec] = Collect_Helen(Helen_numimages);
	images = cat(1, images, img');
	TR_face_size = cat(1, TR_face_size, facesize');
	TR_gt_landmarks = cat(1, TR_gt_landmarks, gt_lm');
	TR_myShape_pRigid = cat(1, TR_myShape_pRigid, pRigid);
	TR_myShape_pNonRigid = cat(1, TR_myShape_pNonRigid, pNonRigid);
	TR_detections = cat(1, TR_detections, detec');

	[img, gt_lm, facesize, pRigid, pNonRigid, detec] = Collect_LFPW(LFPW_numimages);
	images = cat(1, images, img');
	TR_face_size = cat(1, TR_face_size, facesize');
	TR_gt_landmarks = cat(1, TR_gt_landmarks, gt_lm');
	TR_myShape_pRigid = cat(1, TR_myShape_pRigid, pRigid);
	TR_myShape_pNonRigid = cat(1, TR_myShape_pNonRigid, pNonRigid);
	TR_detections = cat(1, TR_detections, detec');

% 	load('CollectedTrainingDataset/TR_detections.mat'); 
% 	load('CollectedTrainingDataset/TR_face_size.mat');
% 	load('CollectedTrainingDataset/TR_gt_landmarks.mat');
% 	load('CollectedTrainingDataset/TR_myShape_pNonRigid.mat');
% 	load('CollectedTrainingDataset/TR_myShape_pRigid.mat');
% 	TR_face_size = face_size;
% 	TR_gt_landmarks = gt_landmarks;
% 	TR_myShape_pRigid = myShape_pRigid;
% 	TR_myShape_pNonRigid = myShape_pNonRigid;
% 	TR_detections = detections;
	
	% save for further use
% 	save('CollectedTrainingDataset/TR_face_size.mat', 'TR_face_size');
% 	save('CollectedTrainingDataset/TR_gt_landmarks.mat', 'TR_gt_landmarks');
% 	save('CollectedTrainingDataset/TR_myShape_pRigid.mat', 'TR_myShape_pRigid');
% 	save('CollectedTrainingDataset/TR_myShape_pNonRigid.mat', 'TR_myShape_pNonRigid');
% 	save('CollectedTrainingDataset/TR_detections.mat', 'TR_detections');
	
	% test correctness
% 	[pt_pt_err1] = plot_and_compute_err(1, images, face_size, gt_landmarks, myShape_pRigid, myShape_pNonRigid, detections);
% 	[pt_pt_err2] = plot_and_compute_err(Helen_numimages + 1, images, face_size, gt_landmarks, myShape_pRigid, myShape_pNonRigid, detections);	
	
end

function [pt_pt_err] = plot_and_compute_err(gg, images, face_size, gt_landmarks, myShape_pRigid, myShape_pNonRigid, detections)
	matfilesDir = [pwd '/matfiles/'];
	myShape = load([matfilesDir 'myShape.mat']); 
	myShape = myShape.myShape;
	if myShape.version ~= VERSIONCHECK
		disp('myShape model is stale');
	end
	num_of_pts = 68;
	figure; imshow(images{gg}); hold on;
	lm = myShape.s0 + myShape.QNonrigid * reshape(myShape_pNonRigid(gg, :), [], 1); 
	lm = reshape(lm , [], 2);
	plot(lm(:,1), lm(:,2),  'Color', 'red');
	lm = lm * myShape_pRigid(gg,1);
	plot(lm(:,1), lm(:,2),  'Color', 'blue');
	lm(:, 1)  = lm(:, 1) + myShape_pRigid(gg,2) * ones(num_of_pts,1); 
	lm(:, 2) = lm(:, 2) + myShape_pRigid(gg,3) * ones(num_of_pts, 1); 
	plot(lm(:,1), lm(:,2),  'Color', 'green');
	rectangle('Position', detections{gg}, 'edgecolor', 'red');
	plot(gt_landmarks(gg, :,1), gt_landmarks(gg, :,2), 'blue');
	pt_pt_err = compute_error(reshape(gt_landmarks(gg, :, :), num_of_pts, 2), lm );
end

function [images, gt_landmarks, face_size, myShape_pRigid, myShape_pNonRigid, detection] = Collect_Helen(n)
	global VERSIONCHECK; 
	addpath([pwd '/matfiles/']);
	addpath([pwd '/functions/']);
	datasetDir = [pwd '/../dataset/'];
	matfilesDir = [pwd '/matfiles/'];
	shapemodel = load([matfilesDir 'shape_model.mat']);
	myShape = load([matfilesDir 'myShape.mat']); 
	myAppearance = load([matfilesDir 'myAppearance']);
	fd_stat = load([matfilesDir 'fd_stat_SM']);
	load([pwd '/../BoundingBoxes/bounding_boxes_helen_trainset.mat']);		
	shapemodel = shapemodel.shape;
	myShape = myShape.myShape;
	if myShape.version ~= VERSIONCHECK
		disp('myShape model is stale');
	end
	myAppearance = myAppearance.myAppearance;
	fd_stat = fd_stat.fd_stat;
	if fd_stat.version ~= VERSIONCHECK
		disp('fd_stat model is stale');
	end
	num_of_pts = 68;
	folder = [datasetDir 'helen/trainset/'];
	what = 'jpg';
	names_img = dir([folder '*.' what]);
	names_lm = dir([folder '*.pts']);
	
	myShape_pNonRigid = myShape.pNonrigid(1:n, :, :);
	myShape_pRigid = myShape.pRigid(1:n, :, :);
	
	gt_landmarks = {};
	face_size = {};
	images = {};
	detection = {};
	
	for gg = 1:n
		pts = read_shape([folder names_lm(gg).name], num_of_pts);   
		gt_landmark = (pts-1);
		gt_landmarks{gg}= reshape(gt_landmark, num_of_pts, 2);
		face_size{gg} =(max(gt_landmark(:,1)) - min(gt_landmark(:,1)) + max(gt_landmark(:,2)) - min(gt_landmark(:,2)))/2;
		images{gg} = imread([folder names_img(gg).name]); 
		bbox = bounding_boxes{gg}.bb_detector;
		bbox(1, 3) = bbox(1, 3) - bbox(1, 1);
		bbox(1, 4) = bbox(1, 4) - bbox(1, 2);
		detection{gg} = bbox;
	end
end


function [images, gt_landmarks, face_size, myShape_pRigid, myShape_pNonRigid, detection] = Collect_LFPW(n)
	global VERSIONCHECK; 
	addpath([pwd '/matfiles/']);
	addpath([pwd '/functions/']);
	datasetDir = [pwd '/../dataset/'];
	matfilesDir = [pwd '/matfiles/'];
	shapemodel = load([matfilesDir 'shape_model.mat']);
	myShape = load([matfilesDir 'myShape.mat']); 
	myAppearance = load([matfilesDir 'myAppearance']);
	fd_stat = load([matfilesDir 'fd_stat_SM']);
	load([pwd '/../BoundingBoxes/bounding_boxes_lfpw_trainset.mat']);		
	shapemodel = shapemodel.shape;
	myShape = myShape.myShape;
	if myShape.version ~= VERSIONCHECK
		disp('myShape model is stale');
	end
	myAppearance = myAppearance.myAppearance;
	fd_stat = fd_stat.fd_stat;
	if fd_stat.version ~= VERSIONCHECK
		disp('fd_stat model is stale');
	end
	num_of_pts = 68;
	folder = [datasetDir 'lfpw/trainset/'];
	what = 'png';
	names_img = dir([folder '*.' what]);
	names_lm = dir([folder '*.pts']);
	
	myShape_pNonRigid = myShape.pNonrigid(2001:(2000+n), :, :);	% Helen dataset size is 2000
	myShape_pRigid = myShape.pRigid(2001:(2000+n), :, :);
	
	gt_landmarks = {};
	face_size = {};
	images = {};
	detection = {};
	
	for gg = 1:n
		pts = read_shape([folder names_lm(gg).name], num_of_pts);   
		gt_landmark = (pts-1);
		gt_landmarks{gg} = reshape(gt_landmark, num_of_pts, 2);
		face_size{gg} =(max(gt_landmark(:,1)) - min(gt_landmark(:,1)) + max(gt_landmark(:,2)) - min(gt_landmark(:,2)))/2;
		images{gg} = imread([folder names_img(gg).name]); 
		bbox = bounding_boxes{gg}.bb_detector;
		bbox(1, 3) = bbox(1, 3) - bbox(1, 1);
		bbox(1, 4) = bbox(1, 4) - bbox(1, 2);
		detection{gg} = bbox;
	end
end













