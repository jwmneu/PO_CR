function [TR_images, TR_face_size, TR_gt_landmarks, TR_myShape_p, TR_detections] = Collect_training_images(nHelen, nLFPW) 
	% inputs:  nHelen: number of images of Helen dataset, nLFPW: number of images of LFPW dataset
	disp(['Collecting training images. ' num2str(nHelen) ' from Helen and ' num2str(nLFPW) ' from LFPW']);
	global VERSIONCHECK; 
	VERSIONCHECK = 'SM_1';
	matfilesDir = 'matfiles/';
	saveDir = 'CollectedTrainingDataset/';
	if (exist(saveDir, 'dir') == 0)
		mkdir(saveDir);
	end
	myShape = load([matfilesDir 'myShape.mat']); 
	myShape = myShape.myShape;
	K = size(myShape.p, 2);
	
	% initialize 
	TR_images =cell(nHelen + nLFPW,1);
	TR_face_size = cell(nHelen + nLFPW,1);
	TR_gt_landmarks = cell(nHelen + nLFPW,1);
	TR_myShape_p = zeros(nHelen + nLFPW, K);
	TR_detections = cell(nHelen + nLFPW,1);
		
	% collect from Helen dataset
	[img, gt_lm, facesize, p, detec] = Collect_Helen(nHelen);
	TR_images(1:nHelen, :) = img;
	TR_face_size(1:nHelen, :)  = facesize;
	TR_gt_landmarks(1:nHelen, :)  = gt_lm;
	TR_myShape_p(1:nHelen, :) = p;
	TR_detections(1:nHelen, :) = detec;

	% collect from LFPW dataset
	[img, gt_lm, facesize, p, detec] = Collect_LFPW(nLFPW);
	TR_images(nHelen + 1 : nHelen + nLFPW, :) = img;
	TR_face_size(nHelen + 1 : nHelen + nLFPW, :)  = facesize;
	TR_gt_landmarks(nHelen + 1 : nHelen + nLFPW, :)  = gt_lm;
	TR_myShape_p(nHelen + 1 : nHelen + nLFPW, :) = p;
	TR_detections(nHelen + 1 : nHelen + nLFPW, :) = detec;

% 	% save for further use
	save([saveDir 'TR_images.mat'], 'TR_images', '-v7.3');
% 	save([saveDir 'TR_face_size.mat'], 'TR_face_size');
% 	save([saveDir 'TR_gt_landmarks.mat'], 'TR_gt_landmarks');
% 	save([saveDir 'TR_myShape_p.mat'], 'TR_myShape_p');
% 	save([saveDir 'TR_detections.mat'], 'TR_detections');

% 	load([saveDir 'TR_detections.mat''); 
% 	load([saveDir 'TR_face_size.mat']);
% 	load([saveDir 'TR_gt_landmarks.mat']);
% 	load([saveDir 'TR_myShape_pNonRigid.mat']);
% 	load([saveDir 'TR_myShape_pRigid.mat']);
	
end

function [images, gt_landmarks, face_size, myShape_p, detection] = Collect_Helen(n)
	global VERSIONCHECK; 
	addpath([pwd '/matfiles/']);
	addpath([pwd '/functions/']);
	datasetDir = [pwd '/../dataset/'];
	matfilesDir = [pwd '/matfiles/'];
	myShape = load([matfilesDir 'myShape.mat']); 
	fd_stat = load([matfilesDir 'fd_stat_SM']);
	load([pwd '/../BoundingBoxes/bounding_boxes_helen_trainset.mat']);		
	myShape = myShape.myShape;
	if myShape.version ~= VERSIONCHECK
		disp('myShape model is stale');
	end
	fd_stat = fd_stat.fd_stat;
	if fd_stat.version ~= VERSIONCHECK
		disp('fd_stat model is stale');
	end
	num_of_pts = 68;
	folder = [datasetDir 'helen/trainset/'];
	what = 'jpg';
	names_img = dir([folder '*.' what]);
	names_lm = dir([folder '*.pts']);
	
	myShape_p = myShape.p(1:n, :);
	
	gt_landmarks = cell(n,1);
	face_size = cell(n,1);
	images = cell(n,1);
	detection = cell(n,1);
	
	for gg = 1:n
		pts = read_shape([folder names_lm(gg).name], num_of_pts);   
		gt_landmark = (pts-1);
		gt_landmarks{gg}= reshape(gt_landmark, num_of_pts, 2);
		face_size{gg} =(max(gt_landmark(:,1)) - min(gt_landmark(:,1)) + max(gt_landmark(:,2)) - min(gt_landmark(:,2)))/2;
		images{gg} = imread([folder names_img(gg).name]); 
		bbox = bounding_boxes{gg}.bb_detector;
		bbox(1, 3) = bbox(1, 3) - bbox(1, 1);		% width
		bbox(1, 4) = bbox(1, 4) - bbox(1, 2);		% height
		detection{gg} = bbox;
	end
end


function [images, gt_landmarks, face_size, myShape_p, detection] = Collect_LFPW(n)
	global VERSIONCHECK; 
	addpath([pwd '/matfiles/']);
	addpath([pwd '/functions/']);
	datasetDir = [pwd '/../dataset/'];
	matfilesDir = [pwd '/matfiles/'];
	myShape = load([matfilesDir 'myShape.mat']); 
	fd_stat = load([matfilesDir 'fd_stat_SM']);
	load([pwd '/../BoundingBoxes/bounding_boxes_lfpw_trainset.mat']);		
	myShape = myShape.myShape;
	if myShape.version ~= VERSIONCHECK
		disp('myShape model is stale');
	end
	fd_stat = fd_stat.fd_stat;
	if fd_stat.version ~= VERSIONCHECK
		disp('fd_stat model is stale');
	end
	num_of_pts = 68;
	folder = [datasetDir 'lfpw/trainset/'];
	what = 'png';
	names_img = dir([folder '*.' what]);
	names_lm = dir([folder '*.pts']);
	
	myShape_p = myShape.p(2001:(2000+n), :); 
	
	gt_landmarks = cell(n,1);
	face_size = cell(n,1);
	images = cell(n,1);
	detection = cell(n,1);
	
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










