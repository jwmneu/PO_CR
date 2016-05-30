function [TR_images, TR_face_size, TR_gt_landmarks, TR_myShape_3D_p, TR_detections] = Collect_training_images_3D(nHelen, nLFPW) 
	% inputs:  nHelen: number of images of Helen dataset, nLFPW: number of images of LFPW dataset
	if nargin == 0
		nHelen = 2000; 
		nLFPW = 811; 
	end
	
	disp(['Collecting training images. ' num2str(nHelen) ' from Helen and ' num2str(nLFPW) ' from LFPW']);

	% initialize 
	n = nHelen + nLFPW;
	TR_images =cell(nHelen + nLFPW,1);
	TR_face_size = cell(nHelen + nLFPW,1);
	TR_gt_landmarks = cell(nHelen + nLFPW,1);
	TR_detections = cell(nHelen + nLFPW,1);
		
	% collect from Helen dataset
	[img, gt_lm, facesize, detec] = Collect_Helen(nHelen);
	TR_images(1:nHelen, :) = img;
	TR_face_size(1:nHelen, :)  = facesize;
	TR_gt_landmarks(1:nHelen, :)  = gt_lm;
	TR_detections(1:nHelen, :) = detec;

	% collect from LFPW dataset
	[img, gt_lm, facesize, detec] = Collect_LFPW(nLFPW);
	TR_images(nHelen + 1 : nHelen + nLFPW, :) = img;
	TR_face_size(nHelen + 1 : nHelen + nLFPW, :)  = facesize;
	TR_gt_landmarks(nHelen + 1 : nHelen + nLFPW, :)  = gt_lm;
	TR_detections(nHelen + 1 : nHelen + nLFPW, :) = detec;

	% collect 3D shape parameters
	gt3DParamDir = 'TR_3D_params/';
	load([gt3DParamDir 'TR_3D_scale.mat']); 
	load([gt3DParamDir 'TR_3D_translation.mat']); 
	load([gt3DParamDir 'TR_3D_rotation.mat']); 
	load([gt3DParamDir 'TR_3D_rotation_euler.mat']); 
	load([gt3DParamDir 'TR_3D_nonrigid_params.mat']); 
	TR_myShape_3D_p = cell(nHelen + nLFPW, 1);

	n1 = 2000;
	for gg = 1 : nHelen
		TR_myShape_3D_p{gg} = [TR_3D_scale{gg}, TR_3D_translation{gg}(1,1), TR_3D_translation{gg}(1,2), TR_3D_rotation_euler{gg}, TR_3D_nonrigid_params{gg}];
	end
	for gg = 1 : nLFPW
		TR_myShape_3D_p{nHelen+ gg} = [TR_3D_scale{n1 + gg}, TR_3D_translation{n1 + gg}(1,1), TR_3D_translation{n1 + gg}(1,2), TR_3D_rotation_euler{n1 + gg}, TR_3D_nonrigid_params{n1 + gg}];
	end
	
	% check correctness of all parameters
	myShapeSM3D = load('matfiles/myShapeSM3D.mat');
	pt_pt_err_image = zeros(nHelen+nLFPW, 1);
	lm_pts = 5;
	lm_ind1 = [34, 37, 46, 61, 65]; 
	lm_ind2 = [34, 37, 46, 61, 65, 102, 105, 114, 129, 133]; 

	for gg = 1 : n
		shape2D = TR_gt_landmarks{gg};
		shape2D = shape2D(lm_ind1, :); 
		face_size = TR_face_size{gg}; 
		[lm] = getShapeFrom3DParam(myShapeSM3D.M, myShapeSM3D.V, TR_myShape_3D_p{gg});
		pt_pt_err_image(gg,1) = my_compute_error(shape2D, lm, face_size );
	end
	
	[pt_pt_err_allimages, cum_err] = Compute_cum_error(pt_pt_err_image, nHelen+ nLFPW, 'cum error of 3D shape model, training dataset', 1);

	
 	% save for further use
	saveDir = 'CollectedTrainingDataset/';
	if (exist(saveDir, 'dir') == 0)
		mkdir(saveDir);
	end
% 	save([saveDir 'TR_images.mat'], 'TR_images', '-v7.3');
% 	save([saveDir 'TR_face_size.mat'], 'TR_face_size');
% 	save([saveDir 'TR_gt_landmarks.mat'], 'TR_gt_landmarks');
% 	save([saveDir 'TR_myShape_3D_p.mat'], 'TR_myShape_3D_p');
% 	save([saveDir 'TR_detections.mat'], 'TR_detections');

% 	load([saveDir 'TR_detections.mat''); 
% 	load([saveDir 'TR_face_size.mat']);
% 	load([saveDir 'TR_gt_landmarks.mat']);
% 	load([saveDir 'TR_myShape_3D_p.mat']);
	
end

function [images, gt_landmarks, face_size, detection] = Collect_Helen(n)
	addpath([pwd '/matfiles/']);
	addpath([pwd '/functions/']);
	datasetDir = [pwd '/../dataset/'];
	matfilesDir = [pwd '/matfiles/'];
	load([pwd '/../BoundingBoxes/bounding_boxes_helen_trainset.mat']);		
	
	num_of_pts = 68;
	folder = [datasetDir 'helen/trainset/'];
	what = 'jpg';
	names_img = dir([folder '*.' what]);
	names_lm = dir([folder '*.pts']);
	
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


function [images, gt_landmarks, face_size, detection] = Collect_LFPW(n)
	addpath([pwd '/matfiles/']);
	addpath([pwd '/functions/']);
	datasetDir = [pwd '/../dataset/'];
	matfilesDir = [pwd '/matfiles/'];
	
	load([pwd '/../BoundingBoxes/bounding_boxes_lfpw_trainset.mat']);		
	
	num_of_pts = 68;
	folder = [datasetDir 'lfpw/trainset/'];
	what = 'png';
	names_img = dir([folder '*.' what]);
	names_lm = dir([folder '*.pts']);
	
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










