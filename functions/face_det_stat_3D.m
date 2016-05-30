function [] = face_det_stat_3D(KTorresani, energy, KNonrigid)
% run this function in folder PO_CR_code_v1
% function that comptues statistics of face detection initialization
% chosed shape model of KTorresani = 15, energy = 0.8, KNonrigid = 7

%% initialization
	saveDir = 'matfiles/'; 		
	
	gtParamDir = 'TR_params/';
	load([gtParamDir 'TR_face_size.mat']); 
	load([gtParamDir 'TR_gt_landmarks.mat']); 
	load([gtParamDir 'TR_detections.mat']);		% detected bounding boxes 
	
	if nargin == 0
		KTorresani = 15; 
		energy = 0.8; 
		KNonrigid = 7;
	end
	gt3DParamDir = ['TR_3D_params/'];
	load([gt3DParamDir 'TR_3D_rotation.mat']);
	load([gt3DParamDir 'TR_3D_rotation_euler.mat']);
	load([gt3DParamDir 'TR_3D_nonrigid_params.mat']);
	KNonrigid = size(TR_3D_nonrigid_params{1}, 2);	
	n = size(TR_3D_nonrigid_params, 1);		% number of images
	
%% compute statistics of 3D shape parameters from all images of Helen and LFPW training dataset
	% nonrigid parameters
	TR_3D_nonrigid_params_mat = cell2mat(TR_3D_nonrigid_params);
	mean_nonrigid_params = mean(TR_3D_nonrigid_params_mat, 1);
	var_nonrigid_params = var(TR_3D_nonrigid_params_mat, 1);
	std_nonrigid_params = sqrt(var_nonrigid_params); 
		
	% rotation euler
	TR_3D_rotation_euler_mat = cell2mat(TR_3D_rotation_euler);
	mean_rotation_euler = mean(TR_3D_rotation_euler_mat, 1);
	var_rotation_euler = var(TR_3D_rotation_euler_mat,1);
	std_rotation_euler = sqrt(var_rotation_euler); 
	
	% scale, translation
	for gg = 1:n
		gt_landmark = TR_gt_landmarks{gg};
		face_size = TR_face_size{gg};
		bb_gt(gg, :) =  [min(gt_landmark(:,1)), min(gt_landmark(:, 2)), max(gt_landmark(:,1)) - min(gt_landmark(:,1)), max(gt_landmark(:,2)) - min(gt_landmark(:,2))]; 
		bb_detection(gg, :) = TR_detections{gg}; 
		delta_translation(gg, 1) = (bb_detection(gg,1) - bb_gt(gg, 1)) / face_size; 
		delta_translation(gg, 2) = (bb_detection(gg, 2) - bb_gt(gg, 2)) / face_size; 
		delta_scale(gg, 1)  = ((bb_detection(gg, 3) / bb_gt(gg, 3)) + (bb_detection(gg, 4) / bb_gt(gg, 4)) )	/2;
	end
	
	mean_delta_translation = mean(delta_translation);
	var_delta_translation = var(delta_translation);
	std_delta_translation = sqrt(var_delta_translation);
	
	mean_delta_scale = mean(delta_scale);
	var_delta_scale = var(delta_scale);
	std_delta_scale = sqrt(var_delta_scale); 
	
	version = 'SM_3D_1';
	save([saveDir 'fd_stat_SM_3D_KTorresani-' num2str(KTorresani) '_energy-' num2str(energy) '_KNonrigid-' num2str(KNonrigid) '.mat'], 'mean_delta_scale', 'var_delta_scale', 'std_delta_scale', ...
		'mean_delta_translation', 'var_delta_translation', 'std_delta_translation', ...
		'mean_rotation_euler', 'var_rotation_euler', 'std_rotation_euler', ...
		'mean_nonrigid_params', 'var_nonrigid_params', 'std_nonrigid_params', 'version', 'KTorresani', 'KNonrigid', 'energy' );

end

