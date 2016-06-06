function [] = face_det_stat_3D(KTorresani, energy, KNonrigid)
% run this function in folder PO_CR_code_v1
% function that comptues statistics of face detection initialization
% chosed shape model of KTorresani = 25, energy = 0.9, KNonrigid = 13, K = 19

%% initialization
	saveDir = 'matfiles/'; 		
	gtParamDir = 'TR_params/';
	load([gtParamDir 'TR_training.mat']); 
	K = KNonrigid + 6; 
	n = size(TR_gt_landmarks, 1);
	
%% compute statistics of 3D shape parameters from all images of Helen and LFPW training dataset
	TR_3D_p = cell2mat(TR_myShape_3D_p);
	
	% nonrigid parameters
	TR_3D_nonrigid_params_mat = TR_3D_p(:, 7:K); 
	mean_nonrigid_params = mean(TR_3D_nonrigid_params_mat, 1);
	std_nonrigid_params = std(TR_3D_nonrigid_params_mat, 1);
		
	% rotation euler
	TR_3D_rotation_euler_mat = TR_3D_p(:, 4:6);
	mean_rotation_euler = mean(TR_3D_rotation_euler_mat, 1);
	std_rotation_euler = std(TR_3D_rotation_euler_mat,1);
	
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
	std_delta_translation = std(delta_translation);
	
	mean_delta_scale = mean(delta_scale);
	std_delta_scale = std(delta_scale);
	
	save([saveDir 'fd_stat_LM_3D_KTorresani-' num2str(KTorresani) '_energy-' num2str(energy) '_KNonrigid-' num2str(KNonrigid) '.mat'], 'mean_delta_scale', ...
		'std_delta_scale', 'mean_delta_translation', 'std_delta_translation', 'mean_rotation_euler', 'std_rotation_euler', 'mean_nonrigid_params', ...
		'std_nonrigid_params', 'version', 'KTorresani', 'KNonrigid', 'energy' );

end

