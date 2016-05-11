function [] = SM_Extract_Perturbed_SIFT_Features()
% this function should run in foler PO_CR_code_v1
clear;
Kpi = 10;
lm_pts = 5;
lm_ind2 = [34, 42, 47, 61, 65]; 
lm_ind = [34,  42, 47, 61, 65, 102, 110, 115, 129, 133]; 
num_feat_per_lm = 128;
start_ind = (lm_ind2 - 1) * num_feat_per_lm + 1 ; 
end_ind = lm_ind2 * num_feat_per_lm;
indvector = [];
for i = 1 : lm_pts
	indvector = [indvector, start_ind(1,i) : end_ind(1,i)];
end
FileDir = '../PerturbationInitialization/'; 
OutputDir = '../PerturbationInitialization_5_lms/';
if(exist(OutputDir, 'dir') == 0)
	mkdir(OutputDir);
end
for SIFT_scale = 10 : 5 : 30
	% extract corresponding features from original perturbed features directly
	Perturbed_SIFT_Feature_labels = load([FileDir 'Perturbed_SIFT_Feature_labels_S-'  num2str(SIFT_scale) '_Kpi-'  num2str(Kpi) '.mat']);
	Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels.b_mat;
	Perturbed_SIFT_Feature_labels_5_lms = Perturbed_SIFT_Feature_labels(:, :, indvector); 
	save([OutputDir 'Perturbed_SIFT_Feature_labels_5_lms_S-'  num2str(SIFT_scale) '_Kpi-'  num2str(Kpi) '.mat'], 'b_mat');
end	
		