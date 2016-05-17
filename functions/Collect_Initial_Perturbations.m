function [p_mat_rigid_initialization_n, p_mat_nonrigid_initialization_n, Perturbed_SIFT_Feature_labels_n] = Collect_Initial_Perturbations(n1, n2, SIFT_scale, Kpi)
	% run this function in foler PO_CP_code_v1
	% the original mat files are on server, so run this function on server
	FileDir = '../PerturbationInitialization/'; 
	saveDir = '../PerturbationInitialization_Smalldataset/';
	if (exist(saveDir, 'dir') == 0)
		mkdir(saveDir);
	end
	SAVE = 1; 
	VERBOSE = 1; 
	
	Perturbed_SIFT_Feature_labels = load([FileDir 'Perturbed_SIFT_Feature_labels_S-'  num2str(SIFT_scale) '_Kpi-'  num2str(Kpi) '.mat']);
	p_mat_rigid_initialization = load([FileDir 'p_mat_rigid_initialization_Kpi-' num2str(Kpi) '.mat']);
	p_mat_nonrigid_initialization = load([FileDir 'p_mat_nonrigid_initialization_Kpi-' num2str(Kpi) '.mat']);
	p_mat_nonrigid_gtperturbed = load([FileDir  'p_mat_nonrigid_gtperturbed_Kpi-' num2str(Kpi) '.mat']);
	p_mat_rigid_gtperturbed = load([FileDir 'p_mat_rigid_gtperturbed_Kpi-' num2str(Kpi) '.mat']);
	
	Perturbed_SIFT_Feature_labels = Perturbed_SIFT_Feature_labels.b_mat;
	p_mat_rigid_initialization = p_mat_rigid_initialization.p_mat_rigid;
	p_mat_nonrigid_initialization = p_mat_nonrigid_initialization.p_mat_nonrigid;
	p_mat_nonrigid_gtperturbed = p_mat_nonrigid_gtperturbed.p_mat_nonrigid_gtperturbed;
	p_mat_rigid_gtperturbed = p_mat_rigid_gtperturbed.p_mat_rigid_gtperturbed;
	
	Perturbed_SIFT_Feature_labels_n = Perturbed_SIFT_Feature_labels([1:n1, 2001:2000+n2], :, :);
	p_mat_rigid_initialization_n = p_mat_rigid_initialization([1:n1, 2001:2000+n2], :, :);
	p_mat_nonrigid_initialization_n = p_mat_nonrigid_initialization([1:n1, 2001:2000+n2], :, :);
	p_mat_nonrigid_gtperturbed_n = p_mat_nonrigid_gtperturbed([1:n1, 2001:2000+n2], :, :);
	p_mat_rigid_gtperturbed_n = p_mat_rigid_gtperturbed([1:n1, 2001:2000+n2], :, :);
	
	if VERBOSE == 1
		size(Perturbed_SIFT_Feature_labels)
		size(p_mat_rigid_initialization)
		size(p_mat_nonrigid_initialization)
		n1
		n2
		[1:n1, 2001:2000+n2]
		disp('size of Perturbed_SIFT_Feature_labels_n is ');
		size(Perturbed_SIFT_Feature_labels_n)
		disp('size of p_mat_rigid_initialization_n is ');
		size(p_mat_rigid_initialization_n)
		disp('size of p_mat_nonrigid_initialization_n is ');
		size(p_mat_nonrigid_initialization_n)
		disp('size of p_mat_nonrigid_gtperturbed_n is ');
		size(p_mat_nonrigid_gtperturbed_n)
		disp('size of p_mat_rigid_gtperturbed_n is ');
		size(p_mat_rigid_gtperturbed_n)
	end
	
	if SAVE == 1
		save([saveDir 'Perturbed_SIFT_Feature_labels_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat'], 'Perturbed_SIFT_Feature_labels_n');
		save([saveDir 'p_mat_rigid_initialization_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat'], 'p_mat_rigid_initialization_n');
		save([saveDir 'p_mat_nonrigid_initialization_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat'], 'p_mat_nonrigid_initialization_n');
		save([saveDir 'p_mat_nonrigid_gtperturbed_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat'], 'p_mat_nonrigid_gtperturbed_n');
		save([saveDir 'p_mat_rigid_gtperturbed_' num2str(n1) '-' num2str(n2) '_Kpi-' num2str(Kpi) '.mat'], 'p_mat_rigid_gtperturbed_n');
	end
end