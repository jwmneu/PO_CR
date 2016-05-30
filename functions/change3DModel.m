function [] = change3DModel(KTorresani, energy)
% this function should be run in folder PO_CR_code_v1
	if nargin == 0
		KTorresani = 15; 
		energy = 0.8; 
	end
	nHelen = 2000;
	nLFPW = 811; 
% 	Reconstruct_Torresani(KTorresani);
	% upload 3D_Shape_Model folder
% 	[KNonrigid] = Create_pdm_wild(K_Torresani, energy);
	% upload 'TR_3D...' folders
% 	face_det_stat_3D(K_Torresani, energy, KNonrigid);
	% upload matfiles folder
	SM_3D_Extract_Perturbed_SIFT_Features(K_Torresani, energy, KNonrigid);
end