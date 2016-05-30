% run this script in folder PO_CR_code_v1
% this script change 2D landmarks to 3D landmarks using Torresani's algorithm
function [] = Reconstruct_Torresani(KTorresani)
	shapeModelDir = '3D_Shape_Model/'; 
	addpath(shapeModelDir);
	addpath([shapeModelDir '/nrsfm-em/']);

	% load big model of 68 landmarks, dataset from Helen and LFPW
	% load('wild_68_pts');

	% load small model of 5 landmarks. collect annotations of Helen and LFPW
	gtParamDir = 'TR_params/';
	load([gtParamDir 'TR_gt_landmarks.mat']);
	all_pts = zeros( 2 * 2811, 68); 
	for gg = 1:2811
		all_pts(gg, :) = TR_gt_landmarks{gg}(:, 1); 
		all_pts(gg + 2811, :) = TR_gt_landmarks{gg}(:, 2); 
	end


	%% Perform NRSFM by Torresani 

	% (T is the number of frames, J is the number of points)

	J = size(all_pts,2);
	T = size(all_pts,1)/2;

	use_lds = 0;				% not modeling a linear dynamic system here
	max_em_iter = 100;
	tol = 0.001;
	MD = zeros(T,J);

	if nargin == 0				    % number of deformation shapes
		KTorresani = 15;		  % default value
	end
	
	[P3, S_hat, V, RO, Tr, Z] = em_sfm(all_pts, MD, KTorresani, use_lds, tol, max_em_iter);

	% save('Torr_wild', 'P3', 'S_hat', 'V', 'RO', 'Tr', 'Z');
	save([shapeModelDir 'Helen_LFPW_5lms_KTorresani-' num2str(KTorresani) ], 'P3', 'S_hat', 'V', 'RO', 'Tr', 'Z');
end