function [] = LM_Appearence_Model()             % features is already normalized
% this function is used to build appearence model for 68 landmarks. 	
	saveDir = 'matfiles/'; 
	plotDir = 'SIFT_Plots/';
	if(exist(plotDir, 'dir') == 0)
		mkdir(plotDir);
	end
	cd('vlfeat-0.9.20/toolbox');
 	vl_setup
	cd('../../');	
	gtParamDir = 'Inputs/';
	TR_train_params = load([gtParamDir 'TR_Training_Params.mat']); 
	TR_images_train = load([gtParamDir 'TR_images_train.mat']);
	
	% get only the training dataset
	TR_images = TR_images_train.TR_images_train; 
	TR_detections = TR_train_params.TR_detections_train;
	TR_face_size = TR_train_params.TR_face_size_train;
	TR_gt_landmarks = TR_train_params.TR_gt_landmarks_train; 
	n = size(TR_detections, 1);
	lm_pts = 68; 
	num_feat_per_lm = 128; 
	
%% extract features

	features = zeros(n, num_feat_per_lm * lm_pts); 
	parfor gg = 1 : n
		gg
		gt_landmark = TR_gt_landmarks{gg}; 
		input_image = TR_images{gg};
		face_size = TR_face_size{gg};
		if size(input_image, 3) == 3
			I = single(rgb2gray(input_image)); 
		else
			I = single(input_image);
		end
		fc = [ gt_landmark'; ones(1, lm_pts) *face_size/100; ones(1, lm_pts) * (-pi/8)];   % scale of SIFT is determined by face_size. rotation is unknown. 
		[f,d] = vl_sift(I,'frames',fc) ;                        % d is the extracted features. f(1) f(2) are x, y axis.
		features(gg, :) = reshape(d, 1, []); 
		
		% plot first 5 images and SIFT features. 
		if mod(gg, 100) == 0
			figure;imagesc(input_image); colormap(gray); hold on; plot(gt_landmark(:,1), gt_landmark(:,2), 'o');     
			h = vl_plotsiftdescriptor(d(:, [35 27 13 9 40 45 50 55 60]), f(:,  [35 27 13 9 40 45 50 55 60])) ;
			set(h,'color','g', 'linewidth', 0.5); 
			savefig([plotDir 'sift_' gg]);
		end
	end
	
%% PCA 
	features = normalize(double(features));
	A0 = ( sum(features, 1) / size(features, 1))';         % mean of feature vector
	[A, C, EiVal] = pca(features);          % A is eigenvectors, C is parameter vectors for each image, Var is eigenvalues for each eigenvector.
	
	% compute PCA_dim as the top 95% variance of eigenvectors
	var_total = sum(EiVal);
	summ = 0;  
	for i = 1:size(EiVal)
		summ = summ + EiVal(i);
		if summ / var_total > 0.95
			PCA_dim = i; 
			break;
		end
	end

	A = A(:, 1:PCA_dim);
	C = C(:, 1:PCA_dim);
	EiVal = EiVal(1:PCA_dim, :);

	% reconstruct features by PCA eigenvectors 
	features_reconstruct =  (repmat(A0, 1, size(features,1)) + A * C')'; 
	save([saveDir 'myAppearenceLM_features.mat'], 'features', 'features_reconstruct');
	
	version = 'LM_1';
	save([saveDir 'myAppearanceLM.mat', 'A0', 'A', 'C', 'EiVal', 'version');
	
%% plot
%     figure;imagesc(input_image); colormap(gray); hold on; plot(init_shape(:,1), init_shape(:,2), 'o');     
%     feature_gg_re = reshape(features_reconstruct(gg, :), 68, []); 
%     face_size = (max(init_shape(:,1)) - min(init_shape(:,1)) + max(init_shape(:,2)) - min(init_shape(:,2)))/2;
%     fc = [ init_shape'; ones(1, num_of_pts) *face_size/100 ; ones(1, num_of_pts) * (-pi/8)];   % scale of SIFT is determined by face_size. rotation is unknown. 
%     h = vl_plotsiftdescriptor(feature_gg_re(1:30, 1:128)', fc(:, 1:30)) ;
%     set(h,'color','b', 'linewidth', 0.5) ;
    
end



