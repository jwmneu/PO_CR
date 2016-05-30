function [myAppearance, Feat, Feat_rec] = SM_appe_model()             % features is already normalized
	addpath('functions/');
	modelDir = 'matfiles/';
	shape = load([modelDir 'shape_model.mat']);
	shape = shape.shape;
	datasetDir = '../dataset/'; 
	testsetDir = '../test_data/'; 
	CLMDir = './';
	folder1 = [datasetDir 'helen/trainset/'];
	what1 = 'jpg';
	folder2 = [datasetDir 'lfpw/trainset/'];
	what2 = 'png';
	names1 = dir([folder1 '*.' what1]);
	names2 = dir([folder1 '*.pts']);
	names3 = dir([folder2 '*.' what2]);
	names4 = dir([folder2 '*.pts']);
	num_of_pts = 68; % num of landmarks in the annotations
	lm_pts = 5;
	lm_ind2 = [34, 37, 46, 61, 65]; 
	lm_ind = [34, 37, 46, 61, 65, 102, 105 ,114, 129, 133]; 
	Feat = []; 
	
 	[TR_images, TR_face_sizes, TR_gt_landmarks, TR_myShape_pRigid, TR_myShape_pNonRigid, ~] = Collect_training_images(n1, n2) ; 
% 	load('CollectedTrainingDataset/TR_face_sizes.mat'); 
% 	load('CollectedTrainingDataset/TR_gt_landmarks.mat'); 
% 	load('CollectedTrainingDataset/TR_myShape_pRigid.mat'); 
% 	load('CollectedTrainingDataset/TR_myShape_pNonRigid.mat'); 
	
%% extract features
	Feat = zeros(n, 128 * 5); 
	for gg = 1 : n 
		gt_landmark = TR_gt_landmarks{gg}; 
		input_image = TR_images{gg};
		face_size = TR_face_sizes{gg};
		if size(input_image, 3) == 3
			I = single(rgb2gray(input_image)); 
		else
			I = single(input_image);
		end
		SM_gt_landmark = gt_landmark(lm_ind2, :);
		fc = [ SM_gt_landmark'; ones(1, lm_pts) *face_size/100; ones(1, lm_pts) * (-pi/8)];   % scale of SIFT is determined by face_size. rotation is unknown. 
		[f,d] = vl_sift(I,'frames',fc) ;                        % d is the extracted features. f(1) f(2) are x, y axis.
		Feat(gg, :) = reshape(d, 1, []); 
		
% 		% plot first 5 images and SIFT features. 
% 		if gg < 5  
% 			figure;imagesc(input_image); colormap(gray); hold on; plot(gt_landmark(:,1), gt_landmark(:,2), 'o');     
% 			h = vl_plotsiftdescriptor(d(:, 1:5), f(:, 1:5)) ;
% 			set(h,'color','g', 'linewidth', 0.5) 
% 		end
	end
	save('Feat.mat', 'Feat');
	
%% PCA 
	Feat = normalize(double(Feat));
	A0 = ( sum(Feat, 1) / size(Feat, 1))';         % mean of feature vector
	[A, C, EiVal] = pca(Feat);          % A is eigenvectors, C is parameter vectors for each image, Var is eigenvalues for each eigenvector.

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
	Feat_rec =  (repmat(A0, 1, size(Feat,1)) + A * C')'; 
	save ('Feat_rec.mat', 'Feat_rec');
	
	myAppearance = struct('A0', A0, 'A', A, 'C', C, 'EiVal', EiVal, 'version', 1);
	save('myAppearance.mat', 'myAppearance');
%% plot
%     figure;imagesc(input_image); colormap(gray); hold on; plot(init_shape(:,1), init_shape(:,2), 'o');     
%     feature_gg_re = reshape(features_reconstruct(gg, :), 68, []); 
%     face_size = (max(init_shape(:,1)) - min(init_shape(:,1)) + max(init_shape(:,2)) - min(init_shape(:,2)))/2;
%     fc = [ init_shape'; ones(1, num_of_pts) *face_size/100 ; ones(1, num_of_pts) * (-pi/8)];   % scale of SIFT is determined by face_size. rotation is unknown. 
%     h = vl_plotsiftdescriptor(feature_gg_re(1:30, 1:128)', fc(:, 1:30)) ;
%     set(h,'color','b', 'linewidth', 0.5) ;
    
end



