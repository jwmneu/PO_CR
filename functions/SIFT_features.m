%% extract SIFT features from one image given the image and landmarks
function [d] = SIFT_features(input_image, lm, SIFT_scale, k, face_size)
	load shape_model;
	load myShape;
	load myAppearance;
	load fd_stat;

	num_of_pts = size(lm, 1);    
	lm_s0 = reshape(myShape.s0, [], 2);
	norm_face_size = (max(lm_s0(:,1)) - min(lm_s0(:,1)) + max(lm_s0(:,2)) - min(lm_s0(:,2)))/2;% num of landmarks in the annotations

	if size(input_image, 3) == 3
		I = single(rgb2gray(input_image)); 
	else
		I = single(input_image);
	end
	fc = [ lm'; ones(1, num_of_pts) * SIFT_scale * (face_size / norm_face_size); ones(1, num_of_pts) * (-pi/8)];   % scale of SIFT is determined by face_size. rotation is unknown. 
	[f,d] = vl_sift(I,'frames',fc) ;                                                                            % d is the extracted features. f(1) f(2) are x, y axis.

%% plot features
% 	if k == 1
% 		figure;imagesc(input_image); colormap(gray); hold on; plot(lm(:,1), lm(:,2), 'o');     
% 		h = vl_plotsiftdescriptor(d(:, 1:5), f(:, 1:5)) ;
% 		set(h,'color','g', 'linewidth', 0.5);
% 	end
end