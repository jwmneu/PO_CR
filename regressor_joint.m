%% initialization
clear;
load shape_model;
load myShape; 
load myAppearance;
load fd_stat;
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
num_of_pts = 68;                                               % num of landmarks in the annotations
P = eye(size(myAppearance.A,1)) - myAppearance.A * myAppearance.A'; 
N = size(myAppearance.A, 1);                            % number of SIFT features
n = length(names1);                                           % number of images
m = size(myAppearance.A, 2);                            % number of eigenvectors of myAppearance.A
K = size(myShape.p, 2);                                      % number of eigenvectors of myShape.Q
T = 2;                                                                 % number of iterations
n = 10;                                                                % for testing correctness by small dataset
% slm_ind = [ 1, 3, 6, 9, 12, 15, 17, 19, 21, 24, 26, 29, 31, 37, 40, 43, 46, 49, 55, 67];         % selected 20 landmarks out of 68 landmarks
num_of_lm = 5;
feat_per_lm = 128;
SN = N * (num_of_lm/num_of_pts);						% choose top 10 landmarks. 
SIFT_scale = 2.5;

%% cascaded regression for only Helen dataset
p_mat = zeros(n, K, K);
delta_p_mat = zeros(n, K, K);
feat = zeros(n, K, SN);
b_mat = zeros(n, K, SN);
bb_gt = zeros(length(names1), 4);
scale = 100; 
non_rigid_std = fd_stat.std(1, 1:4) * scale;
non_rigid_std(1, 3) = non_rigid_std(1, 3) * scale;                                          % ???
pt_pt_err = zeros(n, 1);		     % stores pt-pt error for each image
% figure; hold on;						   % for plotting cumulative error curve

for t = 1 : T
	t
	%% select portion of matrix for selected landmarks
	A = myAppearance.A(1:num_of_lm*feat_per_lm,:);
	P = eye(size(A,1)) - A * A'; 
	A0 = myAppearance.A0(1:num_of_lm*feat_per_lm,:);
	A0P = A0' * P;  
	myShapeS0 = myShape.s0([1:num_of_lm, 69:68+num_of_lm], :);
	myShapeQ = myShape.Q([1:num_of_lm, 69:68+num_of_lm], :);
	fd_stat_std = fd_stat.std;
	myShape_p = myShape.p;
	myShape_p(:, 4) = 1;
	
	for gg = 1 : n																% parfor                               
		gg
		pts = read_shape([folder1 names2(gg).name], num_of_pts);                         % read ground truth landmarks
		gt_landmark = (pts-1);
		gt_landmark = reshape(gt_landmark, 68, 2);
		input_image = imread([folder1 names1(gg).name]); 
 		%figure;imagesc(input_image); colormap(gray); hold on; 
 		%plot(gt_landmark(:,1), gt_landmark(:,2), 'o');  
 		
		% scale ground truth landmarks, image, shape parameters to mean face size
		[~,~,T] = procrustes(shape.s0, gt_landmark);        
		scl = 1/T.b;
		gt_landmark = gt_landmark*(1/scl); 
		input_image = imresize(input_image, (1/scl));
 		%myShape_p(gg, 1:4) = myShape_p(gg, 1:4) * (1/scl);
		%figure;imagesc(input_image); colormap(gray); hold on; 
		%plot(gt_landmark(:,1), gt_landmark(:,2), 'o');  

		p_mat(gg, :, :) = repmat(myShape_p(gg, :), K, 1)+ diag( fd_stat_std .* rand(1, K)  - fd_stat_std / 2 );
		for k = 1 : K
			lm = myShapeS0 + myShapeQ(:, 2:end) * reshape(p_mat(gg, k, 2:end), 1, [])'; 
			lm = reshape(lm, num_of_lm, 2) * (1/scl);
			lm = lm * p_mat(gg, k, 1);				% scale
% 			image = imread([folder1 names1(gg).name]); 
% 			figure;imagesc(input_image); colormap(gray); hold on;
% 			plot(lm(:,1), lm(:,2), 'o');   
			feat(gg, k, :) = reshape(SIFT_features_few(input_image, lm, SIFT_scale, num_of_lm), 1, []); 
			b_mat(gg, k, :) = reshape(feat(gg, k, :), 1, []) - A0P;
		end
	end

	%% duplicate matrices 
	V = K * ones(n);
	f = @(k) repmat(myShape.p(k,:), round(V(k)), 1);
	p_star_mat = cell2mat(arrayfun(f, (1:length(V))', 'UniformOutput', false));
	start = 1; 
	for g = 1 : n
		p_star_mat_t(g, :, :) = p_star_mat(start : start + K - 1, :);
	start = start + K;
	end
	delta_p_mat = p_mat - p_star_mat_t; 

	%% ridge regression to compute Jp jointly
	b_temp = sum(b_mat, 3);
	b_all = reshape(b_temp', [], 1);
	delta_p_all = repmat( reshape(delta_p_mat, size(delta_p_mat,1) * size(delta_p_mat,2), size(delta_p_mat, 3)), 1, SN);
	Jp = ridge(b_all, delta_p_all, 0);
	Jp = reshape(Jp, SN, K);

	%% update shape parameter p
	Hessian = Jp' * Jp; 
	Risk = Hessian \ Jp'; 
	for gg = 1 : n
		for k = 1 : K
			p_mat(gg, k, :) = (reshape(p_mat(gg, k, :), 1, K )' + Risk * reshape((b_mat(gg, k, :)), 1, SN)')';
		end
	end
	
	%% compute pt-pt error
	for gg = 1 : n
		pts = read_shape([folder1 names2(gg).name], num_of_pts);					% read ground truth landmarks
		gt_landmark = (pts-1);
		gt_landmark = reshape(gt_landmark, 68,2);
		face_size = (max(gt_landmark(:,1)) - min(gt_landmark(:,1)) + max(gt_landmark(:,2)) - min(gt_landmark(:,2)))/2;
		gt_landmark = gt_landmark(1:num_of_lm, :);
		for k = 1 : K
			fitted_shape = myShapeQ * reshape(p_mat(gg, k, :), [], 1);
			pt_pt_err1(k) = mean(abs(fitted_shape - reshape(gt_landmark, [], 1))) / face_size;
		end
		pt_pt_err(gg) = sum(pt_pt_err1) / K;
	end
	save(['JPs/joint/Jp_' num2str(t) '.mat'], 'Jp');
		
	%% plot cumulative curve
	var = 0:0.002:0.1;
	cum_err = zeros(size(var));
	for ii = 1:length(cum_err)
		cum_err(ii) = length(find(pt_pt_err<var(ii)))/length(pt_pt_err);
	end
	cum_err_full(t, :) = cum_err;
	
	plot(var, cum_err, 'blue', 'linewidth', 2); grid on
	xtick = 5*var;
	ytick = 0:0.1:1;
	set(gca, 'xtick', xtick);
	set(gca, 'ytick', ytick);
	ylabel('Percentage of Images', 'Interpreter','tex', 'fontsize', 15)
	xlabel('Pt-Pt error normalized by face size', 'Interpreter','tex', 'fontsize', 13)
	legend(['iteration' num2str(t)]);

end

%% save result
save('Jp.mat', 'Jp');
save('cum_err_full.mat', 'cum_err_full');



