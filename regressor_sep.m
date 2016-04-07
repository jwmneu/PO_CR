%% initialization
clear
cd 'vlfeat-0.9.20/toolbox'
vl_setup
cd '../../'
load shape_model;
load myShape; 
load myAppearance;
load fd_stat;
datasetDir = '../dataset/'; 
testsetDir = '../test_data/'; 
outputDir = 'IntermediateResult/'; 
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
N = size(myAppearance.A, 1);				% number of SIFT features
m = size(myAppearance.A, 2);                            % number of eigenvectors of myAppearance.A
K = size(myShape.p, 2);                                      % number of eigenvectors of myShape.Q

small_size = 0; 
full_size = 1;
if small_size == 1
	Kpi = 2;
	T = 2;                                                                 % number of iterations
	n = 3;                                                               % for testing correctness by small dataset
elseif full_size == 0
	Kpi = 10;
	T = 10;
	n = 30;
else
	Kpi = 10;
	T = 10;
	n = length(names1);
end

%% cascaded regression for only Helen dataset
p_mat = zeros(n, Kpi, K);
delta_p_mat = zeros(n, Kpi, K);
feat = zeros(n, Kpi, N);
b_mat = zeros(n, Kpi, N);
% scale = 100; 
% non_rigid_std = fd_stat.std(1, 1:4) * scale;
% non_rigid_std(1, 3) = non_rigid_std(1, 3) * scale;                                          % ???
pt_pt_err = zeros(n, 1);				       % stores pt-pt error for each image
SIFT_scale = 15;
var = 0:0.02:0.9;

for t = 1 : T
	disp(['iteration is ' num2str(t)]);
	if t > 4
		SIFT_scale = 3; 
	end
	myShape_p = myShape.p;
	myShape_p(:, 4) = 0;				       % set rotation as 0
        fd_stat_std = [fd_stat.std(1:4), zeros(1, K-4)];
%  	fd_stat_std = fd_stat_std / 3;			  % reduce deviation for better training
        myShapeS0 = myShape.s0; 
        myShapeQ = myShape.Q; 
        A0P = myAppearance.A0' * P;  
	
	%% parallel task - initialize perturbed shape parameters of image(gg), compute feature matrix
	disp( 'initializing shape parameter and extracting features');
	parfor gg = 1 : n
		p_mat_gg = p_mat(gg, :, :); 
		pts = read_shape([folder1 names2(gg).name], num_of_pts);                         % read ground truth landmarks
		gt_landmark = (pts-1);
		gt_landmark = reshape(gt_landmark, 68, 2);
		input_image = imread([folder1 names1(gg).name]); 
 		%figure;imagesc(input_image); colormap(gray); hold on; 
 		%plot(gt_landmark(:,1), gt_landmark(:,2), 'o');  
 		
		% scale ground truth landmarks, image, shape parameters to mean face size
		[~,~,Tt] = procrustes(shape.s0, gt_landmark);        
		scl = 1/Tt.b;
		gt_landmark = gt_landmark*(1/scl); 
		input_image = imresize(input_image, (1/scl));
 		%myShape_p(gg, 1:4) = myShape_p(gg, 1:4) * (1/scl);
		%figure;imagesc(input_image); colormap(gray); hold on; 
		%plot(gt_landmark(:,1), gt_landmark(:,2), 'o');  

% 		p_mat(gg, :, :) = repmat(myShape_p(gg, :), K, 1)+ ( fd_stat_std .* rand(1, K)  - fd_stat_std / 2 );
		for k = 1 : Kpi
			p_mat_gg(1, k, :) = [myShape_p(gg, 1:4) , zeros(1, K-4)] + fd_stat_std .* rand(1, K)  - fd_stat_std / 2; 
			lm = myShapeS0 + myShapeQ(:, 2:end) * reshape(p_mat_gg(1, k, 2:end), 1, [])'; 
			lm = reshape(lm, 68, 2) * (1/scl);
			lm = lm * p_mat_gg(1, k, 1);				% scale
% 			image = imread([folder1 names1(gg).name]); 
% 			figure;imagesc(input_image); colormap(gray); hold on;
% 			plot(lm(:,1), lm(:,2), 'o');   
			Sfeat = SIFT_features(input_image, lm, SIFT_scale);
			feat(gg, k, :) = reshape(Sfeat, 1, []); 
			b_mat(gg, k, :) =  reshape(feat(gg, k, :), 1, []) - A0P;
		end
		p_mat(gg, :, :) = p_mat_gg;
	end                       

	%% centralized task 
	disp( 'duplicating matrices and doing ridge regresstion');
	V = Kpi * ones(n);
        f = @(k) repmat(myShape.p(k,:), round(V(k)), 1);
        p_star_mat = cell2mat(arrayfun(f, (1:length(V))', 'UniformOutput', false));
        start = 1; 
        for g = 1 : n
            p_star_mat_t(g, :, :) = p_star_mat(start : start + Kpi - 1, :);
            start = start + Kpi;
        end
        delta_p_mat = p_star_mat_t - p_mat; 
	
	% ridge regression to compute Jp seperately
	Jp = zeros(N, K);
	for reg = 1 : N                                                                                             % compute beta_i
		Jp(reg, :) = ridge(reshape( b_mat(:, :, reg), [], 1), reshape(delta_p_mat, size(delta_p_mat,1) * size(delta_p_mat,2), size(delta_p_mat, 3)), 0.05);
	end

	%% parallel task - update shape parameter p and compute pt-pt error
	disp('updating shape parameters and computing pt-pt error')
	Hessian = Jp' * Jp; 
	Risk = Hessian \ Jp'; 
	parfor gg = 1 : n
		% update p_mat
		p_mat_gg = p_mat(gg, :, :);
		for k = 1 : Kpi
			p_mat_gg(1, k, :) = (reshape(p_mat_gg(1, k, :), 1, K )' + Risk * reshape((b_mat(gg, k, :)), 1, N)')';
		end
		ppp(t, gg, :) = (p_mat_gg(1, 1, :));
		p_mat(gg, :, :) = p_mat_gg; 
		
		% compute pt-pt error
		pts = read_shape([folder1 names2(gg).name], num_of_pts);		% read ground truth landmarks
		gt_landmark = (pts-1);
		gt_landmark = reshape(gt_landmark, 68,2);
		face_size = (max(gt_landmark(:,1)) - min(gt_landmark(:,1)) + max(gt_landmark(:,2)) - min(gt_landmark(:,2)))/2;
		pt_pt_err1 = zeros(1, Kpi);
		for k = 1 : Kpi
			fitted_shape = myShape.s0 + myShape.Q * reshape(p_mat_gg(1, k, :), [], 1);
			pt_pt_err1(1, k) = mean(abs(fitted_shape - reshape(gt_landmark, [], 1))) / face_size;
		end
		pt_pt_err(t, gg) = sum(pt_pt_err1) / Kpi;
	end
	
	%% cumulative curve
	cum_err = zeros(size(var));
	for ii = 1:length(cum_err)
		cum_err(ii) = length(find(pt_pt_err(t, :)<var(ii)))/length(pt_pt_err(t, :));
	end
	cum_err_full(t, :) = cum_err;
	
	%% save intermediate results per iteration
	disp( 'saving results to output directory for this iteratoin');
	save([outputDir 'b_mat/b_mat_' num2str(t) '.mat'], 'b_mat');
	save([outputDir 'Risks/Risk_' num2str(t) '.mat'], 'Risk');
	save([outputDir 'JPs/seperate/Jp_' num2str(t) '.mat'], 'Jp');
	save([outputDir 'ppp/ppp_' num2str(t) '.mat'], 'ppp');
	save([outputDir 'cum_err/cum_err_' num2str(t) '.mat'], 'cum_err');
	
end 

%% save result
disp( 'finish all iterations. saving results')
save([outputDir 'Jp.mat'], 'Jp');
save([outputDir 'cum_err_full.mat'], 'cum_err_full');
a = [1 2 3];
save('test_matlab_working_v1.mat', 'a');

%% plot cumulative error curve
% figure; hold on;
% 
% color = [ 0, 0, 0; 1, 0, 0; 1, 1, 0; 0, 1, 0; 0, 0, 1];
% for t = 1 : T
% 	plot(var, cum_err_full, 'Color', color(t,:), 'linewidth', 2); grid on;
% end
% xtick = 5*var;
% ytick = 0:0.05:1;
% set(gca, 'xtick', xtick);
% set(gca, 'ytick', ytick);
% ylabel('Percentage of Images', 'Interpreter','tex', 'fontsize', 15)
% xlabel('Pt-Pt error normalized by face size', 'Interpreter','tex', 'fontsize', 13)
% legend(['iteration' num2str(t)]);

%% visualize iterations
% for gg = 1 : 30
% 	lm1 = myShape.s0 + myShape.Q(:, 2:end) * reshape(ppp(1, gg, 2:end), 1, [])';
% 	lm2 = myShape.s0 + myShape.Q(:, 2:end) * reshape(ppp(2, gg, 2:end), 1, [])';
% 	lm3 = myShape.s0 + myShape.Q(:, 2:end) * reshape(ppp(3, gg,  2:end), 1, [])';
% 	% lm4 = myShape.s0 + myShape.Q(:, 2:end) * reshape(ppp(4, gg,  2:end), 1, [])';
% 	% lm5 = myShape.s0 + myShape.Q(:, 2:end) * reshape(ppp(5, gg,  2:end), 1, [])';
% 	lm1 = reshape(lm1, [],2);
% 	lm2 = reshape(lm2, [],2);
% 	lm3 = reshape(lm3, [],2);
% 	lm3 = reshape(lm3, [],2);
% 	% lm4 = reshape(lm4, [],2);
% 	% lm5 = reshape(lm5, [],2);
% 	pts = read_shape([folder1 names2(gg).name], num_of_pts);                         
% 	gt_landmark = (pts-1);
% 	gt_landmark = reshape(gt_landmark, 68, 2);
% 	input_image = imread([folder1 names1(gg).name]); 
% 	[~,~,Tt] = procrustes(shape.s0, gt_landmark);        
% 	scl = 1/Tt.b;
% 	gt_landmark = gt_landmark*(1/scl); 
% 	input_image = imresize(input_image, (1/scl));
% 	figure;imagesc(input_image); colormap(gray); hold on;
% 	plot(lm1(:,1), lm1(:,2), 'Color', 'black');
% 	plot(lm2(:,1), lm2(:,2), 'Color', 'red');
% 	plot(lm3(:,1), lm3(:,2), 'Color', 'yellow');
% 	% plot(lm4(:,1), lm4(:,2), 'Color', 'green');
% 	% plot(lm5(:,1), lm5(:,2), 'Color', 'blue');
% end




