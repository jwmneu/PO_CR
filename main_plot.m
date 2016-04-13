%% visualize iterations
shapemodel = load('shape_model.mat');
myShape = load('myShape.mat'); 
myAppearance = load('myAppearance');
shapemodel = shapemodel.shape;
myShape = myShape.myShape;
myAppearance = myAppearance.myAppearance;
datasetDir = '../dataset/'; 
testsetDir = '../test_data/'; 
outputDir = 'IntermediateResult/'; 
folder1 = [datasetDir 'helen/trainset/'];
what1 = 'jpg';
folder2 = [datasetDir 'lfpw/trainset/'];
what2 = 'png';
names1 = dir([folder1 '*.' what1]);
names2 = dir([folder1 '*.pts']);
names3 = dir([folder2 '*.' what2]);
names4 = dir([folder2 '*.pts']);
num_of_pts = 68; 
shapemodelS0 = shapemodel.s0;
			
Kpi = 10; 
T = 1;
ridge_param = 0;
learning_rate = 0.5;
SIFT_scale = 15; 

for learning_rate = 0.1:0.1:1
	pp = load([outputDir 'ppp/ppp_initial_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat']);
	pp = pp.pp;
	ppp = load([outputDir 'ppp/ppp_i-' num2str(T) '_S-' num2str(SIFT_scale) '_P-' num2str(Kpi) '_R-' num2str(ridge_param) '_L-' num2str(learning_rate) '.mat']);
	ppp = ppp.ppp;
	figure; hold on;
	for gg = 1:16
		lm0 = myShape.s0 + myShape.Q(:, 2:end) * reshape(pp(1, gg, 2:end), 1, [])';
		lm1 = myShape.s0 + myShape.Q(:, 2:end) * reshape(ppp(1, gg, 2:end), 1, [])';
		lm0 = reshape(lm0, [],2);
		lm1 = reshape(lm1, [],2);
		pts = read_shape([folder1 names2(gg).name], num_of_pts);                         
		gt_landmark = (pts-1);
		gt_landmark = reshape(gt_landmark, 68, 2);
		input_image = imread([folder1 names1(gg).name]); 
		[~,~,Tt] = procrustes(shapemodelS0, gt_landmark);        
		scl = 1/Tt.b;
		gt_landmark = gt_landmark*(1/scl); 
		input_image = imresize(input_image, (1/scl));
		subplot(5,6,gg);
		imagesc(input_image); colormap(gray); hold on;
		plot(lm0(:,1), lm0(:,2), 'Color', 'green');
		plot(lm1(:,1), lm1(:,2), 'Color', 'blue');
	end
end