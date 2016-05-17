function [lm] = Computelm(p_nonrigid, p_rigid, gg, k, plotgg, input_image, gt_landmark)	% plotgg is a vector storing image indexes for plotting
	disp('Computelm should not be called after the new shape model');
	% p_rigid : ( 1, [] ) p_nonrigid : ( 1, [] ) 
	modelDir = 'matfiles/';
	shapemodel = load([modelDir 'shape_model.mat']);
	myShape = load([modelDir 'myShape.mat']); 
	shapemodel = shapemodel.shape;
	myShape = myShape.myShape;
	num_of_pts = size(myShape.s0, 1) / 2;
	
	if any(plotgg==gg) == 1 && k == 1
		figure; 
		imshow(input_image);hold on;
		plot(gt_landmark(:,1), gt_landmark(:,2));
	end
	
	% nonrigid parameters
	lm = myShape.s0 + myShape.QNonrigid * p_nonrigid'; 
	lm = reshape(lm , [], 2);
	if any(plotgg==gg) == 1
		plot(lm(:,1), lm(:,2),  'Color', 'red');
	end

	% scale
	lm = lm * p_rigid(1,1);
	if any(plotgg==gg) == 1
		plot(lm(:,1), lm(:,2),  'Color', 'blue');
	end

	% rotation 
	angle = p_rigid(1,4);
	Rot = [ cos(angle), -sin(angle); sin(angle), cos(angle)];
	lm = (Rot * lm')'; 
	
	% x,y transformation
	lm(:, 1)  = lm(:, 1) + p_rigid(1,2) * ones(num_of_pts,1); 
	lm(:, 2) = lm(:, 2) + p_rigid(1,3) * ones(num_of_pts, 1); 
	if any(plotgg==gg) == 1
		plot(lm(:,1), lm(:,2),  'Color', 'green');
	end
end