%% extract SIFT features from one image given the image and landmarks
function [d] = SIFT_features(image_path, h_scale, w_scale,  landmarks)
    load shape_model;
    load myShape;
    load myAppearance;
    load fd_stat;
    num_of_pts = 68; % num of landmarks in the annotations
   
    input_image = imread(image_path);                                                           % input image must be in [0,255]!!
    input_image = resize(size(input_image, 1) / h_scale, size(input_image, 2) / w_scale, size(input_image, 3)); 
    num_of_pts = 68;                                                                                          % num of landmarks in the annotations
    face_size = (max(landmarks(:,1)) - min(landmarks(:,1)) + max(landmarks(:,2)) - min(landmarks(:,2)))/2;
     
    if size(input_image, 3) == 3
        I = single(rgb2gray(input_image)); 
    else
        I = single(input_image);
    end
    fc = [ landmarks'; ones(1, num_of_pts) *face_size/100; ones(1, num_of_pts) * (-pi/8)];   % scale of SIFT is determined by face_size. rotation is unknown. 
    [f,d] = vl_sift(I,'frames',fc) ;                                                                            % d is the extracted features. f(1) f(2) are x, y axis.
   
%% plot features
    %figure;imagesc(input_image); colormap(gray); hold on; plot(landmarks(:,1), landmarks(:,2), 'o');     
    %h = vl_plotsiftdescriptor(d(:, 1:30), f(:, 1:30)) ;
    %set(h,'color','g', 'linewidth', 0.5);
end