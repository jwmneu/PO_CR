function [init_shape] = initialize_shape(folder, what, gg, shape, std_dr, bounding_boxes)
    

    %% adjust noise level
    std_noise = 5;
    std_init = std_noise*std_dr;

    %% Select image
    %gg = 3;
    names1 = dir([folder '*.' what]);
    names2 = dir([folder '*.pts']);
    input_image = imread([folder names1(gg).name]); % input image must be in [0,255]!!
    num_of_pts = 68; % num of landmarks in the annotations
    pts = read_shape([folder names2(gg).name], num_of_pts);

    %% ground_truth
    gt_s = (pts-1);
    gt_s = reshape(gt_s, 68, 2);
    %face_size = (max(gt_s(:,1)) - min(gt_s(:,1)) + max(gt_s(:,2)) - min(gt_s(:,2)))/2;

    %% initialization
    init_shape = gt_s;
    [~,~,T] = procrustes(shape.s0, init_shape);           % mean landmark & groundtruth landmark. procrustes:compute linear tranformation between two matrices. 
    scl = 1/T.b;                                                           % T has x-position, y-position, scale and rotation. 
    init_shape = init_shape*(1/scl);
    input_image = imresize(input_image, (1/scl));     % scale image and landmark to fit for the shape model (mean shape and eigenvectors).

   % =================== test: plot shape models =================
   % shape.Q(:, 3) and shape.Q(:,4) are linear translation of x-axis and y-axis. 
   % shape.Q(:, 1) and shape.Q(:,2) are plotted as follows. 
   
    r = shape.Q'*(init_shape(:) - shape.s0(:));
    r0 = r;
    r0(1,1) = 0;
    r0(2,1) = 300;
    r1 =  r; 
    r1(1) = 4 * r(1); 
    r1(2) = 0;
    r2 = r0;
    r2(2) = 300 +  r(2);
    r2(1, 1) = 0;
   
    s0 = shape.Q*r0 + shape.s0(:);  
    s0 = reshape(s0, [],2);
    figure;imagesc(input_image); colormap(gray); hold on; plot(s0(:,1), s0(:,2), 'o');     

    s1 = shape.Q*r1 + shape.s0(:);  
    s1 = reshape(s1, [],2);
    plot(s1(:,1), s1(:,2), '*');     

    s2 = shape.Q*r2 + shape.s0(:);  
    s2 = reshape(s2, [],2);
    plot(s2(:,1), s2(:,2), '+');   
%     ==================================================
    
    r = shape.Q'*(init_shape(:) - shape.s0(:));             % r is groundtruth shape parameter vector (p)
    r(1) = r(1) + std_init(1).*rand(1) - std_init(1)/2;
    r(2) = 0;
    r(3) = r(3) + std_init(3).*rand(1) - std_init(3)/2;
    r(4) = r(4) + std_init(4).*rand(1) - std_init(4)/2;    % add noise to shape parameter vector (p)
    init_shape = shape.Q*r + shape.s0(:);                   % compute noisy landmarks
    init_shape = reshape(init_shape, [], 2);                 % get the noisy initial landmarks. 

%     figure;imagesc(input_image); colormap(gray); hold on; plot(init_shape(:,1), init_shape(:,2), 'o');     


end