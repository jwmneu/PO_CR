function [myShape] = shape_model()
%% initialization
    load shape_model;
    folder1 = './train_data/helen/trainset/';
    what1 = 'jpg';
    folder2 = './train_data/lfpw/trainset/';
    what2 = 'png';
    names1 = dir([folder1 '*.' what1]);
    names2 = dir([folder1 '*.pts']);
    names3 = dir([folder2 '*.' what2]);
    names4 = dir([folder2 '*.pts']);
    num_of_pts = 68; % num of landmarks in the annotations
    Z = []; 
    trans = zeros(length(names1), 2);
    scales = zeros(length(names1),1);
    rotate = zeros(length(names1),1); 
    
%% input training images from folder1
    for gg = 1:length(names1)
        %input_image = imread([folder1 names1(gg).name]);                  % input image must be in [0,255]!!
        pts = read_shape([folder1 names2(gg).name], num_of_pts);     % read ground truth landmarks
        gt_landmark = (pts-1);
        gt_landmark = reshape(gt_landmark, 68, 2);

        [~,z,T] = procrustes(shape.s0, gt_landmark);            % mean landmark & groundtruth landmark. z is the transformed result. T captures translation, rotation and scale. 
        z(:, 2) = -1 * z(:, 2); 
        Z = [Z; reshape(z, 1, [])]; 
        trans(gg, :) = T.c(1,:);
        scales(gg, 1) = T.b;
        rotate(gg, 1) = sum(reshape((T.T * gt_landmark'), 1, []) ./ shape.Q(:,2)) / size(shape.Q(:,2), 1) ; 
%         figure;imagesc(input_image); colormap(gray); hold on; plot(gt_landmark(:,1), gt_landmark(:,2), 'o');     
%         plot(gt_landmark_center(:,1), gt_landmark_center(:,2), '*');     
    end

%% input training images from folder2
    for gg = 1:length(names3)
        %input_image = imread([folder2 names3(gg).name]);                  % input image must be in [0,255]!!
        pts = read_shape([folder2 names4(gg).name], num_of_pts);     % read ground truth landmarks
        gt_landmark = (pts-1);
        gt_landmark = reshape(gt_landmark, 68, 2);

        [~,z,~] = procrustes(shape.s0, gt_landmark);            % mean landmark & groundtruth landmark. procrustes:compute linear tranformation between two matrices. 
        z(:, 2) = -1 * z(:, 2); 
        Z = [Z; reshape(z, 1, [])]; 
        trans(gg+length(names1), :) = T.c(1,:);
        scales(gg+length(names1), 1) = T.b;
        rotate(gg+length(names1), 1) = sum(reshape((T.T * gt_landmark'), 1, []) ./ shape.Q(:,2)) / size(shape.Q(:,2), 1) ; 
    end
    
%% build rigid shape model
    rigid = [trans(:,1)'; trans(:,2)'; scales'; rotate']';
    rigidQ = [[ones(num_of_pts,1), zeros(num_of_pts, 1)]; [zeros(num_of_pts, 1) , ones(num_of_pts,1)]; ones(2 * num_of_pts, 1); shape.Q(:,2 )]';
    
%% build non-rigid shape model
    gt_lm = Z; 
    s0 = (sum(gt_lm, 1) / size(gt_lm, 1))';                             % 136 * 1
    [Q, p, EiVal] = pca(gt_lm);
    
    % choose top 95% of eigenvectors
    var_total = sum(EiVal);
    summ = 0;  
    for i = 1:size(EiVal)
        summ = summ + EiVal(i);
        if summ / var_total > 0.95
            PCA_dim = i; 
            break;
        end
    end

    Q = Q(:, 1:PCA_dim);
    p = p(:, 1:PCA_dim);
    EiVal = EiVal(1:PCA_dim, :);
   
%% save shape model to myShape
    Q = [rigidQ; Q']';
    p = [rigid; p']';
    myShape = struct ('s0', s0, 'Q', Q, 'p', p);
    save('myShape.mat', 'myShape');
    
%% plot shape model -- plot mean shape and first three eigenvectors added to the mean shape
    gg = 1;
    
    p1 = zeros(size(Q,2), 1);
    p1(1, 1) = p(gg, 1);
    p2 = zeros(size(Q,2), 1);
    p2(2,1) = p(gg, 2);
    p3 = zeros(size(Q,2), 1);
    p3(2,1) = p(gg, 3);
    
    landmarks = s0 + Q * p(gg, :)';
    landmarks1 = s0 + Q * p1; 
    landmarks2 = s0 + Q * p2; 
    landmarks3 = s0 + Q * p3; 
    landmarks = reshape(landmarks, [], 2);
    landmarks1 = reshape(landmarks1, [], 2);
    landmarks2 = reshape(landmarks2, [], 2);
    landmarks3 = reshape(landmarks3, [], 2);
    
    figure;  hold on; 
    plot(landmarks(:,1), landmarks(:,2), 'o');     
    plot(landmarks1(:,1), landmarks1(:,2), '+');    
    plot(landmarks2(:,1), landmarks2(:,2), '*');    
    plot(landmarks3(:,1), landmarks3(:,2), '-');   
    
end












