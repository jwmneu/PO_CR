function [myAppearance, Feat, Feat_rec] = appe_model()             % features is already normalized
    addpath('functions/');
    load shape_model;
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
    Feat = []; 
     
% %% extract SIFT features from folder1
    for gg = 1:length(names1)
        gg
        % get ground truth landmarks and resize according to shape.s0
        pts = read_shape([folder1 names2(gg).name], num_of_pts);     % read ground truth landmarks
        gt_landmark = (pts-1);
        gt_landmark = reshape(gt_landmark, 68, 2);
        % [~,z,T] = procrustes(shape.s0, gt_landmark);    
        
        % get image and and resize according to shape.s0
        input_image = imread([folder1 names1(gg).name]); % input image must be in [0,255]!!
        face_size = (max(gt_landmark(:,1)) - min(gt_landmark(:,1)) + max(gt_landmark(:,2)) - min(gt_landmark(:,2)))/2;
        if size(input_image, 3) == 3
            I = single(rgb2gray(input_image)); 
        else
            I = single(input_image);
        end
        
        % extract features from images at ground truth landmarks positions.
        fc = [ gt_landmark'; ones(1, num_of_pts) *face_size/100; ones(1, num_of_pts) * (-pi/8)];   % scale of SIFT is determined by face_size. rotation is unknown. 
        [f,d] = vl_sift(I,'frames',fc) ;                        % d is the extracted features. f(1) f(2) are x, y axis.
        Feat = [Feat; reshape(d, 1, [])]; 
        
        % plot first 5 images and SIFT features. 
        if gg < 5  
            figure;imagesc(input_image); colormap(gray); hold on; plot(gt_landmark(:,1), gt_landmark(:,2), 'o');     
            h = vl_plotsiftdescriptor(d(:, 1:30), f(:, 1:30)) ;
            set(h,'color','g', 'linewidth', 0.5) 
        end
    end
    
%% extract SIFT features from folder2
    for gg = 1:length(names3)
        gg
        % get ground truth landmarks and resize according to shape.s0
        pts = read_shape([folder2 names4(gg).name], num_of_pts);     % read ground truth landmarks
        gt_landmark = (pts-1);
        gt_landmark = reshape(gt_landmark, 68, 2);
        % [~,z,T] = procrustes(shape.s0, gt_landmark);    
        
        % get image and and resize according to shape.s0
        input_image = imread([folder2 names3(gg).name]); % input image must be in [0,255]!!
        face_size = (max(gt_landmark(:,1)) - min(gt_landmark(:,1)) + max(gt_landmark(:,2)) - min(gt_landmark(:,2)))/2;
        if size(input_image, 3) == 3
            I = single(rgb2gray(input_image)); 
        else
            I = single(input_image);
        end
        
        % extract features from images at ground truth landmarks positions.
        fc = [ gt_landmark'; ones(1, num_of_pts) *face_size/100; ones(1, num_of_pts) * (-pi/8)];   % scale of SIFT is determined by face_size. rotation is unknown. 
        [f,d] = vl_sift(I,'frames',fc) ;                        % d is the extracted features. f(1) f(2) are x, y axis.
        Feat = [Feat; reshape(d, 1, [])]; 
        
        % plot first 5 images and SIFT features. 
        if gg < 5  
            figure;imagesc(input_image); colormap(gray); hold on; plot(gt_landmark(:,1), gt_landmark(:,2), 'o');     
            h = vl_plotsiftdescriptor(d(:, 1:30), f(:, 1:30)) ;
            set(h,'color','g', 'linewidth', 0.5) 
        end
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
    myAppearance = struct('A0', A0, 'A', A, 'C', C, 'EiVal', EiVal);
    save('myAppearance.mat', 'myAppearance');
%% plot
%     figure;imagesc(input_image); colormap(gray); hold on; plot(init_shape(:,1), init_shape(:,2), 'o');     
%     feature_gg_re = reshape(features_reconstruct(gg, :), 68, []); 
%     face_size = (max(init_shape(:,1)) - min(init_shape(:,1)) + max(init_shape(:,2)) - min(init_shape(:,2)))/2;
%     fc = [ init_shape'; ones(1, num_of_pts) *face_size/100 ; ones(1, num_of_pts) * (-pi/8)];   % scale of SIFT is determined by face_size. rotation is unknown. 
%     h = vl_plotsiftdescriptor(feature_gg_re(1:30, 1:128)', fc(:, 1:30)) ;
%     set(h,'color','b', 'linewidth', 0.5) ;
    
end