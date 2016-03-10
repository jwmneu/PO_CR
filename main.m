%% load models
    load shape_model;
    load std_detector.mat
    load bounding_boxes_iBUG; % initializations produced for noise with sigma = 5; see [1] for more details

%% build shape and appearance model
    [myShape] = shape_model();
    [myAppearance, Feat, Feat_rec] = appe_model();
    % [A0, A, C, EiVal, features_reconstruct] = appe_model(shape, s0, Q, p, EiVal, rigid); 
    [fd_stat] = face_det_stat();
    load myShape; 
    load myAppearance;
    load fd_stat;

%% select database 
    % IBUG
    folder = './test_data/iBUG/';
    what  = 'jpg';
    names1 = dir([folder '*.' what]);
    names2 = dir([folder '*.pts']);
    % gg = 3;         % select image
    features = [];

%% create features for all images 
    %     for gg = 1:length(names1)
    for gg = 1:1
        gg
        [init_shape] = initialize_shape(folder, what, gg, shape, std_dr, bounding_boxes);            % get the initial landmarks
        [features_gg] = SIFT_features(folder, what, gg, init_shape);
        features_gg = reshape(features_gg,1, []);
        features = [features; features_gg]; 
    end
    features = normalize(double(features));              % normalize features to 0 - 1
    %     save('features.mat', 'features');

    load('features.mat', 'features');                               % features is already nomalized
    [A0, A, C, EiVal, features_reconstruct] = appe_model(folder, what, features, init_shape);

%% cascaded regression  
    %     [ave_PJ, ave_H, R, p] = ave_Jacobian( folder, what, shape, A, A0);           % training
    
    
