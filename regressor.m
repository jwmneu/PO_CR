function [Jp] = regressor()
%% initialization
	load shape_model;
	load myShape; 
    load myAppearance;
    load fd_stat;
    folder1 = './train_data/helen/trainset/';
    what1 = 'jpg';
    folder2 = './train_data/lfpw/trainset/';
    what2 = 'png';
    names1 = dir([folder1 '*.' what1]);
    names2 = dir([folder1 '*.pts']);
    names3 = dir([folder2 '*.' what2]);
    names4 = dir([folder2 '*.pts']);
    num_of_pts = 68;                                           % num of landmarks in the annotations
    P = eye(size(myAppearance.A,1)) - myAppearance.A * myAppearance.A'; 
    P_A0 = P * myAppearance.A0;
    N = size(myAppearance.A, 1);                        % number of SIFT features
    n = length(names1);                                       % number of images
    m = size(myAppearance.A, 2);                        % number of eigenvectors of myAppearance.A
    K = size(myShape.p, 2);                                  % number of eigenvectors of myShape.Q
    T = 3;                                                             % number of iterations
%% cascaded regression for only Helen dataset

    n = 30;                                                                                                              % for testing correctness by small dataset
    p_mat = zeros(n, K, K);
    delta_p_mat = zeros(n, K, K);
    feat = zeros(n, K, N);
    b_mat = zeros(n, K, N);
    bb_gt = zeros(length(names1), 4);
    
    scale = 100; 
    non_rigid_std = fd_stat.std(1, 1:4) * scale;
    non_rigid_std(1, 3) = non_rigid_std(1, 3) * scale;                                          % ???
    
    for t = 1 : T
        t
        parfor gg = 1 : n
            gg
            % perturbe shape parameters of image(gg)
            A0P = myAppearance.A0' * P;             
            % scale face to a constant size
            pts = read_shape([folder1 names2(gg).name], num_of_pts);                % read ground truth landmarks
            gt_landmark = (pts-1);
            gt_landmark = reshape(gt_landmark, 68, 2);
            bb_gt(gg, :) = [min(gt_landmark(:,1)), max(gt_landmark(:, 2)), max(gt_landmark(:,2)) - min(gt_landmark(:,2)), max(gt_landmark(:,1)) - min(gt_landmark(:,1))]; 
            h_scale = max(gt_landmark(:,2)) - min(gt_landmark(:,2)) / scale;
            w_scale = max(gt_landmark(:,1)) - min(gt_landmark(:,1)) / scale; 
           
            for k = 1 : K
                k
                if t == 1                                                                                              % initialization for iteration 1
%                     p_mat(gg, k, :) = myShape.p(gg, :);
%                     p_mat(gg, k, 1) = p_mat(gg, k, 1) / h_scale; 
%                     p_mat(gg, k, 2) = p_mat(gg, k, 2) / w_scale;
%                     p_mat(gg, k, 3) = p_mat(gg, k, 3) / (h_scale * w_scale);
                    pp = myShape.p(gg, :); 
                    p_mat(gg, k, :) = [pp(gg, 1) / h_scale; pp(gg, 2) / w_scale, pp(gg, 3) /  (h_scale * w_scale); pp(gg, 4:end)];
                    
%                     if k > 4                                                                                            % non-rigid parameters -- not dependent on image size
%                         p_mat(gg, k, k) = p_mat(gg, k, k) + fd_stat.std(k) * rand(1) -  fd_stat.std(k) / 2;
%                     else                                                                                                 % rigid parameters -- change according to image size
%                         p_mat(gg, k, k) = p_mat(gg, k, k) + non_rigid_std(k) * rand(1) - non_rigid_std(k) / 2;
%                     end
%                  
                    addv = [non_rigid_std(k) * rand(1) * ones(1, 4) - non_rigid_std(k) / 2 * ones(1, 4),  fd_stat.std(k) * rand(1) * ones(1, K-4) - fd_stat.std(k) / 2 * ones(1, K-4)];
                    p_mat(gg, k, :) = p_mat(gg, k, :) +  addv;
                    
                end
%                 tem_p(1, :) = p_mat(gg, k, :);
                lm = myShape.s0 + myShape.Q * reshape(p_mat(gg, k, :), 1, [])'; 
                feat(gg, k, :) = reshape(SIFT_features([folder1 names1(gg).name], h_scale, w_scale, reshape(lm, [], 2)), 1, []); 
                b_mat(gg, k, :) = reshape(feat(gg, k, :), 1, []) - A0P;
            end
        end

        V = K * ones(n);
        f = @(k) repmat(myShape.p(k,:), round(V(k)), 1);
        p_star_mat = cell2mat(arrayfun(f, (1:length(V))', 'UniformOutput', false));
        p_star_mat_t(1, :, :) = p_star_mat(1:17, :);
        p_star_mat_t(2, :, :) = p_star_mat(18:34, :);
        p_star_mat_t(3, :, :) = p_star_mat(35:51, :);
        delta_p_mat = p_mat - p_star_mat_t; 
    
        b_mat = sum(b_mat, 3);
        b_all = reshape(b_mat', [], 1);
        delta_p_all = repmat( reshape(delta_p_mat, size(delta_p_mat,1) * size(delta_p_mat,2), size(delta_p_mat, 3)), 1, N);
        Jp = ridge(b_all, delta_p_all, 0);
        Jp = reshape(Jp, N, K);
        
        % update shape parameter p
        Hessian = Jp' * Jp; 
        Risk = Hessian \ Jp'; 
        parfor gg = 1 : n
            for k = 1 : K
                p_mat(gg, k, :) = (reshape(p_mat(gg, k, :), 1, K )' + Risk * reshape((b_mat(gg, k, :)), 1, N)')';
            end
        end
    end

end