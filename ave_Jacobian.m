function [ave_PJ, ave_H, R, p] = ave_Jacobian( folder, what, shape, A, A0, iter)
    s0 = shape.s0;
    Q = shape.Q; 
    names1 = dir([folder '*.' what]);
    names2 = dir([folder '*.pts']);
    num_of_pts = 68; % num of landmarks in the annotations
    P = eye(size(A,1)) - A * A'; 
   
    ave_J = zeros(size(A, 1), size(Q,2));
    p = zeros(length(names1), size(Q, 2), K);           % initialize K perturbed shape parameters. 
    
    %% training
    for i = 1 : iter
        %% compute average jaccobian of current iteration
        for gg = 1:length(names1)

            pts = read_shape([folder names2(gg).name], num_of_pts);
            s_star = (pts-1);
            s_star = reshape(s_star, 68*2, 1);
            p_star = Q' * (s_star - s0);
            for k = 1:K
                delta_p = p_star - p(gg, :, k); 
                cur_shape = s0 + Q * p(gg, :, k); 
                [I] = SIFT_features(folder, what, gg, cur_shape);
                b = P * I - P * A0; 
                x = ridge(b,P,kk);               % don't know what's kk
                J = x / delta_p;
                ave_J = ave_J + J; 
            end

        end
        ave_J = ave_J / (K * length(names1)); 
        ave_PJ = P * ave_J; 
        ave_H = ave_PJ' * ave_PJ; 
        R = inv(ave_H) * ave_PJ'; 
    
        %% update shape parameters at current iteration
        for gg = 1:length(names1)
            for k = 1:K
                cur_shape = s0 + Q * p(gg, :, k); 
                [I] = SIFT_features(folder, what, gg, cur_shape);
                p(gg, :, k) = p(gg, :, k) + R * (I - A0);
            end

        end
        
    end
    
end

































