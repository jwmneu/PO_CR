% Script for fitting iBUG database using Project-out Cascaded Regression
% Copyright (C) 2015 Georgios Tzimiropoulos
% 
% Matlab Code for paper: 
% [1] G. Tzimiropoulos, "Project-Out Cascaded Regression with an application to Face Alignment," CVPR 2015
% 
% Algorithm and prototype code by Georgios Tzimiropoulos
% C++ implementation by John McDonagh
%
% Model trained on the training set of LFPW and Helen   
%
% Should you use the code, please cite [1] 
% 
% Code released as is **for research purposes only**
% Feel free to distribute but please cite [1]
% 
% contact yorgos.tzimiropoulos@nottingham.ac.uk

clear; clc; close all;
addpath functions;

%% select database and load bb initializations
% IBUG
 folder   = './test_data/iBUG/'; what  = 'jpg';
% folder   = './test_data/AFW/'; what  = 'jpg';
% folder   = './test_data/Helen/'; what  = 'jpg';
% folder   = './test_data/LFPW/'; what  = 'png';

 load bounding_boxes_iBUG; %initializations produced for noise with sigma = 5; see [1] for more details
% load bounding_boxes_AFW;
% load bounding_boxes_Helen;
% load bounding_boxes_LFPW;

%% run for all images
names1 = dir([folder '*.' what]);
names2 = dir([folder '*.pts']);
num_of_pts = 68; % num of landmarks in the annotations
pt_pt_err = zeros(length(names1), 1); % stores pt-pt error for each image

for gg = 1:length(names1)
    gg
    input_image = imread([folder names1(gg).name]); % input image must be in [0,255]!!
    pts = read_shape([folder names2(gg).name], num_of_pts);
        
    %% ground_truth
    gt_s = pts - 1; % provided landmark annotations start from 1, take 1 away to make them start from 0
    gt_s = reshape(gt_s, num_of_pts, 2); % annotations
    face_size = ( max(gt_s(:,1)) - min(gt_s(:,1)) + max(gt_s(:,2)) - min(gt_s(:,2)) )/2;
    
    % keep only 49 points for evaluation
    vec = [1:17 61 65];    
    gt_s(vec, :) = [];
    
    %% initialization
    % Our algorithm assumes as input a bounding box which contains the face.
    % The face region that this bounding box is supposed to cover is shown in "face_region.png".
    % If your face detector was trained on a different face region, it's your
    % responsibility to make the necessary adjustment so that it covers the same region, if you
    % want to obtain the best possible fitting performance.
    % Each bounding box is represented by 4 numbers. bb(1) and bb(2) are the row and column
    % coordinates of its top left corner. bb(3) is the height and bb(4) is the width.
    
    bb = bounding_boxes(gg, :);    
    
    %% fit with Project-Out Cascaded Regression   
    fitted_shape = PO_CR_fit(input_image, bb);    
    
    %% compute pt-pt error
    pt_pt_err1 = zeros(length(gt_s), 1);
    for ii = 1:length(gt_s)
        pt_pt_err1(ii) =  norm(gt_s(ii,:) - fitted_shape(ii,:));                    % abs
    end
    pt_pt_err(gg) = mean(pt_pt_err1)/face_size;    
end

%% plot cumulative curve
var = 0:0.002:0.1;
cum_err = zeros(size(var));
for ii = 1:length(cum_err)
    cum_err(ii) = length(find(pt_pt_err<var(ii)))/length(pt_pt_err);
end

figure; plot(var, cum_err, 'blue', 'linewidth', 4); grid on
xtick = 5*var;
ytick = 0:0.1:1;
set(gca, 'xtick', xtick);
set(gca, 'ytick', ytick);
ylabel('Percentage of Images', 'Interpreter','tex', 'fontsize', 15)
xlabel('Pt-Pt error normalized by face size', 'Interpreter','tex', 'fontsize', 13)
legend('PO-CR')





