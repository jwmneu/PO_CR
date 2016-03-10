%% Script for fitting with Project-Out Cascaded Regression
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
folder   = './test_data/iBUG/';
what  = 'jpg';
load bounding_boxes_iBUG; % initializations produced for noise with sigma = 5; see [1] for more details


%% Select image
gg = 9;
names1 = dir([folder '*.' what]);
names2 = dir([folder '*.pts']);
input_image = imread([folder names1(gg).name]); % input image must be in [0,255]!!
num_of_pts = 68; % num of landmarks in the annotations
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
% To plot the image along with the 4 corners of the bb
figure; imagesc(input_image); 
hold on; rectangle('Position', [bb(2) bb(1) bb(4) bb(3)], 'linewidth', 4)
hold on; plot(bb(2), bb(1), 'ro'); % top left corner
hold on; plot(bb(2), bb(1) + bb(3), 'b+'); % bottom left corner
hold on; plot(bb(2) + bb(4), bb(1), 'y*'); % top right corner
hold on; plot(bb(2) + bb(4), bb(1) + bb(3), 'gs'); % bottom left corner

%% fit with Project-Out Cascaded Regression
fitted_shape = PO_CR_fit(input_image, bb);    

%% plot the result
% add 1 because Matlab starts counting from 1
figure; imagesc(input_image); hold on; plot(fitted_shape(:, 1)+1, fitted_shape(:, 2)+1, '.', 'MarkerSize',11)

%% compute pt-pt error
pt_pt_err1 = zeros(length(gt_s), 1);
for ii = 1:length(gt_s)
    pt_pt_err1(ii) =  norm(gt_s(ii,:) - fitted_shape(ii,:));
end
pt_pt_err = mean(pt_pt_err1)/face_size;

pt_pt_err




