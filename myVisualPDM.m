load('myShape.mat');
load('/Users/christyliyuan/Box/Spring2016/PO_CR_code_v1/CLM-framework-master/matlab_version/pdm_generation/tri_68.mat');
MeanVals = [myShape.s0; zeros(68,1)];
EigenVectors = [-1 * myShape.Q; zeros(68, 17)];
EigenVals = myShape.EiVal;
visualisePDM(  MeanVals, EigenVals, EigenVectors, T, 17, 5);