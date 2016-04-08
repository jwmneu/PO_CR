This is the reimplementation of PO_CR.

Run regressor_sep.m to train a project-out cascaded regressor by training datasets of Helen and LFPW. 

Before running regressor_sep.m, please run following scripts to generate some .mat files. 

run shape_model.m to get myShape.mat

run appe_model.m to get myAppearance.mat

run face_det_stat.m to get fd_stat.mat

To validate the trained model on test set of Helen and LFPW, run validate.m.
