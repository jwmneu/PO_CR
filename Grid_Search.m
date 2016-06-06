% try on different nonrigid parameters 
t = 2; 
for sc_nonrigid = 0.05 : 0.05 : 0.75
	RegressorLM3D(t, 'nonrigid', 0, sc_nonrigid);
end

% try on different rigid parameters
for sc_rigid = 0.05 : 0.05 : 0.75
	RegressorLM3D(t, 'rigid', sc_rigid, 0);
end

% try on different rigid and nonrigid parameters
for sc_rigid = 0.05 : 0.05 : 0.75
	for sc_nonrigid = 0.05 : 0.05 : 0.75
		RegressorLM3D(t, 'all', sc_rigid, sc_nonrigid); 
	end
end
