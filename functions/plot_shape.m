% shape.Q: [scale, rotation, x translation, y translation]

p1 = [100, 0, 0, 0];
p2 = [0,100, 0, 0];
p3 = [0, 0, 100, 0];
p4 = [0, 0, 0, 100];

s0 = shape.s0;
s1 = shape.s0 + reshape(shape.Q * p1', 68,2);
s2 = shape.s0 + reshape(shape.Q * p2', 68,2);
s3 = shape.s0 + reshape(shape.Q * p3', 68,2);
s4 = shape.s0 + reshape(shape.Q * p4', 68,2);

s0(:,2) = -1 * s0(:, 2); 
s1(:,2) = -1 * s1(:,2);
s2(:,2) = -1 * s2(:,2);
s3(:,2) = -1 * s3(:,2);
s4(:,2) = -1 * s4(:,2);


    figure;  hold on; 
    plot(s0(:,1), s0(:,2), 'o'); 
    plot(s1(:,1), s1(:,2), '-');     
  
    figure;  hold on; 
    plot(s0(:,1), s0(:,2), 'o'); 
    plot(s2(:,1), s2(:,2), '-');     
  
    figure;  hold on; 
    plot(s0(:,1), s0(:,2), 'o'); 
    plot(s3(:,1), s3(:,2), '-');     
  
    figure;  hold on; 
    plot(s0(:,1), s0(:,2), 'o'); 
    plot(s4(:,1), s4(:,2), '-');     
  