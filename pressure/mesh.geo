W = 0.6;
H = 0.2;
l0 = 0.114;
h = 0.01; //5e-3;

Point(1) = {-W/2,H/2,0,h};
Point(2) = {-W/2,-H/2,0,h};
Point(3) = {W/2,-H/2,0,h};
Point(4) = {W/2,H/2,0,h};

Line(8) = {1,2};
Line(9) = {2,3};
Line(10) = {3,4};
Line(11) = {4,1};

Line Loop(15) = {8:11};

Plane Surface(16) = {15};
Physical Surface(17) = {16};

Physical Line(19) = {8,9,10,11,14};
Physical Line(18) = {12,13};
