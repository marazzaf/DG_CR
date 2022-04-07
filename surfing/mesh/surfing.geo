L = 5;
H = 1;
l0 = 1;
eps = 0.0005; //0.01; //0.005;

h = 0.0031;
//CG: 0.0043; //0.013
//DG: 0.007 0.013

Point(1) = {0,H/2,0,h};
Point(2) = {0,-H/2,0,h};
Point(3) = {L,-H/2,0,h};
Point(4) = {L,H/2,0,h};
Point(5) = {l0,0,0,h};
Point(6) = {0,eps,0,h};
Point(7) = {0,-eps,0,h};

Line(8) = {1,6};
Line(9) = {2,3};
Line(10) = {3,4};
Line(11) = {4,1};
Line(12) = {6,5};
Line(13) = {7,5};
Line(14) = {7,2};

Line Loop(15) = {8,12,-13,14,9,10,11};

Plane Surface(16) = {15};
Physical Surface(17) = {16};

Physical Line(19) = {8,9,10,11,14};
Physical Line(18) = {12,13};
