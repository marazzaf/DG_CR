W = 0.6;
H = 0.2;
l0 = 0.114;
h = 2.5e-3;
eps = 1e-4;

Point(1) = {-W/2,H/2,0,h};
Point(2) = {-W/2,-H/2,0,h};
Point(3) = {W/2,-H/2,0,h};
Point(4) = {W/2,H/2,0,h};
Point(5) = {-l0/2,0,0,h};
Point(6) = {l0/2,0,0,h};
Point(7) = {0,0,0,h};
Point(8) = {0,-eps,0,h};
Point(9) = {0,eps,0,h};

Line(8) = {1,2};
Line(9) = {2,3};
Line(10) = {3,4};
Line(11) = {4,1};
Ellipse(12) = {5,7,6,8};
Ellipse(13) = {8,7,5,6};
Ellipse(14) = {5,7,6,9};
Ellipse(15) = {9,7,5,6};

Line Loop(16) = {8:11};
Line Loop(17) = {12,13,-15,-14};

Plane Surface(18) = {16, 17};

Physical Surface(19) = {18};

