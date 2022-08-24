L = 10;
l0 = 0.114;
h = 0.5e-3;
hc = 0.25;
W = 0.6;
H = 11*h;

Point(1) = {-W/2,H/2,0,h};
Point(2) = {-W/2,-H/2,0,h};
Point(3) = {W/2,-H/2,0,h};
Point(4) = {W/2,H/2,0,h};
//Point(5) = {-l0/2,0,0,h};
//Point(6) = {l0/2,0,0,h};
Point(7) = {-L/2,-L/2,0,hc};
Point(8) = {-L/2,L/2,0,hc};
Point(9) = {L/2,L/2,0,hc};
Point(10) = {L/2,-L/2,0,hc};

Line(8) = {1,2};
Line(9) = {2,3};
Line(10) = {3,4};
Line(11) = {4,1};
//Line(12) = {5,6};

Line Loop(15) = {8:11};

Line(13) = {7,8};
Line(14) = {8,9};
Line(15) = {9,10};
Line(16) = {10,7};
Line Loop(17) = {13:16};

Plane Surface(16) = {17};
Line{8} In Surface{16};
Line{9} In Surface{16};
Line{10} In Surface{16};
Line{11} In Surface{16};
Physical Surface(17) = {16};

