epsc = 0.01;
L = 1;
W = 1:

hf = 0.03;
hc = 0.7;

Point(1) = {lnotch+a, b, 0, hf};
Point(2) = {lnotch,-epsc,0, hf};
Point(3) = {lnotch-wnotch/2,-wnotch/2,0,hf};
Point(4) = {0,-wnotch/2,0,hf};
Point(5) = {0,wnotch/2,0,hf};
Point(6) = {lnotch-wnotch/2,wnotch/2,0,hf};
Point(7) = {lnotch,epsc,0, hf};

//For i In {1:7}
//	Line(i) = {i,i%7+1};
//EndFor
//Line Loop(1) = {1:7};
//Plane Surface(1) = {1};
//
//Physical Surface(1) = {1};
//Physical Surface(2) = {2};
//Physical Line(30)   = {19};
//Physical Line(40)   = {20};
//Physical Line(50)   = {21,22};
//Physical Point(600) = {21};
