epsc = 0.01;
L = 1;
W = 1;
lnotch = L/2;

hf = 0.001;
hc = 0.01;

Point(1) = {L, W/2, 0, hc};
Point(2) = {L,epsc,0, hf};
Point(3) = {lnotch,0,0,hf};
Point(4) = {L,-epsc,0,hf};
Point(5) = {L, -W/2,0,hc};
Point(6) = {0, -W/2,0,hc};
Point(7) = {0, W/2, 0, hc};

For i In {1:7}
	Line(i) = {i,i%7+1};
EndFor
Line Loop(1) = {1:7};
Plane Surface(1) = {1};

Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Line(7)   = {7};
Physical Line(5)   = {5};
