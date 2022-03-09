SetFactory("OpenCASCADE");
W=63.50;
H=60.96;
Rhole = 10/2;
Rload = 12.7/2;
wnotch = 4;
lnotch = 22.86;//-wnotch/2;
a = 13.47; //Initial crack coord in x
b = 0.35; //Initial crack coord in y
xload = 12.7;
yload = wnotch/2+16.51;
xhole = W-15.8;
yhole = H/2-21.48;
epsc = 0.01;

hf = 0.03; //7.5e-3 in ref //0.08 //0.02
hl = 0.15; //0.3 //0.15
hc = 0.7; //1 //0.7

Point(1) = {lnotch+a, b, 0, hf};
Point(2) = {lnotch,-epsc,0, hf};
Point(3) = {lnotch-wnotch/2,-wnotch/2,0,hf};
Point(4) = {0,-wnotch/2,0,hf};
Point(5) = {0,wnotch/2,0,hf};
Point(6) = {lnotch-wnotch/2,wnotch/2,0,hf};
Point(7) = {lnotch,epsc,0, hf};

For i In {1:7}
	Line(i) = {i,i%7+1};
EndFor
Line Loop(1) = {1:7};
Plane Surface(1) = {1};
Disk(20) = {xload,yload,0,Rload};
Disk(21) = {xload,-yload,0,Rload};
Disk(22) = {xhole,yhole,0,Rhole};

Rectangle(2) = {0,-W/2,0,H,W,0};
BooleanDifference(3) = {Surface{2}; Delete;} {Surface{1,20,21,22}; Delete;};

Rectangle(4) = {lnotch+a/2,-yhole,0,(xhole+Rhole+W-a)/2-lnotch,2*yhole,0};
BooleanFragments{Surface{3,4}; Delete;}{}
Delete{Surface {3,4}; }


// Sizing function:
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;

Field[1]      =  Box;
Field[1].VIn  =  hf;
Field[1].VOut =  hc;
Field[1].XMin =  lnotch+a/2;
Field[1].YMin =  -yhole;
Field[1].ZMin =  0;
Field[1].XMax =  (xhole+Rhole+W)/2;
Field[1].YMax =  yhole;
Field[1].ZMax =  0;

r = wnotch;
Field[2]      = Distance;
Field[2].EdgesList = {19,20};

Field[3] = Threshold;
Field[3].IField = 2;
Field[3].LcMin = hl;
Field[3].LcMax = hc;
Field[3].DistMin = 0;
Field[3].DistMax = r;

Field[4]      = Distance;
Field[4].FacesList = {2};

Field[5] = Threshold;
Field[5].IField = 4;
Field[5].LcMin = hf;
Field[5].LcMax = hc;
Field[5].DistMin = 0;
Field[5].DistMax = 2*r;

Field[6] = Min;
Field[6].FieldsList = {1, 3, 5};

Background Field = 6;


Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Line(30)   = {19};
Physical Line(40)   = {20};
Physical Line(50)   = {21,22};
Physical Point(600) = {21};
