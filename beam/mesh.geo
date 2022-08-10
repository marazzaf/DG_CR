//Définition des valeurs physiques
W = 8; //Longeur selon x
H = 2; //Longuer selon y
l0 = 0.4; //Size initial crack
delta = 0.2; // initil opening	
h = 0.05; //1.5e-3; //2e-3 //Taille du maillage

//Plate
Point(1) = {0,0,0,h};
Point(2) = {W/2-delta/2,0,0,h};
Point(3) = {W/2,delta,0,h};
Point(4) = {W/2+delta/2,0,0,h};
Point(5) = {W,0,0,h};
Point(6) = {W,H,0,h};
Point(7) = {W/2+h,H,0,h};
Point(8) = {W/2-h,H,0,h};
Point(9) = {0,H,0,h};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,1};

Line Loop(1) = {1:9};

//Final Surface
Plane Surface(1) = {1};

//Outputs
Physical Surface(2) = {1};