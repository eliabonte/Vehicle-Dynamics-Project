function ty_data = initialise_tyre_data(R0, Fz0)


% Tyre structure data initialization

ty_data.FZ0             = Fz0; % Fz0  % Normal load
ty_data.R0              = R0; % R0  % nominal radius
ty_data.pCx1            = 1.5; %1; % pCx1
ty_data.pDx1            = 2; %1; % pDx1
ty_data.pDx2            = -1; % pDx2
ty_data.pDx3            = 5; % pDx3
ty_data.pEx1            = 0.2; % pEx1
ty_data.pEx2            = 0.5; % pEx2
ty_data.pEx3            = 1; % pEx3
ty_data.pEx4            = 0.1; % pEx4
ty_data.pHx1            = 0.005; % pHx1
ty_data.pHx2            = 0.005; % pHx2
ty_data.pKx1            = 40; %1; % pKx1
ty_data.pKx2            = -0.04; % pKx2
ty_data.pKx3            = 0.8; % pKx3
ty_data.pVx1            = -0.2; % pVx1
ty_data.pVx2            = -0.1; % pVx2
%ty_data.Fz01            = 220; % Fz01
ty_data.pCy1            = 1.5; % pCy1
ty_data.pDy1            = 2; % pDy1
ty_data.pDy2            = -0.5; % pDy2
ty_data.pDy3            = 10; % pDy3
ty_data.pEy1            = 0.3; % pEy1
ty_data.pEy2            = -1; % pEy2
ty_data.pEy3            = 0.1; % pEy3
ty_data.pEy4            = -7; % pEy4
ty_data.pHy1            = -0.005; % pHy1
ty_data.pHy2            = -0.001; % pHy2
ty_data.pHy3            = -0.1; % pHy3
ty_data.pKy1            = -25; % pKy1
ty_data.pKy2            = -1; % pKy2
ty_data.pKy3            = 0.5; % pKy3
ty_data.pVy1            = 0.03; % pVy1
ty_data.pVy2            = -0.05; % pVy2
ty_data.pVy3            = -1.5; % pVy3
ty_data.pVy4            = -1; % pVy4
ty_data.qBz1            = 5; % qBz1
ty_data.qBz2            = -3; % qBz2
ty_data.qBz3            = -1; % qBz3
ty_data.qBz4            = 0.1; % qBz4
ty_data.qBz5            = 0.1; % qBz5
ty_data.qBz9            = 0.0001; % qBz9
ty_data.qBz10           = 0.0001; % qBz10
ty_data.qCz1            = 1.5; % qCz1
ty_data.qDz1            = 0.2; % qDz1
ty_data.qDz2            = -0.03; % qDz2
ty_data.qDz3            = 1; % qDz3
ty_data.qDz4            = 1; % qDz4
ty_data.qDz6            = 0.005; % qDz6
ty_data.qDz7            = 0.001; % qDz7
ty_data.qDz8            = 1; % qDz8
ty_data.qDz9            = 1; % qDz9
ty_data.qEz1            = 0.4; % qEz1
ty_data.qEz2            = -1; % qEz2
ty_data.qEz3            = -1; % qEz3
ty_data.qEz4            = -0.1; % qEz4
ty_data.qEz5            = 1; % qEz5
ty_data.qHz1            = -0.01; % qHz1
ty_data.qHz2            = 0.005; % qHz2
ty_data.qHz3            = 0.1; % qHz3
ty_data.qHz4            = 0.1; % qHz4
ty_data.rBx1            = 15; % rBx2
ty_data.rBx2            = 15; % rBx2
ty_data.rBy1            = 15; % rBy1
ty_data.rBy2            = 25; % rBy2
ty_data.rBy3            = -0.5; % rBy3
ty_data.rCx1            = 1; % rCx1
ty_data.rCy1            = 1; % rCy1
ty_data.rHx1            = 0.0001; % rHx1
ty_data.rHy1            = 0.03; % rHy1
ty_data.rVy1            = -0.05; % rVy1
ty_data.rVy2            = 0.0001; % rVy2
ty_data.rVy3            = 0.0001; % rVy3
ty_data.rVy4            = -0.1; % rVy4
ty_data.rVy5            = -0.7; % rVy5
ty_data.rVy6            = 50; % rVy6



% scaling factor
ty_data.LCX             = 1; % LCX
ty_data.LCY             = 1; % LCY
ty_data.LEX             = 1; % LEX
ty_data.LEY             = 1; % LEY
ty_data.LFZ0            = 1; % LFZ0
ty_data.LGAMMAY         = 1; % LGAMMAY
ty_data.LHX             = 1; % LHX
ty_data.LHY             = 1; % LHY
ty_data.LKA             = 1; % LKA
ty_data.LKXK            = 1; % LKXK
ty_data.LMUX            = 1; % LMUX
ty_data.LMUY            = 1; % LMUY
ty_data.LVX             = 1; % LVX
ty_data.LVY             = 1; % LVY
ty_data.LVYK            = 1; % LVYK
ty_data.LXA             = 1; % LXA
ty_data.LKY             = 1; % LKY
ty_data.LMR             = 1; % LMR
ty_data.LT             = 1; % LT

end