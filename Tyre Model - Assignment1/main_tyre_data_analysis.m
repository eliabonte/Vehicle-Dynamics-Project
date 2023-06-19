%% Initialisation
clc
clearvars 
close all   

% Set LaTeX as default interpreter for axis labels, ticks and legends
set(0,'defaulttextinterpreter','latex')
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

set(0,'DefaultFigureWindowStyle','docked');
set(0,'defaultAxesFontSize',  16)
set(0,'DefaultLegendFontSize',16)

addpath('tyre_lib/')


to_rad = pi/180;
to_deg = 180/pi;


% tyre geometric data:
% Hoosier	18.0x6.0-10
% 18 diameter in inches
% 6.0 section width in inches
% tread width in inches
diameter = 18*2.56; %
Fz0 = 1120;   % [N] nominal load is given
R0  = diameter/2/100; % [m] get from nominal load R0 (m) *** TO BE CHANGED ***

% initialise tyre data
tyre_coeffs = initialise_tyre_data(R0, Fz0);

%% Select tyre dataset
%dataset path
data_set_path = 'dataset/';
% dataset selection and loading

% compute before later_case and then longitudinal
%data_set is 'later_case' for pure lateral tests and 'longi_case' for braking/traction (pure log. force) + combined
data_set = 'longi_case';


fprintf('Loading dataset ...')
switch data_set
    case 'later_case'
        load ([data_set_path, 'Hoosier_B1464run23.mat']); % pure lateral
        cut_start = 27760;
        cut_end   = 54500;
    case 'longi_case'
        load ([data_set_path, 'Hoosier_B1464run30.mat']); % pure longitudinal
        cut_start = 19028;
        cut_end   = 37643;
    otherwise 
        error('Not found dataset: `%s`\n', data_set) ;
end

% select dataset portion
smpl_range = cut_start:cut_end;

fprintf('completed!\n')
%% Plot raw data

figure
tiledlayout(6,1)

ax_list(1) = nexttile; y_range = [min(min(-FZ),0) round(max(-FZ)*1.1)];
plot(-FZ)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Vertical force')
xlabel('Samples [-]')
ylabel('[N]')

ax_list(2) = nexttile; y_range = [min(min(IA),0) round(max(IA)*1.1)];
plot(IA)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Camber angle')
xlabel('Samples [-]')
ylabel('[deg]')

ax_list(3) = nexttile; y_range = [min(min(SA),0) round(max(SA)*1.1)];
plot(SA)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Side slip')
xlabel('Samples [-]')
ylabel('[deg]')

ax_list(4) = nexttile; y_range = [min(min(SL),0) round(max(SL)*1.1)];
plot(SL)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Longitudinal slip')
xlabel('Samples [-]')
ylabel('[-]')

ax_list(5) = nexttile; y_range = [min(min(P),0) round(max(P)*1.1)];
plot(P)
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Tyre pressure')
xlabel('Samples [-]')
ylabel('[psi]')

ax_list(6) = nexttile;  y_range = [min(min(TSTC),0) round(max(TSTC)*1.1)];
plot(TSTC,'DisplayName','Center')
hold on
plot(TSTI,'DisplayName','Internal')
plot(TSTO,'DisplayName','Outboard')
hold on
plot([cut_start cut_start],y_range,'--r')
plot([cut_end cut_end],y_range,'--r')
title('Tyre temperatures')
xlabel('Samples [-]')
ylabel('[degC]')

linkaxes(ax_list,'x')


%% Select some specific data
% Cut crappy data and select only 12 psi data

vec_samples = 1:1:length(smpl_range);
tyre_data = table(); % create empty table
% store raw data in table
tyre_data.SL =  SL(smpl_range);
tyre_data.SA =  SA(smpl_range)*to_rad;
tyre_data.FZ = -FZ(smpl_range);  % 0.453592  lb/kg
tyre_data.FX =  FX(smpl_range);
tyre_data.FY =  FY(smpl_range);
tyre_data.MZ =  -MZ(smpl_range);
tyre_data.IA =  IA(smpl_range)*to_rad;

% Extract points at constant inclination angle
GAMMA_tol = 0.05*to_rad;
idx.GAMMA_0 = 0.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 0.0*to_rad+GAMMA_tol;
idx.GAMMA_1 = 1.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 1.0*to_rad+GAMMA_tol;
idx.GAMMA_2 = 2.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 2.0*to_rad+GAMMA_tol;
idx.GAMMA_3 = 3.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 3.0*to_rad+GAMMA_tol;
idx.GAMMA_4 = 4.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 4.0*to_rad+GAMMA_tol;
idx.GAMMA_5 = 5.0*to_rad-GAMMA_tol < tyre_data.IA & tyre_data.IA < 5.0*to_rad+GAMMA_tol;
GAMMA_0  = tyre_data( idx.GAMMA_0, : );
GAMMA_1  = tyre_data( idx.GAMMA_1, : );
GAMMA_2  = tyre_data( idx.GAMMA_2, : );
GAMMA_3  = tyre_data( idx.GAMMA_3, : );
GAMMA_4  = tyre_data( idx.GAMMA_4, : );
GAMMA_5  = tyre_data( idx.GAMMA_5, : );

% Extract points at constant vertical load
% Test data done at: 
%  - 50lbf  ( 50*0.453592*9.81 =  223N )
%  - 150lbf (150*0.453592*9.81 =  667N )
%  - 200lbf (200*0.453592*9.81 =  890N )
%  - 250lbf (250*0.453592*9.81 = 1120N )

FZ_tol = 100;
idx.FZ_220  = 220-FZ_tol < tyre_data.FZ & tyre_data.FZ < 220+FZ_tol;
idx.FZ_440  = 440-FZ_tol < tyre_data.FZ & tyre_data.FZ < 440+FZ_tol;
idx.FZ_700  = 700-FZ_tol < tyre_data.FZ & tyre_data.FZ < 700+FZ_tol;
idx.FZ_900  = 900-FZ_tol < tyre_data.FZ & tyre_data.FZ < 900+FZ_tol;
idx.FZ_1120 = 1120-FZ_tol < tyre_data.FZ & tyre_data.FZ < 1120+FZ_tol;
FZ_220  = tyre_data( idx.FZ_220, : );
FZ_440  = tyre_data( idx.FZ_440, : );
FZ_700  = tyre_data( idx.FZ_700, : );
FZ_900  = tyre_data( idx.FZ_900, : );
FZ_1120 = tyre_data( idx.FZ_1120, : );


% switch-case to select the right Fz table based on Fz0 defined at the
% begininning (useful for future sections)
switch Fz0
    case 220
        FZ_table = FZ_220;
    case 440
        FZ_table = FZ_440;
    case 700
        FZ_table = FZ_700;
    case 900
        FZ_table = FZ_900;
    case 1120
        FZ_table = FZ_1120;
    otherwise 
        error('Not found specific Fz dataset: %d\n', Fz0) ;
end



% The slip angle is varied step wise for longitudinal slip tests
% 0° , - 3° , -6 °
SA_tol = 0.5*to_rad;
idx.SA_0    =  0-SA_tol          < tyre_data.SA & tyre_data.SA < 0+SA_tol;
idx.SA_3neg = -(3*to_rad+SA_tol) < tyre_data.SA & tyre_data.SA < -3*to_rad+SA_tol;
idx.SA_6neg = -(6*to_rad+SA_tol) < tyre_data.SA & tyre_data.SA < -6*to_rad+SA_tol;
SA_0     = tyre_data( idx.SA_0, : );
SA_3neg  = tyre_data( idx.SA_3neg, : );
SA_6neg  = tyre_data( idx.SA_6neg, : );

% longitudinal slip is constant 0 for pure lateral slip tests
KAPPA_tol = 0.005;
idx.KAPPA_0 = 0-KAPPA_tol < tyre_data.SL & tyre_data.SL < 0+KAPPA_tol;
KAPPA_0 = tyre_data(idx.KAPPA_0, :);

figure()
tiledlayout(3,1)

ax_list(1) = nexttile;
plot(tyre_data.IA*to_deg)
hold on
plot(vec_samples(idx.GAMMA_0),GAMMA_0.IA*to_deg,'.');
plot(vec_samples(idx.GAMMA_1),GAMMA_1.IA*to_deg,'.');
plot(vec_samples(idx.GAMMA_2),GAMMA_2.IA*to_deg,'.');
plot(vec_samples(idx.GAMMA_3),GAMMA_3.IA*to_deg,'.');
plot(vec_samples(idx.GAMMA_4),GAMMA_4.IA*to_deg,'.');
plot(vec_samples(idx.GAMMA_5),GAMMA_5.IA*to_deg,'.');
title('Camber angle')
xlabel('Samples [-]')
ylabel('[deg]')

ax_list(2) = nexttile;
plot(tyre_data.FZ)
hold on
plot(vec_samples(idx.FZ_220),FZ_220.FZ,'.');
plot(vec_samples(idx.FZ_440),FZ_440.FZ,'.');
plot(vec_samples(idx.FZ_700),FZ_700.FZ,'.');
plot(vec_samples(idx.FZ_900),FZ_900.FZ,'.');
plot(vec_samples(idx.FZ_1120),FZ_1120.FZ,'.');
title('Vertical force')
xlabel('Samples [-]')
ylabel('[N]')


ax_list(3) = nexttile;
plot(tyre_data.SA*to_deg)
hold on
plot(vec_samples(idx.SA_0),   SA_0.SA*to_deg,'.');
plot(vec_samples(idx.SA_3neg),SA_3neg.SA*to_deg,'.');
plot(vec_samples(idx.SA_6neg),SA_6neg.SA*to_deg,'.');
title('Slide slip')
xlabel('Samples [-]')
ylabel('[deg]')


%% Pure LONGITUDINAL SLIP: Fitting with Fz=Fz_nom= 220N and camber=0  alpha = 0 VX= 10

if data_set == 'longi_case'
    %%Intersect tables to obtain specific sub-datasets for pure longitudinal forces
    [TData0, ~] = intersect_table_data( SA_0, GAMMA_0, FZ_table );
    %%plot_selected_data
    figure('Name','Pure Longitudinal case DATA')
    plot_selected_data(TData0);   

    FZ0 = mean(TData0.FZ);
    
    KAPPA_vec = TData0.SL;
    FX_vec    = TData0.FX;
    zeros_vec = zeros(size(TData0.SL));
    ones_vec  = ones(size(TData0.SL));
    
    % [fx0_vec] = MF96_FX0(  kappa_vec, alpha_vec, phi_vec,   Fz_vec,   tyre_data)
    FX0_guess = MF96_FX0_vec(KAPPA_vec,zeros_vec , zeros_vec, FZ0*ones_vec, tyre_coeffs);
    
    % check guess 
    figure('Name','Raw Data Fx0')
    plot(TData0.SL,FX_vec,'.')
    hold on
    plot(TData0.SL,FX0_guess,'-')
    title('Raw Data Fx0 vs initial guess')

    
    % Fit the coeffs {pCx1, pDx1, pEx1, pEx4, pHx1, pKx1, pVx1}
    % Guess values for parameters to be optimised
    %    [pCx1,pDx1,pEx1 pEx4  pHx1  pKx1  pVx1 
    P0 = [1.5, 2, 0.2, 0.1, 0.005, 40, -0.2]; 
    
    % NOTE: many local minima => limits on parameters are fundamentals
    % Limits for parameters to be optimised
    % 1< pCx1 < 2 
    % 0< pEx1 < 1 
    %    [pCx1 pDx1 pEx1 pEx4  pHx1  pKx1  pVx1 
    lb = [1, 0.1,  -1,  -1,  -2,  0, -2];
    ub = [2,   5,   1,   1,   2, 80,  2];
    
    
    % LSM_pure_Fx returns the residual, so minimize the residual varying X. It
    % is an unconstrained minimization problem 
                                   % res = resid_pure_Fx(P,FX,     KAPPA,GAMMA,FZ,tyre_data)
    [P_fz_nom,fval,exitflag] = fmincon(@(P)resid_pure_Fx(P,FX_vec, KAPPA_vec,0,FZ0, tyre_coeffs),...
                                   P0,[],[],[],[],lb,ub);
    
    % Update tyre data with new optimal values                             
    tyre_coeffs.pCx1 = P_fz_nom(1) ; % 1
    tyre_coeffs.pDx1 = P_fz_nom(2) ;  
    tyre_coeffs.pEx1 = P_fz_nom(3) ;
    tyre_coeffs.pEx4 = P_fz_nom(4) ;
    tyre_coeffs.pHx1 = P_fz_nom(5) ; 
    tyre_coeffs.pKx1 = P_fz_nom(6) ;
    tyre_coeffs.pVx1 = P_fz_nom(7) ;
    
    
    SL_vec = -0.3:0.001:0.3;
    FX0_fz_nom_vec = MF96_FX0_vec(SL_vec,zeros(size(SL_vec)) , zeros(size(SL_vec)),FZ0*ones(size(SL_vec)),tyre_coeffs);
    
    figure('Name','Fx0')
    plot(TData0.SL,TData0.FX,'o','MarkerSize',2)
    hold on
    %plot(TDataSub.KAPPA,FX0_fz_nom_vec,'-')
    plot(SL_vec,FX0_fz_nom_vec,'-','LineWidth',2)
    xlabel('$\kappa$ [-]')
    ylabel('$F_{x0}$ [N]')
    titolo = sprintf('Pure Longitudinal slip at vertical load Fz = %d N, zero camber, zero slip angle, press = 12 psi', Fz0);
    title(titolo)

    
    % Compute index of performance

    % Calculate the residuals with the optimal solution found above
    [SL_vec, sortIdx] = sort(TData0.SL,'ascend');
    FX = FX_vec(sortIdx);
    FX0_fz_nom_vec = MF96_FX0_vec(SL_vec,zeros(size(SL_vec)),zeros(size(SL_vec)),FZ0*ones(size(SL_vec)),tyre_coeffs);

    residuals_FX0 = 0;
    for i=1:length(SL_vec)
       residuals_FX0 = residuals_FX0+(FX0_fz_nom_vec(i)-FX(i))^2;     
    end
    
    % Compute the residuals
    residuals_FX0 = residuals_FX0/sum(FX.^2);

    % R-squared is 
    % 1-SSE/SST
    % SSE/SST = residuals_FX0
    
    % SSE is the sum of squared error,  SST is the sum of squared total
    fprintf('Index of Fx0 --> R-squared = %6.3f\n',1-residuals_FX0);


    % RMSE
    FX0_vec_pred = MF96_FX0_vec(KAPPA_vec,zeros(size(KAPPA_vec)),zeros(size(KAPPA_vec)),FZ0*ones(size(KAPPA_vec)),tyre_coeffs);
    E = rmse(FX0_vec_pred,TData0.FX);
    fprintf('Index for FX0 --> RMSE = %6.3f\n',E);

end

%% Pure LATERAL SLIP: Fitting with Fz=Fz_nom= 220N and camber=0  k = 0 Vx= 10

if data_set == 'later_case'

    %%Intersect tables to obtain specific sub-datasets for pure longitudinal forces
    [TData0, ~] = intersect_table_data( KAPPA_0, GAMMA_0, FZ_table );
    %%plot_selected_data
    figure('Name','Pure Lateral case DATA')
    plot_selected_data(TData0);
            
    FZ0 = mean(TData0.FZ);
    
    ALPHA_vec = TData0.SA;
    FY_vec    = TData0.FY;
    zeros_vec = zeros(size(TData0.SA));
    ones_vec  = ones(size(TData0.SA));
    
    FY0_guess = MF96_FY0_vec(zeros_vec,ALPHA_vec,zeros_vec,FZ0*ones_vec,tyre_coeffs);
    
    % check guess 
    figure('Name','Raw Data Fx0')
    plot(TData0.SA,FY_vec,'.')
    hold on
    plot(TData0.SA,FY0_guess,'-')
    title('Raw data Fy0 vs intial guess')
    
    % Guess values for parameters to be optimised
    %    {pCy1,pDy1,pEy1,pHy1,pKy1,pKy2,pVy1}
    P0 = [1, 2, 0.1, -0.01, -25, -1, 0.1]; 
    
    % NOTE: many local minima => limits on parameters are fundamentals
    % Limits for parameters to be optimised
    %{pCy1,pDy1,pEy1,pHy1,pKy1,pKy2,pVy1}
    lb = [ -4, -4, -2, -2, -110, -4, -2];
    ub = [  4,  6,  2,  2, -10,  4,  2];
    
    
    % LSM_pure_Fx returns the residual, so minimize the residual varying X. It
    % is an unconstrained minimization problem 
    [P_fz_nom,fval,exitflag] = fmincon(@(P)resid_pure_Fy(P,FY_vec, ALPHA_vec,0,FZ0, tyre_coeffs),...
                                   P0,[],[],[],[],lb,ub);
    
    % Update tyre data with new optimal values                             
    tyre_coeffs.pCy1 = P_fz_nom(1) ; % 1
    tyre_coeffs.pDy1 = P_fz_nom(2) ;  
    tyre_coeffs.pEy1 = P_fz_nom(3) ;
    tyre_coeffs.pHy1 = P_fz_nom(4) ;
    tyre_coeffs.pKy1 = P_fz_nom(5) ; 
    tyre_coeffs.pKy2 = P_fz_nom(6) ;
    tyre_coeffs.pVy1 = P_fz_nom(7) ;
    
    ALPHA_vec = -0.3:0.001:0.3;
    FY0_fz_nom_vec = MF96_FY0_vec(zeros(size(ALPHA_vec)),ALPHA_vec,zeros(size(ALPHA_vec)),FZ0*ones(size(ALPHA_vec)),tyre_coeffs);

    figure('Name','Fy0')
    plot(TData0.SA,TData0.FY,'o','MarkerSize',2)
    hold on
    plot(ALPHA_vec,FY0_fz_nom_vec,'-','LineWidth',2)
    xlabel('$\alpha$ [rad]')
    ylabel('$F_{y0}$ [N]')
    titolo = sprintf('Pure Lateral slip at vertical load Fz = %d N, zero camber, zero slip angle, press = 12 psi',Fz0);
    title(titolo)


    % Compute index of performance

    % Calculate the residuals with the optimal solution found above
    [ALPHA_vec, sortIdx] = sort(TData0.SA,'ascend');
    FY = FY_vec(sortIdx);
    FY0_fz_nom_vec = MF96_FY0_vec(zeros(size(ALPHA_vec)),ALPHA_vec,zeros(size(ALPHA_vec)),FZ0*ones(size(ALPHA_vec)),tyre_coeffs);

    residuals_FY0 = 0;
    for i=1:length(ALPHA_vec)
       residuals_FY0 = residuals_FY0+(FY0_fz_nom_vec(i)-FY(i))^2;     
    end
    
    % Compute the residuals
    residuals_FY0 = residuals_FY0/sum(FY.^2);

    % R-squared is 
    % 1-SSE/SST
    % SSE/SST = residuals_FX0
    
    % SSE is the sum of squared error,  SST is the sum of squared total
    fprintf('Index of FY0 --> R-squared = %6.3f\n',1-residuals_FY0);


    % RMSE
    ALPHA_vec = TData0.SA;
    FY0_vec_pred = MF96_FY0_vec(zeros(size(ALPHA_vec)),ALPHA_vec,zeros(size(ALPHA_vec)),FZ0*ones(size(ALPHA_vec)),tyre_coeffs);
    E = rmse(FY0_vec_pred,TData0.FY);
    fprintf('Index for FY0 --> RMSE = %6.3f\n',E);
end


%% Fit coeefficients with variable load

% longitudinal case
if data_set == 'longi_case'
    %%Intersect tables to obtain specific sub-datasets
    % extract data with variable load
    [TDataDFz, ~] = intersect_table_data(SA_0, GAMMA_0);
    [TDataFz220, ~] = intersect_table_data(SA_0, GAMMA_0, FZ_220);
    [TDataFz700, ~] = intersect_table_data(SA_0, GAMMA_0, FZ_700);
    [TDataFz900, ~] = intersect_table_data(SA_0, GAMMA_0, FZ_900);
    [TDataFz1120, ~] = intersect_table_data(SA_0, GAMMA_0, FZ_1120);

    %%plot_selected_data
    figure('Name','Selected-data')
    plot_selected_data(TDataDFz);

    
    KAPPA_vec = TDataDFz.SL;
    FX_vec    = TDataDFz.FX;
    FZ_vec    = TDataDFz.FZ;
    zeros_vec = zeros(size(TDataDFz.SL));
    ones_vec  = ones(size(TDataDFz.SL));
    
    FX0_dfz_guess = MF96_FX0_vec(KAPPA_vec,zeros_vec , zeros_vec, FZ_vec, tyre_coeffs);
    
    % check guess 
    figure('Name','Raw Data Fx0(varFz)')
    plot(TDataDFz.SL,TDataDFz.FX,'.')
    hold on
    plot(TDataDFz.SL,FX0_dfz_guess,'-')
    title('Raw data Fx0(varFz) vs initial guess')
    

    % OPTIMIZATION
    % parameters to be optimised
    %  [pDx2 pEx2 pEx3 pHx2  pKx2  pKx3  pVx2] 
    P0 = [-1, 0.5, 1, 0.005, -0.04, 1, -0.1]; 
    
    % NOTE: many local minima => limits on parameters are fundamentals
    % Limits for parameters to be optimised
    % [pDx2 pEx2 pEx3 pHx2  pKx2  pKx3  pVx2] 
    lb = [-5, -2, -5, -0.1, -1, -4, -1];
    ub = [ 5,  2,  5,  0.1,  1,  4,  1];
    
    % LSM_pure_Fx returns the residual, so minimize the residual varying X. It
    % is an unconstrained minimization problem 
    [P_dfz,fval,exitflag] = fmincon(@(P)resid_pure_Fx_varFz(P,FX_vec, KAPPA_vec,0,FZ_vec, tyre_coeffs),...
                           P0,[],[],[],[],lb,ub);
    
    disp(exitflag)
    % Change tyre data with new optimal values
    tyre_coeffs.pDx2 = P_dfz(1) ; 
    tyre_coeffs.pEx2 = P_dfz(2) ;  
    tyre_coeffs.pEx3 = P_dfz(3) ;
    tyre_coeffs.pHx2 = P_dfz(4) ;
    tyre_coeffs.pKx2 = P_dfz(5) ; 
    tyre_coeffs.pKx3 = P_dfz(6) ;
    tyre_coeffs.pVx2 = P_dfz(7) ;
    
    
    SL_vec = -0.3:0.001:0.3;
    tmp_zeros = zeros(size(SL_vec));
    tmp_ones = ones(size(SL_vec));
    
    FX0_fz_var_vec1 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_220.FZ)*tmp_ones,tyre_coeffs);
    FX0_fz_var_vec2 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_700.FZ)*tmp_ones,tyre_coeffs);
    FX0_fz_var_vec3 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_900.FZ)*tmp_ones,tyre_coeffs);
    FX0_fz_var_vec4 = MF96_FX0_vec(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);
    
    figure('Name','Fx0 with variable Fz')
    plot(TDataFz220.SL,TDataFz220.FX,'.','MarkerSize',5,'Color','#0072BD')
    hold on
    plot(TDataFz700.SL,TDataFz700.FX,'.','MarkerSize',5,'Color','#D95319')
    plot(TDataFz900.SL,TDataFz900.FX,'.','MarkerSize',5,'Color','#EDB120')
    plot(TDataFz1120.SL,TDataFz1120.FX,'.','MarkerSize',5,'Color','#77AC30')
    plot(SL_vec,FX0_fz_var_vec1,'-','LineWidth',2,'Color','#0072BD')
    plot(SL_vec,FX0_fz_var_vec2,'-','LineWidth',2,'Color','#D95319')
    plot(SL_vec,FX0_fz_var_vec3,'-','LineWidth',2,'Color','#EDB120')
    plot(SL_vec,FX0_fz_var_vec4,'-','LineWidth',2,'Color','#77AC30')
    xlabel('$\kappa$ [-]')
    ylabel('$F_{x0}$ [N]')
    legend({'Raw $Fz_{220}$','Raw $Fz_{700}$','Raw $Fz_{900}$','Raw $Fz_{1120}$','$Fz_{220}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})
    title('Pure longitudinal slip at different vertical loads')



    % Compute index of performance

    % Calculate the residuals with the optimal solution found above
    [SL_vec, sortIdx] = sort(TDataDFz.SL,'ascend');
    FX = FX_vec(sortIdx);
    FX0_fz_var = MF96_FX0_vec(SL_vec,zeros(size(SL_vec)),zeros(size(SL_vec)),TDataDFz.FZ,tyre_coeffs);

    residuals_FX0_varFz = 0;
    for i=1:length(SL_vec)
       residuals_FX0_varFz = residuals_FX0_varFz+(FX0_fz_var(i)-FX(i))^2;     
    end
    
    % Compute the residuals
    residuals_FX0_varFz = residuals_FX0_varFz/sum(FX.^2);

    % R-squared is 
    % 1-SSE/SST
    % SSE/SST = residuals_FX0
    
    % SSE is the sum of squared error,  SST is the sum of squared total
    fprintf('Index of FX0 varFz --> R-squared = %6.3f\n',1-residuals_FX0_varFz);

    

    % RMSE
    SL_vec = TDataDFz.SL;
    FX0_vec_pred = MF96_FX0_vec(SL_vec,zeros(size(SL_vec)),zeros(size(SL_vec)),TDataDFz.FZ,tyre_coeffs);
    E = rmse(FX0_vec_pred,TDataDFz.FX);
    fprintf('Index for FX0 varFz --> RMSE = %6.3f\n',E);



    %%CORNERING STIFFNESS LONGITUDINAL CASE 
    SL_vec = -0.3:0.001:0.3;
    tmp_zeros = zeros(size(SL_vec));
    tmp_ones = ones(size(SL_vec));

    [kappa__x, Bx, Cx, Dx, Ex, SVx] =MF96_FX0_coeffs(0, 0, 0, mean(FZ_220.FZ), tyre_coeffs);
    Calfa_vec1_0 = magic_formula_stiffness(kappa__x, Bx, Cx, Dx, Ex, SVx);
    [kappa__x, Bx, Cx, Dx, Ex, SVx] =MF96_FX0_coeffs(0, 0, 0, mean(FZ_700.FZ), tyre_coeffs);
    Calfa_vec2_0 = magic_formula_stiffness(kappa__x, Bx, Cx, Dx, Ex, SVx);
    [kappa__x, Bx, Cx, Dx, Ex, SVx] =MF96_FX0_coeffs(0, 0, 0, mean(FZ_900.FZ), tyre_coeffs);
    Calfa_vec3_0 = magic_formula_stiffness(kappa__x, Bx, Cx, Dx, Ex, SVx);
    [kappa__x, Bx, Cx, Dx, Ex, SVx] =MF96_FX0_coeffs(0, 0, 0, mean(FZ_1120.FZ), tyre_coeffs);
    Calfa_vec4_0 = magic_formula_stiffness(kappa__x, Bx, Cx, Dx, Ex, SVx);
    
    Calfa_vec1 = MF96_CorneringStiffness(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_220.FZ)*tmp_ones,tyre_coeffs);
    Calfa_vec2 = MF96_CorneringStiffness(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_700.FZ)*tmp_ones,tyre_coeffs);
    Calfa_vec3 = MF96_CorneringStiffness(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_900.FZ)*tmp_ones,tyre_coeffs);
    Calfa_vec4 = MF96_CorneringStiffness(SL_vec,tmp_zeros ,tmp_zeros, mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);
    
    figure('Name','C_alpha1')
    hold on
    plot([0,mean(FZ_220.FZ)], [0,Calfa_vec1_0],'-k','LineWidth',1)
    plot([mean(FZ_220.FZ),mean(FZ_700.FZ)], [Calfa_vec1_0,Calfa_vec2_0],'-k','LineWidth',1)
    plot([mean(FZ_700.FZ),mean(FZ_900.FZ)], [Calfa_vec2_0,Calfa_vec3_0],'-k','LineWidth',1)
    plot([mean(FZ_900.FZ),mean(FZ_1120.FZ)], [Calfa_vec3_0,Calfa_vec4_0],'-k','LineWidth',1)
    plot(mean(FZ_220.FZ),Calfa_vec1_0,'+','MarkerSize',10,'LineWidth',3,'Color','#0072BD')
    plot(mean(FZ_700.FZ),Calfa_vec2_0,'+','MarkerSize',10,'LineWidth',3,'Color','#D95319')
    plot(mean(FZ_900.FZ),Calfa_vec3_0,'+','MarkerSize',10,'LineWidth',3,'Color','#EDB120')
    plot(mean(FZ_1120.FZ),Calfa_vec4_0,'+','MarkerSize',10,'LineWidth',3,'Color','#7E2F8E')
    legend({'','','','','$Fz_{220}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'},'Location','northwest')
    title('Cornering Stiffness function of vertical load')
    xlabel('Fz [N]');
    ylabel('$CF_{\alpha} [N/rad]$')
    
    
    figure('Name','C_alpha2')
    hold on
    plot(SL_vec,Calfa_vec1,'-','LineWidth',2)
    plot(SL_vec,Calfa_vec2,'-','LineWidth',2)
    plot(SL_vec,Calfa_vec3,'-','LineWidth',2)
    plot(SL_vec,Calfa_vec4,'-','LineWidth',2)
    legend({'$Fz_{220}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})
    title('Cornering Stiffness at different vertical load with variable long slip')
    xlabel('k [-]');
    ylabel('$CF_{\alpha}[N/rad]$')


elseif data_set == 'later_case'
    
    %%Intersect tables to obtain specific sub-datasets
    % extract data with variable load
    [TDataDFz, ~] = intersect_table_data( KAPPA_0, GAMMA_0);
    [TDataFz220, ~] = intersect_table_data(KAPPA_0, GAMMA_0, FZ_220);
    [TDataFz440, ~] = intersect_table_data(KAPPA_0, GAMMA_0, FZ_440);
    [TDataFz700, ~] = intersect_table_data(KAPPA_0, GAMMA_0, FZ_700);
    [TDataFz900, ~] = intersect_table_data(KAPPA_0, GAMMA_0, FZ_900);
    [TDataFz1120, ~] = intersect_table_data(KAPPA_0, GAMMA_0, FZ_1120);
    %%plot_selected_data
    figure('Name','Selected-data')
    plot_selected_data(TDataDFz);

    ALPHA_vec = TDataDFz.SA;
    FY_vec    = TDataDFz.FY;
    FZ_vec    = TDataDFz.FZ;
    zeros_vec = zeros(size(TDataDFz.SA));
    ones_vec  = ones(size(TDataDFz.SA));
    
    FY0_dfz_guess = MF96_FY0_vec(zeros_vec,ALPHA_vec, zeros_vec, FZ_vec, tyre_coeffs);
    
    % check guess 
    figure('Name','Raw Data Fy0(varFz)')
    plot(TDataDFz.SA,TDataDFz.FY,'.')
    hold on
    plot(TDataDFz.SA,FY0_dfz_guess,'-')
    title('Raw Data Fy0(varFz) vs initial guess')
    
    
    % OPTIMIZATION
    % parameters to be optimised
    % {pDy2, pEy2, pHy2, pVy2}
    P0 = [-0.5, -1, -0.001, -0.05];
    
    % NOTE: many local minima => limits on parameters are fundamentals
    % Limits for parameters to be optimised
    %{pCy1, pDy1, pEy1, pHy1, pKy1, pKy2, pVy1, pDy2, pEy2, pHy2, pVy2}
    lb = [-1, -3, -1, -1];
    ub = [ 1,  3,  1,  1];
    
    % LSM_pure_Fx returns the residual, so minimize the residual varying X. It
    % is an unconstrained minimization problem 
    [P_dfz,fval,exitflag] = fmincon(@(P)resid_pure_Fy_varFz(P,FY_vec, ALPHA_vec,0,FZ_vec, tyre_coeffs),...
                           P0,[],[],[],[],lb,ub);
    
    disp(exitflag)
    % Change tyre data with new optimal values
    tyre_coeffs.pDy2 = P_dfz(1) ;
    tyre_coeffs.pEy2 = P_dfz(2) ; 
    tyre_coeffs.pHy2 = P_dfz(3) ;
    tyre_coeffs.pVy2 = P_dfz(4) ;
    

    SA_vec = -0.3:0.001:0.3;
    tmp_zeros = zeros(size(SA_vec));
    tmp_ones = ones(size(SA_vec));
    
    FY0_fz_var_vec1 = MF96_FY0_vec(tmp_zeros,SA_vec ,tmp_zeros, mean(FZ_220.FZ)*tmp_ones,tyre_coeffs);
    FY0_fz_var_vec2 = MF96_FY0_vec(tmp_zeros,SA_vec ,tmp_zeros, mean(FZ_440.FZ)*tmp_ones,tyre_coeffs);
    FY0_fz_var_vec3 = MF96_FY0_vec(tmp_zeros,SA_vec ,tmp_zeros, mean(FZ_700.FZ)*tmp_ones,tyre_coeffs);
    FY0_fz_var_vec4 = MF96_FY0_vec(tmp_zeros,SA_vec ,tmp_zeros, mean(FZ_900.FZ)*tmp_ones,tyre_coeffs);
    FY0_fz_var_vec5 = MF96_FY0_vec(tmp_zeros,SA_vec ,tmp_zeros, mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);
    
    figure('Name','Fy0 with variable Fz')
    plot(TDataFz220.SA,TDataFz220.FY,'.','MarkerSize',5,'Color','#0072BD')
    hold on
    plot(TDataFz440.SA,TDataFz440.FY,'.','MarkerSize',5,'Color','#7E2F8E')
    plot(TDataFz700.SA,TDataFz700.FY,'.','MarkerSize',5,'Color','#D95319')
    plot(TDataFz900.SA,TDataFz900.FY,'.','MarkerSize',5,'Color','#EDB120')
    plot(TDataFz1120.SA,TDataFz1120.FY,'.','MarkerSize',5,'Color','#77AC30')
    plot(SA_vec,FY0_fz_var_vec1,'-','LineWidth',2,'Color','#0072BD')
    plot(SA_vec,FY0_fz_var_vec2,'-','LineWidth',2,'Color','#7E2F8E')
    plot(SA_vec,FY0_fz_var_vec3,'-','LineWidth',2,'Color','#D95319')
    plot(SA_vec,FY0_fz_var_vec4,'-','LineWidth',2,'Color','#EDB120')
    plot(SA_vec,FY0_fz_var_vec5,'-','LineWidth',2,'Color','#77AC30')
    xlabel('$\alpha$ [rad]')
    ylabel('$F_{y0}$ [N]')
    legend({'Raw $Fz_{220}$','Raw $Fz_{440}$','Raw $Fz_{700}$','Raw $Fz_{900}$','Raw $Fz_{1120}$','$Fz_{220}$','$Fz_{440}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})
    title('Pure lateral slip at different vertical loads')


    
    % Compute index of performance

    % Calculate the residuals with the optimal solution found above
    [SA_vec, sortIdx] = sort(TDataDFz.SA,'ascend');
    FY = FY_vec(sortIdx);
    FY0_fz_var = MF96_FY0_vec(zeros(size(SA_vec)),SA_vec ,zeros(size(SA_vec)),TDataDFz.FZ,tyre_coeffs);

    residuals_FY0_varFz = 0;
    for i=1:length(SA_vec)
       residuals_FY0_varFz = residuals_FY0_varFz+(FY0_fz_var(i)-FY(i))^2;     
    end
    
    % Compute the residuals
    residuals_FY0_varFz = residuals_FY0_varFz/sum(FY.^2);

    % R-squared is 
    % 1-SSE/SST
    % SSE/SST = residuals_FY0
    
    % SSE is the sum of squared error,  SST is the sum of squared total
    fprintf('Index of FY0 varFz--> R-squared = %6.3f\n',1-residuals_FY0_varFz);

    
    % RMSE
    SA_vec = TDataDFz.SA;
    FY0_vec_pred =  MF96_FY0_vec(zeros(size(SA_vec)),SA_vec ,zeros(size(SA_vec)),TDataDFz.FZ,tyre_coeffs);
    E = rmse(FY0_vec_pred,TDataDFz.FY);
    fprintf('Index for FY0 varFz --> RMSE = %6.3f\n',E);


    
    %%CORNERING STIFFNESS LATERAL CASE
    SA_vec = -0.3:0.001:0.3;
    tmp_zeros = zeros(size(SA_vec));
    tmp_ones = ones(size(SA_vec));

    [alpha__y, By, Cy, Dy, Ey, SVy, Kya, SHy, mu__y] = MF96_FY0_coeffs(0, 0, 0, mean(FZ_220.FZ), tyre_coeffs);
    Calfa_vec1_0 = magic_formula_stiffness(alpha__y, By, Cy, Dy, Ey, SVy);
    [alpha__y, By, Cy, Dy, Ey, SVy, Kya, SHy, mu__y] = MF96_FY0_coeffs(0, 0, 0, mean(FZ_440.FZ), tyre_coeffs);
    Calfa_vec2_0 = magic_formula_stiffness(alpha__y, By, Cy, Dy, Ey, SVy);
    [alpha__y, By, Cy, Dy, Ey, SVy, Kya, SHy, mu__y] = MF96_FY0_coeffs(0, 0, 0, mean(FZ_700.FZ), tyre_coeffs);
    Calfa_vec3_0 = magic_formula_stiffness(alpha__y, By, Cy, Dy, Ey, SVy);
    [alpha__y, By, Cy, Dy, Ey, SVy, Kya, SHy, mu__y] = MF96_FY0_coeffs(0, 0, 0, mean(FZ_900.FZ), tyre_coeffs);
    Calfa_vec4_0 = magic_formula_stiffness(alpha__y, By, Cy, Dy, Ey, SVy);
    [alpha__y, By, Cy, Dy, Ey, SVy, Kya, SHy, mu__y] = MF96_FY0_coeffs(0, 0, 0, mean(FZ_1120.FZ), tyre_coeffs);
    Calfa_vec5_0 = magic_formula_stiffness(alpha__y, By, Cy, Dy, Ey, SVy);
        
    figure('Name','C_alpha1')
    hold on
    plot([0,mean(FZ_220.FZ)], [0,Calfa_vec1_0],'-k','LineWidth',1)
    plot([mean(FZ_220.FZ),mean(FZ_440.FZ)], [Calfa_vec1_0,Calfa_vec2_0],'-k','LineWidth',1)
    plot([mean(FZ_440.FZ),mean(FZ_700.FZ)], [Calfa_vec2_0,Calfa_vec3_0],'-k','LineWidth',1)
    plot([mean(FZ_700.FZ),mean(FZ_900.FZ)], [Calfa_vec3_0,Calfa_vec4_0],'-k','LineWidth',1)
    plot([mean(FZ_900.FZ),mean(FZ_1120.FZ)], [Calfa_vec4_0,Calfa_vec5_0],'-k','LineWidth',1)
    plot(mean(FZ_220.FZ),Calfa_vec1_0,'+','MarkerSize',10,'LineWidth',3,'Color','#0072BD')
    plot(mean(FZ_440.FZ),Calfa_vec2_0,'+','MarkerSize',10,'LineWidth',3,'Color','#77AC30')
    plot(mean(FZ_700.FZ),Calfa_vec3_0,'+','MarkerSize',10,'LineWidth',3,'Color','#D95319')
    plot(mean(FZ_900.FZ),Calfa_vec4_0,'+','MarkerSize',10,'LineWidth',3,'Color','#EDB120')
    plot(mean(FZ_1120.FZ),Calfa_vec5_0,'+','MarkerSize',10,'LineWidth',3,'Color','#7E2F8E')
    legend({'','','','','','$Fz_{220}$','$Fz_{440}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'},'Location','northwest')
    title('Cornering Stiffness function of vertical load')
    xlabel('Fz [N]');
    ylabel('$CF_{\alpha} [N/rad]$', 'Interpreter','latex')
    
    
    Calfa_vec1 = MF96_CorneringStiffness(SA_vec,tmp_zeros ,tmp_zeros, mean(FZ_220.FZ)*tmp_ones,tyre_coeffs);
    Calfa_vec2 = MF96_CorneringStiffness(SA_vec,tmp_zeros ,tmp_zeros, mean(FZ_440.FZ)*tmp_ones,tyre_coeffs);
    Calfa_vec3 = MF96_CorneringStiffness(SA_vec,tmp_zeros ,tmp_zeros, mean(FZ_700.FZ)*tmp_ones,tyre_coeffs);
    Calfa_vec4 = MF96_CorneringStiffness(SA_vec,tmp_zeros ,tmp_zeros, mean(FZ_900.FZ)*tmp_ones,tyre_coeffs);
    Calfa_vec5 = MF96_CorneringStiffness(SA_vec,tmp_zeros ,tmp_zeros, mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);


    figure('Name','C_alpha2')
    hold on
    plot(SA_vec,Calfa_vec1,'-','LineWidth',2)
    plot(SA_vec,Calfa_vec2,'-','LineWidth',2)
    plot(SA_vec,Calfa_vec3,'-','LineWidth',2)
    plot(SA_vec,Calfa_vec4,'-','LineWidth',2)
    plot(SA_vec,Calfa_vec5,'-','LineWidth',2)
    legend({'$Fz_{220}$','$Fz_{440}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})
    title('Cornering Stiffness at different vertical load with variable side slip')
    xlabel('$\alpha [rad]$', 'Interpreter','latex')
    ylabel('$CF_{\alpha}[N/rad]$', 'Interpreter','latex')

    

end


%% Fit coefficient with variable camber

if data_set == 'longi_case'
    %%Intersect tables to obtain specific sub-datasets
    % extract data with variable camber
    [TDataGamma, ~] = intersect_table_data( SA_0, FZ_table);
    [TDataGamma0, ~] = intersect_table_data( SA_0, FZ_table, GAMMA_0);
    [TDataGamma2, ~] = intersect_table_data( SA_0, FZ_table, GAMMA_2);
    [TDataGamma4, ~] = intersect_table_data( SA_0, FZ_table, GAMMA_4);
    %%plot_selected_data
    figure('Name','Data with variable camber')
    plot_selected_data(TDataGamma);
       
    zeros_vec = zeros(size(TDataGamma.SL));
    ones_vec  = ones(size(TDataGamma.SL));
    
    KAPPA_vec = TDataGamma.SL;
    GAMMA_vec = TDataGamma.IA; 
    FX_vec    = TDataGamma.FX;
    
    FZ0 = mean(TDataGamma.FZ);
    
    FX0_varGamma_guess = MF96_FX0_vec(KAPPA_vec,zeros_vec , GAMMA_vec, FZ0*ones_vec, tyre_coeffs);
    
    % check guess 
    figure('Name','Raw Fx0 with variable Gamma')
    plot(TDataGamma.SL,TDataGamma.FX,'.')
    hold on
    plot(TDataGamma.SL,FX0_varGamma_guess,'-')
    title('Raw Data Fx0(gamma) vs initial guess')
    
    % PARAMETERS' OPRIMIZATION
    
    % Fit the coeffs { pDx3}
    % Guess values for parameters to be optimised
    P0 = 5; 
    
    % NOTE: many local minima => limits on parameters are fundamentals
    % Limits for parameters to be optimised
    lb = -10;
    ub =  50;


    % LSM_pure_Fx returns the residual, so minimize the residual varying X. It
    % is an unconstrained minimization problem 
    [P_varGamma,fval,exitflag] = fmincon(@(P)resid_pure_Fx_varGamma(P,FX_vec, KAPPA_vec,GAMMA_vec,FZ0, tyre_coeffs),...
                                   P0,[],[],[],[],lb,ub);
    
    % Change tyre data with new optimal values                             
    tyre_coeffs.pDx3 = P_varGamma(1) ; % 1
   
    SL_vec = -0.3:0.001:0.3;
    FX0_varGamma_0_vec = MF96_FX0_vec(SL_vec,zeros_vec , mean(GAMMA_0.IA)*ones_vec, FZ0*ones_vec, tyre_coeffs);
    FX0_varGamma_2_vec = MF96_FX0_vec(SL_vec,zeros_vec , mean(GAMMA_2.IA)*ones_vec, FZ0*ones_vec, tyre_coeffs);
    FX0_varGamma_4_vec = MF96_FX0_vec(SL_vec,zeros_vec , mean(GAMMA_4.IA)*ones_vec, FZ0*ones_vec, tyre_coeffs);


    figure('Name','Fx0 with variable Gamma')
    plot(TDataGamma0.SL,TDataGamma0.FX,'.','MarkerSize',5,'Color','#0072BD')
    hold on
    plot(TDataGamma2.SL,TDataGamma2.FX,'.','MarkerSize',5,'Color','#EDB120')
    plot(TDataGamma4.SL,TDataGamma4.FX,'.','MarkerSize',5,'Color','#77AC30')
    plot(SL_vec,FX0_varGamma_0_vec,'-','LineWidth',2,'Color','#0072BD')
    plot(SL_vec,FX0_varGamma_2_vec,'-','LineWidth',2,'Color','#EDB120')
    plot(SL_vec,FX0_varGamma_4_vec,'-','LineWidth',2,'Color','#77AC30')
    xlabel('$\kappa$ [-]')
    ylabel('$F_{x0}$ [N]')
    titolo = sprintf('Pure longitudinal force at different cambers, Fz = %d N, zero slip angle, press = 12psi',Fz0);
    title(titolo)
    legend('Raw $\gamma = 0 \deg$','Raw $\gamma = 2 \deg$','Raw $\gamma = 4 \deg$','$\gamma = 0 \deg$','$\gamma = 2 \deg$','$\gamma = 4 \deg$', 'Interpreter','latex')



    
    % Compute index of performance

    % Calculate the residuals with the optimal solution found above
    [SL_vec, sortIdx] = sort(TDataGamma.SL,'ascend');
    FX = FX_vec(sortIdx);
    FX0_varGamma = MF96_FX0_vec(SL_vec,zeros(size(SL_vec)),TDataGamma.IA,FZ0*ones(size(SL_vec)),tyre_coeffs);

    residuals_FX0_varGamma = 0;
    for i=1:length(SL_vec)
       residuals_FX0_varGamma = residuals_FX0_varGamma+(FX0_varGamma(i)-FX(i))^2;     
    end
    
    % Compute the residuals
    residuals_FX0_varGamma = residuals_FX0_varGamma/sum(FX.^2);

    % R-squared is 
    % 1-SSE/SST
    % SSE/SST = residuals_FX0
    
    % SSE is the sum of squared error,  SST is the sum of squared total
    fprintf('Index of FX0 varGamma --> R-squared = %6.3f\n',1-residuals_FX0_varGamma);


    % RMSE
    SL_vec = TDataGamma.SL;
    FX0_vec_pred =  MF96_FX0_vec(SL_vec,zeros(size(SL_vec)),TDataGamma.IA,FZ0*ones(size(SL_vec)),tyre_coeffs);
    E = rmse(FX0_vec_pred,TDataGamma.FX);
    fprintf('Index for FX0 varGamma --> RMSE = %6.3f\n',E);
    

elseif data_set == 'later_case'
    
    %%Intersect tables to obtain specific sub-datasets
    % extract data with variable camber
    [TDataGamma, ~] = intersect_table_data( KAPPA_0, FZ_table );
    [TDataGamma0, ~] = intersect_table_data( KAPPA_0, FZ_table, GAMMA_0);
    [TDataGamma1, ~] = intersect_table_data( KAPPA_0, FZ_table, GAMMA_1);
    [TDataGamma2, ~] = intersect_table_data( KAPPA_0, FZ_table, GAMMA_2);
    [TDataGamma3, ~] = intersect_table_data( KAPPA_0, FZ_table, GAMMA_3);
    [TDataGamma4, ~] = intersect_table_data( KAPPA_0, FZ_table, GAMMA_4);
    %%plot_selected_data
    figure('Name','Data with variable camber')
    plot_selected_data(TDataGamma);

       
    zeros_vec = zeros(size(TDataGamma.SA));
    ones_vec  = ones(size(TDataGamma.SA));
    
    ALPHA_vec = TDataGamma.SA;
    GAMMA_vec = TDataGamma.IA; 
    FY_vec    = TDataGamma.FY;
    
    FZ0 = mean(TDataGamma.FZ);
    
    FY0_varGamma_guess = MF96_FY0_vec(zeros_vec,ALPHA_vec ,GAMMA_vec, FZ0*ones_vec, tyre_coeffs);
    
    % check guess 
    figure('Name','Raw Data Fy0, varGamma')
    plot(TDataGamma.SA,TDataGamma.FY,'.')
    hold on
    plot(TDataGamma.SA,FY0_varGamma_guess,'-')
    title('Raw Data Fy0(gamma) vs Initial guess')

    
    % OPTIMIZATION
    
    % Fit the coeffs {pDy3, pEy3, pEy4, pHy3, pKy3, pVy3, pVy4}
    % Guess values for parameters to be optimised
    % {pDy3, pEy3, pEy4, pHy3, pKy3, pVy3, pVy4}
    P0 = [10, 0.1, -5, -0.05, 0.5, -2, -1]; 
    
    % NOTE: many local minima => limits on parameters are fundamentals
    % Limits for parameters to be optimised
    %    {pDy3, pEy3, pEy4, pHy3, pKy3, pVy3, pVy4}
    lb = [ 0, -1, -10,  -1,  -3, -5, -4];
    ub = [50,  2,   0, 0.1,   3,  2,  2];


    % LSM_pure_Fx returns the residual, so minimize the residual varying X. It
    % is an unconstrained minimization problem 
    [P_varGamma,fval,exitflag] = fmincon(@(P)resid_pure_Fy_varGamma(P,FY_vec,ALPHA_vec,GAMMA_vec,FZ0, tyre_coeffs),...
                                   P0,[],[],[],[],lb,ub);
    
    % Change tyre data with new optimal values                             
    tyre_coeffs.pDy3 = P_varGamma(1);
    tyre_coeffs.pEy3 = P_varGamma(2);
    tyre_coeffs.pEy4 = P_varGamma(3);
    tyre_coeffs.pHy3 = P_varGamma(4);
    tyre_coeffs.pKy3 = P_varGamma(5);
    tyre_coeffs.pVy3 = P_varGamma(6);
    tyre_coeffs.pVy4 = P_varGamma(7);
   
    SA_vec = -0.3:0.001:0.3;
    tmp_zeros = zeros(size(SA_vec));
    tmp_ones = ones(size(SA_vec));

    FY0_varGamma_0_vec = MF96_FY0_vec(tmp_zeros,SA_vec , mean(GAMMA_0.IA)*tmp_ones, FZ0*tmp_ones,tyre_coeffs);
    FY0_varGamma_1_vec = MF96_FY0_vec(tmp_zeros,SA_vec , mean(GAMMA_1.IA)*tmp_ones, FZ0*tmp_ones,tyre_coeffs);
    FY0_varGamma_2_vec = MF96_FY0_vec(tmp_zeros,SA_vec , mean(GAMMA_2.IA)*tmp_ones, FZ0*tmp_ones,tyre_coeffs);
    FY0_varGamma_3_vec = MF96_FY0_vec(tmp_zeros,SA_vec , mean(GAMMA_3.IA)*tmp_ones, FZ0*tmp_ones,tyre_coeffs);
    FY0_varGamma_4_vec = MF96_FY0_vec(tmp_zeros,SA_vec , mean(GAMMA_4.IA)*tmp_ones, FZ0*tmp_ones,tyre_coeffs);


    figure('Name','Fy0, variable camber')
    plot(TDataGamma0.SA,TDataGamma0.FY,'.','MarkerSize',5,'Color','#0072BD')
    hold on
    plot(TDataGamma1.SA,TDataGamma1.FY,'.','MarkerSize',5,'Color','#D95319')
    plot(TDataGamma2.SA,TDataGamma2.FY,'.','MarkerSize',5,'Color','#EDB120')
    plot(TDataGamma3.SA,TDataGamma3.FY,'.','MarkerSize',5,'Color','#7E2F8E')
    plot(TDataGamma4.SA,TDataGamma4.FY,'.','MarkerSize',5,'Color','#77AC30')
    plot(SA_vec,FY0_varGamma_0_vec,'-','LineWidth',2,'Color','#0072BD')
    plot(SA_vec,FY0_varGamma_1_vec,'-','LineWidth',2,'Color','#D95319')
    plot(SA_vec,FY0_varGamma_2_vec,'-','LineWidth',2,'Color','#EDB120')
    plot(SA_vec,FY0_varGamma_3_vec,'-','LineWidth',2,'Color','#7E2F8E')
    plot(SA_vec,FY0_varGamma_4_vec,'-','LineWidth',2,'Color','#77AC30')
    xlabel('$\alpha$ [rad]')
    ylabel('$F_{y0}$ [N]')
    titolo = sprintf('Pure lateral slip at different cambers, Fz = %d N, k = 0, press = 12psi', Fz0);
    title(titolo)
    legend('Raw $\gamma = 0 \deg$','Raw $\gamma = 1 \deg$','Raw $\gamma = 2 \deg$','Raw $\gamma = 3 \deg$','Raw $\gamma = 4 \deg$','$\gamma = 0 \deg$','$\gamma = 1 \deg$','$\gamma = 2 \deg$','$\gamma = 3 \deg$','$\gamma = 4 \deg$', 'Interpreter','latex')



    % Compute index of performance

    % Calculate the residuals with the optimal solution found above
    [SA_vec, sortIdx] = sort(TDataGamma.SA,'ascend');
    FY = FY_vec(sortIdx);
    FY0_varGamma = MF96_FY0_vec(zeros(size(SA_vec)),SA_vec,TDataGamma.IA,FZ0*ones(size(SA_vec)),tyre_coeffs);

    residuals_FY0_varGamma = 0;
    for i=1:length(SA_vec)
       residuals_FY0_varGamma = residuals_FY0_varGamma+(FY0_varGamma(i)-FY(i))^2;     
    end
    
    % Compute the residuals
    residuals_FY0_varGamma = residuals_FY0_varGamma/sum(FY.^2);

    % R-squared is 
    % 1-SSE/SST
    % SSE/SST = residuals_FY0
    
    % SSE is the sum of squared error,  SST is the sum of squared total
    fprintf('Index of FY0 varGamma --> R-squared = %6.3f\n',1-residuals_FY0_varGamma);


    % RMSE
    SA_vec = TDataGamma.SA;
    FY0_vec_pred =  MF96_FY0_vec(zeros(size(SA_vec)),SA_vec,TDataGamma.IA,FZ0*ones(size(SA_vec)),tyre_coeffs);
    E = rmse(TDataGamma.FY,FY0_vec_pred);
    fprintf('Index for FY0 varGamma --> RMSE = %6.3f\n',E);

end


%% Combined slip, LONGITUDINAL force

if data_set == 'longi_case'

    %%Intersect tables to obtain specific sub-datasets
    % extract data with combined behaviour
    [TDataCombined, ~] = intersect_table_data(GAMMA_0, FZ_table);
    [TDataCombined0, ~] = intersect_table_data(SA_0, GAMMA_0, FZ_table);
    [TDataCombined3, ~] = intersect_table_data(SA_3neg, GAMMA_0, FZ_table);
    [TDataCombined6, ~] = intersect_table_data(SA_6neg, GAMMA_0, FZ_table);
    %%plot_selected_data
    figure('Name','Combined Behaviour DATA')
    plot_selected_data(TDataCombined);

    % Combined slip, LONGITUDINAL force, variable slip angle
    zeros_vec = zeros(size(TDataCombined.SL));
    ones_vec  = ones(size(TDataCombined.SL));
    
    ALPHA_vec = TDataCombined.SA;
    KAPPA_vec = TDataCombined.SL;
    GAMMA_vec = TDataCombined.IA; 
    FX_vec    = TDataCombined.FX;
    FZ0       = mean(TDataCombined.FZ);
    FZ0_vec   = FZ0*ones_vec;

    FX_SA_guess = MF96_FX_vec(KAPPA_vec, ALPHA_vec, zeros_vec, FZ0_vec, tyre_coeffs);
    
    figure('Name', 'Raw data Fx combined');
    plot(KAPPA_vec,FX_vec,'o');
    hold on;
    plot(KAPPA_vec,FX_SA_guess);
    title('Raw combined Fx vs Initial Guess')
    
    
    % PARAMETERS' OPRIMIZATION
    
    % Fit the coeffs { rBx1, rBx2, rCx1, rHx1}
    % Guess values for parameters to be optimised
    P0 = [15, 16, 1, -0.001]; 
    
    % NOTE: many local minima => limits on parameters are fundamentals
    % Limits for parameters to be optimised
    lb = [10,  10, -4, -1];
    ub = [25,  25,  4,  1];


    % LSM_pure_Fx returns the residual, so minimize the residual varying X. It
    % is an unconstrained minimization problem 
    [P_Fx_comb,fval,exitflag] = fmincon(@(P)resid_pure_Fx_comb(P,FX_vec,KAPPA_vec,ALPHA_vec, zeros_vec,FZ0,tyre_coeffs),...
                                   P0,[],[],[],[],lb,ub);
    
    % Change tyre data with new optimal values                             
    tyre_coeffs.rBx1 = P_Fx_comb(1); 
    tyre_coeffs.rBx2 = P_Fx_comb(2);
    tyre_coeffs.rCx1 = P_Fx_comb(3);
    tyre_coeffs.rHx1 = P_Fx_comb(4);

    KAPPA_vec = -0.3:0.001:0.3;
    tmp_zeros = zeros(size(KAPPA_vec));
    tmp_ones = ones(size(KAPPA_vec));
   
    FX_SA0_vec = MF96_FX_vec(KAPPA_vec, mean(SA_0.SA)*tmp_ones, tmp_zeros, FZ0*tmp_ones, tyre_coeffs);
    FX_SA3_vec = MF96_FX_vec(KAPPA_vec, mean(SA_3neg.SA)*tmp_ones, tmp_zeros, FZ0*tmp_ones, tyre_coeffs);
    FX_SA6_vec = MF96_FX_vec(KAPPA_vec, mean(SA_6neg.SA)*tmp_ones, tmp_zeros, FZ0*tmp_ones, tyre_coeffs);


    figure('Name','Combined Fx')
    plot(TDataCombined0.SL,TDataCombined0.FX,'.','MarkerSize',5,'Color',"#0072BD")
    hold on
    plot(TDataCombined3.SL,TDataCombined3.FX,'.','MarkerSize',5,'Color',"#D95319")
    plot(TDataCombined6.SL,TDataCombined6.FX,'.','MarkerSize',5,'Color',"#EDB120")
    plot(KAPPA_vec,FX_SA0_vec,'-','LineWidth',2,'Color',"#0072BD")
    plot(KAPPA_vec,FX_SA3_vec,'-','LineWidth',2,'Color',"#D95319")
    plot(KAPPA_vec,FX_SA6_vec,'-','LineWidth',2,'Color',"#EDB120")
    xlabel('$\kappa$ [-]')
    ylabel('$F_{x}$ [N]')
    titolo = sprintf('Combined slip, longitudinal force Fx, different side slip angle, camber = 0, Fz = %d N, press = 12psi',Fz0);
    title(titolo)
    legend('Raw $\alpha$ = 0 deg','Raw $\alpha$ = 3 deg','Raw $\alpha$ = 6 deg','$\alpha$ = 0 deg','$\alpha$ = 3 deg','$\alpha$ = 6 deg', 'Interpreter', 'latex')
    

    SL_vec = -1:0.001:1;
    Gxa_0 = zeros(size(SL_vec));
    for i = 1:length(SL_vec)
        [Gxa_0(i), ~, ~] = MF96_FXFYCOMB_coeffs(SL_vec(i), 0*to_rad, 0, FZ0, tyre_coeffs);
    end

    Gxa_3 = zeros(size(SL_vec));
    for i = 1:length(SL_vec)
        [Gxa_3(i), ~, ~] = MF96_FXFYCOMB_coeffs(SL_vec(i), -3*to_rad, 0, FZ0, tyre_coeffs);
    end

    Gxa_6 = zeros(size(SL_vec));
    for i = 1:length(SL_vec)
        [Gxa_6(i), ~, ~] = MF96_FXFYCOMB_coeffs(SL_vec(i), -6*to_rad, 0, FZ0, tyre_coeffs);
    end


    figure('Name','Gxa');
    plot(SL_vec, Gxa_0,'-','LineWidth',2)
    hold on
    plot(SL_vec, Gxa_3,'-','LineWidth',2)
    plot(SL_vec, Gxa_6,'-','LineWidth',2)
    xlabel('$k$ [-]')
    ylabel('$G_{xa}$ [-]')
    legend('$\alpha = 0 deg$','$\alpha = -3 deg$','$\alpha = -6 deg$','Interpreter','latex')
    title('Weighting function $G_{xa}$ as a function of k')


    % Compute index of performance

    % Calculate the residuals with the optimal solution found above
    [SL_vec, sortIdx] = sort(TDataCombined.SL,'ascend');
    FX = FX_vec(sortIdx);
    FX_vec = MF96_FX_vec(SL_vec,TDataCombined.SA,zeros(size(SL_vec)),FZ0*ones(size(SL_vec)),tyre_coeffs);

    residuals_FX = 0;
    for i=1:length(SL_vec)
       residuals_FX = residuals_FX+(FX_vec(i)-FX(i))^2;     
    end
    
    % Compute the residuals
    residuals_FX = residuals_FX/sum(FX.^2);

    % R-squared is 
    % 1-SSE/SST
    % SSE/SST = residuals_FX0
    
    % SSE is the sum of squared error,  SST is the sum of squared total
    fprintf('Index of Fx combined --> R-squared = %6.3f\n',1-residuals_FX);
    

    % RMSE 
    FX_vec_pred = MF96_FX_vec(TDataCombined.SL, TDataCombined.SA, zeros(size(TDataCombined.SL)), FZ0*ones(size(TDataCombined.SL)), tyre_coeffs);
    
    E = rmse(FX_vec_pred,TDataCombined.FX);

    fprintf('Index for Fx combined --> RMSE = %6.3f\n',E);

end

%% Combined slip, LATERAL force
 
if data_set == 'longi_case'

    %%Intersect tables to obtain specific sub-datasets
    % extract data with combined behaviour
    [TDataCombined, ~] = intersect_table_data(GAMMA_0, FZ_table);
    [TDataCombined0, ~] = intersect_table_data(SA_0, GAMMA_0, FZ_table);
    [TDataCombined3, ~] = intersect_table_data(SA_3neg,GAMMA_0, FZ_table);
    [TDataCombined6, ~] = intersect_table_data(SA_6neg, GAMMA_0, FZ_table);
    %%plot_selected_data
    figure('Name','Combined Behaviour DATA')
    plot_selected_data(TDataCombined);
          
    zeros_vec = zeros(size(TDataCombined.SL));
    ones_vec  = ones(size(TDataCombined.SL));
    
    ALPHA_vec = TDataCombined.SA;
    KAPPA_vec = TDataCombined.SL; 
    FY_vec    = TDataCombined.FY;
    FZ0       = mean(TDataCombined.FZ);   

    figure('Name','Raw Combined Fy');
    plot(KAPPA_vec,FY_vec,'.');
    

    % PARAMETERS' OPRIMIZATION

    % Fit the coeffs {rBy1, rBy2, rBy3, rCy1, rHy1, rVy1, rVy4, rVy5, rVy6}
    % Guess values for parameters to be optimised
    P0 =[ 14, 13, -0.5, 1, 0.02, -0.2, 4, -0.1, 28]; 
    
    % NOTE: many local minima => limits on parameters are fundamentals
    % Limits for parameters to be optimised
    lb = [ 0, 0, -3, -5, -0.5, -2, -8, -3, 15];
    ub = [30, 30, 3,  5,  0.5,  2,  8,  3, 60];


    % LSM_pure_Fx returns the residual, so minimize the residual varying X. It
    % is an unconstrained minimization problem 
    [P_FY_comb,fval,exitflag] = fmincon(@(P)resid_pure_Fy_comb(P,FY_vec,KAPPA_vec,ALPHA_vec,zeros_vec,FZ0,tyre_coeffs),...
                                  P0,[],[],[],[],lb,ub);

    
    % Change tyre data with new optimal values                             
    tyre_coeffs.rBy1 = P_FY_comb(1); 
    tyre_coeffs.rBy2 = P_FY_comb(2);
    tyre_coeffs.rBy3 = P_FY_comb(3);
    tyre_coeffs.rCy1 = P_FY_comb(4);
    tyre_coeffs.rHy1 = P_FY_comb(5);
    tyre_coeffs.rVy1 = P_FY_comb(6);
    tyre_coeffs.rVy4 = P_FY_comb(7);
    tyre_coeffs.rVy5 = P_FY_comb(8);
    tyre_coeffs.rVy6 = P_FY_comb(9);

    
    KAPPA_vec = -0.3:0.001:0.3;
    tmp_zeros = zeros(size(KAPPA_vec));
    tmp_ones = ones(size(KAPPA_vec));

    FY_0_vec = MF96_FY_vec(KAPPA_vec,    mean(SA_0.SA)*tmp_ones, tmp_zeros, FZ0*tmp_ones, tyre_coeffs);
    FY_3_vec = MF96_FY_vec(KAPPA_vec, mean(SA_3neg.SA)*tmp_ones, tmp_zeros, FZ0*tmp_ones, tyre_coeffs);
    FY_6_vec = MF96_FY_vec(KAPPA_vec, mean(SA_6neg.SA)*tmp_ones, tmp_zeros, FZ0*tmp_ones, tyre_coeffs);


    figure('Name','Combined Fy')
    plot(TDataCombined0.SL,TDataCombined0.FY,'.','MarkerSize',5,'Color',"#0072BD")
    hold on
    plot(TDataCombined3.SL,TDataCombined3.FY,'.','MarkerSize',5,'Color',"#D95319")
    plot(TDataCombined6.SL,TDataCombined6.FY,'.','MarkerSize',5,'Color',"#EDB120")
    plot(KAPPA_vec,FY_0_vec,'-','LineWidth',2,'Color',"#0072BD")
    plot(KAPPA_vec,FY_3_vec,'-','LineWidth',2,'Color',"#D95319")
    plot(KAPPA_vec,FY_6_vec,'-','LineWidth',2,'Color',"#EDB120")
    xlabel('$\kappa$ [-]')
    ylabel('$F_{y}$ [N]')
    titolo = sprintf('Combined slip, lateral force Fy, different side slip angle, camber = 0, Fz = %d N, press = 12psi',Fz0);
    title(titolo)
    legend('Raw $\alpha$ = 0 deg','Raw $\alpha$ = -3 deg','Raw $\alpha$ = -6 deg','$\alpha$ = 0 deg','$\alpha$ = -3 deg','$\alpha$ = -6 deg','Interpreter','latex')



    SL_vec = -1:0.001:1;
    Gyk_0 = zeros(size(SL_vec));
    for i = 1:length(SL_vec)
        [~, Gyk_0(i), ~] = MF96_FXFYCOMB_coeffs(SL_vec(i),  mean(SA_0.SA), 0, FZ0, tyre_coeffs);
    end

    Gyk_3 = zeros(size(SL_vec));
    for i = 1:length(SL_vec)
        [~, Gyk_3(i), ~] = MF96_FXFYCOMB_coeffs(SL_vec(i), mean(SA_3neg.SA), 0, FZ0, tyre_coeffs);
    end


    Gyk_6 = zeros(size(SL_vec));
    for i = 1:length(SL_vec)
        [~, Gyk_6(i), ~] = MF96_FXFYCOMB_coeffs(SL_vec(i), mean(SA_6neg.SA), 0, FZ0, tyre_coeffs);
    end


    figure('Name','Gyk');
    plot(SL_vec, Gyk_0,'-','LineWidth',2)
    hold on
    plot(SL_vec, Gyk_3,'-','LineWidth',2)
    plot(SL_vec, Gyk_6,'-','LineWidth',2)
    xlabel('$k$ [-]')
    ylabel('$G_{yk}$ [-]')
    legend('$\alpha = 0 deg$','$\alpha = -3 deg$','$\alpha = -6 deg$','Interpreter','latex')
    title('Weighting function $G_{yk}$ as a function of k')
    
    
    
    % Compute index of performance

    % Calculate the residuals with the optimal solution found above
    [SL_vec, sortIdx] = sort(TDataCombined.SL,'ascend');
    FY = FY_vec(sortIdx);
    FY_vec = MF96_FY_vec(SL_vec,TDataCombined.SA,zeros(size(SL_vec)),FZ0*ones(size(SL_vec)),tyre_coeffs);

    residuals_FY = 0;
    for i=1:length(SL_vec)
       residuals_FY = residuals_FY+(FY_vec(i)-FY(i))^2;     
    end
    
    % Compute the residuals
    residuals_FY = residuals_FY/sum(FY.^2);

    % R-squared is 
    % 1-SSE/SST
    % SSE/SST = residuals_FX0
    
    % SSE is the sum of squared error,  SST is the sum of squared total
    fprintf('Index of Fy combined --> R-squared = %6.3f\n',1-residuals_FY);

    % RMSE

    FY_vec_pred = MF96_FY_vec(TDataCombined.SL, TDataCombined.SA, zeros(size(TDataCombined.SL)), FZ0*ones(size(TDataCombined.SL)), tyre_coeffs);
    
    E = rmse(FY_vec_pred,TDataCombined.FY);

    fprintf('Index for Fy combined --> RMSE = %6.3f\n',E);

end




%% SELF-ALIGNING MOMENT: Fitting with Fz=Fz_nom, camber=0,  k = 0, press = 12psi

if data_set == 'later_case'
    %%Intersect tables to obtain specific sub-datasets for pure longitudinal forces
    [TDataMoment0, ~] = intersect_table_data(KAPPA_0, GAMMA_0, FZ_table);
    %%plot_selected_data
    figure('Name','Self-Aligning Moment DATA')
    plot_selected_data(TDataMoment0);
        
    
    FZ0 = mean(TDataMoment0.FZ);
    
    ALPHA_vec = TDataMoment0.SA;
    MZ_vec    = TDataMoment0.MZ;
    zeros_vec = zeros(size(TDataMoment0.SA));
    ones_vec  = ones(size(TDataMoment0.SA));
    
    MZ_varFz_guess = MF96_MZ0_vec(zeros_vec,ALPHA_vec,zeros_vec,FZ_vec,tyre_coeffs);
    MZ0_guess = MF96_MZ0_vec(zeros_vec,ALPHA_vec,zeros_vec,FZ0*ones_vec,tyre_coeffs);
    
    % check guess 
    figure('Name','Raw data Mz0')
    plot(TDataMoment0.SA,MZ_vec,'.')
    hold on
    plot(TDataMoment0.SA,MZ0_guess,'-')
    title('Raw Data Mz0 vs Initial Guess')

           
    % Fit the coeffs {qBz1,qBz9,qBz10,qCz1,qDz1,qDz2,qDz6,qEz1,qEz4,qHz1}
    % Guess values for parameters to be optimised
    %  
    P0 = [7, 0.0001, 0.0001, 2, 0.2, -0.05, 0.003, 0.5, -0.1, -0.01]; 
    
    % NOTE: many local minima => limits on parameters are fundamentals
    % Limits for parameters to be optimised
    % 1< pCx1 < 2 
    % 0< pEx1 < 1 
    %    [{qBz1,qBz9,qBz10,qCz1,qDz1,qDz2,qDz6,qEz1,qEz4,qHz1]
    lb = [ 1, -0.1,-0.1, -5, -2, -1, -0.1, -3, -2, -0.5];
    ub = [10,  0.1, 0.1,  5,  2,  1,  0.1,  3,  2,  0.5];
    
    
    % LSM_pure_Fx returns the residual, so minimize the residual varying X. It
    % is an unconstrained minimization problem 
                                   
    [P_Mz0,fval,exitflag] = fmincon(@(P)resid_pure_Mz(P,MZ_vec,ALPHA_vec,0,FZ0, tyre_coeffs),...
                                   P0,[],[],[],[],lb,ub);
    
    % Update tyre data with new optimal values                             
    tyre_coeffs.qBz1 = P_Mz0(1);
    tyre_coeffs.qBz9 = P_Mz0(2);
    tyre_coeffs.qBz10 = P_Mz0(3);
    tyre_coeffs.qCz1 = P_Mz0(4);
    tyre_coeffs.qDz1 = P_Mz0(5);
    tyre_coeffs.qDz2 = P_Mz0(6);
    tyre_coeffs.qDz6 = P_Mz0(7);
    tyre_coeffs.qEz1 = P_Mz0(8);
    tyre_coeffs.qEz4 = P_Mz0(9);
    tyre_coeffs.qHz1 = P_Mz0(10);
    
    
    SA_vec = -0.3:0.001:0.3;
    MZ0_vec = MF96_MZ0_vec(zeros(size(SA_vec)),SA_vec,zeros(size(SA_vec)),FZ0*ones(size(SA_vec)),tyre_coeffs);
    
    figure('Name','Mz0')
    plot(TDataMoment0.SA,TDataMoment0.MZ,'o','MarkerSize',2)
    hold on
    plot(SA_vec,MZ0_vec,'-','LineWidth',2)
    xlabel('$\alpha$ [rad]')
    ylabel('$M_{z0}$ [N]')
    titolo = sprintf('Self-Aligning Moment with Fz = %d N, zero camber, k = 0, press = 12 psi',Fz0);
    title(titolo);

    % Compute index of performance

    % Calculate the residuals with the optimal solution found above
    [SA_vec, sortIdx] = sort(TDataMoment0.SA,'ascend');
    MZ = MZ_vec(sortIdx);
    MZ_vec = MF96_MZ0_vec(zeros(size(SA_vec)),SA_vec,zeros(size(SA_vec)),FZ0*ones(size(SA_vec)),tyre_coeffs);

    residuals_MZ = 0;
    for i=1:length(SA_vec)
       residuals_MZ = residuals_MZ+(MZ_vec(i)-MZ(i))^2;     
    end
    
    % Compute the residuals
    residuals_MZ = residuals_MZ/sum(MZ.^2);

    % R-squared is 
    % 1-SSE/SST
    % SSE/SST = residuals_MZ0
    
    % SSE is the sum of squared error,  SST is the sum of squared total
    fprintf('Index of MZ0 --> R-squared = %6.3f\n',1-residuals_MZ);

    
    % RMSE
    MZ0_vec_pred = MF96_MZ0_vec(zeros(size(ALPHA_vec)),ALPHA_vec,zeros(size(ALPHA_vec)),FZ0*ones(size(ALPHA_vec)),tyre_coeffs);
    
    E = rmse(MZ0_vec_pred,TDataMoment0.MZ);

    fprintf('Index for Mz0 --> RMSE = %6.3f\n',E);


end

%% SELF-ALIGNING MOMENT at different vertical loads, camber=0,  k = 0, press = 12psi

if data_set == 'later_case'
    %%Intersect tables to obtain specific sub-datasets for pure longitudinal forces
    [TDataMoment_varFz, ~] = intersect_table_data(KAPPA_0, GAMMA_0);
    [TDataMoment_Fz220, ~] = intersect_table_data(KAPPA_0, GAMMA_0, FZ_220);
    [TDataMoment_Fz440, ~] = intersect_table_data(KAPPA_0, GAMMA_0, FZ_440);
    [TDataMoment_Fz700, ~] = intersect_table_data(KAPPA_0, GAMMA_0, FZ_700);
    [TDataMoment_Fz900, ~] = intersect_table_data(KAPPA_0, GAMMA_0, FZ_900);
    [TDataMoment_Fz1120, ~] = intersect_table_data(KAPPA_0, GAMMA_0, FZ_1120);
    %%plot_selected_data
    figure('Name','Self-Aligning Moment DATA - variable Fz')
    plot_selected_data(TDataMoment_varFz);
   
    
    ALPHA_vec = TDataMoment_varFz.SA;
    MZ_vec    = TDataMoment_varFz.MZ;
    FZ_vec    = TDataMoment_varFz.FZ;
    zeros_vec = zeros(size(TDataMoment_varFz.SA));
    ones_vec  = ones(size(TDataMoment_varFz.SA));
    

    MZ_varFz_guess = MF96_MZ0_vec(zeros_vec,ALPHA_vec,zeros_vec,FZ_vec,tyre_coeffs);
    
    % check guess 
    figure('Name','Raw Data Mz0 with variable Fz')
    plot(TDataMoment_varFz.SA,MZ_vec,'.')
    hold on
    plot(TDataMoment_varFz.SA,MZ_varFz_guess,'-')
    

    % Fit the coeffs {qBz2,qBz3,qDz7,qEz2,qEz3,qHz2}
    % Guess values for parameters to be optimised
    %  
    P0 = [-3, -1, 0.01, -1, -1, 0.005]; 
    
    % NOTE: many local minima => limits on parameters are fundamentals
    % Limits for parameters to be optimised
    % 1< pCx1 < 2 
    % 0< pEx1 < 1 
    %    [qBz2,qBz3,qDz7,qEz2,qEz3,qHz2]
    lb = [-10, -5, -0.01, -5, -5, -0.01];
    ub = [  1,  2,   0.1,  5,  5,  0.01];
    
    
    % LSM_pure_Fx returns the residual, so minimize the residual varying X. It
    % is an unconstrained minimization problem 
                                   
    [P_Mz_varFz,fval,exitflag] = fmincon(@(P)resid_pure_Mz_varFz(P,MZ_vec,ALPHA_vec,0,FZ_vec, tyre_coeffs),...
                                   P0,[],[],[],[],lb,ub);
    
    % Update tyre data with new optimal values                             
    tyre_coeffs.qBz2 = P_Mz_varFz(1);
    tyre_coeffs.qBz3 = P_Mz_varFz(2);
    tyre_coeffs.qDz7 = P_Mz_varFz(3);
    tyre_coeffs.qEz2 = P_Mz_varFz(4);
    tyre_coeffs.qEz3 = P_Mz_varFz(5);
    tyre_coeffs.qHz2 = P_Mz_varFz(6);
    
    
    SA_vec = -0.3:0.001:0.3;
    tmp_zeros = zeros(size(SA_vec));
    tmp_ones = ones(size(SA_vec));
    
    MZ0_varFz1_vec = MF96_MZ0_vec(tmp_zeros,SA_vec,tmp_zeros,mean(FZ_220.FZ)*tmp_ones,tyre_coeffs);
    MZ0_varFz2_vec = MF96_MZ0_vec(tmp_zeros,SA_vec,tmp_zeros,mean(FZ_440.FZ)*tmp_ones,tyre_coeffs);
    MZ0_varFz3_vec = MF96_MZ0_vec(tmp_zeros,SA_vec,tmp_zeros,mean(FZ_700.FZ)*tmp_ones,tyre_coeffs);
    MZ0_varFz4_vec = MF96_MZ0_vec(tmp_zeros,SA_vec,tmp_zeros,mean(FZ_900.FZ)*tmp_ones,tyre_coeffs);
    MZ0_varFz5_vec = MF96_MZ0_vec(tmp_zeros,SA_vec,tmp_zeros,mean(FZ_1120.FZ)*tmp_ones,tyre_coeffs);

    
    figure('Name','Mz0 with variable Fz')
    plot(TDataMoment_Fz220.SA,TDataMoment_Fz220.MZ,'.','MarkerSize',5,'Color','#0072BD')
    hold on
    plot(TDataMoment_Fz440.SA,TDataMoment_Fz440.MZ,'.','MarkerSize',5,'Color','#D95319')
    plot(TDataMoment_Fz700.SA,TDataMoment_Fz700.MZ,'.','MarkerSize',5,'Color','#EDB120')
    plot(TDataMoment_Fz900.SA,TDataMoment_Fz900.MZ,'.','MarkerSize',5,'Color','#7E2F8E')
    plot(TDataMoment_Fz1120.SA,TDataMoment_Fz1120.MZ,'.','MarkerSize',5,'Color','#77AC30')
    
    plot(SA_vec,MZ0_varFz1_vec,'-','LineWidth',2,'Color','#0072BD')
    plot(SA_vec,MZ0_varFz2_vec,'-','LineWidth',2,'Color','#D95319')
    plot(SA_vec,MZ0_varFz3_vec,'-','LineWidth',2,'Color','#EDB120')
    plot(SA_vec,MZ0_varFz4_vec,'-','LineWidth',2,'Color','#7E2F8E')
    plot(SA_vec,MZ0_varFz5_vec,'-','LineWidth',2,'Color','#77AC30')
    xlabel('$\alpha$ [rad]')
    ylabel('$M_{z0}$ [N]')
    legend({'Raw $Fz_{220}$','Raw $Fz_{440}$','Raw $Fz_{700}$','Raw $Fz_{900}$','Raw $Fz_{1120}$','$Fz_{220}$','$Fz_{440}$','$Fz_{700}$','$Fz_{900}$','$Fz_{1120}$'})
    title('Self-Aligning Moment Mz at different vertical loads, zero camber, k = 0, press = 12 psi')


   
    % Compute index of performance

    % Calculate the residuals with the optimal solution found above
    [SA_vec, sortIdx] = sort(TDataMoment_varFz.SA,'ascend');
    MZ = MZ_vec(sortIdx);
    MZ_vec = MF96_MZ0_vec(zeros(size(SA_vec)),SA_vec,zeros(size(SA_vec)),FZ_vec,tyre_coeffs);

    residuals_MZ = 0;
    for i=1:length(SA_vec)
       residuals_MZ = residuals_MZ+(MZ_vec(i)-MZ(i))^2;     
    end
    
    % Compute the residuals
    residuals_MZ = residuals_MZ/sum(MZ.^2);

    % R-squared is 
    % 1-SSE/SST
    % SSE/SST = residuals_MZ0
    
    % SSE is the sum of squared error,  SST is the sum of squared total
    fprintf('Index of MZ0 varFz--> R-squared = %6.3f\n',1-residuals_MZ);
   

    % RMSE

    MZ0_vec_pred = MF96_MZ0_vec(zeros(size(ALPHA_vec)),ALPHA_vec,zeros(size(ALPHA_vec)),FZ_vec,tyre_coeffs);
    
    E = rmse(MZ0_vec_pred,TDataMoment_varFz.MZ);

    fprintf('Index for Mz0 varFz--> RMSE = %6.3f\n',E);



end
%% Save tyre data structure to mat file

save('results\optimal_tyre_coeffs.mat','tyre_coeffs');

