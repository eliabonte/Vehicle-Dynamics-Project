function res = resid_pure_Fx_comb(P,FX,KAPPA,ALPHA,GAMMA,FZ,tyre_data)

    % ----------------------------------------------------------------------
    %% Compute the residuals - least squares approach - to fit the Fx curve 
    %  with Fz=Fz_nom, IA=0. Pacejka 1996 Magic Formula
    % ----------------------------------------------------------------------

    % Define MF coefficients

    %Fz0 = 200*4.44822; % Nominal load 200 lbf
    
    tmp_tyre_data = tyre_data;
      
    tmp_tyre_data.rBx1 = P(1); 
    tmp_tyre_data.rBx2 = P(2);
    tmp_tyre_data.rCx1 = P(3);
    tmp_tyre_data.rHx1 = P(4);

    res = 0;
    for i=1:length(KAPPA)
       fx  = MF96_FX(KAPPA(i), ALPHA(i), GAMMA(i), FZ, tmp_tyre_data);
       res = res+(fx-FX(i))^2;     
    end
    
    % Compute the residuals
    res = res/sum(FX.^2);
    

end
