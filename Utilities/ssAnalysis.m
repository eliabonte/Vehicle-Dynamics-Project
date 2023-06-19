function ssAnalysis(model_sim,vehicle_data,Ts)

    % ----------------------------------------------------------------
    %% Post-Processing and Data Analysis
    % ----------------------------------------------------------------

    % ---------------------------------
    %% Load vehicle data
    % ---------------------------------
    Lf = vehicle_data.vehicle.Lf;  % [m] Distance between vehicle CoG and front wheels axle
    Lr = vehicle_data.vehicle.Lr;  % [m] Distance between vehicle CoG and front wheels axle
    L  = vehicle_data.vehicle.L;   % [m] Vehicle length
    Wf = vehicle_data.vehicle.Wf;  % [m] Width of front wheels axle 
    Wr = vehicle_data.vehicle.Wr;  % [m] Width of rear wheels axle                   
    m  = vehicle_data.vehicle.m;   % [kg] Vehicle Mass
    g  = vehicle_data.vehicle.g;   % [m/s^2] Gravitational acceleration
    hs = vehicle_data.vehicle.hGs;

    tau_D = vehicle_data.steering_system.tau_D;  % [-] steering system ratio (pinion-rack)

    Ks_r = vehicle_data.rear_suspension.Ks_r;
    Ks_f = vehicle_data.front_suspension.Ks_f;
    hrr  = vehicle_data.rear_suspension.h_rc_r;
    hrf  = vehicle_data.front_suspension.h_rc_f;

    eps_s = Ks_f/(Ks_f + Ks_r);

    % ---------------------------------
    %% Extract data from simulink model
    % ---------------------------------
    time_sim = model_sim.states.u.time;
    dt = time_sim(2)-time_sim(1);

    t_steer = model_sim.inputs.t_steer.data;
    
    time_sim_transient = time_sim(time_sim < t_steer);
    index_ss = length(time_sim_transient);
    time_sim_ss        = time_sim(index_ss:end);

    % -----------------
    % Inputs
    % -----------------
    delta_D       = model_sim.inputs.delta_D.data;
    delta_D_ss    = delta_D(index_ss:end);

    % -----------------
    % States
    % -----------------
    u          = model_sim.states.u.data;
    v          = model_sim.states.v.data;
    Omega      = model_sim.states.Omega.data;
    Fz_rr      = model_sim.states.Fz_rr.data;
    Fz_rl      = model_sim.states.Fz_rl.data;
    Fz_fr      = model_sim.states.Fz_fr.data;
    Fz_fl      = model_sim.states.Fz_fl.data;

    u_ss          = u(index_ss:end);
    v_ss          = v(index_ss:end);
    Omega_ss      = Omega(index_ss:end);
    Fz_rr_ss      = Fz_rr(index_ss:end);
    Fz_rl_ss      = Fz_rl(index_ss:end);
    Fz_fr_ss      = Fz_fr(index_ss:end);
    Fz_fl_ss      = Fz_fl(index_ss:end);

    % -----------------
    % Extra Parameters
    % -----------------

    Fx_fr      = model_sim.extra_params.Fx_fr.data;
    Fx_fl      = model_sim.extra_params.Fx_fl.data;
    Fy_rr      = model_sim.extra_params.Fy_rr.data;
    Fy_rl      = model_sim.extra_params.Fy_rl.data;
    Fy_fr      = model_sim.extra_params.Fy_fr.data;
    Fy_fl      = model_sim.extra_params.Fy_fl.data;
    gamma_rr   = model_sim.extra_params.gamma_rr.data;
    gamma_rl   = model_sim.extra_params.gamma_rl.data;
    gamma_fr   = model_sim.extra_params.gamma_fr.data;
    gamma_fl   = model_sim.extra_params.gamma_fl.data;
    delta_fr   = model_sim.extra_params.delta_fr.data;
    delta_fl   = model_sim.extra_params.delta_fl.data;

    Fx_fr_ss      = Fx_fr(index_ss:end);
    Fx_fl_ss      = Fx_fl(index_ss:end);
    Fy_rr_ss      = Fy_rr(index_ss:end);
    Fy_rl_ss      = Fy_rl(index_ss:end);
    Fy_fr_ss      = Fy_fr(index_ss:end);
    Fy_fl_ss      = Fy_fl(index_ss:end);
    gamma_rr_ss   = gamma_rr(index_ss:end);
    gamma_rl_ss   = gamma_rl(index_ss:end);
    gamma_fr_ss   = gamma_fr(index_ss:end);
    gamma_fl_ss   = gamma_fl(index_ss:end);
    delta_fr_ss   = delta_fr(index_ss:end);
    delta_fl_ss   = delta_fl(index_ss:end);

    % Chassis side slip angle beta [rad]
    beta_ss = atan(v_ss./u_ss);

    % -----------------
    % Accelerations
    
    % Steady state longitudinal acceleration
    Ax_ss = - Omega_ss.*v_ss;
    % Steady state lateral acceleration
    Ay_ss = Omega_ss.*u_ss;

    % -----------------
    % Other parameters
    % -----------------
    % Total CoM speed [m/s]
    vG = sqrt(u_ss.^2 + v_ss.^2);
    % Steady state and transient curvature [m]
    rho_ss   = Omega_ss./vG;
    
    % Steering angle at the wheel
    delta_f = delta_D_ss/tau_D;

    % Normalized lateral acceleration
    norm_acc = Ay_ss./g;

    % --- VERTICAL LOADS ---
    Fz_r = Fz_rr_ss + Fz_rl_ss;
    Fz_f = Fz_fr_ss + Fz_fl_ss;
    
    % --- AXLE CHARACTERISTICS ---
    Fy_R = Fy_rl_ss + Fy_rr_ss;
    Fy_F = sin(delta_fl_ss).*Fx_fl_ss + Fy_fl_ss + sin(delta_fr_ss).*Fx_fr_ss + Fy_fr_ss;


    % ---- axle side slips ----
    alpha_r = - (v_ss - Omega_ss*Lr)./u_ss;
    alpha_f =  deg2rad(delta_f) - (v_ss + Omega_ss*Lf)./u_ss;

    alpha_r = rad2deg(alpha_r);
    alpha_f = rad2deg(alpha_f);

    % ---- normalized axle lateral forces ----
    Fzr0 = m*9.81*Lf/L;
    Fzf0 = m*9.81*Lr/L;

    mu_f = Fy_F./Fzf0;
    mu_r = Fy_R./Fzr0;

    % --- LATERAL LOAD TRANSFER ---
    dFz_r = m*Ay_ss*(((Lf/L)*(hrr/Wr)) + (hs/Wr)*(1 - eps_s));
    dFz_f = m*Ay_ss*(((Lr/L)*(hrf/Wr)) + (hs/Wf)*(eps_s));
    
    % Fz_rl_data = Fz_r/2 - dFz_r;
    % Fz_fl_data = Fz_f/2 - dFz_f;
    % Fz_rr_data = Fz_r/2 + dFz_r;
    % Fz_fr_data = Fz_f/2 + dFz_f;

    % ---------------------------------
    %% PLOTS
    % ---------------------------------

    % ---------------------------------
    %% Vertical loads
        
    figure('Name','Lateral Load Transfer','NumberTitle','off'), clf   
    % --- REAR --- %
    ax(1) = subplot(211);
    hold on
    plot(Ay_ss, dFz_r, 'LineWidth',2)
    xlabel('$a_y$')
    grid on
    ylim([0 1000])
    title('Rear Lateral Load Transfer vs Ay')
    % --- FRONT --- %
    ax(2) = subplot(212);
    plot(Ay_ss, dFz_f, 'LineWidth',2)
    xlabel('$a_y$')
    grid on
    ylim([0 1000])
    title('Front Lateral Load Transfer vs Ay')


    figure('Name','Wheel Individual LLT','NumberTitle','off'), clf   
    % --- REAR LEFT--- %
    ax(1) = subplot(221);
    % hold on
    % plot(Ay_ss, Fz_rl_data, 'LineWidth',2)
    hold on
    plot(Ay_ss, Fz_rl_ss, 'LineWidth',2)
    xlabel('$a_y$')
    grid on
    ylim([100 1000])
    title('Fz rear left')
    ax(2) = subplot(222);
    % hold on
    % plot(Ay_ss, Fz_rr_data, 'LineWidth',2)
    hold on
    plot(Ay_ss, Fz_rr_ss, 'LineWidth',2)
    xlabel('$a_y$')
    grid on
    ylim([100 1000])
    title('Fz rear right')
    ax(3) = subplot(223);
    % hold on
    % plot(Ay_ss, Fz_fl_data, 'LineWidth',2)
    hold on
    plot(Ay_ss, Fz_fl_ss, 'LineWidth',2)
    xlabel('$a_y$')
    grid on
    ylim([100 1000])
    title('Fz front left')
    ax(4) = subplot(224);
    % hold on
    % plot(Ay_ss, Fz_fr_data, 'LineWidth',2)
    hold on
    plot(Ay_ss, Fz_fr_ss, 'LineWidth',2)
    xlabel('$a_y$')
    grid on
    ylim([100 1000])
    title('Fz front right')

    %% Plot Axle characteristics

    figure('Name','Axle Characteristics','NumberTitle','off'), clf   
    % --- alpha_R --- %
    ax(1) = subplot(221);
    hold on
    plot(time_sim_ss, alpha_r, 'LineWidth',2)
    hold on
    plot(time_sim_ss, alpha_f, 'LineWidth',2)
    grid on
    title('$\alpha_R$ [deg]')
    xlim([0 time_sim(end)])
    ylim([0 6])
    % --- alpha_F --- %
    ax(2) = subplot(222);
    plot(time_sim_ss, alpha_f, 'LineWidth',2)
    grid on
    title('$\alpha_F$ [deg]')
    ylim([0 6])
    xlim([0 time_sim(end)])
    % --- Fy_R --- %
    ax(3) = subplot(223);
    hold on
    plot(time_sim_ss, Fy_R, 'LineWidth',2)
    grid on
    title('$Fy_R$ [N]')
    ylim([0 3000])
    xlim([0 time_sim(end)])
    % --- alpha_F --- %
    ax(4) = subplot(224);
    plot(time_sim_ss, Fy_F, 'LineWidth',2)
    grid on
    title('$Fy_F$ [N]')
    ylim([0 3000])
    xlim([0 time_sim(end)])


    clear ax

    % -- Plot axle characteristics ---
    
    mu_r_lin = mu_r(mu_r < 0.38001);
    mu_f_lin = mu_f(mu_f < 0.35001);

    i_r = length(mu_r_lin);
    i_f = length(mu_f_lin);

    Cyr = mu_r_lin(end)/alpha_r(i_r);
    Cyf = mu_f_lin(end)/alpha_f(i_f);

    lin_Fyr = alpha_r.*Cyr;
    lin_Fyf = alpha_f.*Cyf;

    lin_range = round(length(alpha_r)/2);

    theta_r = atan(lin_Fyr(200)/alpha_r(200));
    th_r = 0:1/360:theta_r;  r_r = 0.5;
    xx_r = r_r*cos(th_r); yy_r = r_r*sin(th_r);

    theta_f = atan(lin_Fyf(200)/alpha_f(200));
    th_f = 0:1/360:theta_f;  r_f = 0.35;
    xx_f = r_f*cos(th_f); yy_f = r_f*sin(th_f);

    figure('Name','Fy normalized vs alpha','NumberTitle','off'), clf
    hold on
    plot(alpha_r, mu_r,'LineWidth',2,'Color',"#EDB120")
    hold on
    plot(alpha_f, mu_f,'LineWidth',2,'Color',"#0072BD")
    hold on
    plot(alpha_r(1:lin_range), lin_Fyr(1:lin_range),'LineWidth',2,'Color',"#EDB120")
    hold on
    plot(alpha_f(1:lin_range), lin_Fyf(1:lin_range),'LineWidth',2,'Color',"#0072BD")
    hold on
    %plot(alpha_r, ones(size(alpha_r)),'LineWidth',1,'Color','red')
    %hold on
    text(alpha_r(end)+0.03, mu_r(end)+0.03, 'R','Color',"#EDB120",'FontSize',15)  %letter R
    hold on
    text(alpha_f(end)+0.03, mu_f(end)+0.03, 'F','Color',"#0072BD",'FontSize',15)  %letter R
    hold on
    %text(alpha_r(end)+0.03, 1, '$\frac{a_y}{g}$','Color',"red")
    %hold on
    plot(xx_r,yy_r,'LineWidth',1,'Color',"#EDB120")
    hold on
    text(r_r*cos(theta_r)+0.03,r_r*sin(theta_r)/2,'$C_yr$','Color',"#EDB120",'FontSize',10)
    hold on
    plot(xx_f,yy_f,'LineWidth',1,'Color',"#0072BD")
    hold on
    text((r_f*cos(theta_f))+0.03,r_f*sin(theta_f)/2,'$C_yf$','Color',"#0072BD",'FontSize',10)
    grid on
    title('Normalized Lateral Forces vs Axle Side-Slip angles')
    xlabel('$\alpha_R$, $\alpha_F$')
    ylabel('$\mu_r$, $\mu_f$')
    xlim([0 6]);
    ylim([0 3]);
    legend('Rear Axle','Front Axle')

    %% Plot behaviour of vehicle - Handling diagram for steer ramp test with constant velocity
    % Kus e fit polynomial
    
    % bug sul primo elemento di alpha
    alpha_r(1) = 0;
    alpha_f(1) = 0;
    
    %hand_curve_alpha = - (deg2rad(alpha_r) - deg2rad(alpha_f)); 
    
    hand_curve =  deg2rad(delta_f) - rho_ss.*L;
    %hand_curve = rad2deg(hand_curve);

    hand_curve(1) = 0;

    % -- fit linear range ---

    norm_acc_lin = norm_acc(norm_acc < 0.7);
    linear_index = length(norm_acc_lin);
    
    coeffs_lin = polyfit(norm_acc_lin, hand_curve(1:linear_index),1);
    K_us = coeffs_lin(1);

    fprintf('Understeering Gradient K_us = %d',K_us);
    
    hand_lin_curve = K_us.*norm_acc(1:linear_index);

    % -- fit non-linear range
    c_nonlin = polyfit(norm_acc(linear_index:end), hand_curve(linear_index:end),3);


    norm_acc_nonlin = norm_acc(linear_index:end);

    hand_nonlin_curve = c_nonlin(1).*(norm_acc_nonlin.^3) + c_nonlin(2).*(norm_acc_nonlin.^2) + c_nonlin(3).*norm_acc_nonlin + c_nonlin(4);
    
    %hand_fitted_curve = c_nonlin(1).*(norm_acc_nonlin.^3) + c_nonlin(2).*(norm_acc_nonlin.^2) + c_nonlin(3).*norm_acc_nonlin + c_nonlin(4);

    theta_k = atan(K_us);
    th_k = theta_k:1/360:0;  r_k = 0.3;
    xx_k = r_k*cos(th_k); yy_k = r_k*sin(th_k);
    
    figure('Name','Handling curve','NumberTitle','off'), clf
    set(gca,'fontsize',16)
    hold on
    xlabel('$ \frac{a_y}{g}$')
    ylabel('$\delta\tau - \rho L$')
    title('Handling Diagram','FontSize',18)
    plot(norm_acc,hand_curve,'Color',"#D95319",'LineWidth',2)
    hold on
    plot(norm_acc(1:linear_index),hand_lin_curve,'-.c','LineWidth',1.5)
    hold on
    plot(norm_acc_nonlin,hand_nonlin_curve,'-.b','LineWidth',1.5)
    hold on
    plot(norm_acc,zeros(size(norm_acc)),'LineWidth',1,'Color',"#77AC30")
    hold on
    text(norm_acc(end) + 0.07, 0, "NS", 'Color',"#77AC30",'FontSize', 10)
    hold on
    plot(xx_k,yy_k,'LineWidth',1,'Color',"#A2142F")
    hold on
    text(r_k*cos(theta_k)+0.05,r_k*sin(theta_k)/2,'$K_{US}$','Color',"#A2142F",'FontSize',10)
    xlim([0 1.2])
    ylim auto
    % axis squar
    legend('Handling curve','Linear Fitting', 'Polynomial Fitting')


    %% Yaw Rate Gain

    % %yr_gain = (u/L)./(1 + K_us.*(u.^2));
    % 
    % delta = delta_f*pi/180;
    % 
    % data_yr_gain = Omega_ss./delta;
    % 
    % figure('Name','Yaw Rate Gain','NumberTitle','off'), clf
    % set(gca,'fontsize',16)
    % hold on
    % axis equal
    % xlabel('$u$')
    % ylabel('$\frac{\Omega}{\delta}$','Interpreter','latex')
    % title('Yaw Rate Gain','FontSize',18)
    % %plot(u,yr_gain,'LineWidth',2,'Color',"#A2142F")
    % hold on
    % plot(u_ss,data_yr_gain,'LineWidth',2,'Color',"#77AC30")
    % 
    % %% Beta Gain
    % 
    % %beta_gain = Lr/L - (m/L^3)*((Lf/Cyr)+(Lr/Cyf))*((u.^2)./(1 + K_us.*(u.^2)));
    % 
    % data_beta_gain = beta_ss./delta;
    % 
    % figure('Name','Beta Gain','NumberTitle','off'), clf
    % set(gca,'fontsize',16)
    % hold on
    % axis equal
    % xlabel('$u$')
    % ylabel('$\frac{\beta}{\delta}$','Interpreter','latex')
    % title('$\beta$ Gain','FontSize',18)
    % %plot(u,beta_gain,'LineWidth',2,'Color',"#A2142F")
    % hold on
    % plot(u_ss,data_beta_gain,'LineWidth',2,'Color',"#77AC30")
    % ylim([-1 1])


end
    
