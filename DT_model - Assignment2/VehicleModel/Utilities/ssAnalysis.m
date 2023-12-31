function ssAnalysis(model_sim,vehicle_data,type_test, t_steer, Ts)

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
    tau_H = vehicle_data.steering_system.tau_H;

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
    %t_steer = model_sim.inputs.t_steer.data;

    type_test = 1; % -> If I'm performing test with constant velocity and steer ramp
    % type_test = 2; % -> If I'm performing test with constant steering and speed ramp

    % STEADY-STATE time changes with the two different tests
    if type_test == 1
        time_sim_transient = time_sim(time_sim < t_steer);
        index_ss = length(time_sim_transient);
        time_sim_ss        = time_sim(index_ss:end);
    else
        time_sim_transient = time_sim(time_sim < (20+Ts));
        index_ss = length(time_sim_transient);
        time_sim_ss        = time_sim(index_ss:end);
    end

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
    delta      = model_sim.states.delta.data;

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
    beta_tot = atan(v./u);
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
    delta_ss   = deg2rad(delta_D_ss/tau_D);

    % Normalized lateral acceleration
    norm_acc = Ay_ss./g;

    % --- VERTICAL LOADS ---
    Fz_r = Fz_rr_ss + Fz_rl_ss;
    Fz_f = Fz_fr_ss + Fz_fl_ss;
    
    % --- AXLE CHARACTERISTICS ---
    Fy_R = Fy_rl_ss + Fy_rr_ss;
    Fy_F = sin(delta_fl_ss).*Fx_fl_ss + cos(delta_fl_ss).*Fy_fl_ss + sin(delta_fr_ss).*Fx_fr_ss + cos(delta_fr_ss).*Fy_fr_ss;

    % ---- axle side slips ----
    alpha_r = - (v_ss - Omega_ss*Lr)./u_ss;
    alpha_f =  delta_ss - (v_ss + Omega_ss*Lf)./u_ss;
    alpha_r(1) = 0; alpha_r(2) = 0;
    alpha_f(1) =  0; alpha_f(2) = 0;


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
    color_rear = "#EDB120";
    color_front = "#0072BD";
    % ---------------------------------
    %% Vertical loads
        
    figure('Name','Lateral Load Transfer','NumberTitle','off'), clf   
    hold on
    plot(Ay_ss, dFz_r, 'LineWidth',2,'Color',color_rear)
    plot(Ay_ss, dFz_f, 'LineWidth',2,'Color',color_front)
    grid on
    ylabel('$\Delta Fz$ [N]')
    xlabel('$a_y [\frac{m}{s^2}]$')
    ylim([0 1000])
    legend('$\Delta Fz_r$','$\Delta Fz_f$')
    title('Lateral Load Transfers')
    hold off

    figure('Name','Wheel Individual LLT','NumberTitle','off'), clf   
    % --- REAR --- %
    ax(1) = subplot(211);
    hold on
    plot(Ay_ss, Fz_rl_ss, 'LineWidth',2)
    plot(Ay_ss, Fz_rr_ss, 'LineWidth',2)
    grid on
    xlabel('$a_y [\frac{m}{s^2}]$')
    ylim([0 1400])
    ylabel('Fz[N]')
    legend('$Fz_rl$','$Fz_rr$')
    title('Fz rear')
    hold off

    % --- FRONT --- %
    ax(2) = subplot(212);
    hold on
    plot(Ay_ss, Fz_fl_ss, 'LineWidth',2)
    plot(Ay_ss, Fz_fr_ss, 'LineWidth',2)
    grid on
    ylabel('Fz[N]')
    xlabel('$a_y [\frac{m}{s^2}]$')
    ylim([0 1400])
    legend('$Fz_fl$','$Fz_fr$')
    title('Fz front')
    hold off


    %% Plot Axle characteristics

    figure('Name','Axle Characteristics','NumberTitle','off'), clf   
    % --- alpha_R --- %
    ax(1) = subplot(211);
    hold on
    plot(time_sim_ss, alpha_r, 'LineWidth',2,'Color',color_rear)
    plot(time_sim_ss, alpha_f, 'LineWidth',2,'Color',color_front)
    grid on
    title('Axle Side-Slip')
    legend('$\alpha_R$','$\alpha_F$')
    xlabel('t[s]')
    ylabel('$\alpha$[rad]')
    xlim([0 time_sim(end)])
    ylim([0 0.12])
    hold off

    % --- Fy --- %
    ax(2) = subplot(212);
    hold on
    plot(time_sim_ss, Fy_R, 'LineWidth',2,'Color',color_rear)
    plot(time_sim_ss, Fy_F, 'LineWidth',2,'Color',color_front)
    grid on
    title('Axle Lateral Forces')
    legend('$\Fy_R$','$\Fy_F$')
    xlabel('t[s]')
    ylabel('Fy[N]')
    xlim([0 time_sim(end)])
    ylim([0 3000])
    hold off

    clear ax


    % -- Plot normalized axle characteristics --    

    figure('Name','muR,muF vs alphaR,alphaF','NumberTitle','off'), clf
    hold on
    plot(alpha_r, mu_r,'LineWidth',2,'Color',"#EDB120")
    plot(alpha_f, mu_f,'LineWidth',2,'Color',"#0072BD")
    %plot(alpha_r, ones(size(alpha_r)),'LineWidth',1,'Color','red')
    text(alpha_r(end)+0.003, mu_r(end)+0.003, 'R','Color',color_rear,'FontSize',15)  %letter R
    text(alpha_f(end)+0.003, mu_f(end)+0.003, 'F','Color',color_front,'FontSize',15)  %letter R
    grid on
    title('Normalized Lateral Forces vs Axle Side-Slip')
    xlabel('$\alpha_R$, $\alpha_F$ [rad]')
    ylabel('$\mu_r$, $\mu_f$')
    xlim([0 0.12]);
    ylim([0 1.7]);
    legend('$\mu_R$', '$\mu_F$')
    hold off

    %% Plot behaviour of vehicle - Handling diagram for steer ramp test with constant velocity
    % Kus e fit polynomial
    
    % -- COMPUTE HANDLING CURVE and fitting with KUS--
    Delta_alpha = (alpha_r - alpha_f);
    hand_curve = - Delta_alpha; 
    %hand_curve =  delta_ss - rho_ss.*L;

    hand_curve(1) = 0;

    % -- fit linear range ---
    
    max_lin_acc = 0.4;
    norm_acc_lin = norm_acc(norm_acc < max_lin_acc);
    linear_index = length(norm_acc_lin);
    
    coeffs_lin = polyfit(norm_acc_lin, hand_curve(1:linear_index),1);
    K_us = coeffs_lin(1);
    
    hand_lin_curve = K_us.*norm_acc(1:linear_index);

    % -- fit non-linear range
    c_nonlin = polyfit(norm_acc(linear_index:end), hand_curve(linear_index:end),3);

    norm_acc_nonlin = norm_acc(linear_index:end);

    hand_nonlin_curve = c_nonlin(1).*(norm_acc_nonlin.^3) + c_nonlin(2).*(norm_acc_nonlin.^2) + c_nonlin(3).*norm_acc_nonlin + c_nonlin(4);
    
    
    % -- UNDERSTEERING GRADIENT THEORETICAL --
    K_us_th1 = -diff(Delta_alpha)./diff(Ay_ss);

    Cy_r = diff(mu_r) ./ diff(alpha_r);
    Cy_r = Cy_r(10:end);
    Cy_f = diff(mu_f) ./ diff(alpha_f);
    Cy_f = Cy_f(10:end);
    K_us_th2 = (-1/(L*g*tau_H))*((1./Cy_r)-(1./Cy_f));
    
    index_linear = find(norm_acc<0.35);
    index_linear = index_linear(end);

    figure('Name','Kus VS A_y','NumberTitle','off'), clf
    hold on
    plot(norm_acc(20:end), K_us_th1(19:end),'LineWidth',2)
    plot(norm_acc(20:end), K_us_th2(10:end),'LineWidth',2)
    plot(norm_acc(20:end), K_us*ones(size(norm_acc(20:end))),'LineWidth',2)
    xlabel('ay/g')
    ylabel('$K_{us}$')
    title('$K_{us}$ vs $a_y$')
    legend('Kus $delta_{ay}$','Kus formula','Kus fitted')
    ylim([-0.009 0])
   
    K_us_th1 = K_us_th1(index_linear);
    K_us_th2 = K_us_th2(index_linear);

    fprintf('\n')
    fprintf('Understeering Gradient theoretically: K_us_th1 = %d',K_us_th1);
    fprintf('\n')
    fprintf('Understeering Gradient theoretically: K_us_th2 = %d',K_us_th2);
    fprintf('\n')
    fprintf('\n')
    fprintf('Understeering Gradient with fitting: K_us = %d',K_us);
    fprintf('\n')


    % fprintf('\n')
    % fprintf('Understeering Gradient theoretically: K_us_th2 = %d',K_us_th2);
    % fprintf('\n')

    
    theta_k = atan(K_us);
    r_k = 0.3;
    x = [r_k r_k*cos(theta_k)];
    y = [0 r_k*sin(theta_k)];                    
    
    figure('Name','Handling curve','NumberTitle','off'), clf
    set(gca,'fontsize',16)
    hold on
    xlabel('$ \frac{a_y}{g}$')
    ylabel('$\delta - \rho L$ [rad]')
    title('Handling Diagram','FontSize',18)
    plot(norm_acc,hand_curve,'Color',"#D95319",'LineWidth',2)
    %plot(norm_acc,hand_curve_a,'--y','LineWidth',2)
    plot(norm_acc(1:linear_index),hand_lin_curve,'-.c','LineWidth',1.5)
    plot(norm_acc_nonlin,hand_nonlin_curve,'-.b','LineWidth',1.5)
    plot(norm_acc,zeros(size(norm_acc)),'LineWidth',1,'Color',"#77AC30")
    text(norm_acc(end) + 0.03, 0, "NS", 'Color',"#77AC30",'FontSize', 15)
    quiver(x(1),y(1),x(2)-x(1),y(2)-y(1),0,'MaxHeadSize',15,'LineWidth',1.5,'Color','#A2142F');
    text(r_k*cos(theta_k)+0.02,r_k*sin(theta_k)/2,'$K_{US}$','Color',"#A2142F",'FontSize',20)
    xlim([0 1.7])
    ylim([-11*10^(-3) 10^(-3)])
    legend('Handling curve','Linear Fitting', 'Polynomial Fitting','Location','east')
    hold off

    %% Yaw Rate Gain - Beta Gain - Acceleration Gain
    % Compute the gains only with the test with speed ramp and constant
    % steering angle 

    if type_test == 2

        %%Yaw Rate Gain
    
        yr_gain = Omega_ss./(delta_ss.*tau_D);
    
        figure('Name','Yaw Rate Gain','NumberTitle','off'), clf
        set(gca,'fontsize',16)
        hold on
        xlabel('$u[\frac{km}{h}]$')
        ylabel('$\frac{\Omega}{\delta}$','Interpreter','latex')
        title('Yaw Rate Gain','FontSize',18)
        plot(u_ss*3.6,  yr_gain,'LineWidth',2)
    
        %%Beta Gain
        % % 
        % beta_gain_formula = Lr/L - (m/L^3)*((Lf/Cyr)+(Lr/Cyf))*((u_ss.^2)./(1 + K_us.*(u_ss.^2)));
        
        beta_gain = beta_ss./(delta_ss.*tau_D);
    
        figure('Name','Beta Gain','NumberTitle','off'), clf
        set(gca,'fontsize',16)
        hold on
        xlabel('$u [\frac{km}{h}]$')
        ylabel('$\frac{\beta}{\delta}$','Interpreter','latex')
        title('$\beta$ Gain','FontSize',18)
        plot(u_ss*3.6,beta_gain,'LineWidth',2)
        % plot(u_ss,beta_gain_formula,'LineWidth',2,'--y')
        

        %%Acceleration Gain
    
        acc_gain = Ay_ss./(delta_ss.*tau_D);
    
        figure('Name','Acceleration Gain','NumberTitle','off'), clf
        set(gca,'fontsize',16)
        hold on
        xlabel('$u[\frac{km}{h}]$')
        ylabel('$\frac{a_y}{\delta}$','Interpreter','latex')
        title('Acceleration Gain','FontSize',18)
        plot(u_ss*3.6,acc_gain,'LineWidth',2)
    end

end
    
