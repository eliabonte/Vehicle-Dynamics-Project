function suspAnalysis(sim_vec,vehicle_data,type_test, t_steer, Ts,Ks_r0,Ks_f0)

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
    hrr  = vehicle_data.rear_suspension.h_rc_r;
    hrf  = vehicle_data.front_suspension.h_rc_f;

    tau_D = vehicle_data.steering_system.tau_D;  % [-] steering system ratio (pinion-rack)

    Ks_f_vec = [Ks_f0-(20*Ks_f0/100), Ks_f0, Ks_f0+(20*Ks_f0/100)];

    for i = 1:length(sim_vec)
        Ks_f(i) = Ks_f_vec(i);
        eps_s(i) = Ks_f(i)/(Ks_r0 + Ks_f(i));
    end
    % ---------------------------------
    %% Extract data from simulink model
    % ---------------------------------
    time_sim = sim_vec(1).states.u.time;

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


    for i = 1:length(sim_vec)

        delta_D{i}       = sim_vec(i).inputs.delta_D.data;
        delta_D_ss{i}    = delta_D{i}(index_ss:end);

        u{i}          = sim_vec(i).states.u.data;
        v{i}          = sim_vec(i).states.v.data;
        Omega{i}      = sim_vec(i).states.Omega.data;
        Fz_rr{i}      = sim_vec(i).states.Fz_rr.data;
        Fz_rl{i}      = sim_vec(i).states.Fz_rl.data;
        Fz_fr{i}      = sim_vec(i).states.Fz_fr.data;
        Fz_fl{i}      = sim_vec(i).states.Fz_fl.data;
        delta{i}      = sim_vec(i).states.delta.data;
        
        u_ss{i}          = u{i}(index_ss:end);
        v_ss{i}          = v{i}(index_ss:end);
        Omega_ss{i}      = Omega{i}(index_ss:end);
        Fz_rr_ss{i}      = Fz_rr{i}(index_ss:end);
        Fz_rl_ss{i}      = Fz_rl{i}(index_ss:end);
        Fz_fr_ss{i}      = Fz_fr{i}(index_ss:end);
        Fz_fl_ss{i}      = Fz_fl{i}(index_ss:end);
        delta_ss{i}      = delta{i}(index_ss:end);

        % -----------------
        % Extra Parameters
        % -----------------
        Fx_fr{i}      = sim_vec(i).extra_params.Fx_fr.data;
        Fx_fl{i}      = sim_vec(i).extra_params.Fx_fl.data;
        Fy_rr{i}      = sim_vec(i).extra_params.Fy_rr.data;
        Fy_rl{i}      = sim_vec(i).extra_params.Fy_rl.data;
        Fy_fr{i}      = sim_vec(i).extra_params.Fy_fr.data;
        Fy_fl{i}      = sim_vec(i).extra_params.Fy_fl.data;
        delta_fr{i}   = sim_vec(i).extra_params.delta_fr.data;
        delta_fl{i}   = sim_vec(i).extra_params.delta_fl.data;

        Fx_fr_ss{i}       = Fx_fr{i}(index_ss:end);
        Fx_fl_ss{i}       = Fx_fl{i}(index_ss:end);
        Fy_rr_ss{i}       = Fy_rr{i}(index_ss:end);
        Fy_rl_ss{i}       = Fy_rl{i}(index_ss:end);
        Fy_fr_ss{i}       = Fy_fr{i}(index_ss:end);
        Fy_fl_ss{i}       = Fy_fl{i}(index_ss:end);
        delta_fr_ss{i}    = delta_fr{i}(index_ss:end);
        delta_fl_ss{i}    = delta_fl{i}(index_ss:end);

        
        delta_ss{i}   = deg2rad(delta_D_ss{i}./tau_D);

        % Chassis side slip angle beta [rad]
        beta_ss{i} = atan(v_ss{i}./u_ss{i});
    
        % -----------------
        % Accelerations
        % -----------------
     
        % Steady state lateral acceleration
        Ay_ss{i} = Omega_ss{i}.*u_ss{i};
    
        % -----------------
        % Other parameters
        % -----------------
        % Steady state and transient curvature [m]
        rho_ss{i}   = Omega_ss{i}./u_ss{i};

    
        % Normalized lateral acceleration
        norm_acc{i} = Ay_ss{i}./g;

        
        % --- VERTICAL LOADS ---
        Fz_r{i} = Fz_rr_ss{i} + Fz_rl_ss{i};
        Fz_f{i} = Fz_fr_ss{i} + Fz_fl_ss{i};
        
        % --- AXLE CHARACTERISTICS ---
        Fy_R_ss{i} = Fy_rl_ss{i} + Fy_rr_ss{i};
        Fy_F_ss{i} = sin(delta_fl_ss{i}).*Fx_fl_ss{i} + cos(delta_fl_ss{i}).*Fy_fl_ss{i} + sin(delta_fr_ss{i}).*Fx_fr_ss{i} + cos(delta_fr_ss{i}).*Fy_fr_ss{i};
    
        % ---- axle side slips ----
        alpha_r_ss{i} = - (v_ss{i} - Omega_ss{i}*Lr)./u_ss{i};
        alpha_f_ss{i} =  delta_ss{i} - (v_ss{i} + Omega_ss{i}*Lf)./u_ss{i};
        alpha_r_ss{i}(1,1) = 0;
        alpha_f_ss{i}(1,1) = 0;
    
        % ---- normalized axle lateral forces ----
        Fzr0 = m*9.81*Lf/L;
        Fzf0 = m*9.81*Lr/L;
    
        mu_f_ss{i} = Fy_F_ss{i}./Fzf0;
        mu_r_ss{i} = Fy_R_ss{i}./Fzr0;
    
        % --- LATERAL LOAD TRANSFER ---
        dFz_r_ss{i} = m*Ay_ss{i}*(((Lf/L)*(hrr/Wr)) + (hs/Wr)*(1 - eps_s(i)));
        dFz_f_ss{i} = m*Ay_ss{i}*(((Lr/L)*(hrf/Wr)) + (hs/Wf)*(eps_s(i)));
    
    end

    % ---------------------------------
    %% PLOTS
    % ---------------------------------
    color_rear = "#EDB120";
    color_front = "#0072BD";
    % ---------------------------------
    %% Vertical loads
    
    figure('Name','Lateral Load Transfer','NumberTitle','off'), clf   

    % --- REAR --- %
    ax(1) = subplot(211);
    hold on
    for i = 1:length(sim_vec)
        plot(Ay_ss{i}, dFz_r_ss{i}, 'LineWidth',2)
        hold on
    end
    ylabel('$\Delta Fz_r$')
    xlabel('$a_y$')
    grid on
    ylim([0 1000])
    legend('$Ks_{f0} - 20\%$','$Ks_{f0}$','$Ks_{f0} + 20\%$')
    % title('Rear LLT with variable Rear Stiffness','FontSize',18)
    title('Rear LLT with variable Front Stiffness','FontSize',18)

    % --- FRONT --- %
    ax(2) = subplot(212);
    hold on
    for i = 1:length(sim_vec)
        plot(Ay_ss{i}, dFz_f_ss{i}, 'LineWidth',2)
        hold on
    end
    ylabel('$\Delta Fz_f$')
    xlabel('$a_y$')
    grid on
    ylim([0 1000])
    legend('$Ks_{f0} - 20\%$','$Ks_{f0}$','$Ks_{f0} + 20\%$')
    % title('Front LLT with variable Rear Stiffness','FontSize',18)
    title('Front LLT with variable Front Stiffness','FontSize',18)


    %% Plot Axle characteristics

    % -- Plot axle characteristics ---

    figure('Name','Fy_R and Fy_F normalized','NumberTitle','off'), clf
    hold on
    for i = 1:length(sim_vec)
        plot(alpha_r_ss{i}, mu_r_ss{i},'LineWidth',2)
        plot(alpha_f_ss{i}, mu_f_ss{i},'LineWidth',2)
        hold on
    end
    grid on
    xlabel('$\alpha_R,\alpha_R[rad]$')
    ylabel('$\mu_R,\mu_F$')
    legend('$\mu_R, Ks_{f0}-20\%$','$\mu_F, Ks_{f0}-20\%$','$\mu_R, Ks_{f0}$','$\mu_F, Ks_{f0}$','$\mu_R, Ks_{f0}+20\%$','$\mu_F, Ks_{f0}+20\%$')
    title('$\mu_R$ and $\mu_F$ with variable Front Stiffness','FontSize',18)
    xlim([0 0.1]);
    ylim([0 1.7]);

    % figure('Name','Fy_F normalized','NumberTitle','off'), clf
    % hold on
    % for i = 1:length(sim_vec)
    %     plot(alpha_f_ss{i}, mu_f_ss{i},'LineWidth',2)
    %     hold on
    % end
    % grid on
    % xlabel('$\alpha_F [rad]$')
    % ylabel('$\mu_F$')
    % legend('$Ks_{f0} - 20\%$','$Ks_{f0}$','$Ks_{f0} + 20\%$')
    % title('$\mu_F$ with variable Front Stiffness','FontSize',18)
    % xlim([0 0.1]);
    % ylim([0 1.7]);


    %% Plot behaviour of vehicle - Handling diagram for steer ramp test with constant velocity
   
    for i = 1:length(sim_vec)
        %hand_curve_ss{i} =  delta_ss{i} - rho_ss{i}.*L;
        hand_curve_ss{i} =  - (alpha_r_ss{i} - alpha_f_ss{i});
    end
   
    figure('Name','Handling curve','NumberTitle','off'), clf
    set(gca,'fontsize',16)
    hold on
    xlabel('$ \frac{a_y}{g}$')
    ylabel('$\delta - \rho L$')
    %title('Handling Diagram with variable Rear Stiffness','FontSize',18)
    title('Handling Diagram with variable Front Stiffness','FontSize',18)
    for i = 1:length(sim_vec)   
        plot(norm_acc{i},hand_curve_ss{i},'LineWidth',2)
    end
    plot(norm_acc{i},zeros(size(norm_acc{1})),'LineWidth',1,'Color',"#77AC30")
    text(norm_acc{i}(end) + 0.03, 0, "NS", 'Color',"#77AC30",'FontSize', 15)
    xlim([0 1.7])
    %legend('$Ks_{f0} - 20\%$','$Ks_{f0}$','$Ks_{f0} + 20\%$')
    %legend('$Ks_{f0} - 20\%$','$Ks_{f0}$','$Ks_{f0} + 20\%$')
    legend('$Ks_{f0}, Ks_{r0}$','$Ks_{f0} + 40\%, Ks_{r0} - 40\$')
    hold off
    

end
    
