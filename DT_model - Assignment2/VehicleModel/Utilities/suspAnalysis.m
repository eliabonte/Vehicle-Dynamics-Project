function suspAnalysis(sim_vec,vehicle_data,Ts)

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
    
    Ks_f = vehicle_data.front_suspension.Ks_f;
    Ks_r_vec = [15000 vehicle_data.rear_suspension.Ks_r 25000];

    for i = 1:length(sim_vec)
        Ks_r(i) = Ks_r_vec(i);
        eps_s(i) = Ks_f/(Ks_f + Ks_r_vec(i));
    end
    % ---------------------------------
    %% Extract data from simulink model
    % ---------------------------------
    time_sim = sim_vec(1).states.u.time;

    % -----------------
    % Inputs
    % -----------------
    delta_D(1,:) = sim_vec(1).inputs.delta_D.data; delta_D(2,:) = sim_vec(2).inputs.delta_D.data; delta_D(3,:) = sim_vec(3).inputs.delta_D.data;

    % -----------------
    % States
    % -----------------
    for i = 1:length(sim_vec)    
        u(i,:)          = sim_vec(i).states.u.data;
        v(i,:)          = sim_vec(i).states.v.data;
        Omega(i,:)      = sim_vec(i).states.Omega.data;
        Fz_rr(i,:)      = sim_vec(i).states.Fz_rr.data;
        Fz_rl(i,:)      = sim_vec(i).states.Fz_rl.data;
        Fz_fr(i,:)      = sim_vec(i).states.Fz_fr.data;
        Fz_fl(i,:)      = sim_vec(i).states.Fz_fl.data;

        % -----------------
        % Extra Parameters
        % -----------------
        Fx_fr(i,:)      = sim_vec(i).extra_params.Fx_fr.data;
        Fx_fl(i,:)      = sim_vec(i).extra_params.Fx_fl.data;
        Fy_rr(i,:)      = sim_vec(i).extra_params.Fy_rr.data;
        Fy_rl(i,:)      = sim_vec(i).extra_params.Fy_rl.data;
        Fy_fr(i,:)      = sim_vec(i).extra_params.Fy_fr.data;
        Fy_fl(i,:)      = sim_vec(i).extra_params.Fy_fl.data;
        delta_fr(i,:)   = sim_vec(i).extra_params.delta_fr.data;
        delta_fl(i,:)   = sim_vec(i).extra_params.delta_fl.data;
    
        % Chassis side slip angle beta [rad]
        beta(i,:) = atan(v(i,:)./u(i,:));
    
        % -----------------
        % Accelerations
        % -----------------
     
        % Steady state lateral acceleration
        Ay_ss(i,:) = Omega(i,:).*u(i,:);
    
        % -----------------
        % Other parameters
        % -----------------
        % Total CoM speed [m/s]
        vG(i,:) = sqrt(u(i,:).^2 + v(i,:).^2);
        % Steady state and transient curvature [m]
        rho_ss(i,:)   = Omega(i,:)./vG(i,:);

        % Steering angle at the wheel
        delta_f(i,:) = delta_D(i,:)/tau_D;
    
        % Normalized lateral acceleration
        norm_acc(i,:) = Ay_ss(i,:)./g;

        
        % --- VERTICAL LOADS ---
        Fz_r(i,:) = Fz_rr(i,:) + Fz_rl(i,:);
        Fz_f(i,:) = Fz_fr(i,:) + Fz_fl(i,:);
        
        % --- AXLE CHARACTERISTICS ---
        Fy_R(i,:) = Fy_rl(i,:) + Fy_rr(i,:);
        Fy_F(i,:) = sin(delta_fl(i,:)).*Fx_fl(i,:) + Fy_fl(i,:) + sin(delta_fr(i,:)).*Fx_fr(i,:) + Fy_fr(i,:);
    
        % ---- axle side slips ----
        alpha_r(i,:) = - (v(i,:) - Omega(i,:)*Lr)./u(i,:);
        alpha_f(i,:) =  delta_f(i,:)*pi/180 - (v(i,:) + Omega(i,:)*Lf)./u(i,:);
    
        alpha_r(i,:) = rad2deg(alpha_r(i,:));
        alpha_f(i,:) = rad2deg(alpha_f(i,:));

        % ---- normalized axle lateral forces ----
        Fzr0 = m*9.81*Lf/L;
        Fzf0 = m*9.81*Lr/L;

        Fy_f(i,:) = Fy_R(i,:)./Fzf0;
        Fy_r(i,:) = Fy_F(i,:)./Fzr0;
    
        % --- LATERAL LOAD TRANSFER ---
        dFz_r(i,:) = m*Ay_ss(i,:)*(((Lf/L)*(hrr/Wr)) + (hs/Wr)*(1 - eps_s(i)));
        dFz_f(i,:) = m*Ay_ss(i,:)*(((Lr/L)*(hrf/Wr)) + (hs/Wf)*(eps_s(i)));
    
    end

    % ---------------------------------
    %% PLOTS
    % ---------------------------------

    % ---------------------------------
    %% Vertical loads
    
    figure('Name','Lateral Load Transfer','NumberTitle','off'), clf   
    
    % --- REAR --- %
    ax(1) = subplot(211);
    hold on
    for i = 1:length(sim_vec)
        plot(Ay_ss(i,:), dFz_r(i,:), 'LineWidth',2)
        hold on
    end
    xlabel('$a_y$')
    grid on
    ylim([0 1000])
    legend('$Ks_r1$','$Ks_r2$','$Ks_r3$')
    title('Rear Lateral Load Transfer vs Ay (with $Ks_r1 \less Ks_r2 \less Ks_r3$)')
    % --- FRONT --- %
    ax(2) = subplot(212);
    hold on
    for i = 1:length(sim_vec)
        plot(Ay_ss(i,:), dFz_f(i,:), 'LineWidth',2)
        hold on
    end
    xlabel('$a_y$')
    grid on
    ylim([0 1000])
    legend('$Ks_r1$','$Ks_r2$','$Ks_r3$')
    title('Front Lateral Load Transfer vs Ay (with $Ks_r1 \less Ks_r2 \less Ks_r3$)')


    %% Plot Axle characteristics

    % -- Plot axle characteristics ---


    figure('Name','Fy_R normalized vs alpha_R','NumberTitle','off'), clf
    hold on
    for i = 1:length(sim_vec)
        plot(alpha_r(i,:), Fy_r(i,:),'LineWidth',2)
        hold on
    end
    grid on
    xlabel('$\alpha_R$')
    ylabel('$\mu_R$')
    legend('$Ks_r1$','$Ks_r2$','$Ks_r3$')
    title('$\mu_R$ (with $Ks_r1 \less Ks_r2 \less Ks_r3$)')
    xlim([0 6]);
    ylim([0 3]);

    figure('Name','Fy_F normalized vs alpha_F','NumberTitle','off'), clf
    hold on
    for i = 1:length(sim_vec)
        plot(alpha_f(i,:), Fy_f(i,:),'LineWidth',2)
        hold on
    end
    grid on
    xlabel('$\alpha_F$')
    ylabel('$\mu_F$')
    legend('$Ks_r1$','$Ks_r2$','$Ks_r3$')
    title('$\mu_F$ (with $Ks_r1 \less Ks_r2 \less Ks_r3$)')
    xlim([0 6]);
    ylim([0 3]);


    %% Plot behaviour of vehicle - Handling diagram for steer ramp test with constant velocity
   
    for i = 1:length(sim_vec)
        % bug sul primo elemento di alpha
        alpha_r(1,i) = 0;
        alpha_f(1,i) = 0;
        
        hand_curve(i,:) = - (alpha_r(i,:) - alpha_f(i,:)); 
        hand_curve(1,i) = 0;
    end
   
    
    figure('Name','Handling curve','NumberTitle','off'), clf
    set(gca,'fontsize',16)
    hold on
    axis equal
    xlabel('$ \frac{a_y}{g}$')
    ylabel('$\delta\tau - \rho L$')
    title('Handling Diagram (with $Ks_r1 \less Ks_r2 \less Ks_r3$)','FontSize',18)
    for i = 1:length(sim_vec)   
        plot(norm_acc(i,:),hand_curve(i,:),'LineWidth',2)
        hold on
    end
    xlim([0 2])
    legend('$Ks_r1$','$Ks_r2$','$Ks_r3$')



end
    
