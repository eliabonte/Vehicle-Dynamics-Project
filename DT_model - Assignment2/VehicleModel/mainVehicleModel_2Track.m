% ----------------------------------------------------------------
%% Main script for a basic simulation framework with a double track vehcile model
%  authors: 
%  rev. 1.0 Mattia Piccinini & Gastone Pietro Papini Rosati
%  rev. 2.0 Edoardo Pagot
%  date:
%  rev 1.0:    13/10/2020
%  rev 2.0:    16/05/2022
%  rev 2.1:    08/07/2022 (Biral)
%       - added Fz saturation. Correceted error in Fx
%       - initial condition is now parametric in initial speed
%       - changed the braking torque parameters to adapt to a GP2 model
% ----------------------------------------------------------------

% ----------------------------
%% Initialization
% ----------------------------
initialize_environment;

% ----------------------------
%% Load vehicle data
% ----------------------------

%test_tyre_model; % some plot to visualize the curvers resulting from the
%   loaded data

vehicle_data = getVehicleDataStruct();
% pacejkaParam = loadPacejkaParam();


% ----------------------------
%% Define initial conditions for the simulation
% ----------------------------
V0 = 0; % Initial speed
X0 = loadInitialConditions(V0);

% ----------------------------
%% Define the desired speed
% ----------------------------
V_des = 80/3.6; % Initial speed

% ----------------------------
%% Simulation parameters
% ----------------------------
simulationPars = getSimulationParams(); 
Ts = simulationPars.times.step_size;  % integration step for the simulation (fixed step)
T0 = simulationPars.times.t0;         % starting time of the simulation
Tf = simulationPars.times.tf;         % stop time of the simulation

% ----------------------------
%% Start Simulation
% % ----------------------------

%gamma_f0 = vehicle_data.front_wheel.static_camber;
% vehicle_data.front_wheel.static_camber = -2;
% 
fprintf('Starting Simulation\n')
tic;
model_sim = sim('Vehicle_Model_2Track');
elapsed_time_simulation = toc;
fprintf('Simulation completed\n')
fprintf('The total simulation time was %.2f seconds\n',elapsed_time_simulation)

% ----------------------------
%% Post-Processing with basic data
% ----------------------------
dataAnalysis(model_sim,vehicle_data,Ts);

%% Steady-State analysis - Handling Behaviour

ssAnalysis(model_sim,vehicle_data,Ts);

%% Further analysis on the behaviour of the vehicle - Different suspension stiffness
% Ks_r0 = vehicle_data.rear_suspension.Ks_r;
% Ks_f0 = vehicle_data.front_suspension.Ks_f;
% %Ks_r_vec = [Ks_r0-(20*Ks_r0/100), Ks_r0, Ks_r0+(20*Ks_r0/100)];
% Ks_f_vec = [Ks_f0-(20*Ks_f0/100), Ks_f0, Ks_f0+(20*Ks_f0/100)];
% 
% for i=1:3
%     fprintf('Starting Simulation %d\n',i)
%     tic;
%     %vehicle_data.rear_suspension.Ks_f = Ks_r_vec(i);
%     vehicle_data.front_suspension.Ks_f = Ks_f_vec(i);
%     model_sim = sim('Vehicle_Model_2Track');
%     sim_vec(i) = model_sim;
%     elapsed_time_simulation = toc;
%     fprintf('Simulation %d completed\n', i)
%     fprintf('The total simulation time was %.2f seconds\n',elapsed_time_simulation)
% end
% %%
% suspAnalysis(sim_vec,vehicle_data,Ts,Ks_r0,Ks_f0);


%% Further analysis on the behaviour of the vehicle - Different camber 

% gamma_f0 = vehicle_data.front_wheel.static_camber;
% gamma_vec = [-2 0 2];
% 
% for i=1:3
%     fprintf('Starting Simulation %d\n',i)
%     tic;
%     vehicle_data.front_wheel.static_camber = gamma_vec(i);
%     model_sim = sim('Vehicle_Model_2Track');
%     sim_vec(i) = model_sim;
%     elapsed_time_simulation = toc;
%     fprintf('Simulation %d completed\n', i)
%     fprintf('The total simulation time was %.2f seconds\n',elapsed_time_simulation)
% end
%%
% camberAnalysis(sim_vec,vehicle_data,Ts);  

%% Further analysis on the behaviour of the vehicle - Different toe angles 

% vehicle_data.steering_system.toe = 0;
% toe_vec = [-2 0 2];
% 
% for i=1:3
%     fprintf('Starting Simulation %d\n',i)
%     tic;
%     vehicle_data.steering_system.toe = toe_vec(i);
%     model_sim = sim('Vehicle_Model_2Track');
%     sim_vec(i) = model_sim;
%     elapsed_time_simulation = toc;
%     fprintf('Simulation %d completed\n', i)
%     fprintf('The total simulation time was %.2f seconds\n',elapsed_time_simulation)
% end
% 
% %%
% toeAnalysis(sim_vec,vehicle_data,Ts);


% %vehicleAnimation(model_sim,vehicle_data,Ts);
