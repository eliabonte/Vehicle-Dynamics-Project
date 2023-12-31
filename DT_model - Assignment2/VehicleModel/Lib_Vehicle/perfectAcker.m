function [delta__fr, delta__fl] = perfectAcker(delta,vehicle_data)

    % ----------------------------------------------------------------
    %% Function purpose: compute steering angles for front wheels, with 
    %%                   perfect Ackerman steering model
    % ----------------------------------------------------------------
    
    % Load vehicle data
    Lf = vehicle_data.vehicle.Lf;        
    Lr = vehicle_data.vehicle.Lr;                 
    Wf = vehicle_data.vehicle.Wf;     
    toe = vehicle_data.steering_system.toe;
    
    % Perfect Ackermann steering law
    delta__fr = deg2rad(toe) + atan((2*Lr+2*Lf)*tan(delta)/(Wf*tan(delta)+2*Lr+2*Lf));
    delta__fl = -deg2rad(toe) - atan((2*Lr+2*Lf)*tan(delta)/(Wf*tan(delta)-2*Lr-2*Lf));

end

