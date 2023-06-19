% Self_aligning Moment Mz
% this function remap the scalar function to its vectorial form
function [mz0_vec] = MF96_MZ0_vec(kappa_vec, alpha_vec, phi_vec, Fz_vec, tyre_data)
  
    mz0_vec = zeros(size(kappa_vec));
    for i = 1:length(kappa_vec)
    
        mz0_vec(i) = MF96_MZ0(kappa_vec(i), alpha_vec(i), phi_vec(i), Fz_vec(i), tyre_data);
    
    end
  
 end
