% Implement the Pneumatic Trail formula
function [trail] = pneumatic_trail(alpha__t, alpha, Bt, Ct, Dt, Et)

 % precode

  

 % main code

  t1 = Bt * alpha__t;
  t2 = atan(t1);
  t6 = atan(-t1 + (t1 - t2) * Et);
  t8 = cos(t6 * Ct);
  t10 = cos(alpha);
  trail = t10 * t8 * Dt;
  
 end
