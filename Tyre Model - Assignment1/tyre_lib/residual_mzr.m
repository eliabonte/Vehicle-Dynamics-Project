% Implement the self-aligning residual moment formula
function [mzr] = residual_mzr(alpha__r, alpha, Br, Dr)

 % precode

  

 % main code

  t1 = (Br ^ 2);
  t2 = (alpha__r ^ 2);
  t5 = sqrt((t2 * t1 + 1));
  t8 = cos(alpha);
  mzr = t8 / t5 * Dr;
  
 end
