function [B,T,U] = KerNIPALS(K,Y,Fac,show);

  %%% Kernel Partial Least Squares Regression - kernel-based NIPALS-PLS 
  %%%  
  %%%
  %%% Inputs  
    %     K    : kernel (Gram) matrix (number of samples  x number of samples) 
    %     y    : training outputs (number of samples  x dimY) 
    %     Fac  : number of score vectors (components,latent vectors)  to extract 
    %     show : 1 - print number of iterations needed (default) / 0 - do not print 
    %
    %     Outputs: 
    %     B      : matrix of dual-form regression coefficients (number of samples x dimY)     
    %     T,U    : matrix of latent vectors (number of samples x Fac)  

if ~exist('show')==1
  show=0;
end

%%%% max number of iterations  
maxit=20;
%%%% criterion for stopping 
crit=1e-8;

[n,n]=size(K);

T=[];
U=[];
Ie=eye(n);

Kres=K;
Yres=Y;

for num_lv=1:Fac

 %initialization 
 u=randn(n,1);t=randn(n,1);tgl=t+2;it=0;
  
 while (norm(t-tgl)/norm(t))>crit & (it<maxit)   
     tgl=t;
     it=it+1;
    
     w=Kres*u;
     t=w/norm(w);
     c=Yres'*t;
     u=Yres*c;
     u=u/norm(u);
 end
 if (num_lv > 1) 
   T=[T t];
   U=[U u];
 else 
   T=t;U=u; 
 end   

 %%% deflation procedures  
 %tt=t*t';
 %G=(Ie-tt);
 %Kres=G*Kres*G;
 Ktt=(Kres*t)*t';
 Kres=Kres-Ktt'-Ktt+(t*(t'*Ktt));
 
 Yres=Yres-t*(t'*Yres);
 
 if show~=0
     disp(' ') 
     fprintf('number of iterations: %g',it);
     disp(' ')
 end
 
end

%%% matrix of regression coefficients 
temp = T'*K*U;
%B=U*inv(temp + 1e-5 * eye(size(temp)) )*T'*Y;
B=U * (eye(size(temp)) / (temp + 1e-5 * eye(size(temp) )))*T'*Y;


