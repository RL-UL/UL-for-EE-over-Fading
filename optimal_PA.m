% Source code provided: “F. Zhou, N. C. Beaulieu, Z. Li, J. Si, and P. Qi, “Energy-efficient optimal power allocation for fading cognitive radio channels: Ergodic capacity, outage capacity, and minimum-rate capacity,” IEEE Transactions on Wireless Communications, vol. 15, no. 4, pp. 2741–2755, 2016.”. please cite.

function [P_optimal,EE_rate,EE] = optimal_PA(g, noise_pwer,P_tr_aver,Pmax,P_p,P_inter_av1,zeta,Pc)
g_ss=g(3,:);
h_ps=g(1,:);
g_sp=g(2,:);

tole=10^(-4);
tole2=10^(-4);
tole3=10^(-4); 
% zeta=0.2; 
P_c=Pc; 
u_step=1e-1; 
tao_step=1e-1; 
eta=0.1;
u=0.1;
tao=0.1;
f=1;
n=1;
iter1=10;
while (f>tole)&&(n<=iter1)
     f2=1;
     while (1)
       P_optimal=max(0,((1./((eta*zeta+u*g_sp+tao)*log(2)))-(h_ps*P_p+noise_pwer)./g_ss));
       tao=tao-tao_step*(P_tr_aver-mean(P_optimal)); 
       u=u-u_step*(P_inter_av1-mean(g_sp.*P_optimal));  
       u=max(u,0);
       tao=max(tao,0);
       f2=abs(u*(P_inter_av1-mean(g_sp.*P_optimal)));
       f3=abs(tao*(P_tr_aver-mean(P_optimal)));
       if (f2<=tole2)&&(f3<=tole3)
           break;
       end   
     end
     f=mean(log2(1+(g_ss.*P_optimal)./(h_ps*P_p+noise_pwer)))-eta*mean(zeta* P_optimal+P_c);
     eta=mean(log2(1+(g_ss.*P_optimal)./(h_ps*P_p+noise_pwer)))/mean(zeta* P_optimal+P_c);
     n=n+1;
end
EE_rate=mean(log2(1+(g_ss.*P_optimal)./(h_ps*P_p+noise_pwer)));
EE=eta;

end