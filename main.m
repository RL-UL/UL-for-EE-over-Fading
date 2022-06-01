% Source code provided: ¡°C. Sun and C. Yang, ¡°Unsupervised deep learning for ultra-reliable and low-latency communications,¡± in 2019 IEEE Global Communications Conference (GLOBECOM), 2019, pp. 1¨C6.¡±. please cite.
% The code has been modified BY Hao.

clc,clear;
%% Parameters
W=20e6;                     % Bandwidth (Hz)
N0=10^((-173-30)/10);       % Noise power spectrum density (W/Hz)
Pmax=40;                    % Maximum transmit power (W)
Pmean=0.1;                  % Average transmit power (W)

Pp=0.06;                      % PU transmit power (W)
% Pint=[5:5:40];              % average interference power (W)
Pint=[0.01];                  % average interference power (W)

zeta = 0.2;                 % Power Eff
Pc = 0.05;

dist_min=50;                % Minimum distance from user to the BS (m)
dist_max=500;               % Maximum distance from user to the BS (m)
PathLoss=@(x) (10.^-(3.53+3.76*log10(x)));  % Path loss function

%% System Setup
alpha=PathLoss(dist_min);
% Neff=W*N0/alpha;

Neff=0.01;

for in = 1:length(Pint)
    Pin = Pint(in)
    
    test_num=100000;
    % G=1e-2:1e-2:10;
    % G=10.^(-2:1e-2:1);
    H=(1/sqrt(2))*(randn(3,test_num)+1j*randn(3,test_num)); G=abs(H).^2;
    
    %% Optimal Solution
    [PC_Opt_temp,R_Opt_temp,EE_Opt_temp] = optimal_PA(G,Neff,Pmean,Pmax,Pp,Pin,zeta,Pc);
%     [PC_Opt_temp,R_Opt_temp] = optimal_PA2(G,Neff,Pmean,Pmax,Pp,Pin);
    Pint_Opt_temp = mean(G(2,:).*PC_Opt_temp);
    Pmean_Opt_temp = mean(PC_Opt_temp);
    
    %% Learnt Solution
    [PQNN,PQ_NN_mean,PQ_NN_int,EEQ_NN_iter,PNN,R_NN_iter,iter_num,P_NN_mean,P_NN_int,EE_NN_iter]=pwrCtrl_MB(G,Pmean,Pmax,Pp,Pin,Neff,zeta,Pc);
    %PNN
    PC_Lrn=@(g) (ForwardProp(PNN,reglr(g)));
    Power_Lrn=PC_Lrn(G);
    R_Lrn_temp=log2(1+(G(3,:).*Power_Lrn)./(Neff+G(1,:)*Pp));
    R_Lrn_temp=mean(R_Lrn_temp);
    Pint_Lrn_temp = mean(G(2,:).*Power_Lrn);
    Pmean_Lrn_temp = mean(Power_Lrn);
    %PQNN
    PCQ_Lrn=@(g) (ForwardProp(PQNN,reglr(g)));
    PowerQ_Lrn=PCQ_Lrn(G);
    RQ_Lrn_temp=log2(1+(G(3,:).*PowerQ_Lrn)./(Neff+G(1,:)*Pp));
    RQ_Lrn_temp=mean(RQ_Lrn_temp);
    PintQ_Lrn_temp = mean(G(2,:).*PowerQ_Lrn);
    PmeanQ_Lrn_temp = mean(PowerQ_Lrn);
    
    % [PNN,LNN,xi,RNN]=pwrCtrl_MF(Pmean,Pmax,Neff);
    % [R_ave_Lrn,P_ave_Lrn]=perfMetric(PC_Lrn,Neff);
    % R_Lrn=ForwardProp(RNN,reglr([Power_Lrn;G]));
    
    
    R_Opt(in) = R_Opt_temp;
    EE_Opt(in) = EE_Opt_temp;
    Pint_Opt(in) = Pint_Opt_temp;
    Pmean_Opt(in) = Pmean_Opt_temp;
    
    R_Lrn(in) = R_Lrn_temp;
    Pint_Lrn(in) = Pint_Lrn_temp;
    Pmean_Lrn(in) = Pmean_Lrn_temp;
    
    RQ_Lrn(in) = RQ_Lrn_temp;
    PintQ_Lrn(in) = PintQ_Lrn_temp;
    PmeanQ_Lrn(in) = PmeanQ_Lrn_temp;
    

end

%% Plot

figure;
EE_opt_iter = EE_Opt_temp.*ones(length(iter_num),1);
plot(iter_num,EE_opt_iter, 'LineWidth',1.5,'Color', 'r');
hold on;
plot(iter_num,EE_NN_iter, 'LineWidth',1.5,'Color', 'b');
hold on;
plot(iter_num,EEQ_NN_iter, 'LineWidth',1.5,'Color', 'k');
grid on
title('EE')
xlabel('Iteration')
ylabel('Average EE (bps/Hz/Joule)')
legend('Optimal','Dinkelbach','Quadratic','Location','best')
%
figure;
P_opt_mean = Pmean_Opt_temp.*ones(length(iter_num),1);
plot(iter_num,P_opt_mean, 'LineWidth',1.5,'Color', 'r');
hold on;
plot(iter_num,P_NN_mean, 'LineWidth',1.5,'Color', 'b');
hold on;
plot(iter_num,PQ_NN_mean, 'LineWidth',1.5,'Color', 'k');
grid on
title('ATP')
xlabel('Iteration')
ylabel('Power (W)')
legend('Optimal','Dinkelbach','Quadratic','Location','best')
%
figure;
P_opt_int = Pint_Opt_temp.*ones(length(iter_num),1);
plot(iter_num,P_opt_int, 'LineWidth',1.5,'Color', 'r');
hold on;
plot(iter_num,P_NN_int, 'LineWidth',1.5,'Color', 'b');
hold on;
plot(iter_num,PQ_NN_int, 'LineWidth',1.5,'Color', 'k');
grid on
title('AIP')
xlabel('Iteration')
ylabel('Power (W)')
legend('Optimal','Dinkelbach','Quadratic','Location','best')




















