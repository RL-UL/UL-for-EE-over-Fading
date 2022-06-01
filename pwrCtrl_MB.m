function [PQNN,PQ_mea,PQ_in,EEQ_ave,PNN,R_ave,iter_num,P_mea,P_in,EE_ave]=pwrCtrl_MB(G_test,Pmean,Pmax,Pp,Pin,Neff,zeta,Pc,PNN0,xi0,eta0)
%% pwrCtrl_MB

% Find the suboptimal power control policy to maximize the ergodic capacity
% based on the Shannon's capacity.
% 
% *Input*
%   |Pmean|     Average transmit power (W)
%   |Pmax|     	Maximum transmit power (W)
%   |Neff|      Power of the effective noise at the transmitter
%   |PNN0|   	Initial network parameters of the power control
%   |LNN0|   	Initial network parameters of the Lagrange multipliers
%   |xi0|       Initial values of the Lagrange multipliers
% *Output*
%   |PNN|       Neural network for power control
%   |LNN|       State variant Lagrange multiplier
%   |xi|        Lagrange multiplier
%   |R_ave|   	Average capacity (bits/block)

%% Parameters
lrnRt0=1;
decayRt=0;

LrnRtxi=5e-4;  % 干扰
LrnRteta=5e-4; % 平均
nnLrnRtP=5e-4; % PNN网络

nnLrnRtPQ=5e-4; % PQNN网络

ScalarX=1;
ScalarE=1;

Tmax_iter = 1e0;% 大循环
batchSz = 1e2;
numiter = 1e3; % 这里不是epoch其实是一个epoch的迭代次数
train_sample = numiter*batchSz;
numepoch = 1e3;

rho=0.1;
Kmax=10;

Tmax = Tmax_iter*numiter*numepoch;
iter_num = [1:1:Tmax];
%% Initializations
if nargin<9 || isempty(PNN0) || isempty(xi0) || isempty(eta0) % nargin是“number of input arguments”的缩写，判断输入变量个数，可能是为了判断是不是有预训练的模型，我觉得这里应该用&&
    PNN=PNNGen; % 初始化P网络
    PQNN=PNNGen; % 初始化PQ网络
%     xi=0.5+0.5*rand; % 初始化因子xi  %%这个一定要初始化不能为0不然很难去满足平均功率约束
%     eta=0.5+0.5*rand; % 初始化因子eta
    xi=0.1; % 初始化因子xi  %%这个一定要初始化不能为0不然很难去满足平均功率约束
    eta=0.1; % 初始化因子eta

%     xi=0; % 初始化因子xi
%     eta=0; % 初始化因子eta
    xiQ=0.1;
    etaQ=0.1;
else
    PNN=PNN0; PQNN=PNN0; xi=xi0; eta=eta0; xiQ=xi0; etaQ=eta0;
end

PNN_best = [];
PQNN_best = [];
%% SGD  
t=0; 
R_ave=zeros(Tmax,1);P_mea=zeros(Tmax,1);P_in=zeros(Tmax,1);EE_ave=zeros(Tmax,1);
RQ_ave=zeros(Tmax,1);PQ_mea=zeros(Tmax,1);PQ_in=zeros(Tmax,1);EEQ_ave=zeros(Tmax,1);

%% 代码是错的，这里要先预定义一下训练集 
H_total=(1/sqrt(2))*(randn(3,train_sample)+1j*randn(3,train_sample)); G_total=abs(H_total).^2; % 随机信道的幅度的平方
tt=1;
while t<Tmax_iter, t=t+1; lrnRt=lrnRt0/(1+decayRt*(t-1));
    scaltao=t-1;
    scaltao(scaltao~=0)=1;
%     H=(1/sqrt(2))*(randn(3,batchSz)+1j*randn(3,batchSz)); G=abs(H).^2; % 随机信道的幅度的平方
    G=G_total;
    %PNN
    P_tao=ForwardProp(PNN,reglr(G));
    R_tao=log2(1+(G(3,:).*P_tao)./(Neff+G(1,:)*Pp));
    tao=mean(R_tao)/mean(zeta*P_tao+Pc);
    tao=scaltao*tao;
    %PQNN
    PQ_tao=ForwardProp(PQNN,reglr(G));
    RQ_tao=log2(1+(G(3,:).*PQ_tao)./(Neff+G(1,:)*Pp));
    taoQ=sqrt(mean(RQ_tao))/mean(zeta*PQ_tao+Pc);
    taoQ=taoQ;
    
    for nepo=1:numepoch
        G_total_chaos = G_total(:,randperm(size(G_total,2)));
        for niter=1:numiter
            G = G_total_chaos(:,niter:(niter-1+batchSz));
            %PNN
            P=ForwardProp(PNN,reglr(G)); % reglr是归一化输入，因为最大取10，这个时候概率已经很低了，按照10归一化就可以了
            [gradP,gradX,gradE]=gradLossFunc(Pmean,Pmax,Pp,Pin,Neff,G,P,xi/ScalarX,eta/ScalarE,zeta,Pc,tao);
            gradX_ave=sum(gradX,2)/batchSz;
            gradE_ave=sum(gradE,2)/batchSz;
            
            dxi=LrnRtxi*lrnRt*gradX_ave;
            xi_nxt=xi+dxi;
            %         k=0;
            %         while any(xi_nxt<=0)
            %             k=k+1; if k>Kmax, xi_nxt=xi; break; end
            %             dxi=rho*dxi; xi_nxt=xi+dxi;
            %         end, xi=xi_nxt;
            xi=max(xi_nxt,0);
            
            deta=LrnRteta*lrnRt*gradE_ave;
            eta_nxt=eta+deta;
            %         k=0;
            %         while any(eta_nxt<=0)
            %             k=k+1; if k>Kmax, eta_nxt=eta; break; end
            %             deta=rho*deta; eta_nxt=eta+deta;
            %         end, eta=eta_nxt;
            eta=max(eta_nxt,0);
            %         eta=0;
            
            BackwardProp(PNN,-gradP,nnLrnRtP*lrnRt);
            
            %PQNN
            PQ=ForwardProp(PQNN,reglr(G)); % reglr是归一化输入，因为最大取10，这个时候概率已经很低了，按照10归一化就可以了
            [gradPQ,gradXQ,gradEQ]=gradLossFuncQ(Pmean,Pmax,Pp,Pin,Neff,G,PQ,xiQ/ScalarX,etaQ/ScalarE,zeta,Pc,taoQ);
            gradXQ_ave=sum(gradXQ,2)/batchSz;
            gradEQ_ave=sum(gradEQ,2)/batchSz;
            
            dxiQ=LrnRtxi*lrnRt*gradXQ_ave;
            xiQ_nxt=xiQ+dxiQ;
            xiQ=max(xiQ_nxt,0);
            
            detaQ=LrnRteta*lrnRt*gradEQ_ave;
            etaQ_nxt=etaQ+detaQ;
            etaQ=max(etaQ_nxt,0);
            
            BackwardProp(PQNN,-gradPQ,nnLrnRtPQ*lrnRt);
            
            %PNN
            P_temp=ForwardProp(PNN,reglr(G_test));
            R_temp=log2(1+(G_test(3,:).*P_temp)./(Neff+G_test(1,:)*Pp));
            R_ave(tt,1)=sum(R_temp,2)/100000; %100000是测试样本数
            
            EE_ave(tt,1)=R_ave(tt,1)/mean(zeta*P_temp+Pc);
            
            P_mea(tt,1)=mean(P_temp);
            P_in(tt,1)=mean(G_test(2,:).*P_temp);
            
            %PQNN
            PQ_temp=ForwardProp(PQNN,reglr(G_test));
            RQ_temp=log2(1+(G_test(3,:).*PQ_temp)./(Neff+G_test(1,:)*Pp));
            RQ_ave(tt,1)=sum(RQ_temp,2)/100000; %100000是测试样本数
            
            EEQ_ave(tt,1)=RQ_ave(tt,1)/mean(zeta*PQ_temp+Pc);
            
            PQ_mea(tt,1)=mean(PQ_temp);
            PQ_in(tt,1)=mean(G_test(2,:).*PQ_temp);
            
            %保存最好的PNN模型
            if isempty(PNN_best)
                if mean(P_temp)<=Pmean
                    if P_in(tt,1)<=Pin
                        PNN_best = PNN;
                        EE_best = EE_ave(tt,1);
                    end
                end
            else
                if mean(P_temp)<=Pmean
                    if P_in(tt,1)<=Pin
                        if EE_best <= EE_ave(tt,1)
                            PNN_best = PNN;
                            EE_best = EE_ave(tt,1);
                        end
                    end
                end
            end
            
            %保存最好的PQNN模型
            if isempty(PQNN_best)
                if mean(PQ_temp)<=1.02*Pmean
                    if PQ_in(tt,1)<=1.02*Pin
                        PQNN_best = PQNN;
                        EEQ_best = EEQ_ave(tt,1);
                    end
                end
            else
                if mean(PQ_temp)<=1.02*Pmean
                    if PQ_in(tt,1)<=1.02*Pin
                        if EEQ_best <= EEQ_ave(tt,1)
                            PQNN_best = PQNN;
                            EEQ_best = EEQ_ave(tt,1);
                        end
                    end
                end
            end
                        
            tt=tt+1;
        end
        
    end
    if isempty(PNN_best)
        PNN = PNN;
    else
        PNN = PNN_best;
    end
    
    if isempty(PQNN_best)
        PQNN = PQNN;
    else
        PQNN = PQNN_best;
    end

end
end



function [gradP,gradX,gradE]=gradLossFunc(Pmean,Pmax,Pp,Pin,Neff,g,P,xi,eta,zeta,Pc,tao)
%% gradLossFunc

% The gradient of the loss function w.r.t. the transmit power and the
% Lagrange multiplier.
% 
% *Input*
%   |Pmean|     Average transmit power (W)
%   |Pmax|     	Maximum transmit power (W)
%   |Neff|      Power of the effective noise at the transmitter
%   |g|         Channel gain
%   |P|         Transmit power (W)
%   |L|         State variant Lagrange multiplier
%   |xi|        Lagrange multiplier
% *Output*
%   |gradP|     Gradient w.r.t. transmit power
%   |gradL|     Gradient w.r.t. the state variant Lagrange multiplier
%   |gradX|     Gradient w.r.t. Lagrange multiplier
%   |R|      	Channel capacity

%% % dinkelbach based Gradients
R=log2(1+(g(3,:).*P)./(Neff+g(1,:)*Pp));
tao=mean(R)/mean(zeta*P+Pc);

gradP=(1./((g(1,:)*Pp+Neff)./g(3,:)+P))/log(2)-tao*zeta-eta-xi*g(2,:);
gradX=g(2,:).*P/Pin-1;
gradE=P/Pmean-1;

end

% quadratic based
function [gradPQ,gradXQ,gradEQ]=gradLossFuncQ(Pmean,Pmax,Pp,Pin,Neff,g,PQ,xiQ,etaQ,zeta,Pc,taoQ)
R = log2(1+(g(3,:).*PQ)./(Neff+g(1,:)*Pp));
A = mean(R);
taoQ = sqrt(A)/mean(zeta*PQ+Pc);

gradPQ=(taoQ./((g(1,:)*Pp+Neff)./g(3,:)+PQ))/log(2)/sqrt(A)-(taoQ^2)*zeta-etaQ-xiQ*g(2,:);
gradXQ=g(2,:).*PQ/Pin-1;
gradEQ=PQ/Pmean-1;
end
