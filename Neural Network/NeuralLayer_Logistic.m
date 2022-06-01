%% NeuralLayer_Logistic
% _ChanGingSuny_ 2019-01-16 v1.0
% 
% Logistic layer (a.k.a. Sigmoid or Soft step).

classdef NeuralLayer_Logistic < NeuralLayer
    methods
        %% Constructor
        function NL=NeuralLayer_Logistic(Arg1,Arg2)
            NL@NeuralLayer(Arg1,Arg2);
        end
        
        %% Active Function
        function ActiveFunc(NL)
            NL.outputArg=1./(1+exp(-NL.activeArg));
        end
        
        %% Gradient of the Active Function
        function grad_a=gradActiveFunc(NL)
            grad_a=NL.outputArg.*(1-NL.outputArg).*NL.outputGrad;
        end
        
    end
end
