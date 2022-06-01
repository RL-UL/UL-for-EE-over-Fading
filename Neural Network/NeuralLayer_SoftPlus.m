%% NeuralLayer_SoftPlus
% _ChanGingSuny_ 2019-04-19 v1.0
% 
% TanH layer.

classdef NeuralLayer_SoftPlus < NeuralLayer
    methods
        %% Constructor
        function NL=NeuralLayer_SoftPlus(Arg1,Arg2)
            NL@NeuralLayer(Arg1,Arg2);
        end
        
        %% Active Function
        function ActiveFunc(NL)
            NL.outputArg=log(1+exp(NL.activeArg));
        end
        
        %% Gradient of the Active Function
        function grad_a=gradActiveFunc(NL)
            grad_a=NL.outputGrad./(1+exp(-NL.activeArg));
        end
        
    end
end
