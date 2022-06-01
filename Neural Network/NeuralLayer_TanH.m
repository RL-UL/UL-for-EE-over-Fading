%% NeuralLayer_TanH
% _ChanGingSuny_ 2019-01-16 v1.0
% 
% TanH layer.

classdef NeuralLayer_TanH < NeuralLayer
    methods
        %% Constructor
        function NL=NeuralLayer_TanH(Arg1,Arg2)
            NL@NeuralLayer(Arg1,Arg2);
        end
        
        %% Active Function
        function ActiveFunc(NL)
            NL.outputArg=tanh(NL.activeArg);
        end
        
        %% Gradient of the Active Function
        function grad_a=gradActiveFunc(NL)
            grad_a=(1-NL.outputArg.*NL.outputArg).*NL.outputGrad;
        end
        
    end
end
