%% NeuralLayer_Softmax
% _ChanGingSuny_ 2019-01-15 v1.0
% 
% Softmax layer.

classdef NeuralLayer_Softmax < NeuralLayer
    methods
        %% Constructor
        function NL=NeuralLayer_Softmax(Arg1,Arg2)
            NL@NeuralLayer(Arg1,Arg2);
        end
        
        %% Active Function
        function ActiveFunc(NL)
            expa=exp(NL.activeArg);
            NL.outputArg=expa./(sum(expa));
        end
        
        %% Gradient of the Active Function
        function grad_a=gradActiveFunc(NL)
            prodOut=NL.outputArg.*NL.outputGrad;
            grad_a=prodOut-NL.outputArg.*sum(prodOut);
        end
        
    end
end
