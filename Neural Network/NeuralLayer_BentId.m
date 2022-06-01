%% NeuralLayer_BentId
% _ChanGingSuny_ 2019-04-19 v1.0
% 
% TanH layer.

classdef NeuralLayer_BentId < NeuralLayer
    methods
        %% Constructor
        function NL=NeuralLayer_BentId(Arg1,Arg2)
            NL@NeuralLayer(Arg1,Arg2);
        end
        
        %% Active Function
        function ActiveFunc(NL)
            NL.outputArg=(sqrt(NL.activeArg.*NL.activeArg+1)-1)/2+NL.activeArg;
        end
        
        %% Gradient of the Active Function
        function grad_a=gradActiveFunc(NL)
            grad_a=(NL.activeArg./(2*sqrt(NL.activeArg.*NL.activeArg+1))+1).*NL.outputGrad;
        end
        
    end
end
