%% NeuralLayer_PReLU
% _ChanGingSuny_ 2019-01-16 v1.0
% 
% Parameteric rectified linear unit layer.

classdef NeuralLayer_PReLU < NeuralLayer
    properties
        leakage=0.01	% Leakage coefficient
    end
    
    
    methods
        %% Constructor
        function NL=NeuralLayer_PReLU(Arg1,Arg2,alpha)
            NL@NeuralLayer(Arg1,Arg2);
            if nargin>2 && alpha>=0, NL.leakage=alpha; end
        end
        
        %% Active Function
        function ActiveFunc(NL)
            NL.outputArg=NL.activeArg+(NL.activeArg<0).*(NL.leakage-1).*NL.activeArg;
        end
        
        %% Gradient of the Active Function
        function grad_a=gradActiveFunc(NL)
            grad_a=NL.outputGrad+(NL.activeArg<0).*(NL.leakage-1).*NL.outputGrad;
        end
        
    end
end
