%% NeuralLayer
% _ChanGingSuny_ 2019-01-15 v1.0
% 
% A layer in a neural network.

classdef NeuralLayer < handle
    properties
        inputDim		% Number of input elements
        outputDim		% Number of output elements
        Weight       	% Weights
        bias           	% Bias
        prev=[]     	% Previous Layer
        next=[]     	% Next layer
        
        inputArg        % Input of the layer
        outputArg    	% Output of the layer
        activeArg       % The input of the active function of the layer
        inputGrad   	% The gradient of loss function w.r.t. the input
        outputGrad   	% The gradient of loss function w.r.t. the output
        weightGrad   	% The gradient of loss function w.r.t. the weight
        biasGrad        % The gradient of loss function w.r.t. the bias
    end
    
    
    
    methods
        %% Constructor and Destructor
        % Constructor
        function NL=NeuralLayer(Arg1,Arg2)
            % _ChanGingSuny_ 2019-01-15 v1.0
            % 
            % Construct a new layer.
            % 
            % *Input*
            %   |Arg1|      First input argument
            %                   1. Input dimension (scalar, integer)
            %                   2. Weights (matrix)
            %   |Arg2|      Scecond input argument
            %                   1. Output dimension (scalar, integer)
            %                   2. Bias (vector)
            % *Output*
            %   |NL|        New layer
            
            % Initialization
            if isscalar(Arg1) && isscalar(Arg2)
                NL.inputDim=Arg1; NL.outputDim=Arg2;
                NL.Weight=1e1*randn(NL.outputDim,NL.inputDim)/NL.inputDim;
                NL.bias=1e0*randn(NL.outputDim,1);
            elseif ismatrix(Arg1) && isvector(Arg2)
                mustHaveSameDim(Arg1,Arg2);
                NL.Weight=Arg1; NL.bias=Arg2(:);
                [NL.outputDim,NL.inputDim]=size(Arg1);
            else
                error('Unsupport input arguments.');
            end
        end
        
        % Destructor
        function ClearLayer(NL)
            % _ChanGingSuny_ 2019-02-09 v1.0
            % 
            % Destruct the layers.
            % 
            % *Input*
            %   |NL|        The last layer
            
            % Destruct the layers
            while ~isempty(NL)
                NL_next=NL.next;
                delete(NL);
                NL=NL_next;
            end
        end
        
        % Copy
        function copy(NL_copy,NL)
            % _ChanGingSuny_ 2019-04-18 v1.0
            % 
            % Copy the layer.
            % 
            % *Input*
            %   |NL_copy|   The copy of all layers
            %   |NL|        The first original layer
            
            % Copy
            NL_copy.inputDim   = NL.inputDim;
            NL_copy.outputDim  = NL.outputDim;
            NL_copy.Weight     = NL.Weight;
            NL_copy.bias       = NL.bias;
            
            NL_copy.inputArg   = NL.inputArg;
            NL_copy.outputArg  = NL.outputArg;
            NL_copy.activeArg  = NL.activeArg;
            NL_copy.inputGrad  = NL.inputGrad;
            NL_copy.outputGrad = NL.outputGrad;
            NL_copy.weightGrad = NL.weightGrad;
            NL_copy.biasGrad   = NL.biasGrad;
            
            if ~isempty(NL_copy.next) && ~isempty(NL.next)
                copy(NL_copy.next,NL.next);
            end
        end
        
        
        %% Connect and Break
        % Connect two layers
        function ConnectLayer(NL_prev,NL_next)
            % _ChanGingSuny_ 2019-01-16 v1.0
            % 
            % Connect two layers.
            % 
            % *Input*
            %   |NL_prev|	The tail of the front packet list
            %   |NL_next|	The head of the back packet list
            
            % Connect the layers
            mustBeEqual(NL_prev.outputDim,NL_next.inputDim);
            NL_prev.next=NL_next;
            NL_next.prev=NL_prev;
        end
        
        % Break the connection
        function NL_next=BreakLayer(NL)
            % _ChanGingSuny_ 2019-01-16 v1.0
            % 
            % Break the connection between two layers.
            % 
            % *Input*
            %   |NL|        The start side of the link to be broken
            % *Output*
            %   |NL_next|	The end side of the broken link
            
            % Break the Link
            NL_next=NL.next;
            NL.next=packet.empty;
            NL_next.prev=packet.empty;
        end
        
        %% Forward & Backward Propagation
        % Forward Propagation
        function ForwardProp(NL)
            if ~isempty(NL.prev)
                ForwardProp(NL.prev);
                NL.inputArg=NL.prev.outputArg;
            end
            
            NL.activeArg=NL.Weight*NL.inputArg+NL.bias;
            ActiveFunc(NL);
        end
        
        % Backward Propagation
        function BackwardProp(NL,learningRate)
            if ~isempty(NL.next)
                BackwardProp(NL.next,learningRate);
                NL.outputGrad=NL.next.inputGrad;
            end
            
            numBatch=size(NL.inputArg,2);
            
            grad_b=gradActiveFunc(NL);
            NL.weightGrad=grad_b*NL.inputArg'/numBatch;
            NL.biasGrad=sum(grad_b,2)/numBatch;
            NL.inputGrad=NL.Weight'*grad_b;
            
            % NL.Weight=max(NL.Weight-learningRate*NL.weightGrad,0);
            NL.Weight=NL.Weight-learningRate*NL.weightGrad;
            NL.bias  =NL.bias  -learningRate*NL.biasGrad;
        end
        
        %% Norm
        % Norm of the gradient w.r.t. the parameters
        function norm_grad=NormGrad(NL,p)
            if nargin<2, p=2; end
            norm_grad=norm([NL.weightGrad,NL.biasGrad],p);
        end
        
    end
    
    methods(Abstract)
        ActiveFunc(NL)              % Active function
        grad_a=gradActiveFunc(NL)   % Gradient of the active function
    end
end





%% Validation Functions

function mustHaveSameDim(W,b)
% _ChanGingSuny_ 2019-01-15 v1.0
% 
% The column dimension of the matrix |W| must equal to the dimension of the
% vector |b|.

% Validation
if size(W) ~= length(b)
    error('The weight and the bias must have the same output dimension.');
end
end

function mustBeEqual(outputDim,inputDim)
% _ChanGingSuny_ 2019-01-16 v1.0
% 
% The output dimension of the front layer must equal to the input dimension
% of the back layer.

% Validation
if outputDim ~= inputDim
    error(['The output dimension of the previous layer must equal to ',...
           'the input dimension of the next layer.']);
end
end









