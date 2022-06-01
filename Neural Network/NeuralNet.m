%% NeuralNet
% _ChanGingSuny_ 2019-01-16 v1.0
% 
% Neural network.

classdef NeuralNet < handle
    properties
        dim             % Dimension of each layer
        typeLayer       % Type of the active function in each layer
        Layers          % Layers
        learningRate=1	% Learning rate
    end
    
    
    
    methods
        %% Constructor and Destructor
        % Constructor
        function NN=NeuralNet(Dims,Types,learningRate)
            % _ChanGingSuny_ 2019-01-16 v1.0
            % 
            % Construct a new neural network.
            % 
            % *Input*
            %   |dims|          Dimension of each layer
            %   |types|         Type of the active function in each layer
            %   |learningRate|  Learning rate
            % *Output*
            %   |NN|            New neural network
            
            % Initialization
            numDim=numel(Dims);
            numType=numel(Types);
            mustBeLargerByOne(numDim,numType);
            
            NN.dim=Dims(:);
            NN.typeLayer{numType,1}=[];
            if nargin>2, NN.learningRate=learningRate; end
            
            % Construction
            NN.Layers{numType,1}=[];
            for l=1:numType, type=Types{l};
                if      any(strcmpi(type,{'softmax','soft max'}))
                    NN.typeLayer{l}='Softmax';
                    constructor=@NeuralLayer_Softmax;
                elseif  any(strcmpi(type,'tanh'))
                    NN.typeLayer{l}='TanH';
                    constructor=@NeuralLayer_TanH;
                elseif  any(strcmpi(type,{'logistic','sigmoid','softstep','soft step'}))
                    NN.typeLayer{l}='Logistic';
                    constructor=@NeuralLayer_Logistic;
                elseif  any(strcmpi(type,{'prelu','leakyrelu','leaky relu'}))
                    NN.typeLayer{l}='PReLU';
                    constructor=@NeuralLayer_PReLU;
                elseif  any(strcmpi(type,'relu'))
                    NN.typeLayer{l}='ReLU';
                    constructor=@(arg1,arg2) NeuralLayer_PReLU(arg1,arg2,0);
                elseif  any(strcmpi(type,{'bentidentity','bent identity','bentid'}))
                    NN.typeLayer{l}='BentId';
                    constructor=@NeuralLayer_BentId;
                elseif  any(strcmpi(type,{'softplus','soft plus'}))
                    NN.typeLayer{l}='SoftPlus';
                    constructor=@NeuralLayer_SoftPlus;
                else
                    error('Unsupported active function.');
                end
                NN.Layers{l}=constructor(Dims(l),Dims(l+1));
                if l>1, ConnectLayer(NN.Layers{l-1},NN.Layers{l}); end
            end
        end
        
        % Destructor
        function ClearNet(NN)
            % _ChanGingSuny_ 2019-01-16 v1.0
            % 
            % Destruct the neural network.
            % 
            % *Input*
            %   |NN|        The neural network
            
            % Destruct the neural network
            ClearLayer(NN.Layers{1});
            delete(NN);
        end
        
        % Copy
        function NN_copy=copy(NN)
            % _ChanGingSuny_ 2019-04-18 v1.0
            % 
            % Copy the neural network.
            % 
            % *Input*
            %   |NN|        The original neural network
            % *Output*
            %   |NN_copy| 	The copy of the neural network
            
            % Copy
            NN_copy=NeuralNet(NN.dim,NN.typeLayer,NN.learningRate);
            copy(NN_copy.Layers{1},NN.Layers{1});
        end
        
        
        %% Connect
        % Connect two neural networks
        function NN=ConnectNet(NN_front,NN_back)
            % _ChanGingSuny_ 2019-01-16 v1.0
            % 
            % Connect two neural networks.
            % 
            % *Input*
            %   |NN_prev|	Front neural network
            %   |NN_next|	Back neural network
            % *Output*
            %   |NN|        The new neural network
            
            % Connect the neural networks
            ConnectLayer(NN_front.outputLayer,NN_back.inputLayer);
            NN.dim=[NN_front.dim(:);NN_back.dim(2:end)];
            NN.typeLayer={NN_front.typeLayer{:};NN_back.dimtypeLayer{:}};
            NN.Layers={NN_front.Layers;NN_back.Layers};
            NN.learningRate=max(NN_front.learningRate,NN_back.learningRate);
        end
        
        
        %% Forward & Backward Propagation
        % Forward Propagation
        function arg_out=ForwardProp(NN,arg_input)
            NN.Layers{1}.inputArg=arg_input;
            ForwardProp(NN.Layers{end});
            arg_out=NN.Layers{end}.outputArg;
        end
        
        % Backward Propagation
        function grad_input=BackwardProp(NN,grad_output,alpha)
            if nargin>2, NN.learningRate=alpha; end
            NN.Layers{end}.outputGrad=grad_output;
            BackwardProp(NN.Layers{1},NN.learningRate);
            grad_input=NN.Layers{1}.inputGrad;
        end
        
        %% Gradient
        % Norm of the gradient w.r.t. the parameters
        function norm_grad=NormGrad(NN,p)
            if nargin<2, p=2; end
            norm_p=0;
            Layer=NN.Layers{1};
            
            while ~isempty(Layer)
                normG=NormGrad(Layer,p);
                norm_p=norm_p+normG^p;
                Layer=Layer.next;
            end
            norm_grad=norm_p^(1/p);
        end
        
        % Get the gradient w.r.t. the parameters
        function grad=GetGrad(NN)
            numParam=sum((NN.dim(1:end-1)+1).*NN.dim(2:end));
            grad=zeros(numParam,1);
            
            Layer=NN.Layers{1}; nEnd=1;
            while ~isempty(Layer), nSrt=nEnd;
                nEnd=nSrt-1+(Layer.inputDim+1)*Layer.outputDim;
                grad(nSrt:nEnd)=[Layer.weightGrad(:);Layer.biasGrad(:)];
                Layer=Layer.next;
            end
        end
        
    end
end





%% Validation Functions

function mustBeLargerByOne(numDim,numType)
% _ChanGingSuny_ 2019-01-16 v1.0
% 
% |dim| must have one more element than |Type|.

% Validation
if numDim ~= numType + 1
    error('The dimension array must have one more element than the type array.');
end
end









