function loss=NNTest(NN,SmpTest,LblTest)
%% NNTest
% _ChanGingSuny_ 2019-06-11 v1.0
% 
% Test the neural network.
% 
% *Input*
%   |NN|     	Neural network
%   |SmpTest|	Test samples
%   |LblTest| 	Labels for the test samples
% *Output*
%   |loss|     	Loss

%% Loss
y=ForwardProp(NN,reglr(SmpTest));
err=y-LblTest;
loss=sqrt(sum(err(:).*err(:)));
% loss=sqrt(sum(err(:).*err(:))/sum(LblTest(:).*LblTest(:)));
end
