function [output, dv_input, grad] = fn_tanh(input, backprop, dv_output)
% Rectified linear unit activation function

output = tanh(input);

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
        dv_input = sech(dv_output).*sech(dv_output);
end