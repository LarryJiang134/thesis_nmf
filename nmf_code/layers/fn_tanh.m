function [output, dv_input, grad] = fn_tanh(input, params, hyper_params, backprop, dv_output)
% tanh activation

% tmp1 = exp(input);
% tmp2 = exp(-input);
% output = (exp(input) - exp(-input)) ./ (exp(input) + exp(-input));
output = tanh(input);

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
%     tmp1 = exp(dv_output);
%     tmp2 = exp(-dv_output);
%     tmp3 = (tmp1 - tmp2) ./ (tmp1 + tmp2);
%     tmp4 = 1 - tmp3.*tmp3;
%     disp(tmp4);
    sech_t = sech(dv_output);
    dv_input = sech_t.*sech_t;
%     disp(dv_input);
end