function pce_list = compute_pce_with_fingerprint(res_list, fp)
% For each residual in a list of noise residuals, compute its pce correlation with a fingerpint.
% INPUT:
%   res_list      A list of residual file paths
%   fp            the fingerprint
% OUTPUT:
%   pce_list      a list of PCE values
pce_list = zeros(1, length(res_list));
for i = 1 : length(res_list)
    residual_path = res_list(i);
    Residual = load(char(residual_path));
    ccorr = crosscorr(Residual.Noise,fp);
    out = PCE(ccorr);
    pce_list(1, i) = out.PCE;
end
