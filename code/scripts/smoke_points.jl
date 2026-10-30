using Kappa3Certificate
using Printf

level = parse(Int, get(ENV, "KAPPA3_LEVEL", "3"))
basis = Symbol(get(ENV, "KAPPA3_BASIS", "haar"))

cases = [
    (:star, (60.0, 60.0, 60.0)),
    (:star, (10.0, 10.0, 10.0)),
]

for (family, parameters) in cases
    @printf("Computing %s %s at red-refinement level %d\n",
            String(family), repr(parameters), level)
    cert = point_certificate(family, parameters; level, basis,
                             initial_backoff = 1e-7)
    @show cert
    @printf("lambda_h >= %.15g, hmax^2 <= %.15g, C(K) <= %.15g\n\n",
            cert.lambda_lower, cert.hmax2_upper, cert.C_upper)
end
