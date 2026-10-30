using Kappa3Certificate
using Printf

family = Symbol(get(ENV, "KAPPA3_FAMILY", "star"))
parameters = Tuple(parse.(Float64, split(get(ENV, "KAPPA3_PARAMETERS", "5,5,5"), ',')))
length(parameters) == 3 || error("KAPPA3_PARAMETERS must contain three comma-separated angles")
params = (parameters[1], parameters[2], parameters[3])
level = parse(Int, get(ENV, "KAPPA3_LEVEL", "5"))
basis = Symbol(get(ENV, "KAPPA3_BASIS", "haar"))
backoff = parse(Float64, get(ENV, "KAPPA3_INITIAL_BACKOFF", "1e-4"))

@printf("Computing %s %s at red-refinement level %d\n",
        String(family), repr(params), level)
cert = point_certificate(family, params; level, basis,
                         initial_backoff = backoff)
@show cert
@printf("lambda_h >= %.17g\n", cert.lambda_lower)
@printf("hmax^2 <= %.17g\n", cert.hmax2_upper)
@printf("C(K)^2 <= %.17g\n", cert.u2_upper)
@printf("C(K) <= %.17g\n", cert.C_upper)
@printf("SPD residual shift = %.17g (roundoff %.17g, radius %.17g, margin %.17g)\n",
        cert.spd_certificate.rho,
        cert.spd_certificate.roundoff_bound,
        cert.spd_certificate.input_radius_bound,
        cert.spd_certificate.strict_margin_lower)
