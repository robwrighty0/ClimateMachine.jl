"""
    PointwiseDiagnostic

A diagnostic with the same dimensions as the original grid (DG or interpolated).
"""
abstract type PointwiseDiagnostic <: DiagnosticVar end
dv_PointwiseDiagnostic(
    ::ClimateMachineConfigType,
    ::Union{PointwiseDiagnostic},
    ::BalanceLaw,
    ::States,
    ::AbstractFloat,
) = nothing

function dv_dimnames(
    ::ClimateMachineConfigType,
    ::PointwiseDiagnostic,
    out_dims::OrderedDict,
)
    tuple(collect(keys(out_dims))...)
end

macro pointwise_diagnostic(config_type, name)
    iex = generate_dv_interface(:PointwiseDiagnostic, config_type, name)
    esc(MacroTools.prewalk(unblock, iex))
end

macro pointwise_diagnostic(config_type, name, units, long_name, standard_name)
    iex = generate_dv_interface(
        :PointwiseDiagnostic,
        config_type,
        name,
        units,
        long_name,
        standard_name,
    )
    esc(MacroTools.prewalk(unblock, iex))
end

macro pointwise_diagnostic(
    impl,
    config_type,
    name,
    units,
    long_name,
    standard_name,
)
    iex = quote
        $(generate_dv_interface(
            :PointwiseDiagnostic,
            config_type,
            name,
            units,
            long_name,
            standard_name,
        ))
        $(generate_dv_function(:PointwiseDiagnostic, config_type, [name], impl))
    end
    esc(MacroTools.prewalk(unblock, iex))
end

macro pointwise_diagnostics(impl, config_type, names...)
    exprs = [
        generate_dv_interface(:PointwiseDiagnostic, config_type, name)
        for name in names
    ]
    fex = generate_dv_function(:PointwiseDiagnostic, config_type, names, impl)
    push!(exprs, fex)
    iex = quote
        $(exprs...)
    end
    esc(MacroTools.prewalk(unblock, iex))
end

macro pointwise_diagnostic_impl(impl, config_type, names...)
    iex = generate_dv_function(:PointwiseDiagnostic, config_type, names, impl)
    esc(MacroTools.prewalk(unblock, iex))
end
