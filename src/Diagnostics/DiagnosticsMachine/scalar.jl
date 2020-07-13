"""
    ScalarDiagnostic

A reduction into a scalar value.
"""
abstract type ScalarDiagnostic <: DiagnosticVar end
dv_ScalarDiagnostic(
    ::ClimateMachineConfigType,
    ::Union{ScalarDiagnostic},
    ::BalanceLaw,
    ::States,
    ::AbstractFloat,
) = nothing

dv_dimnames(::ClimateMachineConfigType, ::ScalarDiagnostic, ::Any) = ()

macro scalar_diagnostic(config_type, name)
    iex = generate_dv_interface(:ScalarDiagnostic, config_type, name)
    esc(MacroTools.prewalk(unblock, iex))
end

macro scalar_diagnostic(config_type, name, units, long_name, standard_name)
    iex = generate_dv_interface(
        :ScalarDiagnostic,
        config_type,
        name,
        units,
        long_name,
        standard_name,
    )
    esc(MacroTools.prewalk(unblock, iex))
end

macro scalar_diagnostic(
    impl,
    config_type,
    name,
    units,
    long_name,
    standard_name,
)
    iex = quote
        $(generate_dv_interface(
            :ScalarDiagnostic,
            config_type,
            name,
            units,
            long_name,
            standard_name,
        ))
        $(generate_dv_function(:ScalarDiagnostic, config_type, [name], impl))
    end
    esc(MacroTools.prewalk(unblock, iex))
end

macro scalar_diagnostics(impl, config_type, names...)
    exprs = [
        generate_dv_interface(:ScalarDiagnostic, config_type, name)
        for name in names
    ]
    fex = generate_dv_function(:ScalarDiagnostic, config_type, names, impl)
    push!(exprs, fex)
    iex = quote
        $(exprs...)
    end
    esc(MacroTools.prewalk(unblock, iex))
end

macro scalar_diagnostic_impl(impl, config_type, names...)
    iex = generate_dv_function(:ScalarDiagnostic, config_type, names, impl)
    esc(MacroTools.prewalk(unblock, iex))
end
