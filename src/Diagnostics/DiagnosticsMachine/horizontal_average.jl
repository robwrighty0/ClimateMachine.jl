"""
    HorizontalAverage

A horizontal reduction into a single vertical dimension.
"""
abstract type HorizontalAverage <: DiagnosticVar end
dv_HorizontalAverage(
    ::ClimateMachineConfigType,
    ::Union{HorizontalAverage},
    ::BalanceLaw,
    ::States,
    ::AbstractFloat,
) = nothing

dv_dg_points_range(::ClimateMachineConfigType, ::Type{HorizontalAverage}) = :(Nqk)
dv_dg_points_index(::ClimateMachineConfigType, ::Type{HorizontalAverage}) = :(k)

dv_dg_elems_range(::ClimateMachineConfigType, ::Type{HorizontalAverage}) = :(nvertelem)
dv_dg_elems_index(::ClimateMachineConfigType, ::Type{HorizontalAverage}) = :(ev)

dv_dimnames(::ClimateMachineConfigType, ::HorizontalAverage, ::Any) = ("z",)

dv_op(::ClimateMachineConfigType, ::HorizontalAverage, x, y) = x += y

macro horizontal_average(impl, config_type, name)
    iex = quote
        $(generate_dv_interface(:HorizontalAverage, config_type, name))
        $(generate_dv_function(:HorizontalAverage, config_type, [name], impl))
    end
    esc(MacroTools.prewalk(unblock, iex))
end

macro horizontal_average(
    impl,
    config_type,
    name,
    units,
    long_name,
    standard_name,
)
    iex = quote
        $(generate_dv_interface(
            :HorizontalAverage,
            config_type,
            name,
            units,
            long_name,
            standard_name,
        ))
        $(generate_dv_function(:HorizontalAverage, config_type, [name], impl))
    end
    esc(MacroTools.prewalk(unblock, iex))
end
