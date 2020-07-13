@pointwise_diagnostic(
    AtmosGCMConfigType,
    u,
    "m s^-1",
    "zonal wind",
    "eastward_wind",
) do (atmos::AtmosModel, states::States, curr_time)
    u = states.prognostic.ρu[1] / states.prognostic.ρ
    return u
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    v,
    "m s^-1",
    "meridional wind",
    "northward_wind",
) do (atmos::AtmosModel, states::States, curr_time)
    v = states.prognostic.ρu[2] / states.prognostic.ρ
    return v
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    w,
    "m s^-1",
    "vertical wind",
    "upward_air_velocity",
) do (atmos::AtmosModel, states::States, curr_time)
    w = states.prognostic.ρu[3] / states.prognostic.ρ
    return w
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    rho,
    "kg m^-3",
    "air density",
    "air_density",
) do (atmos::AtmosModel, states::States, curr_time)
    rho = states.prognostic.ρ
    return rho
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    et,
    "J kg^-1",
    "total specific energy",
    "specific_dry_energy_of_air",
) do (atmos::AtmosModel, states::States, curr_time)
    et = states.prognostic.ρe / states.prognostic.ρ
    return et
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    temp,
    "K",
    "air temperature",
    "air_temperature",
)
@pointwise_diagnostic(
    AtmosGCMConfigType,
    pres,
    "Pa",
    "air pressure",
    "air_pressure",
)
@pointwise_diagnostic(
    AtmosGCMConfigType,
    thd,
    "K",
    "dry potential temperature",
    "air_potential_temperature",
)
@pointwise_diagnostic(
    AtmosGCMConfigType,
    ei,
    "J kg^-1",
    "specific internal energy",
    "internal_energy",
)
@pointwise_diagnostic(
    AtmosGCMConfigType,
    ht,
    "J kg^-1",
    "specific enthalpy based on total energy",
    "",
)
@pointwise_diagnostic(
    AtmosGCMConfigType,
    hi,
    "J kg^-1",
    "specific enthalpy based on internal energy",
    "atmosphere_enthalpy_content",
)

@pointwise_diagnostic_impl(
    AtmosGCMConfigType,
    temp,
    pres,
    thd,
    ei,
    ht,
    hi,
) do (atmos::AtmosModel, states::States, curr_time)
    ts = recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    temp = air_temperature(ts)
    pres = air_pressure(ts)
    thd = dry_pottemp(ts)
    ei = internal_energy(ts)
    ht = total_specific_enthalpy(ts, e_tot)
    hi = specific_enthalpy(ts)
    return temp, pres, thd, ei, ht, hi
end

#= TODO
@XXX_diagnostic(
    "vort",
    AtmosGCMConfigType,
    GridInterpolated,
    "s^-1",
    "vertical component of relative velocity",
    "atmosphere_relative_velocity",
) do (atmos::AtmosModel, states::States, curr_time)
end
=#

@pointwise_diagnostic(
    AtmosGCMConfigType,
    qt,
    "kg kg^-1",
    "mass fraction of total water in air (qv+ql+qi)",
    "mass_fraction_of_water_in_air",
) do (
    m::Union{EquilMoist, NonEquilMoist},
    atmos::AtmosModel,
    states::States,
    curr_time,
)
    qt = states.prognostic.moisture.ρq_tot / states.prognostic.ρ
    return qt
end

@pointwise_diagnostic(
    AtmosGCMConfigType,
    ql,
    "kg kg^-1",
    "mass fraction of liquid water in air",
    "mass_fraction_of_cloud_liquid_water_in_air",
)
@pointwise_diagnostic(
    AtmosGCMConfigType,
    qv,
    "kg kg^-1",
    "mass fraction of water vapor in air",
    "specific_humidity",
)
@pointwise_diagnostic(
    AtmosGCMConfigType,
    qi,
    "kg kg^-1",
    "mass fraction of ice in air",
    "mass_fraction_of_cloud_ice_in_air",
)
@pointwise_diagnostic(
    AtmosGCMConfigType,
    thv,
    "K",
    "virtual potential temperature",
    "virtual_potential_temperature",
)
@pointwise_diagnostic(
    AtmosGCMConfigType,
    thl,
    "K",
    "liquid-ice potential temperature",
    "",
)

@pointwise_diagnostic_impl(
    AtmosGCMConfigType,
    ql,
    qv,
    qi,
    thv,
    thl,
) do (
    m::Union{EquilMoist, NonEquilMoist},
    atmos::AtmosModel,
    states::States,
    curr_time,
)
    ts = recover_thermo_state(atmos, states.prognostic, states.auxiliary)
    ql = liquid_specific_humidity(ts)
    qi = ice_specific_humidity(ts)
    qv = vapor_specific_humidity(ts)
    thv = virtual_pottemp(ts)
    thl = liquid_ice_pottemp(ts)
    has_condensate = has_condensate(ts)
    return ql, qv, qi, thv, thl
end
