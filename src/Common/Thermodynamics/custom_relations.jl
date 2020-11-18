#####
##### CustomPhaseDry_ρp
#####

const CustomPhase_ρp =
    Union{CustomPhaseDry_ρp, CustomPhaseEquil_ρpq, CustomPhaseNonEquil_ρpq}

PhasePartition(ts::CustomPhaseDry_ρp{FT}) where {FT <: Real} = q_pt_0(FT)
PhasePartition(ts::CustomPhaseEquil_ρpq) = PhasePartition_equil(
    ts.param_set,
    air_temperature(ts),
    air_density(ts),
    total_specific_humidity(ts),
    typeof(ts),
)
PhasePartition(ts::CustomPhaseNonEquil_ρpq) = ts.q

air_pressure(ts::CustomPhase_ρp) = ts.p

internal_energy(ts::CustomPhase_ρp) =
    internal_energy(ts.param_set, air_temperature(ts), PhasePartition(ts))

air_temperature(ts::CustomPhase_ρp) = air_temperature_from_ideal_gas_law(
    ts.param_set,
    air_pressure(ts),
    air_density(ts),
    PhasePartition(ts),
)
