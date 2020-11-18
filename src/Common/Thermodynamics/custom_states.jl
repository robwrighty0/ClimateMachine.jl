# Custom thermodynamic states
export CustomPhaseDry_ρp, CustomPhaseEquil_ρpq, CustomPhaseNonEquil_ρpq

#####
##### Dry states
#####

"""
    CustomPhaseDry_ρp{FT, PS} <: AbstractPhaseDry{FT}

A dry thermodynamic state (`q_tot = 0`).

# Constructors

    CustomPhaseDry_ρp(param_set, ρ, p)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct CustomPhaseDry_ρp{FT, PS} <: AbstractPhaseDry{FT}
    "parameter set, used to dispatch planet parameter function calls"
    param_set::PS
    "density of dry air"
    ρ::FT
    "pressure"
    p::FT
end
CustomPhaseDry_ρp(param_set::APS, ρ::FT, p::FT) where {FT} =
    CustomPhaseDry_ρp{FT, typeof(param_set)}(param_set, ρ, p)

#####
##### Equilibrium states
#####

"""
    CustomPhaseEquil_ρpq{FT, PS} <: AbstractPhaseEquil{FT}

A thermodynamic state assuming thermodynamic equilibrium.

# Constructors

    CustomPhaseEquil_ρpq(param_set, ρ, p, q_tot)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct CustomPhaseEquil_ρpq{FT, PS} <: AbstractPhaseEquil{FT}
    "parameter set, used to dispatch planet parameter function calls"
    param_set::PS
    "density of air (potentially moist)"
    ρ::FT
    "pressure"
    p::FT
    "total specific humidity"
    q_tot::FT
end

"""
    CustomPhaseEquil_ρpq(param_set, ρ, p, q_tot)

Constructs a [`CustomPhaseEquil_ρpq`](@ref) thermodynamic state from:

 - `param_set` an `AbstractParameterSet`, see the [`Thermodynamics`](@ref) for more details
 - `ρ` (moist-)air density
 - `p` pressure
 - `q_tot` total specific humidity
"""
function CustomPhaseEquil_ρpq(
    param_set::APS,
    ρ::FT,
    p::FT,
    q_tot::FT,
) where {FT <: Real}
    return CustomPhaseEquil_ρpq{FT, typeof(param_set)}(param_set, ρ, p, q_tot)
end

#####
##### Non-equilibrium states
#####

"""
     CustomPhaseNonEquil_ρpq{FT} <: AbstractPhaseNonEquil{FT}

A thermodynamic state assuming thermodynamic non-equilibrium
(therefore, temperature can be computed directly).

# Constructors

    CustomPhaseNonEquil_ρpq(param_set, ρ, p, q::PhasePartition, ρ)

# Fields

$(DocStringExtensions.FIELDS)

"""
struct CustomPhaseNonEquil_ρpq{FT, PS} <: AbstractPhaseNonEquil{FT}
    "parameter set, used to dispatch planet parameter function calls"
    param_set::PS
    "density of air (potentially moist)"
    ρ::FT
    "pressure"
    p::FT
    "phase partition"
    q::PhasePartition{FT}
end
function CustomPhaseNonEquil_ρpq(
    param_set::APS,
    ρ::FT,
    p::FT,
    q::PhasePartition{FT} = q_pt_0(FT),
) where {FT}
    return CustomPhaseNonEquil_ρpq{FT, typeof(param_set)}(param_set, ρ, p, q)
end
