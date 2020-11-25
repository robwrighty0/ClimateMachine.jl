#####
##### Tendency specification
#####

import ..BalanceLaws: eq_tends
using ..BalanceLaws: AbstractTendencyType

# Main entry point, this calls the eq_tends(::Tuple) definitions
eq_tends(m::AtmosModel, tt::AbstractTendencyType) = filter(x->x≠(),(
  eq_tends(prognostic_vars(m), m, tt)...,
  eq_tends(prognostic_vars(m.moisture), m.moisture, tt)...,
  eq_tends(prognostic_vars(m.turbconv), m.turbconv, tt)...,
  ))

# Main entry point, this calls the eq_tends(::PrognosticVariable) definitions
eq_tends(pv::Tuple, m::AtmosModel, tt::AbstractTendencyType) =
  (eq_tends(Mass(), m, tt)...,
   eq_tends(Momentum(), m, tt)...,
   eq_tends(Energy(), m, tt)...)

# eq_tends(pv::Tuple, m::AtmosModel, tt::AbstractTendencyType) =
#   (eq_tends(Mass(), m, tt)...,
#    eq_tends(Momentum(), m, tt)...,
#    eq_tends(Energy(), m, tt)...)

# MoistureModels
eq_tends(pv::Tuple, m::DryModel, tt::AbstractTendencyType) = ()
eq_tends(pv::Tuple, m::EquilMoist, tt::AbstractTendencyType) =
  (eq_tends(TotalMoisture(), m, tt)...,)
eq_tends(pv::Tuple, m::NonEquilMoist, tt::AbstractTendencyType) =
  (eq_tends(TotalMoisture(), m, tt)...,
    eq_tends(LiquidMoisture(), m, tt)...,
    eq_tends(IceMoisture(), m, tt)...)

# Fallback to no tendencies. This allows us
# to not define every entry.
eq_tends(::PV, bl, ::AbstractTendencyType) where {PV<:PrognosticVariable} =
    ()
eq_tends(::Tuple, bl, ::AbstractTendencyType) = ()

#####
##### Sources
#####

# --------- Some of these methods are generic or
#           temporary during transition to new specification:
filter_source(pv, m, s) = nothing
# Sources that have been added to new specification:
filter_source(pv::PV, m, s::Subsidence{PV}) where {PV <: PrognosticVariable} = s
filter_source(pv::PV, m, s::Gravity{PV}) where {PV <: Momentum} = s
filter_source(pv::PV, m, s::GeostrophicForcing{PV}) where {PV <: Momentum} = s
filter_source(pv::PV, m, s::Coriolis{PV}) where {PV <: Momentum} = s
filter_source(pv::PV, m, s::RayleighSponge{PV}) where {PV <: Momentum} = s
filter_source(pv::PV, m, s::CreateClouds{PV}) where {PV <: LiquidMoisture} = s
filter_source(pv::PV, m, s::CreateClouds{PV}) where {PV <: IceMoisture} = s
filter_source(
    pv::PV,
    m,
    s::RemovePrecipitation{PV},
) where {PV <: Union{Mass, Energy, TotalMoisture}} = s

filter_source(
    pv::PV,
    ::NonEquilMoist,
    s::Rain_1M{PV},
) where {PV <: LiquidMoisture} = s
filter_source(
    pv::PV,
    ::MoistureModel,
    s::Rain_1M{PV},
) where {PV <: LiquidMoisture} = nothing
filter_source(pv::PV, m::MoistureModel, s::Rain_1M{PV}) where {PV} = s

filter_source(pv::PV, m::AtmosModel, s::Rain_1M{PV}) where {PV} =
    filter_source(pv, m.moisture, s)

# Filter sources / empty elements
filter_sources(t::Tuple) = filter(x -> !(x == nothing), t)
filter_sources(pv::PrognosticVariable, m, srcs) =
    filter_sources(map(s -> filter_source(pv, m, s), srcs))

# Entry point
eq_tends(pv::PrognosticVariable, m::AtmosModel, ::Source) =
    filter_sources(pv, m, m.source)
# ---------

#####
##### First order fluxes
#####

# Mass
eq_tends(pv::PV, ::AtmosModel, ::Flux{FirstOrder}) where {PV <: Mass} =
    (Advect{PV}(),)

# Momentum
eq_tends(pv::PV, m::AtmosModel, ::Flux{FirstOrder}) where {PV <: Momentum} =
    (Advect{PV}(), PressureGradient{PV}())

# Energy
eq_tends(pv::PV, m::AtmosModel, tt::Flux{FirstOrder}) where {PV <: Energy} =
    (Advect{PV}(), Pressure{PV}())

# Moisture
eq_tends(pv::PV, ::AtmosModel, ::Flux{FirstOrder}) where {PV <: Moisture} =
    (Advect{PV}(),)

# Precipitation
eq_tends(pv::PV, ::AtmosModel, ::Flux{FirstOrder}) where {PV <: Precipitation} =
    ()

#####
##### Second order fluxes
#####

# Mass
moist_diffusion(pv::PV, ::DryModel, ::Flux{SecondOrder}) where {PV <: Mass} = ()
moist_diffusion(pv::PV, ::MoistureModel, ::Flux{SecondOrder}) where {PV <: Mass} =
    (MoistureDiffusion{PV}(),)

eq_tends(pv::PV, m::AtmosModel, tt::Flux{SecondOrder}) where {PV <: Mass} =
    (moist_diffusion(pv, m.moisture, tt)...,)

# Momentum
eq_tends(pv::PV, m::AtmosModel, tt::Flux{SecondOrder}) where {PV <: Momentum} =
    (ViscousStress{PV}(), eq_tends(pv, m.turbconv, tt)...)

# Energy
eq_tends(pv::PV, m::AtmosModel, tt::Flux{SecondOrder}) where {PV <: Energy} =
    (ViscousProduction{PV}(), EnthalpyProduction{PV}(), eq_tends(pv, m.turbconv, tt)...)

# Moisture
eq_tends(pv::PV, ::AtmosModel, ::Flux{SecondOrder}) where {PV <: Moisture} =
    (eq_tends(pv, m.turbconv, tt)...,)

# Precipitation
eq_tends(
    pv::PV,
    ::AtmosModel,
    ::Flux{SecondOrder},
) where {PV <: Precipitation} = ()
