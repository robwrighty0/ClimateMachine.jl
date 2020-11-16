using ..Ocean: surface_flux

"""
    ocean_temperature_boundary_state!(
        ::Union{NumericalFlux{FirstOrder},
        NumericalFlux{Gradient}},
        ::Insulating,
        ::HBModel)

apply insulating boundary condition for temperature
sets transmissive ghost point
"""
function ocean_temperature_boundary_state!(
    nf::Union{NumericalFlux{FirstOrder}, NumericalFlux{Gradient}},
    bc_temperature::Insulating,
    ocean,
    Q⁺,
    A⁺,
    n,
    Q⁻,
    A⁻,
    t,
)
    Q⁺.θ = Q⁻.θ

    return nothing
end

"""
    ocean_temperature_boundary_state!(::NumericalFlux{SecondOrder}, ::Insulating, ::HBModel)

apply insulating boundary condition for velocity
sets ghost point to have no numerical flux on the boundary for κ∇θ
"""
@inline function ocean_temperature_boundary_state!(
    nf::NumericalFlux{SecondOrder},
    bc_temperature::Insulating,
    ocean,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    Q⁺.θ = Q⁻.θ
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    ocean_temperature_boundary_state!(
        ::Union{NumericalFlux{FirstOrder},
        NumericalFlux{Gradient}},
        ::TemperatureFlux,
        ::HBModel)

apply temperature flux boundary condition for velocity
applies insulating conditions for first-order and gradient fluxes
"""
function ocean_temperature_boundary_state!(
    nf::Union{NumericalFlux{FirstOrder}, NumericalFlux{Gradient}},
    bc_velocity::TemperatureFlux,
    ocean,
    args...,
)
    return ocean_temperature_boundary_state!(nf, Insulating(), ocean, args...)
end

"""
    ocean_temperature_boundary_state!(
        ::NumericalFlux{SecondOrder},
        ::TemperatureFlux,
        ::HBModel)

apply insulating boundary condition for velocity
sets ghost point to have specified flux on the boundary for κ∇θ
"""
@inline function ocean_temperature_boundary_state!(
    nf::NumericalFlux{SecondOrder},
    bc_temperature::TemperatureFlux,
    ocean,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    Q⁺.θ = Q⁻.θ
    D⁺.κ∇θ = n⁻ * surface_flux(ocean.problem, A⁻.y, Q⁻.θ)

    return nothing
end
