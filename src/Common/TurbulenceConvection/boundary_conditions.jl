#### Boundary conditions

export TurbConvBC,
    NoTurbConvBC,
    turbconv_bcs,
    turbconv_boundary_state!,
    turbconv_normal_boundary_flux_second_order!

abstract type AbstractTurbConvBC end

"""
    NoTurbConvBC <: AbstractTurbConvBC

Boundary conditions are not applied
"""
struct NoTurbConvBC <: AbstractTurbConvBC end

turbconv_bcs(::NoTurbConv) = NoTurbConvBC()

function turbconv_boundary_state!(nf, bc_turbulence::NoTurbConvBC, bl, args...)
    nothing
end

function turbconv_normal_boundary_flux_second_order!(
    nf,
    bc_turbulence::NoTurbConvBC,
    bl,
    args...,
)
    nothing
end
