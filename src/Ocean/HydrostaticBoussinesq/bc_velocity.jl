using ..Ocean: kinematic_stress

"""
    ocean_velocity_boundary_state!(
        ::NumericalFlux{FirstOrder},
        ::Impenetrable{NoSlip},
        ::HBModel)

apply no slip boundary condition for velocity
sets reflective ghost point
"""
function ocean_velocity_boundary_state!(
    nf::NumericalFlux{FirstOrder},
    bc_velocity::Impenetrable{NoSlip},
    ocean,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    Q⁺.u = -Q⁻.u
    A⁺.w = -A⁻.w

    return nothing
end

"""
    ocean_velocity_boundary_state!(
        ::NumericalFlux{Gradient},
        ::Impenetrable{NoSlip},
        ::HBModel)

apply no slip boundary condition for velocity
set numerical flux to zero for u
"""
function ocean_velocity_boundary_state!(
    nf::NumericalFlux{Gradient},
    bc_velocity::Impenetrable{NoSlip},
    ocean,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    FT = eltype(Q⁺)
    Q⁺.u = SVector(-zero(FT), -zero(FT))
    A⁺.w = -zero(FT)

    return nothing
end

"""
    ocean_velocity_boundary_state!(
        ::NumericalFlux{SecondOrder},
        ::Impenetrable{NoSlip},
        ::HBModel)

apply no slip boundary condition for velocity
sets ghost point to have no numerical flux on the boundary for u
"""
@inline function ocean_velocity_boundary_state!(
    nf::NumericalFlux{SecondOrder},
    bc_velocity::Impenetrable{NoSlip},
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
    Q⁺.u = -Q⁻.u
    A⁺.w = -A⁻.w
    D⁺.ν∇u = D⁻.ν∇u

    return nothing
end

"""
    ocean_velocity_boundary_state!(
        ::NumericalFlux{FirstOrder},
        ::Impenetrable{FreeSlip},
        ::HBModel)

apply free slip boundary condition for velocity
sets reflective ghost point
"""
function ocean_velocity_boundary_state!(
    nf::NumericalFlux{FirstOrder},
    bc_velocity::Impenetrable{FreeSlip},
    ocean,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    v⁻ = @SVector [Q⁻.u[1], Q⁻.u[2], A⁻.w]
    v⁺ = v⁻ - 2 * n⁻ ⋅ v⁻ .* SVector(n⁻)
    Q⁺.u = @SVector [v⁺[1], v⁺[2]]
    A⁺.w = v⁺[3]

    return nothing
end

"""
    ocean_velocity_boundary_state!(
        ::NumericalFlux{Gradient},
        ::Impenetrable{FreeSlip},
        ::HBModel)

apply free slip boundary condition for velocity
sets non-reflective ghost point
"""
function ocean_velocity_boundary_state!(
    nf::NumericalFlux{Gradient},
    bc_velocity::Impenetrable{FreeSlip},
    ocean,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    v⁻ = @SVector [Q⁻.u[1], Q⁻.u[2], A⁻.w]
    v⁺ = v⁻ - n⁻ ⋅ v⁻ .* SVector(n⁻)
    Q⁺.u = @SVector [v⁺[1], v⁺[2]]
    A⁺.w = v⁺[3]

    return nothing
end

"""
    ocean_velocity_normal_boundary_flux_second_order!(
        ::NumericalFlux{SecondOrder},
        ::Impenetrable{FreeSlip},
        ::HBModel)

apply free slip boundary condition for velocity
apply zero numerical flux in the normal direction
"""
function ocean_velocity_boundary_state!(
    nf::NumericalFlux{SecondOrder},
    bc_velocity::Impenetrable{FreeSlip},
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
    Q⁺.u = Q⁻.u
    A⁺.w = A⁻.w
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'

    return nothing
end

"""
    ocean_velocity_boundary_state!(
        ::Union{NumericalFlux{FirstOrder}, NumericalFlux{Gradient}},
        ::Penetrable{FreeSlip},
        ::HBModel)

apply free slip boundary condition for velocity
sets non-reflective ghost point
"""
function ocean_velocity_boundary_state!(
    nf::Union{NumericalFlux{FirstOrder}, NumericalFlux{Gradient}},
    bc_velocity::Penetrable{FreeSlip},
    ocean,
    args...,
)
    return nothing
end

"""
    ocean_velocity_boundary_state!(
        ::NumericalFlux{SecondOrder},
        ::Penetrable{FreeSlip},
        ::HBModel)

apply free slip boundary condition for velocity
sets non-reflective ghost point
"""
function ocean_velocity_boundary_state!(
    nf::NumericalFlux{SecondOrder},
    bc_velocity::Penetrable{FreeSlip},
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
    Q⁺.u = Q⁻.u
    A⁺.w = A⁻.w
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'

    return nothing
end

"""
    ocean_velocity_boundary_state!(
        ::Union{NumericalFlux{FirstOrder}, NumericalFlux{Gradient}},
        ::Impenetrable{KinematicStress},
        ::HBModel)

apply kinematic stress boundary condition for velocity
applies free slip conditions for first-order and gradient fluxes
"""
function ocean_velocity_boundary_state!(
    nf::Union{NumericalFlux{FirstOrder}, NumericalFlux{Gradient}},
    bc_velocity::Impenetrable{KinematicStress},
    ocean,
    args...,
)
    return ocean_velocity_boundary_state!(
        nf,
        Impenetrable(FreeSlip()),
        ocean,
        args...,
    )
end

"""
    ocean_velocity_boundary_state!(
        ::NumericalFlux{SecondOrder},
        ::Impenetrable{KinematicStress},
        ::HBModel)

apply kinematic stress boundary condition for velocity
sets ghost point to have specified flux on the boundary for ν∇u
"""
@inline function ocean_velocity_boundary_state!(
    nf::NumericalFlux{SecondOrder},
    bc_velocity::Impenetrable{KinematicStress},
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
    Q⁺.u = Q⁻.u
    D⁺.ν∇u = n⁻ * kinematic_stress(ocean.problem, A⁻.y, ocean.ρₒ)'

    return nothing
end

"""
    ocean_velocity_boundary_state!(
        ::Union{NumericalFlux{FirstOrder}, NumericalFlux{Gradient}},
        ::Penetrable{KinematicStress},
        ::HBModel)

apply kinematic stress boundary condition for velocity
applies free slip conditions for first-order and gradient fluxes
"""
function ocean_velocity_boundary_state!(
    nf::Union{NumericalFlux{FirstOrder}, NumericalFlux{Gradient}},
    bc_velocity::Penetrable{KinematicStress},
    ocean,
    args...,
)
    return ocean_velocity_boundary_state!(
        nf,
        Penetrable(FreeSlip()),
        ocean,
        args...,
    )
end

"""
    ocean_velocity_boundary_state!(
        ::NumericalFlux{SecondOrder},
        ::Penetrable{KinematicStress},
        ::HBModel)

apply kinematic stress boundary condition for velocity
sets ghost point to have specified flux on the boundary for ν∇u
"""
@inline function ocean_velocity_boundary_state!(
    nf::NumericalFlux{SecondOrder},
    bc_velocity::Penetrable{KinematicStress},
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
    Q⁺.u = Q⁻.u
    D⁺.ν∇u = n⁻ * kinematic_stress(ocean.problem, A⁻.y, ocean.ρₒ)'

    return nothing
end
