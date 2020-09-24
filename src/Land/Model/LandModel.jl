module Land

using DocStringExtensions
using LinearAlgebra, StaticArrays
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

using CLIMAParameters
using CLIMAParameters.Planet:
    ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0, T_freeze

using ..Mesh.Grids
using ..VariableTemplates
using ..MPIStateArrays
using ..BalanceLaws
import ..BalanceLaws:
    BalanceLaw,
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    nodal_update_auxiliary_state!
using ..DGMethods: LocalGeometry, DGModel, post_tendency_hook!
import ..DGMethods: post_tendency_hook!
export LandModel

"""
    LandModel{PS, S, SRC, IS} <: BalanceLaw

A BalanceLaw for land modeling.
Users may over-ride prescribed default values for each field.

# Usage

    LandModel(
        param_set,
        soil,
        source
        init_state_prognostic
    )

# Fields
$(DocStringExtensions.FIELDS)
"""
struct LandModel{PS, S, SRC, IS} <: BalanceLaw
    "Parameter set"
    param_set::PS
    "Soil model"
    soil::S
    "Source Terms (Problem specific source terms)"
    source::SRC
    "Initial Condition (Function to assign initial values of state variables)"
    init_state_prognostic::IS
end

"""
    LandModel(
        param_set::AbstractParameterSet,
        soil::BalanceLaw;
        source::SRC = (),
        init_state_prognostic::IS = nothing
    ) where {SRC, IS}

Constructor for the LandModel structure. 
"""
function LandModel(
    param_set::AbstractParameterSet,
    soil::BalanceLaw;
    source::SRC = (),
    init_state_prognostic::IS = nothing,
) where {SRC, IS}
    @assert init_state_prognostic ≠ nothing
    land = (param_set, soil, source, init_state_prognostic)
    return LandModel{typeof.(land)...}(land...)
end


function vars_state(land::LandModel, st::Prognostic, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end


function vars_state(land::LandModel, st::Auxiliary, FT)
    @vars begin
        z::FT
        soil::vars_state(land.soil, st, FT)
    end
end

function vars_state(land::LandModel, st::Gradient, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end

function vars_state(land::LandModel, st::GradientFlux, FT)
    @vars begin
        soil::vars_state(land.soil, st, FT)
    end
end

function nodal_init_state_auxiliary!(
    land::LandModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.z = geom.coord[3]
    land_init_aux!(land, land.soil, aux, geom)
end

function flux_first_order!(
    land::LandModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    directions,
) end


function compute_gradient_argument!(
    land::LandModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)

    compute_gradient_argument!(land, land.soil, transform, state, aux, t)
end

function compute_gradient_flux!(
    land::LandModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)

    compute_gradient_flux!(
        land,
        land.soil,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )

end

function flux_second_order!(
    land::LandModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux_second_order!(
        land,
        land.soil,
        flux,
        state,
        diffusive,
        hyperdiffusive,
        aux,
        t,
    )

end

function nodal_update_auxiliary_state!(
    land::LandModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    land_nodal_update_auxiliary_state!(land, land.soil, state, aux, t)
end


function source!(
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    land_source!(land.source, land, source, state, diffusive, aux, t, direction)
end


function init_state_prognostic!(
    land::LandModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
    args...,
)
    land.init_state_prognostic(land, state, aux, coords, t, args...)
end

include("land_bc.jl")
include("SoilWaterParameterizations.jl")
using .SoilWaterParameterizations
include("SoilHeatParameterizations.jl")
using .SoilHeatParameterizations
include("soil_model.jl")
include("soil_water.jl")
include("soil_heat.jl")
include("soil_bc.jl")
include("source.jl")

function post_tendency_hook!(
    dg,
    land::LandModel,
    tendency,
    state_prognostic,
    t,
    α,
    β;
    event,
)

    device = array_device(state_prognostic)

    grid = dg.grid
    topology = grid.topology

    dim = dimensionality(grid)
    N = polynomialorder(grid)
    Nq = N + 1

    realelems = topology.realelems
    nelem = length(realelems)

    Np = dofs_per_element(grid)

    event = kernel_post_tendency_source!(device, (Np,))(
        land,
        Val(dim),
        Val(N),
        tendency.data,
        state_prognostic.data,
        dg.state_auxiliary.data,
        t,
        α,
        realelems;
        ndrange = Np * nelem,
        dependencies = (event,),
    )

    return event
end

@kernel function kernel_post_tendency_source!(
    land::LandModel,
    ::Val{dim},
    ::Val{N},
    tendency,
    state_prognostic,
    state_auxiliary,
    t,
    α,
    elems,
) where {dim, N}
    @uniform begin
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(land, Prognostic())
        num_state_auxiliary = number_states(land, Auxiliary())

        Nq = N + 1

        Nqk = dim == 2 ? 1 : Nq

        Np = Nq * Nq * Nqk

        local_tendency = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
    end

    I = @index(Global, Linear)
    eI = (I - 1) ÷ Np + 1
    n = (I - 1) % Np + 1

    @inbounds begin
        e = elems[eI]

        # XXX: Should we only read the values we need?
        #      If so we *could* use FilterIndices
        @unroll for s in 1:num_state_prognostic
            local_tendency[s] = tendency[n, s, e]
        end

        @unroll for s in 1:num_state_prognostic
            local_state_prognostic[s] = state_prognostic[n, s, e]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end

        @unroll for s in 1:num_state_prognostic
            local_tendency[s] /= α
        end


        land_post_tendency_source!(
            land.source,
            land,
            Vars{vars_state(land, Prognostic(), FT)}(local_tendency),
            Vars{vars_state(land, Prognostic(), FT)}(local_state_prognostic),
            Vars{vars_state(land, Auxiliary(), FT)}(local_state_auxiliary),
            t,
        )

        # XXX: Should we write only read the values we need to?
        #      If so we *could* use FilterIndices
        @unroll for s in 1:num_state_auxiliary
            tendency[n, s, e] = local_tendency[s] * α
        end
    end
end

end # Module
