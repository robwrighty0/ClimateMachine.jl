module Atmos

export AtmosEquations, AtmosAcousticLinearEquations, AtmosAcousticGravityLinearEquations

using CLIMAParameters
using CLIMAParameters.Planet: grav, cp_d
using CLIMAParameters.Atmos.SubgridScale: C_smag
using DocStringExtensions
using LinearAlgebra, StaticArrays
using ..ConfigTypes
using ..Orientations
import ..Orientations:
    vertical_unit_vector,
    altitude,
    latitude,
    longitude,
    projection_normal,
    gravitational_potential,
    ∇gravitational_potential,
    projection_tangential

using ..VariableTemplates
using ..Thermodynamics
using ..TemperatureProfiles

using ..TurbulenceClosures
import ..TurbulenceClosures: turbulence_tensors
using ..TurbulenceConvection

import ..Thermodynamics: internal_energy
using ..MPIStateArrays: MPIStateArray
using ..Mesh.Grids:
    VerticalDirection,
    HorizontalDirection,
    min_node_distance,
    EveryDirection,
    Direction

using ClimateMachine.BalanceLaws
using ClimateMachine.Problems

import ClimateMachine.BalanceLaws:
    vars_state,
    flux_first_order!,
    flux_second_order!,
    source!,
    wavespeed,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    transform_post_gradient_laplacian!,
    init_state_auxiliary!,
    init_state_prognostic!,
    update_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

import ClimateMachine.DGMethods:
    LocalGeometry, lengthscale, resolutionmetric, DGModel

import ..DGMethods.NumericalFluxes:
    boundary_state!,
    boundary_flux_second_order!,
    normal_boundary_flux_second_order!,
    NumericalFluxFirstOrder,
    NumericalFluxGradient,
    NumericalFluxSecondOrder,
    CentralNumericalFluxHigherOrder,
    CentralNumericalFluxDivergence,
    CentralNumericalFluxFirstOrder,
    numerical_flux_first_order!,
    NumericalFluxFirstOrder
using ..DGMethods.NumericalFluxes:
    RoeNumericalFlux, HLLCNumericalFlux, RusanovNumericalFlux

import ..Courant: advective_courant, nondiffusive_courant, diffusive_courant


"""
    AtmosEquations <: BalanceLaw

A `BalanceLaw` for atmosphere equations. Users may over-ride prescribed
default values for each field.

# Usage

    AtmosEquations(
        param_set,
        problem,
        orientation,
        ref_state,
        turbulence,
        hyperdiffusion,
        spongelayer,
        moisture,
        radiation,
        source,
        tracers,
        data_config,
    )

# Fields
$(DocStringExtensions.FIELDS)
"""
struct AtmosEquations{FT, PS, PR, O, RS, T, TC, HD, VS, M, P, R, S, TR, DC} <:
       BalanceLaw
    "Parameter Set (type to dispatch on, e.g., planet parameters. See CLIMAParameters.jl package)"
    param_set::PS
    "Problem (initial and boundary conditions)"
    problem::PR
    "An orientation model"
    orientation::O
    "Reference State (For initial conditions, or for linearisation when using implicit solvers)"
    ref_state::RS
    "Turbulence Closure (Equations for dynamics of under-resolved turbulent flows)"
    turbulence::T
    "Turbulence Convection Closure (e.g., EDMF)"
    turbconv::TC
    "Hyperdiffusion Equations (Equations for dynamics of high-order spatial wave attenuation)"
    hyperdiffusion::HD
    "Viscous sponge layers"
    viscoussponge::VS
    "Moisture Equations (Equations for dynamics of moist variables)"
    moisture::M
    "Precipitation Equations (Equations for dynamics of precipitating species)"
    precipitation::P
    "Radiation Equations (Equations for radiative fluxes)"
    radiation::R
    "Source Terms (Problem specific source terms)"
    source::S
    "Tracer Terms (Equations for dynamics of active and passive tracers)"
    tracers::TR
    "Data Configuration (Helper field for experiment configuration)"
    data_config::DC
end

"""
    AtmosEquations{FT}()

Constructor for `AtmosEquations` (where `AtmosEquations <: BalanceLaw`) for LES
and single stack configurations.
"""
function AtmosEquations{FT}(
    ::Union{Type{AtmosLESConfigType}, Type{SingleStackConfigType}},
    param_set::AbstractParameterSet;
    init_state_prognostic::ISP = nothing,
    problem::PR = AtmosProblem(init_state_prognostic = init_state_prognostic),
    orientation::O = FlatOrientation(),
    ref_state::RS = HydrostaticState(DecayingTemperatureProfile{FT}(param_set),),
    turbulence::T = SmagorinskyLilly{FT}(0.21),
    turbconv::TC = NoTurbConv(),
    hyperdiffusion::HD = NoHyperDiffusion(),
    viscoussponge::VS = NoViscousSponge(),
    moisture::M = EquilMoist{FT}(),
    precipitation::P = NoPrecipitation(),
    radiation::R = NoRadiation(),
    source::S = (
        Gravity(),
        Coriolis(),
        GeostrophicForcing{FT}(7.62e-5, 0, 0),
        turbconv_sources(turbconv)...,
    ),
    tracers::TR = NoTracers(),
    data_config::DC = nothing,
) where {FT <: AbstractFloat, ISP, PR, O, RS, T, TC, HD, VS, M, P, R, S, TR, DC}

    atmos = (
        param_set,
        problem,
        orientation,
        ref_state,
        turbulence,
        turbconv,
        hyperdiffusion,
        viscoussponge,
        moisture,
        precipitation,
        radiation,
        source,
        tracers,
        data_config,
    )

    return AtmosEquations{FT, typeof.(atmos)...}(atmos...)
end

"""
    AtmosEquations{FT}()

Constructor for `AtmosEquations` (where `AtmosEquations <: BalanceLaw`) for GCM
configurations.
"""
function AtmosEquations{FT}(
    ::Type{AtmosGCMConfigType},
    param_set::AbstractParameterSet;
    init_state_prognostic::ISP = nothing,
    problem::PR = AtmosProblem(init_state_prognostic = init_state_prognostic),
    orientation::O = SphericalOrientation(),
    ref_state::RS = HydrostaticState(DecayingTemperatureProfile{FT}(param_set),),
    turbulence::T = SmagorinskyLilly{FT}(C_smag(param_set)),
    turbconv::TC = NoTurbConv(),
    hyperdiffusion::HD = NoHyperDiffusion(),
    viscoussponge::VS = NoViscousSponge(),
    moisture::M = EquilMoist{FT}(),
    precipitation::P = NoPrecipitation(),
    radiation::R = NoRadiation(),
    source::S = (Gravity(), Coriolis(), turbconv_sources(turbconv)...),
    tracers::TR = NoTracers(),
    data_config::DC = nothing,
) where {FT <: AbstractFloat, ISP, PR, O, RS, T, TC, HD, VS, M, P, R, S, TR, DC}

    atmos = (
        param_set,
        problem,
        orientation,
        ref_state,
        turbulence,
        turbconv,
        hyperdiffusion,
        viscoussponge,
        moisture,
        precipitation,
        radiation,
        source,
        tracers,
        data_config,
    )

    return AtmosEquations{FT, typeof.(atmos)...}(atmos...)
end

"""
    vars_state(atmos::AtmosEquations, ::Prognostic, FT)

Conserved state variables (prognostic variables).
"""
function vars_state(atmos::AtmosEquations, st::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        turbulence::vars_state(atmos.turbulence, st, FT)
        turbconv::vars_state(atmos.turbconv, st, FT)
        hyperdiffusion::vars_state(atmos.hyperdiffusion, st, FT)
        moisture::vars_state(atmos.moisture, st, FT)
        radiation::vars_state(atmos.radiation, st, FT)
        tracers::vars_state(atmos.tracers, st, FT)
    end
end

"""
    vars_state(atmos::AtmosEquations, ::Gradient, FT)

Pre-transform gradient variables.
"""
function vars_state(atmos::AtmosEquations, st::Gradient, FT)
    @vars begin
        u::SVector{3, FT}
        h_tot::FT
        turbulence::vars_state(atmos.turbulence, st, FT)
        turbconv::vars_state(atmos.turbconv, st, FT)
        hyperdiffusion::vars_state(atmos.hyperdiffusion, st, FT)
        moisture::vars_state(atmos.moisture, st, FT)
        tracers::vars_state(atmos.tracers, st, FT)
    end
end

"""
    vars_state(atmos::AtmosEquations, ::GradientFlux, FT)

Post-transform gradient variables.
"""
function vars_state(atmos::AtmosEquations, st::GradientFlux, FT)
    @vars begin
        ∇h_tot::SVector{3, FT}
        turbulence::vars_state(atmos.turbulence, st, FT)
        turbconv::vars_state(atmos.turbconv, st, FT)
        hyperdiffusion::vars_state(atmos.hyperdiffusion, st, FT)
        moisture::vars_state(atmos.moisture, st, FT)
        tracers::vars_state(atmos.tracers, st, FT)
    end
end

"""
    vars_state(atmos::AtmosEquations, ::GradientLaplacian, FT)

Pre-transform hyperdiffusive variables.
"""
function vars_state(atmos::AtmosEquations, st::GradientLaplacian, FT)
    @vars begin
        hyperdiffusion::vars_state(atmos.hyperdiffusion, st, FT)
    end
end

"""
    vars_state(atmos::AtmosEquations, ::Hyperdiffusive, FT)

Post-transform hyperdiffusive variables.
"""
function vars_state(atmos::AtmosEquations, st::Hyperdiffusive, FT)
    @vars begin
        hyperdiffusion::vars_state(atmos.hyperdiffusion, st, FT)
    end
end

"""
    vars_state(atmos::AtmosEquations, ::Auxiliary, FT)

Auxiliary variables, such as vertical (stack) integrals, coordinates,
orientation information, reference states, subcomponent auxiliary vars,
debug variables.
"""
function vars_state(atmos::AtmosEquations, st::Auxiliary, FT)
    @vars begin
        ∫dz::vars_state(atmos, UpwardIntegrals(), FT)
        ∫dnz::vars_state(atmos, DownwardIntegrals(), FT)
        coord::SVector{3, FT}
        orientation::vars_state(atmos.orientation, st, FT)
        ref_state::vars_state(atmos.ref_state, st, FT)
        turbulence::vars_state(atmos.turbulence, st, FT)
        turbconv::vars_state(atmos.turbconv, st, FT)
        hyperdiffusion::vars_state(atmos.hyperdiffusion, st, FT)
        moisture::vars_state(atmos.moisture, st, FT)
        tracers::vars_state(atmos.tracers, st, FT)
        radiation::vars_state(atmos.radiation, st, FT)
    end
end

"""
    vars_state(atmos::AtmosEquations, ::UpwardIntegrals, FT)
"""
function vars_state(atmos::AtmosEquations, st::UpwardIntegrals, FT)
    @vars begin
        radiation::vars_state(atmos.radiation, st, FT)
        turbconv::vars_state(atmos.turbconv, st, FT)
    end
end

"""
    vars_state(atmos::AtmosEquations, ::DownwardIntegrals, FT)
"""
function vars_state(atmos::AtmosEquations, st::DownwardIntegrals, FT)
    @vars begin
        radiation::vars_state(atmos.radiation, st, FT)
    end
end

####
#### Forward orientation methods
####
projection_normal(bl, aux, u⃗) =
    projection_normal(bl.orientation, bl.param_set, aux, u⃗)
projection_tangential(bl, aux, u⃗) =
    projection_tangential(bl.orientation, bl.param_set, aux, u⃗)
latitude(bl, aux) = latitude(bl.orientation, aux)
longitude(bl, aux) = longitude(bl.orientation, aux)
altitude(bl, aux) = altitude(bl.orientation, bl.param_set, aux)
vertical_unit_vector(bl, aux) =
    vertical_unit_vector(bl.orientation, bl.param_set, aux)
gravitational_potential(bl, aux) = gravitational_potential(bl.orientation, aux)
∇gravitational_potential(bl, aux) =
    ∇gravitational_potential(bl.orientation, aux)

turbulence_tensors(atmos::AtmosEquations, args...) =
    turbulence_tensors(atmos.turbulence, atmos, args...)

###
### Abstract base type for various components of `AtmosEquations`
###
abstract type AbstractAtmosComponent end

include("problem.jl")
include("ref_state.jl")
include("moisture.jl")
include("thermo_states.jl")
include("precipitation.jl")
include("radiation.jl")
include("source.jl")
include("tracers.jl")
include("linear.jl")
include("courant.jl")
include("filters.jl")

"""
    flux_first_order!(
        atmos::AtmosEquations,
        flux::Grad,
        state::Vars,
        aux::Vars,
        t::Real
    )

Computes and assembles non-diffusive fluxes in the model
equations.
"""
@inline function flux_first_order!(
    atmos::AtmosEquations,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρ = state.ρ
    ρinv = 1 / ρ
    ρu = state.ρu
    u = ρinv * ρu

    # advective terms
    flux.ρ = ρ * u
    flux.ρu = ρ * u .* u'
    flux.ρe = u * state.ρe

    # pressure terms
    ts = recover_thermo_state(atmos, state, aux)
    p = air_pressure(ts)
    if atmos.ref_state isa HydrostaticState
        flux.ρu += (p - aux.ref_state.p) * I
    else
        flux.ρu += p * I
    end
    flux.ρe += u * p
    flux_radiation!(atmos.radiation, atmos, flux, state, aux, t)
    flux_moisture!(atmos.moisture, atmos, flux, state, aux, t)
    flux_tracers!(atmos.tracers, atmos, flux, state, aux, t)
    flux_first_order!(atmos.turbconv, atmos, flux, state, aux, t)
end

function compute_gradient_argument!(
    atmos::AtmosEquations,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.u = ρinv * state.ρu
    ts = recover_thermo_state(atmos, state, aux)
    e_tot = state.ρe * (1 / state.ρ)
    transform.h_tot = total_specific_enthalpy(ts, e_tot)

    compute_gradient_argument!(atmos.moisture, transform, state, aux, t)
    compute_gradient_argument!(atmos.turbulence, transform, state, aux, t)
    compute_gradient_argument!(
        atmos.hyperdiffusion,
        atmos,
        transform,
        state,
        aux,
        t,
    )
    compute_gradient_argument!(atmos.tracers, transform, state, aux, t)
    compute_gradient_argument!(atmos.turbconv, atmos, transform, state, aux, t)
end

function compute_gradient_flux!(
    atmos::AtmosEquations,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.∇h_tot = ∇transform.h_tot

    # diffusion terms required for SGS turbulence computations
    compute_gradient_flux!(
        atmos.turbulence,
        atmos.orientation,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
    # diffusivity of moisture components
    compute_gradient_flux!(atmos.moisture, diffusive, ∇transform, state, aux, t)
    compute_gradient_flux!(atmos.tracers, diffusive, ∇transform, state, aux, t)
    compute_gradient_flux!(
        atmos.turbconv,
        atmos,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
end

function transform_post_gradient_laplacian!(
    atmos::AtmosEquations,
    hyperdiffusive::Vars,
    hypertransform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform_post_gradient_laplacian!(
        atmos.hyperdiffusion,
        atmos,
        hyperdiffusive,
        hypertransform,
        state,
        aux,
        t,
    )
end

"""
    flux_second_order!(
        atmos::AtmosEquations,
        flux::Grad,
        state::Vars,
        diffusive::Vars,
        hyperdiffusive::Vars,
        aux::Vars,
        t::Real
    )
Diffusive fluxes in AtmosEquations. Viscosity, diffusivity are calculated
in the turbulence subcomponent and accessed within the diffusive flux
function. Contributions from subcomponents are then assembled (pointwise).
"""
@inline function flux_second_order!(
    atmos::AtmosEquations,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    ν, D_t, τ = turbulence_tensors(atmos, state, diffusive, aux, t)
    sponge_viscosity_modifier!(atmos, atmos.viscoussponge, ν, D_t, τ, aux)
    d_h_tot = -D_t .* diffusive.∇h_tot
    flux_second_order!(atmos, flux, state, τ, d_h_tot)
    flux_second_order!(atmos.moisture, flux, state, diffusive, aux, t, D_t)
    flux_second_order!(
        atmos.hyperdiffusion,
        flux,
        state,
        diffusive,
        hyperdiffusive,
        aux,
        t,
    )
    flux_second_order!(atmos.tracers, flux, state, diffusive, aux, t, D_t)
    flux_second_order!(atmos.turbconv, atmos, flux, state, diffusive, aux, t)
end

#TODO: Consider whether to not pass ρ and ρu (not state), foc BCs reasons
@inline function flux_second_order!(
    atmos::AtmosEquations,
    flux::Grad,
    state::Vars,
    τ,
    d_h_tot,
)
    flux.ρu += τ * state.ρ
    flux.ρe += τ * state.ρu
    flux.ρe += d_h_tot * state.ρ
end

@inline function wavespeed(
    atmos::AtmosEquations,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρinv = 1 / state.ρ
    u = ρinv * state.ρu
    uN = abs(dot(nM, u))
    ts = recover_thermo_state(atmos, state, aux)
    ss = soundspeed_air(ts)

    FT = typeof(state.ρ)
    ws = fill(uN + ss, MVector{number_states(atmos, Prognostic()), FT})
    vars_ws = Vars{vars_state(atmos, Prognostic(), FT)}(ws)

    wavespeed_tracers!(atmos.tracers, vars_ws, nM, state, aux, t)

    return ws
end


function update_auxiliary_state!(
    dg::DGModel,
    atmos::AtmosEquations,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    FT = eltype(Q)
    state_auxiliary = dg.state_auxiliary

    if number_states(atmos, UpwardIntegrals()) > 0
        indefinite_stack_integral!(dg, atmos, Q, state_auxiliary, t, elems)
        reverse_indefinite_stack_integral!(dg, atmos, Q, state_auxiliary, t, elems)
    end

    update_auxiliary_state!(nodal_update_auxiliary_state!, dg, atmos, Q, t, elems)

    # TODO: Remove this hook. This hook was added for implementing
    # the first draft of EDMF, and should be removed so that we can
    # rely on a single vertical element traversal. This hook allows
    # us to compute globally vertical quantities specific to EDMF
    # until we're able to remove them or somehow incorporate them
    # into a higher level hierarchy.
    update_auxiliary_state!(dg, atmos.turbconv, atmos, Q, t, elems)

    return true
end

function nodal_update_auxiliary_state!(
    atmos::AtmosEquations,
    state::Vars,
    aux::Vars,
    t::Real,
)
    atmos_nodal_update_auxiliary_state!(atmos.moisture, atmos, state, aux, t)
    atmos_nodal_update_auxiliary_state!(atmos.radiation, atmos, state, aux, t)
    atmos_nodal_update_auxiliary_state!(atmos.tracers, atmos, state, aux, t)
    turbulence_nodal_update_auxiliary_state!(atmos.turbulence, atmos, state, aux, t)
    turbconv_nodal_update_auxiliary_state!(atmos.turbconv, atmos, state, aux, t)
end

function integral_load_auxiliary_state!(
    atmos::AtmosEquations,
    integ::Vars,
    state::Vars,
    aux::Vars,
)
    integral_load_auxiliary_state!(atmos.radiation, integ, state, aux)
    integral_load_auxiliary_state!(atmos.turbconv, atmos, integ, state, aux)
end

function integral_set_auxiliary_state!(atmos::AtmosEquations, aux::Vars, integ::Vars)
    integral_set_auxiliary_state!(atmos.radiation, aux, integ)
    integral_set_auxiliary_state!(atmos.turbconv, atmos, aux, integ)
end

function reverse_integral_load_auxiliary_state!(
    atmos::AtmosEquations,
    integ::Vars,
    state::Vars,
    aux::Vars,
)
    reverse_integral_load_auxiliary_state!(atmos.radiation, integ, state, aux)
end

function reverse_integral_set_auxiliary_state!(
    atmos::AtmosEquations,
    aux::Vars,
    integ::Vars,
)
    reverse_integral_set_auxiliary_state!(atmos.radiation, aux, integ)
end

function atmos_nodal_init_state_auxiliary!(
    atmos::AtmosEquations,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.coord = geom.coord
    init_aux_turbulence!(atmos.turbulence, atmos, aux, geom)
    atmos_init_aux!(atmos.ref_state, atmos, aux, tmp, geom)
    init_aux_hyperdiffusion!(atmos.hyperdiffusion, atmos, aux, geom)
    atmos_init_aux!(atmos.tracers, atmos, aux, geom)
    init_aux_turbconv!(atmos.turbconv, atmos, aux, geom)
    atmos.problem.init_state_auxiliary(atmos.problem, atmos, aux, geom)
end

"""
    init_state_auxiliary!(
        atmos::AtmosEquations,
        aux::Vars,
        grid,
        direction
    )

Initialise auxiliary variables for each AtmosEquations component.
Store Cartesian coordinate information in `aux.coord`.
"""
function init_state_auxiliary!(
    atmos::AtmosEquations,
    state_auxiliary::MPIStateArray,
    grid,
    direction,
)
    init_aux!(atmos, atmos.orientation, state_auxiliary, grid, direction)

    init_state_auxiliary!(
        atmos,
        (atmos, aux, tmp, geom) ->
            atmos_init_ref_state_pressure!(atmos.ref_state, atmos, aux, geom),
        state_auxiliary,
        grid,
        direction,
    )

    ∇p = ∇reference_pressure(atmos.ref_state, state_auxiliary, grid)

    init_state_auxiliary!(
        atmos,
        atmos_nodal_init_state_auxiliary!,
        state_auxiliary,
        grid,
        direction;
        state_temporary = ∇p,
    )
end

"""
    source!(
        atmos::AtmosEquations,
        source::Vars,
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        direction::Direction,
    )

Computes (and assembles) source terms `S(Y)` in:
```
∂Y
-- = - ∇ • F + S(Y)
∂t
```
"""
function source!(
    atmos::AtmosEquations,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    atmos_source!(atmos.source, atmos, source, state, diffusive, aux, t, direction)
end

"""
    init_state_prognostic!(
        atmos::AtmosEquations,
        state::Vars,
        aux::Vars,
        coords,
        t,
        args...,
    )

Initialise state variables. `args...` provides an option to include
configuration data (current use cases include problem constants,
spline-interpolants).
"""
function init_state_prognostic!(
    atmos::AtmosEquations,
    state::Vars,
    aux::Vars,
    coords,
    t,
    args...,
)
    atmos.problem.init_state_prognostic(
        atmos.problem,
        atmos,
        state,
        aux,
        coords,
        t,
        args...,
    )
end

roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function numerical_flux_first_order!(
    numerical_flux::RoeNumericalFlux,
    balance_law::AtmosEquations,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    @assert balance_law.moisture isa DryEquations

    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    FT = eltype(fluxᵀn)
    param_set = balance_law.param_set
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)

    Φ = gravitational_potential(balance_law, state_auxiliary⁻)

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.ρe
    ts⁻ = recover_thermo_state(
        balance_law,
        balance_law.moisture,
        state_prognostic⁻,
        state_auxiliary⁻,
    )

    u⁻ = ρu⁻ / ρ⁻
    uᵀn⁻ = u⁻' * normal_vector
    e⁻ = ρe⁻ / ρ⁻
    h⁻ = total_specific_enthalpy(ts⁻, e⁻)
    p⁻ = air_pressure(ts⁻)
    c⁻ = soundspeed_air(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.ρe

    # TODO: state_auxiliary⁺ is not up-to-date
    # with state_prognostic⁺ on the boundaries
    ts⁺ = recover_thermo_state(
        balance_law,
        balance_law.moisture,
        state_prognostic⁺,
        state_auxiliary⁺,
    )

    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    e⁺ = ρe⁺ / ρ⁺
    h⁺ = total_specific_enthalpy(ts⁺, e⁺)
    p⁺ = air_pressure(ts⁺)
    c⁺ = soundspeed_air(ts⁺)

    ρ̃ = sqrt(ρ⁻ * ρ⁺)
    ũ = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    h̃ = roe_average(ρ⁻, ρ⁺, h⁻, h⁺)
    c̃ = sqrt(roe_average(ρ⁻, ρ⁺, c⁻^2, c⁺^2))

    ũᵀn = ũ' * normal_vector

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δuᵀn = Δu' * normal_vector

    w1 = abs(ũᵀn - c̃) * (Δp - ρ̃ * c̃ * Δuᵀn) / (2 * c̃^2)
    w2 = abs(ũᵀn + c̃) * (Δp + ρ̃ * c̃ * Δuᵀn) / (2 * c̃^2)
    w3 = abs(ũᵀn) * (Δρ - Δp / c̃^2)
    w4 = abs(ũᵀn) * ρ̃

    fluxᵀn.ρ -= (w1 + w2 + w3) / 2
    fluxᵀn.ρu -=
        (
            w1 * (ũ - c̃ * normal_vector) +
            w2 * (ũ + c̃ * normal_vector) +
            w3 * ũ +
            w4 * (Δu - Δuᵀn * normal_vector)
        ) / 2
    fluxᵀn.ρe -=
        (
            w1 * (h̃ - c̃ * ũᵀn) +
            w2 * (h̃ + c̃ * ũᵀn) +
            w3 * (ũ' * ũ / 2 + Φ - _T_0 * _cv_d) +
            w4 * (ũ' * Δu - ũᵀn * Δuᵀn)
        ) / 2

    if !(balance_law.tracers isa NoTracers)
        ρχ⁻ = state_prognostic⁻.tracers.ρχ
        χ⁻ = ρχ⁻ / ρ⁻

        ρχ⁺ = state_prognostic⁺.tracers.ρχ
        χ⁺ = ρχ⁺ / ρ⁺

        χ̃ = roe_average(ρ⁻, ρ⁺, χ⁻, χ⁺)
        Δρχ = ρχ⁺ - ρχ⁻

        wt = abs(ũᵀn) * (Δρχ - χ̃ * Δp / c̃^2)

        fluxᵀn.tracers.ρχ -= ((w1 + w2) * χ̃ + wt) / 2
    end
end

"""
    NumericalFluxFirstOrder()
        ::HLLCNumericalFlux,
        balance_law::AtmosEquations,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

An implementation of the numerical flux based on the HLLC method for
the AtmosEquations. For more information on this particular implementation,
see Chapter 10.4 in the provided reference below.

## References
    @book{toro2013riemann,
        title={Riemann solvers and numerical methods for fluid dynamics: a practical introduction},
        author={Toro, Eleuterio F},
        year={2013},
        publisher={Springer Science & Business Media}
    }
"""
function numerical_flux_first_order!(
    ::HLLCNumericalFlux,
    balance_law::AtmosEquations,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    param_set = balance_law.param_set

    # Extract the first-order fluxes from the AtmosEquations (underlying BalanceLaw)
    # and compute normals on the positive + and negative - sides of the
    # interior facets
    flux⁻ = similar(parent(fluxᵀn), Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    flux_first_order!(
        balance_law,
        Grad{S}(flux⁻),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )
    fluxᵀn⁻ = flux⁻' * normal_vector

    flux⁺ = similar(flux⁻)
    fill!(flux⁺, -zero(FT))
    flux_first_order!(
        balance_law,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    fluxᵀn⁺ = flux⁺' * normal_vector

    # Extract relevant fields and thermodynamic variables defined on
    # the positive + and negative - sides of the interior facets
    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.ρe
    ts⁻ = recover_thermo_state(
        balance_law,
        balance_law.moisture,
        state_prognostic⁻,
        state_auxiliary⁻,
    )

    u⁻ = ρu⁻ / ρ⁻
    c⁻ = soundspeed_air(ts⁻)

    uᵀn⁻ = u⁻' * normal_vector
    e⁻ = ρe⁻ / ρ⁻
    p⁻ = air_pressure(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.ρe
    ts⁺ = recover_thermo_state(
        balance_law,
        balance_law.moisture,
        state_prognostic⁺,
        state_auxiliary⁺,
    )

    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    e⁺ = ρe⁺ / ρ⁺
    p⁺ = air_pressure(ts⁺)
    c⁺ = soundspeed_air(ts⁺)

    # Wave speeds estimates S⁻ and S⁺
    S⁻ = min(uᵀn⁻ - c⁻, uᵀn⁺ - c⁺)
    S⁺ = max(uᵀn⁻ + c⁻, uᵀn⁺ + c⁺)

    # Compute the middle wave speed S⁰ in the contact/star region
    S⁰ =
        (p⁺ - p⁻ + ρ⁻ * uᵀn⁻ * (S⁻ - uᵀn⁻) - ρ⁺ * uᵀn⁺ * (S⁺ - uᵀn⁺)) /
        (ρ⁻ * (S⁻ - uᵀn⁻) - ρ⁺ * (S⁺ - uᵀn⁺))

    p⁰ =
        (
            p⁺ +
            p⁻ +
            ρ⁻ * (S⁻ - uᵀn⁻) * (S⁰ - uᵀn⁻) +
            ρ⁺ * (S⁺ - uᵀn⁺) * (S⁰ - uᵀn⁺)
        ) / 2

    # Compute p * D = p * (0, n₁, n₂, n₃, S⁰)
    pD = @MVector zeros(FT, num_state_prognostic)
    if balance_law.ref_state isa HydrostaticState
        # pressure should be continuous but it doesn't hurt to average
        ref_p⁻ = state_auxiliary⁻.ref_state.p
        ref_p⁺ = state_auxiliary⁺.ref_state.p
        ref_p⁰ = (ref_p⁻ + ref_p⁺) / 2

        momentum_p = p⁰ - ref_p⁰
    else
        momentum_p = p⁰
    end

    pD[2] = momentum_p * normal_vector[1]
    pD[3] = momentum_p * normal_vector[2]
    pD[4] = momentum_p * normal_vector[3]
    pD[5] = p⁰ * S⁰

    # Computes both +/- sides of intermediate flux term flux⁰
    flux⁰⁻ =
        (S⁰ * (S⁻ * parent(state_prognostic⁻) - fluxᵀn⁻) + S⁻ * pD) / (S⁻ - S⁰)
    flux⁰⁺ =
        (S⁰ * (S⁺ * parent(state_prognostic⁺) - fluxᵀn⁺) + S⁺ * pD) / (S⁺ - S⁰)

    if 0 <= S⁻
        parent(fluxᵀn) .= fluxᵀn⁻
    elseif S⁻ < 0 <= S⁰
        parent(fluxᵀn) .= flux⁰⁻
    elseif S⁰ < 0 <= S⁺
        parent(fluxᵀn) .= flux⁰⁺
    else # 0 > S⁺
        parent(fluxᵀn) .= fluxᵀn⁺
    end
end

end # module
