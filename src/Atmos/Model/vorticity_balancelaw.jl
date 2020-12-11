# Insider Model allows computation of variables using model kernels using mini balance laws
using DocStringExtensions
using ..TemperatureProfiles
using ..DGMethods: init_ode_state
export ReferenceState, NoReferenceState, HydrostaticState
const TD = Thermodynamics
using CLIMAParameters.Planet: R_d, MSLP, cp_d, grav, T_surf_ref, T_min_ref

using ClimateMachine.BalanceLaws: AbstractStateType, Auxiliary, Gradient

import ClimateMachine.BalanceLaws: vars_state

"""
    VorticityModel

A mini balance law that is used to take the gradient of u and v to obtain vorticity

"""

# specify VorticityModel as a balance law struct
struct VorticityModel <: BalanceLaw end

# declare gradient, diffusice and auxillary variables for the VorticityModel balance law
vars_state(::VorticityModel, ::Prognostic, FT) = @vars(Ω_dg::SVector{3, FT}); # output vorticity vector
vars_state(::VorticityModel, ::Auxiliary, FT) = @vars(u::SVector{3, FT}); # input velocity vector
vars_state(::VorticityModel, ::Gradient, FT) = @vars();
vars_state(::VorticityModel, ::GradientFlux, FT) = @vars();



# list initiate kernels and compute kernels, as required for each balance law 
function init_state_auxiliary!(
    m::VorticityModel,
    state_auxiliary::MPIStateArray,
    grid,
    direction,
) end
function init_state_prognostic!(
    ::VorticityModel,
    state::Vars,
    aux::Vars,
    localgeo,
    t,
) end
function nodal_update_auxiliary_state!(
    m::VorticityModel,
    state::Vars,
    aux::Vars,
    t::Real,
) end;
function flux_first_order!(
    ::VorticityModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    u = aux.u
    @inbounds begin
        flux.Ω_dg = @SMatrix [
            0 u[3] -u[2]
            -u[3] 0 u[1]
            u[2] -u[1] 0
        ]
    end
end
function compute_gradient_argument!(
    ::VorticityModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) end
flux_second_order!(::VorticityModel, _...) = nothing
source!(::VorticityModel, _...) = nothing
boundary_conditions(::VorticityModel) = ntuple(i -> nothing, 6)
boundary_state!(nf, ::Nothing, ::VorticityModel, _...) = nothing
