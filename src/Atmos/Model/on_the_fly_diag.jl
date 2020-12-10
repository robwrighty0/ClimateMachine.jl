# Allows on-the-fly computation of variables using model kernels using mini balance laws that use ClimateMachine's DG kernels
using DocStringExtensions
using ..TemperatureProfiles
using ..DGMethods: init_ode_state
export ReferenceState, NoReferenceState, HydrostaticState
const TD = Thermodynamics
using CLIMAParameters.Planet: R_d, MSLP, cp_d, grav, T_surf_ref, T_min_ref

using ClimateMachine.BalanceLaws: AbstractStateType, Auxiliary, Gradient

import ClimateMachine.BalanceLaws: vars_state

include("vorticity_balancelaw.jl")

# function by which AtmosModel calls mini balance laws related to DG gradients, e.g., VorticityModel. 
# (plugs into update_auxiliary_state! in AtmosModel.jl)
function ∇diagnostics(
    NoAdditionalDiagnostics,
    m,
    state_prognostic,
    state_auxiliary,
    grid,
)
    return nothing
end
function ∇diagnostics(
    AdditionalDiagnostics,
    m,
    state_prognostic,
    state_auxiliary,
    grid,
)

    grad_model = VorticityModel()
    # Note that the choice of numerical fluxes doesn't matter
    # for taking the gradient of a continuous field
    grad_dg = DGModel(
        grad_model,
        grid,
        CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    # input data
    FT = eltype(state_auxiliary)

    # get the 3D u vector from AtmosModel's prognostic MPIStateArray
    # and save it as VerticityModel's auxiliary MPIStateArray
    ix_ρu = varsindex(vars(state_prognostic), :ρu)
    ix_ρ = varsindex(vars(state_prognostic), :ρ)

    ρ = state_prognostic.data[:, ix_ρ, :]
    u = state_prognostic.data[:, ix_ρu, :] ./ ρ

    grad_dg.state_auxiliary.data = u

    # init output data
    Ω_dg =
        similar(state_auxiliary; vars = @vars(Ω_dg::SVector{3, FT}), nstate = 3)

    # FIXME: this isn'tt used but needs to be passed in (?)
    gradQ = init_ode_state(grad_dg, FT(0))

    grad_dg(Ω_dg, gradQ, nothing, FT(0))

    # include other DGModel variables here...
    return Ω_dg
end

# init the above ∇diagnostics function (plugs into atmos_nodal_init_state_auxiliary!)
function ∇diagnostics_init(NoAdditionalDiagnostics, m, FT)
    return nothing
end
