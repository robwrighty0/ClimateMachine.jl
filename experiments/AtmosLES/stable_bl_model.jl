#!/usr/bin/env julia --project
#=
# This experiment file establishes the initial conditions, boundary conditions,
# source terms and simulation parameters (domain size + resolution) for the
# GABLS LES case ([Beare2006](@cite); [Kosovic2000](@cite)).
#
## [Kosovic2000](@cite)
#
# To simulate the experiment, type in
#
# julia --project experiments/AtmosLES/stable_bl_les.jl
=#

using ArgParse
using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra
using Printf
using UnPack

using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.ODESolvers
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.TurbulenceConvection
using ClimateMachine.VariableTemplates
using ClimateMachine.BalanceLaws
import ClimateMachine.BalanceLaws: source
import ClimateMachine.Atmos: filter_source, atmos_source!

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav, day
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine.Atmos: altitude, recover_thermo_state

"""
  StableBL Geostrophic Forcing (Source)
"""
struct StableBLGeostrophic{PV <: Momentum, FT} <: TendencyDef{Source, PV}
    "Coriolis parameter [s⁻¹]"
    f_coriolis::FT
    "Eastward geostrophic velocity `[m/s]` (Base)"
    u_geostrophic::FT
    "Eastward geostrophic velocity `[m/s]` (Slope)"
    u_slope::FT
    "Northward geostrophic velocity `[m/s]`"
    v_geostrophic::FT
end
StableBLGeostrophic(::Type{FT}, args...) where {FT} =
    StableBLGeostrophic{Momentum, FT}(args...)

function source(
    s::StableBLGeostrophic{Momentum},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    @unpack f_coriolis, u_geostrophic, u_slope, v_geostrophic = s

    z = altitude(m, aux)
    # Note z dependence of eastward geostrophic velocity
    u_geo = SVector(u_geostrophic + u_slope * z, v_geostrophic, 0)
    ẑ = vertical_unit_vector(m, aux)
    fkvector = f_coriolis * ẑ
    # Accumulate sources
    return -fkvector × (state.ρu .- state.ρ * u_geo)
end

"""
  StableBL Sponge (Source)
"""
struct StableBLSponge{PV <: Momentum, FT} <: TendencyDef{Source, PV}
    "Maximum domain altitude (m)"
    z_max::FT
    "Altitude at with sponge starts (m)"
    z_sponge::FT
    "Sponge Strength 0 ⩽ α_max ⩽ 1"
    α_max::FT
    "Sponge exponent"
    γ::FT
    "Eastward geostrophic velocity `[m/s]` (Base)"
    u_geostrophic::FT
    "Eastward geostrophic velocity `[m/s]` (Slope)"
    u_slope::FT
    "Northward geostrophic velocity `[m/s]`"
    v_geostrophic::FT
end

StableBLSponge(::Type{FT}, args...) where {FT} =
    StableBLSponge{Momentum, FT}(args...)

function source(
    s::StableBLSponge{Momentum},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    @unpack z_max, z_sponge, α_max, γ = s
    @unpack u_geostrophic, u_slope, v_geostrophic = s

    z = altitude(m, aux)
    u_geo = SVector(u_geostrophic + u_slope * z, v_geostrophic, 0)
    ẑ = vertical_unit_vector(m, aux)
    # Accumulate sources
    if z_sponge <= z
        r = (z - z_sponge) / (z_max - z_sponge)
        β_sponge = α_max * sinpi(r / 2)^s.γ
        return -β_sponge * (state.ρu .- state.ρ * u_geo)
    else
        FT = eltype(state)
        return SVector{3, FT}(0, 0, 0)
    end
end

filter_source(pv::PV, m, s::StableBLGeostrophic{PV}) where {PV} = s
filter_source(pv::PV, m, s::StableBLSponge{PV}) where {PV} = s
atmos_source!(::StableBLGeostrophic, args...) = nothing
atmos_source!(::StableBLSponge, args...) = nothing

"""
  Initial Condition for StableBoundaryLayer LES
"""
function init_problem!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord
    # Problem floating point precision
    FT = eltype(state)
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v
    # Initialise speeds [u = Eastward, v = Northward, w = Vertical]
    u::FT = 8
    v::FT = 0
    w::FT = 0
    # Assign piecewise quantities to θ_liq and q_tot
    θ_liq::FT = 0
    q_tot::FT = 0
    # Piecewise functions for potential temperature and total moisture
    z1 = FT(100)
    if z <= z1
        θ_liq = FT(265)
    else
        θ_liq = FT(265) + FT(0.01) * (z - z1)
    end
    θ = θ_liq
    π_exner = FT(1) - _grav / (c_p * θ) * z # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas) # density
    # Establish thermodynamic state and moist phase partitioning
    if bl.moisture isa DryModel
        TS = PhaseDry_ρθ(bl.param_set, ρ, θ_liq)
    else
        TS = PhaseEquil_ρθq(bl.param_set, ρ, θ_liq, q_tot)
    end
    # Compute momentum contributions
    ρu = ρ * u
    ρv = ρ * v
    ρw = ρ * w

    # Compute energy contributions
    e_kin = FT(1 // 2) * (u^2 + v^2 + w^2)
    e_pot = _grav * z
    ρe_tot = ρ * total_energy(e_kin, e_pot, TS)

    # Assign initial conditions for prognostic state variables
    state.ρ = ρ
    state.ρu = SVector(ρu, ρv, ρw)
    state.ρe = ρe_tot
    if !(bl.moisture isa DryModel)
        state.moisture.ρq_tot = ρ * q_tot
    end
    if z <= FT(50) # Add random perturbations to bottom 50m of model
        state.ρe += rand() * ρe_tot / 100
    end
    init_state_prognostic!(bl.turbconv, bl, state, aux, localgeo, t)
end

function surface_temperature_variation(bl, state, t)
    FT = eltype(state)
    ρ = state.ρ
    θ_liq_sfc = FT(265) - FT(1 / 4) * (t / 3600)
    if bl.moisture isa DryModel
        TS = PhaseDry_ρθ(bl.param_set, ρ, θ_liq_sfc)
    else
        q_tot = state.moisture.ρq_tot / ρ
        TS = PhaseEquil_ρθq(bl.param_set, ρ, θ_liq_sfc, q_tot)
    end
    return air_temperature(TS)
end

function stable_bl_model(
    ::Type{FT},
    config_type,
    zmax,
    surface_flux;
    turbconv = NoTurbConv(),
    moisture_model = "dry",
) where {FT}

    ics = init_problem!     # Initial conditions

    C_smag = FT(0.23)     # Smagorinsky coefficient
    C_drag = FT(0.001)    # Momentum exchange coefficient
    u_star = FT(0.30)

    z_sponge = FT(300)     # Start of sponge layer
    α_max = FT(0.75)       # Strength of sponge layer (timescale)
    γ = 2                  # Strength of sponge layer (exponent)

    u_geostrophic = FT(8)        # Eastward relaxation speed
    u_slope = FT(0)              # Slope of altitude-dependent relaxation speed
    v_geostrophic = FT(0)        # Northward relaxation speed
    f_coriolis = FT(1.39e-4) # Coriolis parameter at 73N

    q_sfc = FT(0)

    # Assemble source components
    source_default = (
        Gravity(),
        StableBLSponge(
            FT,
            zmax,
            z_sponge,
            α_max,
            γ,
            u_geostrophic,
            u_slope,
            v_geostrophic,
        ),
        StableBLGeostrophic(
            FT,
            f_coriolis,
            u_geostrophic,
            u_slope,
            v_geostrophic,
        ),
    )
    if moisture_model == "dry"
        moisture = DryModel()
    elseif moisture_model == "equilibrium"
        source = source_default
        moisture = EquilMoist{FT}(; maxiter = 5, tolerance = FT(0.1))
    elseif moisture_model == "nonequilibrium"
        source = (source_default..., CreateClouds()...)
        moisture = NonEquilMoist()
    else
        @warn @sprintf(
            """
%s: unrecognized moisture_model in source terms, using the defaults""",
            moisture_model,
        )
        source = source_default
    end
    # Set up problem initial and boundary conditions
    if surface_flux == "prescribed"
        energy_bc = PrescribedEnergyFlux((state, aux, t) -> LHF + SHF)
        moisture_bc = PrescribedMoistureFlux((state, aux, t) -> moisture_flux)
    elseif surface_flux == "bulk"
        energy_bc = BulkFormulaEnergy(
            (bl, state, aux, t, normPu_int) -> C_drag,
            (bl, state, aux, t) ->
                (surface_temperature_variation(bl, state, t), q_sfc),
        )
        moisture_bc = BulkFormulaMoisture(
            (state, aux, t, normPu_int) -> C_drag,
            (state, aux, t) -> q_sfc,
        )
    else
        @warn @sprintf(
            """
%s: unrecognized surface flux; using 'prescribed'""",
            surface_flux,
        )
    end

    if moisture_model == "dry"
        boundary_conditions = (
            AtmosBC(
                momentum = Impenetrable(DragLaw(
                    # normPu_int is the internal horizontal speed
                    # P represents the projection onto the horizontal
                    (state, aux, t, normPu_int) -> (u_star / normPu_int)^2,
                )),
                energy = energy_bc,
            ),
            AtmosBC(),
        )
    else
        boundary_conditions = (
            AtmosBC(
                momentum = Impenetrable(DragLaw(
                    # normPu_int is the internal horizontal speed
                    # P represents the projection onto the horizontal
                    (state, aux, t, normPu_int) -> (u_star / normPu_int)^2,
                )),
                energy = energy_bc,
                moisture = moisture_bc,
            ),
            AtmosBC(),
        )
    end

    moisture_flux = FT(0)
    problem = AtmosProblem(
        init_state_prognostic = ics,
        boundaryconditions = boundary_conditions,
    )

    # Assemble model components
    model = AtmosModel{FT}(
        config_type,
        param_set;
        problem = problem,
        turbulence = SmagorinskyLilly{FT}(C_smag),
        moisture = moisture,
        source = source_default,
        turbconv = turbconv,
    )

    return model
end

function config_diagnostics(driver_config)
    default_dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        "2500steps",
        driver_config.name,
    )
    core_dgngrp = setup_atmos_core_diagnostics(
        AtmosLESConfigType(),
        "2500steps",
        driver_config.name,
    )
    return ClimateMachine.DiagnosticsConfiguration([
        default_dgngrp,
        core_dgngrp,
    ])
end
