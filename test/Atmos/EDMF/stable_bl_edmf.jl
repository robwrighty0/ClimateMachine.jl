# ACTIVE DRIVER
using ClimateMachine
ClimateMachine.init(;
    parse_clargs = true,
    output_dir = get(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR", "output"),
    fix_rng_seed = true,
)
using ClimateMachine.SingleStackUtils
using ClimateMachine.Checkpoint
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.Atmos: AtmosModel
using ClimateMachine.BalanceLaws: vars_state
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
using ClimateMachine.Atmos: PressureGradientModel
using ClimateMachine.BalanceLaws
using ClimateMachine.Mesh.Filters: apply!
import ClimateMachine.DGMethods: custom_filter!, rhs_prehook_filters, calculate_dt
using ClimateMachine.DGMethods: courant
using ClimateMachine.DGMethods: RemBL
using ClimateMachine.Courant
using ClimateMachine.DGMethods: AbstractCustomFilter, apply!

function calculate_dt(dg, model::AtmosModel, Q, Courant_number, t, direction)
    # Δt = one(eltype(Q))
    # CFL = courant(nondiffusive_courant, dg, model, Q, Δt, t, direction)
    # return Courant_number / CFL
    return eltype(Q)(2.64e-01)
end


ENV["CLIMATEMACHINE_SETTINGS_FIX_RNG_SEED"] = true
include(joinpath(clima_dir, "experiments", "AtmosLES", "stable_bl_model.jl"))
include(joinpath(clima_dir, "docs", "plothelpers.jl"))
include("edmf_model.jl")
include("edmf_kernels.jl")

"""
    init_state_prognostic!(
            turbconv::EDMF{FT},
            m::AtmosModel{FT},
            state::Vars,
            aux::Vars,
            localgeo,
            t::Real,
        ) where {FT}

Initialize EDMF state variables.
This method is only called at `t=0`.
"""
function init_state_prognostic!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    state::Vars,
    aux::Vars,
    localgeo,
    t::Real,
) where {FT}
    # Aliases:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    N_up = n_updrafts(turbconv)
    # GCM setting - Initialize the grid mean profiles of prognostic variables (ρ,e_int,q_tot,u,v,w)
    z = altitude(m, aux)

    # SCM setting - need to have separate cases coded and called from a folder - see what LES does
    # a thermo state is used here to convert the input θ to e_int profile
    e_int = internal_energy(m, state, aux)

    ts = PhaseDry(m.param_set, e_int, state.ρ)
    T = air_temperature(ts)
    p = air_pressure(ts)
    q = PhasePartition(ts)
    θ_liq = liquid_ice_pottemp(ts)

    a_min = turbconv.subdomains.a_min
    @unroll_map(N_up) do i
        up[i].ρa = gm.ρ * a_min
        up[i].ρaw = gm.ρu[3] * a_min
        up[i].ρaθ_liq = gm.ρ * a_min * θ_liq
        up[i].ρaq_tot = FT(0)
    end

    # initialize environment covariance with zero for now
    if z <= FT(250)
        en.ρatke =
            gm.ρ *
            FT(0.4) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0)
        en.ρaθ_liq_cv =
            gm.ρ *
            FT(0.4) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0)
    else
        en.ρatke = FT(0)
        en.ρaθ_liq_cv = FT(0)
    end
    en.ρaq_tot_cv = FT(0)
    en.ρaθ_liq_q_tot_cv = FT(0)
    return nothing
end;


struct ZeroVerticalVelocityFilter <: AbstractCustomFilter end
function custom_filter!(::ZeroVerticalVelocityFilter, bl, state, aux)
    state.ρu = SVector(state.ρu[1], state.ρu[2], 0)
end

rhs_prehook_filters(tc::EDMF) = EDMFFilter()
rhs_prehook_filters(tc::NoTurbConv) = nothing

rhs_prehook_filters(bl::AtmosModel) = rhs_prehook_filters(bl.turbconv)
rhs_prehook_filters(bl::AtmosAcousticGravityLinearModel) = GridMeanFilter()
rhs_prehook_filters(bl::RemBL) = rhs_prehook_filters(bl.main)
rhs_prehook_filters(bl::PressureGradientModel) = nothing

struct GridMeanFilter <: AbstractCustomFilter end
function custom_filter!(::GridMeanFilter, bl, state, aux)
    state.ρu = SVector(state.ρu[1],state.ρu[2],0)
end

struct EDMFFilter <: AbstractCustomFilter end
function custom_filter!(::EDMFFilter, bl, state, aux)
    if bl isa AtmosAcousticGravityLinearModel
        atmos = bl.atmos
    elseif bl isa RemBL
        atmos = bl.main
    else
        atmos = bl
    end
    # println("------------------- Using prehook filter!")
    # @show nameof(typeof(bl))
    # @show propertynames(bl)
    # @show propertynames(atmos)
    # @show propertynames(state)
    if hasproperty(atmos, :turbconv)
        # println("hasproperty(atmos, :turbconv)")
        state.ρu = SVector(state.ρu[1],state.ρu[2],0)
        # @show bl isa AtmosAcousticGravityLinearModel
        # if atmos.turbconv isa EDMF
        if :turbconv in propertynames(state)
            # println(":turbconv in propertynames(state)")
            @show state

            # FT = eltype(state)
            # this ρu[3]=0 is only for single_stack
            # state.ρu = SVector(state.ρu[1],state.ρu[2],0)
            # up = state.turbconv.updraft
            # en = state.turbconv.environment
            # N_up = n_updrafts(atmos.turbconv)
            # ρ_gm = state.ρ
            # ρa_min = ρ_gm * atmos.turbconv.subdomains.a_min
            # ρa_max = ρ_gm-ρa_min
            # ts = recover_thermo_state(atmos, state, aux)
            # θ_liq_gm    = liquid_ice_pottemp(ts)
            # ρaθ_liq_ups = sum(vuntuple(i->up[i].ρaθ_liq, N_up))
            # ρa_ups      = sum(vuntuple(i->up[i].ρa, N_up))
            # ρaw_ups     = sum(vuntuple(i->up[i].ρaw, N_up))
            # ρa_en       = ρ_gm - ρa_ups
            # ρaw_en      = - ρaw_ups
            # θ_liq_en    = (θ_liq_gm - ρaθ_liq_ups) / ρa_en
            # w_en        = ρaw_en / ρa_en
            # @unroll_map(N_up) do i
            #     if !(ρa_min <= up[i].ρa <= ρa_max)
            #         up[i].ρa = min(max(up[i].ρa,ρa_min),ρa_max)
            #         up[i].ρaθ_liq = up[i].ρa * θ_liq_gm
            #         up[i].ρaw     = FT(0)
            #     end
            # end
            # en.ρatke = max(en.ρatke,FT(0))
            # en.ρaθ_liq_cv = max(en.ρaθ_liq_cv,FT(0))

            FT = eltype(state)
            # this ρu[3]=0 is only for single_stack
            up = state.turbconv.updraft
            en = state.turbconv.environment
            N_up = n_updrafts(atmos.turbconv)
            ρ_gm = state.ρ
            ρa_min = ρ_gm * atmos.turbconv.subdomains.a_min
            ρa_max = ρ_gm-ρa_min
            ts = recover_thermo_state(atmos, state, aux)
            θ_liq_gm    = liquid_ice_pottemp(ts)
            ρaθ_liq_ups = sum(vuntuple(i->up[i].ρaθ_liq, N_up))
            ρa_ups      = sum(vuntuple(i->up[i].ρa, N_up))
            ρaw_ups     = sum(vuntuple(i->up[i].ρaw, N_up))
            ρa_en       = ρ_gm - ρa_ups
            ρaw_en      = - ρaw_ups
            θ_liq_en    = (θ_liq_gm - ρaθ_liq_ups) / ρa_en
            w_en        = ρaw_en / ρa_en
            @unroll_map(N_up) do i
                up[i].ρa = ρa_min
                up[i].ρaθ_liq = up[i].ρa * θ_liq_gm
                up[i].ρaw     = FT(0)
            end
            en.ρatke = max(en.ρatke,FT(0))
            en.ρaθ_liq_cv = max(en.ρaθ_liq_cv,FT(0))
            en.ρaq_tot_cv = FT(0)
            en.ρaθ_liq_q_tot_cv = FT(0)
            validate_variables(atmos, state, aux, "custom_filter!")
        end
    end
end

function main(::Type{FT}) where {FT}
    # add a command line argument to specify the kind of surface flux
    # TODO: this will move to the future namelist functionality
    sbl_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(sbl_args, "StableBoundaryLayer")
    @add_arg_table! sbl_args begin
        "--surface-flux"
        help = "specify surface flux for energy and moisture"
        metavar = "prescribed|bulk"
        arg_type = String
        default = "bulk"
    end

    cl_args =
        ClimateMachine.init(parse_clargs = true, custom_clargs = sbl_args)

    surface_flux = cl_args["surface_flux"]
    config_type = SingleStackConfigType

    # DG polynomial order
    N = 4
    nelem_vert = 20

    # Prescribe domain parameters
    zmax = FT(400)
    t0 = FT(0)
    # Simulation time
    timeend = FT(3600*3)

    # Charlie's solver
    use_explicit_stepper_with_small_Δt = false
    if use_explicit_stepper_with_small_Δt
        CFLmax = FT(0.9)
        # ode_solver_type = ClimateMachine.IMEXSolverType()

         ode_solver_type = ClimateMachine.ExplicitSolverType(
            solver_method = LSRK144NiegemannDiehlBusch,
        )
    else
        CFLmax = FT(25)
        ode_solver_type = ClimateMachine.IMEXSolverType(
            implicit_model = AtmosAcousticGravityLinearModel,
            implicit_solver = SingleColumnLU,
            solver_method = ARK2GiraldoKellyConstantinescu,
            split_explicit_implicit = true,
            discrete_splitting = false,
        )
    end

    N_updrafts = 1
    N_quad = 3 # Using N_quad = 1 leads to norm(Q) = NaN at init.
    # turbconv = EDMF(FT, N_updrafts, N_quad)
    turbconv = NoTurbConv()

    model = stable_bl_model(
        FT,
        config_type,
        zmax,
        surface_flux;
        turbconv = turbconv,
    )

    # Assemble configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        "SBL_EDMF",
        N,
        nelem_vert,
        zmax,
        param_set,
        model;
        hmax = zmax,
        solver_type = ode_solver_type,
    )

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFLmax,
    )
    println("-------- Check ICs (1)")
    @show all(isfinite.(solver_config.Q.data))
    @show all(isfinite.(solver_config.dg.state_auxiliary.data))

    # --- Zero-out horizontal variations:
    vsp = vars_state(model, Prognostic(), FT)
    horizontally_average!(
        driver_config.grid,
        solver_config.Q,
        varsindex(vsp, :turbconv),
    )
    horizontally_average!(
        driver_config.grid,
        solver_config.Q,
        varsindex(vsp, :ρe),
    )
    vsa = vars_state(model, Auxiliary(), FT)
    horizontally_average!(
        driver_config.grid,
        solver_config.dg.state_auxiliary,
        varsindex(vsa, :turbconv),
    )
    # ---

    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            (),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    # State variable
    Q = solver_config.Q
    # Volume geometry information
    vgeo = driver_config.grid.vgeo
    M = vgeo[:, Grids._M, :]
    # Unpack prognostic vars
    ρ₀ = Q.ρ
    ρe₀ = Q.ρe
    # DG variable sums
    Σρ₀ = sum(ρ₀ .* M)
    Σρe₀ = sum(ρe₀ .* M)

    grid = driver_config.grid

    # state_types = (Prognostic(), Auxiliary(), GradientFlux())
    state_types = (Prognostic(), Auxiliary())
    dons_arr = [dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0]

    # Define the number of outputs from `t0` to `timeend`
    n_outputs = 5
    # This equates to exports every ceil(Int, timeend/n_outputs) time-step:
    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    cb_data_vs_time =
        GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
            push!(
                dons_arr,
                dict_of_nodal_states(solver_config, state_types; interp = true),
            )
            push!(time_data, gettime(solver_config.solver))
            nothing
        end

    cb_check_cons = GenericCallbacks.EveryXSimulationSteps(1) do
        Q = solver_config.Q
        if !isfinite(norm(Q))
            show_not_finite_fields(Q)
        end
        δρ = (sum(Q.ρ .* M) - Σρ₀) / Σρ₀
        δρe = (sum(Q.ρe .* M) .- Σρe₀) ./ Σρe₀
        @show (abs(δρ))
        @show (abs(δρe))
        @test (abs(δρ) <= 0.001)
        @test (abs(δρe) <= 0.025)
        nothing
    end

    cb_print_step = GenericCallbacks.EveryXSimulationSteps(100) do
        @show getsteps(solver_config.solver)
        nothing
    end

    println("-------- Check ICs (2)")
    @show all(isfinite.(solver_config.Q.data))
    @show all(isfinite.(solver_config.dg.state_auxiliary.data))

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (
            cbtmarfilter,
            cb_check_cons,
            cb_data_vs_time,
            cb_print_step,
        ),
        check_euclidean_distance = true,
    )

    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, dons_arr, time_data, state_types
end

solver_config, dons_arr, time_data, state_types = main(Float64)

export_state_plots(
    solver_config,
    dons_arr,
    time_data,
    joinpath("output", "stable_bl_edmf");
    z = Array(get_z(solver_config.dg.grid; rm_dupes = true)),
)

export_state_contours(
    solver_config,
    dons_arr,
    time_data,
    joinpath("output", "stable_bl_edmf");
    z = Array(get_z(solver_config.dg.grid; rm_dupes = true)),
)
