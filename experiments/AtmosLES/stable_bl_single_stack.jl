include("stable_bl_model.jl")
using ClimateMachine.SingleStackUtils

function main(::Type{FT}) where {FT}

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

    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = sbl_args)

    surface_flux = cl_args["surface_flux"]

    config_type = SingleStackConfigType

    # DG polynomial order
    N = 4
    # Domain resolution and size
    nelem_vert = 20

    # Prescribe domain parameters
    xmax = FT(100)
    ymax = FT(100)
    zmax = FT(400)

    t0 = FT(0)

    # Required simulation time == 9hours
    timeend = FT(3600 * 0.1)
    CFLmax = FT(0.4)

    # Choose default IMEX solver
    ode_solver_type = ClimateMachine.ExplicitSolverType()
    # ode_solver_type = ClimateMachine.ImplicitSolverType()

    model = stable_bl_model(FT, config_type, zmax, surface_flux)
    ics = model.problem.init_state_prognostic

    # Assemble configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        "StableBoundaryLayer",
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

    state_types = (Prognostic(), Auxiliary())
    dons_arr = [dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0]

    # Define the number of outputs from `t0` to `timeend`
    n_outputs = 10
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



    dgn_config = config_diagnostics(driver_config)

    check_cons = (
        ClimateMachine.ConservationCheck("ρ", "1mins", FT(0.0001)),
        ClimateMachine.ConservationCheck("ρe", "1mins", FT(0.0025)),
    )

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        check_cons = check_cons,
        user_callbacks = (cb_data_vs_time, ),
        check_euclidean_distance = true,
    )

    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, dons_arr, time_data, state_types
end

solver_config, dons_arr, time_data, state_types = main(Float64)

nothing
