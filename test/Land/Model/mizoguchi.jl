# Test that freeze thaw alone reproduces expected behavior: exponential behavior
# for liquid water content, ice content, and total water conserved

#To be fixed - the grid spacing part, there is some onus on the user to define the τft function appropriately in the PrescribedTemperatureCase. We will define it in the SoilHeatModel case.
# another issue - passing spacing?
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
using CLIMAParameters.Planet: ρ_cloud_liq
using CLIMAParameters.Planet: ρ_cloud_ice

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.SoilHeatParameterizations
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state

@testset "Freeze thaw alone" begin
    FT = Float64
    struct tmp_model <: BalanceLaw end
    struct tmp_param_set <: AbstractParameterSet end

    function get_grid_spacing(
        N_poly::Int64,
        nelem_vert::Int64,
        zmax::FT,
        zmin::FT,
    )
        test_config = ClimateMachine.SingleStackConfiguration(
            "TmpModel",
            N_poly,
            nelem_vert,
            zmax,
            tmp_param_set(),
            tmp_model();
            zmin = zmin,
        )

        Δ = min_node_distance(test_config.grid)
        return Δ
    end

    function init_soil_water!(land, state, aux, coordinates, time)
        myf = eltype(state)
        state.soil.water.ϑ_l = myf(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_i = myf(land.soil.water.initialθ_i(aux))
    end

    ClimateMachine.init()

    N_poly = 5
    nelem_vert = 50
    zmax = FT(0)
    zmin = FT(-0.2)
    t0 = FT(0)
    timeend = FT(30)
   ### dt = FT(0.05)

    n_outputs = 30
    every_x_simulation_time = ceil(Int, timeend / n_outputs)
    Δ = get_grid_spacing(N_poly, nelem_vert, zmax, zmin)
    cs = FT(3e6)
    κ = FT(1.5)
    τLTE = FT(cs * Δ^FT(2.0) / κ)
    
    freeze_thaw_source = FreezeThaw{FT}(Δt = dt,
                                    τLTE = τLTE)
    ν_ss_quartz = FT(0.58)
    ν_ss_minerals = FT(0.42)
    ν_ss_om = FT(0.0)
    ν_ss_gravel = FT(0.0);
    κ_quartz = FT(7.7) # W/m/K
    κ_minerals = FT(2.5) # W/m/K
    κ_om = FT(0.25) # W/m/K
    κ_liq = FT(0.57) # W/m/K
    κ_ice = FT(2.29); # W/m/K

    ρp = FT(2700) # kg/m^3
    κ_solid = k_solid(ν_ss_om, ν_ss_quartz, κ_quartz, κ_minerals, κ_om)
    κ_sat_frozen = ksat_frozen(κ_solid, porosity, κ_ice)
    κ_sat_unfrozen = ksat_unfrozen(κ_solid, porosity, κ_liq);

    ρc_ds = FT((1 - porosity) * 1.926e06) # J/m^3/K
    
    soil_param_functions =
        SoilParamFunctions{FT}(porosity = 0.535,
                               Ksat = 3.2e-6,
                               S_s = 1e-3,
                               ν_ss_gravel = ν_ss_gravel,
                               ν_ss_om = ν_ss_om,
                               ν_ss_quartz = ν_ss_quartz,
                               ρc_ds = ρc_ds,
                               ρp = ρp,
                               κ_solid = κ_solid,
                               κ_sat_unfrozen = κ_sat_unfrozen,
                               κ_sat_frozen = κ_sat_frozen,
                               );

    bottom_flux = (aux, t) -> eltype(aux)(0.0)
    surface_flux = (aux, t) -> eltype(aux)(0.0)
    surface_state = nothing
    bottom_state = nothing
    ϑ_l0 = (aux) -> eltype(aux)(0.33)
    soil_water_model = SoilWaterModel(
        FT;
        initialϑ_l = ϑ_l0,
        dirichlet_bc = Dirichlet(
            surface_state = surface_state,
            bottom_state = bottom_state,
        ),
        neumann_bc = Neumann(
            surface_flux = surface_flux,
            bottom_flux = bottom_flux,
        ),
    )

    surface_heat_flux = (aux, t) -> eltype(aux)(-28)*(aux.T-eltype(aux)(273.15-6))
    T_init = aux -> eltype(aux)(279.85)
    soil_heat_model = SoilHeatModel(
        FT;
        initialT = T_init,
        dirichlet_bc = Dirichlet(
            surface_state = nothing,
            bottom_state = nothing,
        ),
        neumann_bc = Neumann(
            surface_flux = surface_heat_flux,
            bottom_flux = bottom_flux,
        ),
    );
    #temperature_value = FT(267.15)
    #soil_heat_model = PrescribedTemperatureModel((aux, t) -> temperature_value)

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = (freeze_thaw_source,)
    m = LandModel(
        param_set,
        m_soil;
        source = sources,
        init_state_prognostic = init_soil_water!,
    )

    driver_config = ClimateMachine.SingleStackConfiguration(
        "LandModel",
        N_poly,
        nelem_vert,
        zmax,
        param_set,
        m;
        zmin = zmin,
        numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
    )



    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    mygrid = solver_config.dg.grid
    Q = solver_config.Q

    ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
    θ_i_ind =
        varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :θ_i)

    all_data = Dict([k => Dict() for k in 1:n_outputs]...)
    step = [1]
    callback = GenericCallbacks.EveryXSimulationTime(
        every_x_simulation_time,
    ) do (init = false)
        t = ODESolvers.gettime(solver_config.solver)
        ϑ_l = Q[:, ϑ_l_ind, :]
        θ_i = Q[:, θ_i_ind, :]
        all_vars =
            Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l, "θ_i" => θ_i)
        all_data[step[1]] = all_vars
        step[1] += 1
        nothing
    end
    
    ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

    t = ODESolvers.gettime(solver_config.solver)
    ϑ_l = Q[:, ϑ_l_ind, :]
    θ_i = Q[:, θ_i_ind, :]
    all_vars = Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l, "θ_i" => θ_i)

    all_data[n_outputs] = all_vars

    m_liq =
        [ρ_cloud_liq(param_set) * mean(all_data[k]["ϑ_l"]) for k in 1:n_outputs]
    m_ice = [
        ρ_cloud_ice(param_set) * mean(all_data[k]["θ_i"])
        for k in 1:n_outputs
    ]
    t = [all_data[k]["t"][1] for k in 1:n_outputs]
    total_water = m_ice + m_liq
    τft = max(dt, τLTE)
    m_liq_of_t = m_liq[1] * exp.(-1.0 .* (t .- t[1]) ./ τft)
    m_ice_of_t = -m_liq_of_t .+ (m_ice[1] + m_liq[1])

    @test mean(abs.(m_ice + m_liq .- total_water)) < 1e-9
    @test mean(abs.(m_liq .- m_liq_of_t)) < 1e-9
    @test mean(abs.(m_ice .- m_ice_of_t)) < 1e-9
end
