# Update this to use dict_of_nodal_states
#Test that freezing front agrees with lab data from Mizoguchi.
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test
using Logging
disable_logging(Logging.Warn)
using DelimitedFiles
using Plots

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
using ClimateMachine.SystemSolvers
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state


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

function init_soil!(land, state, aux, coordinates, time)
    myFT = eltype(state)
    ϑ_l = myFT(land.soil.water.initialϑ_l(aux))
    θ_i = myFT(land.soil.water.initialθ_i(aux))
    state.soil.water.ϑ_l = ϑ_l
    state.soil.water.θ_i = θ_i

    θ_l = volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
    ρc_ds = land.soil.param_functions.ρc_ds
    ρc_s = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, land.param_set)

    state.soil.heat.ρe_int = volumetric_internal_energy(
        θ_i,
        ρc_s,
        land.soil.heat.initialT(aux),
        land.param_set,
    )
end;


ClimateMachine.init()
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(
    clima_dir,
    "tutorials",
    "Land",
    "Soil",
    "interpolation_helper.jl",
));

N_poly = 1
nelem_vert = 40
zmax = FT(0)
zmin = FT(-0.2)
t0 = FT(0)
timeend = FT(3600*51)
n_outputs = 60
every_x_simulation_time = ceil(Int, timeend / n_outputs)
Δ = get_grid_spacing(N_poly, nelem_vert, zmax, zmin)

porosity = FT(0.535)
ν_ss_quartz = FT(0.2)
ν_ss_minerals = FT(0.6)
ν_ss_om = FT(0.2)
ν_ss_gravel = FT(0.0);
κ_quartz = FT(7.7) # W/m/K
κ_minerals = FT(2.5) # W/m/K
κ_om = FT(0.25) # W/m/K
κ_liq = FT(0.57) # W/m/K
κ_ice = FT(2.29); # W/m/Kall_

ρp = FT(2700) # kg/m^3
κ_solid = k_solid(ν_ss_om, ν_ss_quartz, κ_quartz, κ_minerals, κ_om)
κ_sat_frozen = ksat_frozen(κ_solid, porosity, κ_ice)
κ_sat_unfrozen = ksat_unfrozen(κ_solid, porosity, κ_liq);

ρc_ds = FT((1 - porosity) * 2.3e6) # plJ/m^3/K

soil_param_functions =
    SoilParamFunctions{FT}(porosity = porosity,
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
vg_α = 1.11
vg_n = 1.48
soil_water_model = SoilWaterModel(
    FT;
    viscosity_factor = TemperatureDependentViscosity{FT}(),
    moisture_factor = MoistureDependent{FT}(),
    impedance_factor= IceImpedance{FT}(Ω = 7.0),
    hydraulics = vanGenuchten{FT}(α = vg_α, n = vg_n),
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

ρc_s = volumetric_heat_capacity(FT(0.33), FT(0.0), ρc_ds, param_set)
κ = FT(1.0)
τLTE = FT(ρc_s * Δ^FT(2.0) / κ)
dt = FT(2.0)
explicit = true


#changing this factor makes front move more slowly but doesnt change top value.
surface_heat_flux = (aux, t) -> eltype(aux)(28)*(aux.soil.heat.T-eltype(aux)(273.15-6))
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
freeze_thaw_source = Variableτ_FreezeThaw{FT}(Δt = dt, τLTE = τLTE)
m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model;
                   phase_change_source = freeze_thaw_source
                   )
sources = (freeze_thaw_source,)
m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_prognostic = init_soil!,
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

dg = solver_config.dg
Q = solver_config.Q

if explicit == false
    vdg = DGModel(
        driver_config.bl,
        driver_config.grid,
        driver_config.numerical_flux_first_order,
        driver_config.numerical_flux_second_order,
        driver_config.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        direction = VerticalDirection(),
    )
    
    linearsolver = BatchedGeneralizedMinimalResidual(
        dg,
        Q;
        max_subspace_size = 30,
        atol = -1.0,
        rtol = 1e-4,
    )
    
    nonlinearsolver =
        JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-4)
    
    ode_solver = ARK548L2SA2KennedyCarpenter(
        dg,
        vdg,
        NonLinearBackwardEulerSolver(
            nonlinearsolver;
            isadjustable = true,
            preconditioner_update_freq = 100,
        ),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = false,
        variant = NaiveVariant(),
    )
    
    solver_config.solver = ode_solver
end



aux = solver_config.dg.state_auxiliary
grads = solver_config.dg.state_gradient_flux
ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
θ_i_ind =
    varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :θ_i)
ρe_int_ind =
    varsindex(vars_state(m, Prognostic(), FT), :soil, :heat, :ρe_int)
T_ind =
    varsindex(vars_state(m, Auxiliary(), FT), :soil, :heat, :T)
h_ind =
    varsindex(vars_state(m, Auxiliary(), FT), :soil, :water, :h)
all_data = Dict([k => Dict() for k in 1:n_outputs]...)
z = aux[:,1:1,:]
# Specify interpolation grid:
zres = FT(abs(zmin/nelem_vert))
boundaries = [
    FT(0) FT(0) zmin
    FT(1) FT(1) zmax
]
resolution = (FT(2), FT(2), zres)
thegrid = solver_config.dg.grid
intrp_brck = create_interpolation_grid(boundaries, resolution, thegrid);
step = [1]
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
) do (init = false)
    t = ODESolvers.gettime(solver_config.solver)
   iQ, iaux, igrads = interpolate_variables((Q, aux, grads), intrp_brck)
    ϑ_l = iQ[:, ϑ_l_ind, :]
    θ_i = iQ[:, θ_i_ind, :]
    ρe_int = iQ[:, ρe_int_ind, :]
    T = iaux[:, T_ind, :]
    h = iaux[:, h_ind, :]
    all_vars =
        Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l, "θ_i" => θ_i, "ρe" => ρe_int, "T" => T, "h" => h)
    all_data[step[1]] = all_vars
    step[1] += 1
    nothing
end

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))
t = ODESolvers.gettime(solver_config.solver)
iQ, iaux, igrads = interpolate_variables((Q, aux, grads), intrp_brck)
iz = iaux[:, 1:1, :][:]


data_12h = readdlm("/Users/katherinedeck/Desktop/Papers/freeze:thaw/Data/freeze_thaw/mizoguchi_12h_vwc_depth.csv", ',')
first_plot = scatter(data_12h[:,1]./100.0,-data_12h[:,2], label = "data, 12h")
k = Int(round(12*3600/timeend*n_outputs))
scatter!(all_data[k]["θ_i"][:]+all_data[k]["ϑ_l"][:], iz[:], label = "model, 12h")
plot!(legend = :bottomright)

data_24h = readdlm("/Users/katherinedeck/Desktop/Papers/freeze:thaw/Data/freeze_thaw/mizoguchi_24h.csv", ',')
second = scatter(data_24h[:,1]./100.0,-data_24h[:,2], label = "data, 24h")
k = Int(round(24*3600/timeend*n_outputs))
scatter!(all_data[k]["θ_i"][:]+all_data[k]["ϑ_l"][:], iz[:], label = "model, 24h")
plot!(legend = :bottomright)
data_50h = readdlm("/Users/katherinedeck/Desktop/Papers/freeze:thaw/Data/freeze_thaw/mizoguchi_50h.csv", ',')
third = scatter(data_50h[:,1]./100.0,-data_50h[:,2], label = "data, 50h")
k = Int(round(50*3600/timeend*n_outputs))

scatter!(all_data[k]["θ_i"][:]+all_data[k]["ϑ_l"][:], iz[:], label = "model, 50h")
plot!(legend = :bottomright)
plot(first_plot,second,third, layout = (1,3))
savefig("./freeze_thaw_plots/mizoguchi_vt.png")

function f(k)
    T = all_data[k]["T"][:]
    θ_l = all_data[k]["ϑ_l"][:]
    θ_i = all_data[k]["θ_i"][:]
    τ = abs.(τLTE*100*0.535./(T.-273.15))[:]

    mask = (θ_l.>0.01) .* (T.<273.15)
    plot1 = scatter(all_data[k]["θ_i"][:]+all_data[k]["ϑ_l"][:], iz[:], xlim = [0.2,0.535],ylim = [-0.2,0], xlabel = "Total Vol. Water", label = "")
    plot2 = scatter(all_data[k]["ϑ_l"][:], iz[:], xlim = [0.0,0.35],ylim = [-0.2,0], xlabel = "Vol. Liquid", label = "")
    plot4 = scatter(T.-273.15, iz[:], xlim = [-6.5,8], xlabel = "T-T_freeze", label = "")
    plot(plot1, plot2, plot4, layout = (1,3))
end

anim = @animate for i ∈ 1:60
    f(i)
end
(gif(anim, "./freeze_thaw_plots/mizoguchi_vt.gif",fps = 8))
