# Test that freeze thaw alone reproduces expected behavior: exponential behavior
# for liquid water content, ice content, and total water conserved

#To be fixed - the grid spacing part, there is some onus on the user to define the τft function appropriately in the PrescribedTemperatureCase. We will define it in the SoilHeatModel case.
# another issue - passing spacing?
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test
using Logging
disable_logging(Logging.Warn)
using DelimitedFiles

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
    state.soil.heat.∇κ∇T = myFT(0.0)
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
timeend = FT(3600*50)
n_outputs = 50
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
cs = FT(3e6)
κ = FT(1.0)
#making this larger makes a big difference! in the top value of total water.
# but also makes freezing front behavior less kink-y
τLTE = FT(cs * Δ^FT(2.0) / κ * 6)
dt = FT(10)

freeze_thaw_source = FreezeThaw{FT}(Δt = dt,
                                    τLTE = τLTE)

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
#changing this factor makes front move more slowly but doesnt change top value.
surface_heat_flux = (aux, t) -> eltype(aux)(40)*(aux.soil.heat.T-eltype(aux)(273.15-6))
surface_T = nothing#(aux, t) -> eltype(aux)(273.15-6.0)
T_init = aux -> eltype(aux)(279.85)
soil_heat_model = SoilHeatModel(
    FT;
    initialT = T_init,
    dirichlet_bc = Dirichlet(
        surface_state = surface_T,
        bottom_state = nothing,
    ),
    neumann_bc = Neumann(
        surface_flux = surface_heat_flux,
        bottom_flux = bottom_flux,
    ),
);

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
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

#dg = solver_config.dg
#Q = solver_config.Q

#vdg = DGModel(
#    driver_config.bl,
#    driver_config.grid,
#    driver_config.numerical_flux_first_order,
#    driver_config.numerical_flux_second_order,
#    driver_config.numerical_flux_gradient,
#    state_auxiliary = dg.state_auxiliary,
#    direction = VerticalDirection(),
#)



#linearsolver = BatchedGeneralizedMinimalResidual(
#    dg,
#    Q;
#    max_subspace_size = 30,
#    atol = -1.0,
#    rtol = 1e-4,
#)

"""
N(q)(Q) = Qhat  => F(Q) = N(q)(Q) - Qhat

F(Q) == 0
||F(Q^i) || / ||F(Q^0) || < tol

"""
#nonlinearsolver =
#    JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-4)

#ode_solver = ARK548L2SA2KennedyCarpenter(
#    dg,
#    vdg,
#    NonLinearBackwardEulerSolver(
#        nonlinearsolver;
#        isadjustable = true,
#        preconditioner_update_freq = 100,
#    ),
#    Q;
#    dt = dt,
#    t0 = 0,
#    split_explicit_implicit = false,
#    variant = NaiveVariant(),
#)

#solver_config.solver = ode_solver

Q = solver_config.Q
aux = solver_config.dg.state_auxiliary
grads = solver_config.dg.state_gradient_flux
ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
θ_i_ind =
    varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :θ_i)
ρe_int_ind =
    varsindex(vars_state(m, Prognostic(), FT), :soil, :heat, :ρe_int)
T_ind =
    varsindex(vars_state(m, Auxiliary(), FT), :soil, :heat, :T)
divT_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :heat, :∇κ∇T)

all_data = Dict([k => Dict() for k in 1:n_outputs]...)
z = aux[:,1:1,:]
# Specify interpolation grid:
zres = FT(0.01)
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
    divT = iQ[:, divT_ind, :]
    T = iaux[:, T_ind, :]
    all_vars =
        Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l, "θ_i" => θ_i, "ρe" => ρe_int, "T" => T, "divT" => divT)
    all_data[step[1]] = all_vars
    step[1] += 1
    nothing
end

ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))
t = ODESolvers.gettime(solver_config.solver)
iQ, iaux, igrads = interpolate_variables((Q, aux, grads), intrp_brck)
ϑ_l = iQ[:, ϑ_l_ind, :]
θ_i = iQ[:, θ_i_ind, :]
ρe_int = iQ[:, ρe_int_ind, :]
divT = iQ[:, divT_ind, :]
T = iaux[:, T_ind, :]
all_vars =
    Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l, "θ_i" => θ_i, "ρe" => ρe_int, "T" => T, "divT" => divT)
iz = iaux[:, 1:1, :][:]

all_data[n_outputs] = all_vars

m_liq =
    [ρ_cloud_liq(param_set) * mean(all_data[k]["ϑ_l"]) for k in 1:n_outputs]
m_ice = [
    ρ_cloud_ice(param_set) * mean(all_data[k]["θ_i"])
    for k in 1:n_outputs]


data_12h = readdlm("/Users/katherinedeck/Desktop/Papers/freeze:thaw/Data/freeze_thaw/mizoguchi_12h_vwc_depth.csv", ',')
first_plot = scatter(data_12h[:,1]./100.0,-data_12h[:,2], label = "data, 12h")
k = 12
scatter!(all_data[k]["θ_i"][:]+all_data[k]["ϑ_l"][:], iz[:], label = "model, 12h")
plot!(legend = :bottomright)

data_24h = readdlm("/Users/katherinedeck/Desktop/Papers/freeze:thaw/Data/freeze_thaw/mizoguchi_24h.csv", ',')
second = scatter(data_24h[:,1]./100.0,-data_24h[:,2], label = "data, 24h")
k = 24
scatter!(all_data[k]["θ_i"][:]+all_data[k]["ϑ_l"][:], iz[:], label = "model, 24h")
plot!(legend = :bottomright)
data_50h = readdlm("/Users/katherinedeck/Desktop/Papers/freeze:thaw/Data/freeze_thaw/mizoguchi_50h.csv", ',')
third = scatter(data_50h[:,1]./100.0,-data_50h[:,2], label = "data, 50h")
k = 50
scatter!(all_data[k]["θ_i"][:]+all_data[k]["ϑ_l"][:], iz[:], label = "model, 50h")
plot!(legend = :bottomright)
plot(first_plot,second,third, layout = (1,3))
