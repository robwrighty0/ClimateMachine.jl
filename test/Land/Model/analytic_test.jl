using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test
using Logging
disable_logging(Logging.Warn)
using DelimitedFiles
#using Plots

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
using CLIMAParameters.Planet: ρ_cloud_liq
using CLIMAParameters.Planet: ρ_cloud_ice
using CLIMAParameters.Planet: LH_f0

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
using SpecialFunctions

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
zmin = FT(-2)
t0 = FT(0)
timeend = FT(3600*24*20)
n_outputs = 540
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
                           Ksat = 0.0,
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
τLTE = FT(cs * Δ^FT(2.0) / κ)
dt = FT(50)

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

surface_T = (aux, t) -> eltype(aux)(273.15-10.0)
T_init = aux -> eltype(aux)(275.15)
soil_heat_model = SoilHeatModel(
    FT;
    initialT = T_init,
    dirichlet_bc = Dirichlet(
        surface_state = surface_T,
        bottom_state = nothing,
    ),
    neumann_bc = Neumann(
        surface_flux = nothing,
        bottom_flux = bottom_flux,
    ),
);

phase_change_source = Variableτ_FreezeThaw{FT}(Δt = dt, τLTE = τLTE)
m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model;
                   phase_change_source = phase_change_source
                   )
sources = (phase_change_source,)
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
aux = solver_config.dg.state_auxiliary
grads = solver_config.dg.state_gradient_flux
ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
θ_i_ind =
    varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :θ_i)
ρe_int_ind =
    varsindex(vars_state(m, Prognostic(), FT), :soil, :heat, :ρe_int)
T_ind =
    varsindex(vars_state(m, Auxiliary(), FT), :soil, :heat, :T)

all_data = Dict([k => Dict() for k in 1:n_outputs]...)
z = aux[:,1:1,:]
# Specify interpolation grid:
zres = FT(0.05)
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
    all_vars =
        Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l, "θ_i" => θ_i, "ρe" => ρe_int, "T" => T)
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
T = iaux[:, T_ind, :]
all_vars =
    Dict{String, Array}("t" => [t], "ϑ_l" => ϑ_l, "θ_i" => θ_i, "ρe" => ρe_int, "T" => T)
iz = iaux[:, 1:1, :][:]

all_data[n_outputs] = all_vars

#analytic solution
kdry = k_dry(param_set, soil_param_functions)
# Frozen region:
ksat = saturated_thermal_conductivity(FT(0.0), FT(0.33)*ρ_cloud_liq(param_set)/ρ_cloud_ice(param_set), κ_sat_unfrozen, κ_sat_frozen)
kersten = kersten_number(FT(0.33)*ρ_cloud_liq(param_set)/ρ_cloud_ice(param_set), FT(0.33)*ρ_cloud_liq(param_set)/ρ_cloud_ice(param_set)/FT(0.535), soil_param_functions)
λ1 = thermal_conductivity(kdry, kersten, ksat)
c1 = volumetric_heat_capacity(FT(0.0), FT(0.33)*ρ_cloud_liq(param_set)/ρ_cloud_ice(param_set), ρc_ds, param_set)
d1 = λ1/c1

ksat = saturated_thermal_conductivity(FT(0.33), FT(0.0), κ_sat_unfrozen, κ_sat_frozen)
kersten = kersten_number(FT(0.0), FT(0.33)/FT(0.535), soil_param_functions)
λ2 = thermal_conductivity(kdry, kersten, ksat)
c2 = volumetric_heat_capacity(FT(0.33), FT(0.0), ρc_ds, param_set)
d2 = λ2/c2
Ti = FT(275.15-273.15)
Ts = FT(-10.0)

function implicit(ζ)
    term1 = exp(-ζ^2)/ζ/erf(ζ)
    term2 = -λ2*sqrt(d1)*(Ti-0)/(λ1*sqrt(d2)*(0-Ts)*ζ*erfc(ζ*sqrt(d1/d2)))*exp(-d1/d2*ζ^2)
    term3 = -LH_f0(param_set)*ρ_cloud_liq(param_set)*0.33*sqrt(π)/c1/(0-Ts)
    return (term1+term2+term3)
end


ζ = 0.265       


function f2(k;ζ = ζ)
    T = all_data[k]["T"][:].-273.15
    θ_l = all_data[k]["ϑ_l"][:]
    θ_i = all_data[k]["θ_i"][:]
    plot!(θ_i, iz[:], xlim = [0,0.4], ylim = [-2,0],xlabel = "volumetric water content", label = "θ_i")
end





function f(k;ζ = ζ)
    T = all_data[k]["T"][:].-273.15
    θ_l = all_data[k]["ϑ_l"][:]
    θ_i = all_data[k]["θ_i"][:]
    plot(T, iz[:], xlim = [-10.5,3], ylim = [-2,0],xlabel = "T", label = "simulation")
    t = all_data[k]["t"][1]
    zf = 2.0*ζ*sqrt(d1*t)
    z = -2:0.01:0
    spatially_varying = (erfc.(abs.(z)./(zf/ζ/(d1/d2)^0.5)))./erfc(ζ*(d1/d2)^0.5)
    mask = abs.(z) .>zf
    plot!(Ti.-(Ti-0.0).*spatially_varying[mask], z[mask], label = "analytic", color = "green")
    
    spatially_varying = ((erf.(abs.(z)./(zf/ζ))))./erf(ζ)
    mask = abs.(z) .< zf
    plot!(Ts.+(0.0-Ts).*spatially_varying[mask], z[mask], label = "", color = "green")

    
end

k = n_outputs
ds = readdlm("../../../../bonan_sp/bonanmodeling/sp_05_03/bonan_data.csv", ',')
ds2 = readdlm("../../../../bonan_sp/bonanmodeling/sp_05_03/bonan_data_ahc.csv", ',')
f(k)
plot!(ds[:,2], ds[:,1], label = "Excess Heat")
plot!(ds2[:,2], ds2[:,1], label = "Apparent heat capacity")
savefig("./freeze_thaw_plots/analytic_comparison.png")
