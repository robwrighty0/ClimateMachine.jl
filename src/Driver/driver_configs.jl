# ClimateMachine driver configurations
#
# Contains helper functions to establish simulation configurations to be
# used with the ClimateMachine driver. Currently:
# - AtmosLESConfiguration
# - AtmosGCMConfiguration
# - OceanBoxGCMConfiguration
# - SingleStackConfiguration
#
# User-customized configurations can use these as templates.

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius

abstract type ConfigSpecificInfo end
struct AtmosLESSpecificInfo <: ConfigSpecificInfo end
struct AtmosGCMSpecificInfo{FT} <: ConfigSpecificInfo
    domain_height::FT
    nelem_vert::Int
    nelem_horz::Int
end
struct OceanBoxGCMSpecificInfo <: ConfigSpecificInfo end
struct SingleStackSpecificInfo <: ConfigSpecificInfo end

include("SolverTypes/SolverTypes.jl")

"""
    ClimateMachine.DriverConfiguration

Collects all parameters necessary to set up a ClimateMachine simulation.
"""
struct DriverConfiguration{FT}
    config_type::ClimateMachineConfigType

    name::String
    # polynomial order tuple (polyorder_horz, polyorder_vert)
    polyorders::NTuple{2, Int}
    array_type::Any
    solver_type::AbstractSolverType
    #
    # Model details
    param_set::AbstractParameterSet
    bl::BalanceLaw
    #
    # execution details
    mpicomm::MPI.Comm
    #
    # mesh details
    grid::DiscontinuousSpectralElementGrid
    #
    # DGModel details
    numerical_flux_first_order::NumericalFluxFirstOrder
    numerical_flux_second_order::NumericalFluxSecondOrder
    numerical_flux_gradient::NumericalFluxGradient
    #
    # configuration-specific info
    config_info::ConfigSpecificInfo

    function DriverConfiguration(
        config_type,
        name::String,
        polyorders::NTuple{2, Int},
        FT,
        array_type,
        solver_type::AbstractSolverType,
        param_set::AbstractParameterSet,
        bl::BalanceLaw,
        mpicomm::MPI.Comm,
        grid::DiscontinuousSpectralElementGrid,
        numerical_flux_first_order::NumericalFluxFirstOrder,
        numerical_flux_second_order::NumericalFluxSecondOrder,
        numerical_flux_gradient::NumericalFluxGradient,
        config_info::ConfigSpecificInfo,
    )
        return new{FT}(
            config_type,
            name,
            polyorders,
            array_type,
            solver_type,
            param_set,
            bl,
            mpicomm,
            grid,
            numerical_flux_first_order,
            numerical_flux_second_order,
            numerical_flux_gradient,
            config_info,
        )
    end
end

function print_model_info(model)
    msg = "Model composition\n"
    for key in fieldnames(typeof(model))
        msg =
            msg * @sprintf(
                "    %s = %s\n",
                string(key),
                string((getproperty(model, key)))
            )
    end
    @info msg
    show_tendencies(model)
end

function AtmosLESConfiguration(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    (Δx, Δy, Δz)::NTuple{3, FT},
    xmax::FT,
    ymax::FT,
    zmax::FT,
    param_set::AbstractParameterSet,
    init_LES!;
    xmin = zero(FT),
    ymin = zero(FT),
    zmin = zero(FT),
    array_type = ClimateMachine.array_type(),
    solver_type = IMEXSolverType(
        implicit_solver = SingleColumnLU,
        implicit_solver_adjustable = false,
    ),
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        init_state_prognostic = init_LES!,
    ),
    mpicomm = MPI.COMM_WORLD,
    boundary = ((0, 0), (0, 0), (1, 2)),
    periodicity = (true, true, false),
    meshwarp = (x...) -> identity(x),
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
) where {FT <: AbstractFloat}

    (polyorder_horz, polyorder_vert) = isa(N, Int) ? (N, N) : N

    print_model_info(model)

    brickrange = (
        grid1d(xmin, xmax, elemsize = Δx * polyorder_horz),
        grid1d(ymin, ymax, elemsize = Δy * polyorder_horz),
        grid1d(zmin, zmax, elemsize = Δz * polyorder_vert),
    )
    topology = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = periodicity,
        boundary = boundary,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horz, polyorder_vert),
        meshwarp = meshwarp,
    )

    @info @sprintf(
        """
Establishing Atmos LES configuration for %s
    precision              = %s
    horiz polynomial order = %d
    vert polynomial order  = %d
    domain                 = %.2f m x%.2f m x%.2f m
    resolution             = %dx%dx%d
    MPI ranks              = %d
    min(Δ_horz)            = %.2f m
    min(Δ_vert)            = %.2f m""",
        name,
        FT,
        polyorder_horz,
        polyorder_vert,
        xmax,
        ymax,
        zmax,
        Δx,
        Δy,
        Δz,
        MPI.Comm_size(mpicomm),
        min_node_distance(grid, HorizontalDirection()),
        min_node_distance(grid, VerticalDirection())
    )

    return DriverConfiguration(
        AtmosLESConfigType(),
        name,
        (polyorder_horz, polyorder_vert),
        FT,
        array_type,
        solver_type,
        param_set,
        model,
        mpicomm,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        AtmosLESSpecificInfo(),
    )
end

function AtmosGCMConfiguration(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    (nelem_horz, nelem_vert)::NTuple{2, Int},
    domain_height::FT,
    param_set::AbstractParameterSet,
    init_GCM!;
    array_type = ClimateMachine.array_type(),
    solver_type = DefaultSolverType(),
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        init_state_prognostic = init_GCM!,
    ),
    mpicomm = MPI.COMM_WORLD,
    meshwarp::Function = cubedshellwarp,
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
) where {FT <: AbstractFloat}

    (polyorder_horz, polyorder_vert) = isa(N, Int) ? (N, N) : N

    print_model_info(model)

    _planet_radius::FT = planet_radius(param_set)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + domain_height),
        nelem = nelem_vert,
    )

    topology = StackedCubedSphereTopology(
        mpicomm,
        nelem_horz,
        vert_range;
        boundary = (1, 2),
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horz, polyorder_vert),
        meshwarp = meshwarp,
    )

    @info @sprintf(
        """
Establishing Atmos GCM configuration for %s
    precision              = %s
    horiz polynomial order = %d
    vert polynomial order  = %d
    # horiz elem           = %d
    # vert elems           = %d
    domain height          = %.2e m
    MPI ranks              = %d
    min(Δ_horz)            = %.2f m
    min(Δ_vert)            = %.2f m""",
        name,
        FT,
        polyorder_horz,
        polyorder_vert,
        nelem_horz,
        nelem_vert,
        domain_height,
        MPI.Comm_size(mpicomm),
        min_node_distance(grid, HorizontalDirection()),
        min_node_distance(grid, VerticalDirection())
    )

    return DriverConfiguration(
        AtmosGCMConfigType(),
        name,
        (polyorder_horz, polyorder_vert),
        FT,
        array_type,
        solver_type,
        param_set,
        model,
        mpicomm,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        AtmosGCMSpecificInfo(domain_height, nelem_vert, nelem_horz),
    )
end

function OceanBoxGCMConfiguration(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    (Nˣ, Nʸ, Nᶻ)::NTuple{3, Int},
    param_set::AbstractParameterSet,
    model;
    FT = Float64,
    array_type = ClimateMachine.array_type(),
    solver_type = ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    ),
    mpicomm = MPI.COMM_WORLD,
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
    periodicity = (false, false, false),
    boundary = ((1, 1), (1, 1), (2, 3)),
)

    (polyorder_horz, polyorder_vert) = isa(N, Int) ? (N, N) : N

    brickrange = (
        range(FT(0); length = Nˣ + 1, stop = model.problem.Lˣ),
        range(FT(0); length = Nʸ + 1, stop = model.problem.Lʸ),
        range(FT(-model.problem.H); length = Nᶻ + 1, stop = 0),
    )

    topology = StackedBrickTopology(
        mpicomm,
        brickrange;
        periodicity = periodicity,
        boundary = boundary,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horz, polyorder_vert),
    )

    return DriverConfiguration(
        OceanBoxGCMConfigType(),
        name,
        (polyorder_horz, polyorder_vert),
        FT,
        array_type,
        solver_type,
        param_set,
        model,
        mpicomm,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        OceanBoxGCMSpecificInfo(),
    )
end

function SingleStackConfiguration(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    nelem_vert::Int,
    zmax::FT,
    param_set::AbstractParameterSet,
    model::BalanceLaw;
    zmin = zero(FT),
    hmax = one(FT),
    array_type = ClimateMachine.array_type(),
    solver_type = ExplicitSolverType(),
    mpicomm = MPI.COMM_WORLD,
    boundary = ((0, 0), (0, 0), (1, 2)),
    periodicity = (true, true, false),
    meshwarp = (x...) -> identity(x),
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
) where {FT <: AbstractFloat}

    (polyorder_horz, polyorder_vert) = isa(N, Int) ? (N, N) : N

    print_model_info(model)

    xmin, xmax = zero(FT), hmax
    ymin, ymax = zero(FT), hmax
    brickrange = (
        grid1d(xmin, xmax, nelem = 1),
        grid1d(ymin, ymax, nelem = 1),
        grid1d(zmin, zmax, nelem = nelem_vert),
    )
    topology = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = periodicity,
        boundary = boundary,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horz, polyorder_vert),
        meshwarp = meshwarp,
    )

    @info @sprintf(
        """
Establishing single stack configuration for %s
    precision              = %s
    horiz polynomial order = %d
    vert polynomial order  = %d
    domain_min             = %.2f m x%.2f m x%.2f m
    domain_max             = %.2f m x%.2f m x%.2f m
    # vert elems           = %d
    MPI ranks              = %d
    min(Δ_horz)            = %.2f m
    min(Δ_vert)            = %.2f m""",
        name,
        FT,
        polyorder_horz,
        polyorder_vert,
        xmin,
        ymin,
        zmin,
        xmax,
        ymax,
        zmax,
        nelem_vert,
        MPI.Comm_size(mpicomm),
        min_node_distance(grid, HorizontalDirection()),
        min_node_distance(grid, VerticalDirection())
    )

    return DriverConfiguration(
        SingleStackConfigType(),
        name,
        (polyorder_horz, polyorder_vert),
        FT,
        array_type,
        solver_type,
        param_set,
        model,
        mpicomm,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        SingleStackSpecificInfo(),
    )
end

import ..DGMethods: DGModel

"""
    DGModel(driver_config; kwargs...)

Initialize a [`DGModel`](@ref) given a
[`DriverConfiguration`](@ref) and keyword
arguments supported by [`DGModel`](@ref).
"""
DGModel(driver_config; kwargs...) = DGModel(
    driver_config.bl,
    driver_config.grid,
    driver_config.numerical_flux_first_order,
    driver_config.numerical_flux_second_order,
    driver_config.numerical_flux_gradient;
    kwargs...,
)
