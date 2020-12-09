# ClimateMachine driver configurations
#
# Contains helper functions to establish simulation configurations to be
# used with the ClimateMachine driver. Currently:
# - AtmosLESConfiguration
# - AtmosGCMConfiguration
# - OceanBoxGCMConfiguration
# - OceanSplitExplicitConfiguration
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
    nelem_horiz::Int
end
struct OceanBoxGCMSpecificInfo <: ConfigSpecificInfo end
struct OceanSplitExplicitSpecificInfo <: ConfigSpecificInfo
    model_2D::BalanceLaw
    grid_2D::DiscontinuousSpectralElementGrid
    dg::DGModel
end
struct SingleStackSpecificInfo <: ConfigSpecificInfo end

include("SolverTypes/SolverTypes.jl")

"""
    ClimateMachine.DriverConfiguration

Collects all parameters necessary to set up a ClimateMachine simulation.
"""
struct DriverConfiguration{FT}
    config_type::ClimateMachineConfigType

    name::String
    # polynomial order tuple (polyorder_horiz, polyorder_vert)
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

    (polyorder_horiz, polyorder_vert) = isa(N, Int) ? (N, N) : N
    # Check if polynomial degree was passed as a CL option
    if ClimateMachine.Settings.degree != "(-1,-1)"
        (polyorder_horiz, polyorder_vert) = parse_tuple(ClimateMachine.Settings.degree)
    end

    print_model_info(model)

    brickrange = (
        grid1d(xmin, xmax, elemsize = Δx * polyorder_horiz),
        grid1d(ymin, ymax, elemsize = Δy * polyorder_horiz),
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
        polynomialorder = (polyorder_horiz, polyorder_vert),
        meshwarp = meshwarp,
    )

    @info @sprintf(
        """
Establishing Atmos LES configuration for %s
    precision               = %s
    horiz polynomial order  = %d
    vert polynomial order   = %d
    domain                  = %.2f m x%.2f m x%.2f m
    resolution              = %dx%dx%d
    MPI ranks               = %d
    min(Δ_horiz)            = %.2f m
    min(Δ_vert)             = %.2f m""",
        name,
        FT,
        polyorder_horiz,
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
        (polyorder_horiz, polyorder_vert),
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
    (nelem_horiz, nelem_vert)::NTuple{2, Int},
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

    (polyorder_horiz, polyorder_vert) = isa(N, Int) ? (N, N) : N
    # Check if polynomial degree was passed as a CL option
    if ClimateMachine.Settings.degree != "(-1,-1)"
        (polyorder_horiz, polyorder_vert) = parse_tuple(ClimateMachine.Settings.degree)
    end

    print_model_info(model)

    _planet_radius::FT = planet_radius(param_set)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + domain_height),
        nelem = nelem_vert,
    )

    topology = StackedCubedSphereTopology(
        mpicomm,
        nelem_horiz,
        vert_range;
        boundary = (1, 2),
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horiz, polyorder_vert),
        meshwarp = meshwarp,
    )

    @info @sprintf(
        """
Establishing Atmos GCM configuration for %s
    precision               = %s
    horiz polynomial order  = %d
    vert polynomial order   = %d
    # horiz elem            = %d
    # vert elems            = %d
    domain height           = %.2e m
    MPI ranks               = %d
    min(Δ_horiz)            = %.2f m
    min(Δ_vert)             = %.2f m""",
        name,
        FT,
        polyorder_horiz,
        polyorder_vert,
        nelem_horiz,
        nelem_vert,
        domain_height,
        MPI.Comm_size(mpicomm),
        min_node_distance(grid, HorizontalDirection()),
        min_node_distance(grid, VerticalDirection())
    )

    return DriverConfiguration(
        AtmosGCMConfigType(),
        name,
        (polyorder_horiz, polyorder_vert),
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
        AtmosGCMSpecificInfo(domain_height, nelem_vert, nelem_horiz),
    )
end

function OceanBoxGCMConfiguration(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    (Nˣ, Nʸ, Nᶻ)::NTuple{3, Int},
    param_set::AbstractParameterSet,
    model::HydrostaticBoussinesqModel;
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

    (polyorder_horiz, polyorder_vert) = isa(N, Int) ? (N, N) : N
    # Check if polynomial degree was passed as a CL option
    if ClimateMachine.Settings.degree != "(-1,-1)"
        (polyorder_horiz, polyorder_vert) = parse_tuple(ClimateMachine.Settings.degree)
    end

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
        polynomialorder = (polyorder_horiz, polyorder_vert),
    )

    return DriverConfiguration(
        OceanBoxGCMConfigType(),
        name,
        (polyorder_horiz, polyorder_vert),
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

function OceanSplitExplicitConfiguration(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    (Nˣ, Nʸ, Nᶻ)::NTuple{3, Int},
    param_set::AbstractParameterSet,
    model_3D::OceanModel;
    FT = Float64,
    array_type = ClimateMachine.array_type(),
    solver_type = SplitExplicitSolverType{FT}(90.0 * 60.0, 240.0),
    mpicomm = MPI.COMM_WORLD,
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
    periodicity = (false, false, false),
    boundary = ((1, 1), (1, 1), (2, 3)),
)

    (polyorder_horiz, polyorder_vert) = isa(N, Int) ? (N, N) : N
    # Check if polynomial degree was passed as a CL option
    if ClimateMachine.Settings.degree != "(-1,-1)"
        (polyorder_horiz, polyorder_vert) = parse_tuple(ClimateMachine.Settings.degree)
    end

    xrange = range(FT(0); length = Nˣ + 1, stop = model_3D.problem.Lˣ)
    yrange = range(FT(0); length = Nʸ + 1, stop = model_3D.problem.Lʸ)
    zrange = range(FT(-model_3D.problem.H); length = Nᶻ + 1, stop = 0)

    brickrange_2D = (xrange, yrange)
    brickrange_3D = (xrange, yrange, zrange)

    topology_2D = BrickTopology(
        mpicomm,
        brickrange_2D;
        periodicity = (periodicity[1], periodicity[2]),
        boundary = (boundary[1], boundary[2]),
    )
    topology_3D = StackedBrickTopology(
        mpicomm,
        brickrange_3D;
        periodicity = periodicity,
        boundary = boundary,
    )

    grid_2D = DiscontinuousSpectralElementGrid(
        topology_2D,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = polyorder_horiz,
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topology_3D,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horiz, polyorder_vert),
    )

    model_2D = BarotropicModel(model_3D)

    dg_2D = DGModel(
        model_2D,
        grid_2D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
    )

    Q_2D = init_ode_state(dg_2D, FT(0); init_on_cpu = true)

    vert_filter = CutoffFilter(grid_3D, polyorder_vert - 1)
    exp_filter = ExponentialFilter(grid_3D, 1, 8)

    flowintegral_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.FlowIntegralModel(model_3D),
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
    )

    tendency_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.TendencyIntegralModel(model_3D),
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
    )

    conti3d_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.Continuity3dModel(model_3D),
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
    )
    conti3d_Q = init_ode_state(conti3d_dg, FT(0); init_on_cpu = true)

    ivdc_dg = DGModel(
        ClimateMachine.Ocean.SplitExplicit01.IVDCModel(model_3D),
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient;
        direction = VerticalDirection(),
    )
    # Not sure this is needed since we set values later,
    # but we'll do it just in case!
    ivdc_Q = init_ode_state(ivdc_dg, FT(0); init_on_cpu = true)
    ivdc_RHS = init_ode_state(ivdc_dg, FT(0); init_on_cpu = true)

    ivdc_bgm_solver = BatchedGeneralizedMinimalResidual(
        ivdc_dg,
        ivdc_Q;
        max_subspace_size = 10,
    )

    modeldata = (
        dg_2D = dg_2D,
        Q_2D = Q_2D,
        vert_filter = vert_filter,
        exp_filter = exp_filter,
        flowintegral_dg = flowintegral_dg,
        tendency_dg = tendency_dg,
        conti3d_dg = conti3d_dg,
        conti3d_Q = conti3d_Q,
        ivdc_dg = ivdc_dg,
        ivdc_Q = ivdc_Q,
        ivdc_RHS = ivdc_RHS,
        ivdc_bgm_solver = ivdc_bgm_solver,
    )

    dg_3D = DGModel(
        model_3D,
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient;
        modeldata = modeldata,
    )


    return DriverConfiguration(
        OceanSplitExplicitConfigType(),
        name,
        (polyorder_horiz, polyorder_vert),
        FT,
        array_type,
        solver_type,
        param_set,
        model_3D,
        mpicomm,
        grid_3D,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        OceanSplitExplicitSpecificInfo(model_2D, grid_2D, dg_3D),
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

    (polyorder_horiz, polyorder_vert) = isa(N, Int) ? (N, N) : N
    # Check if polynomial degree was passed as a CL option
    if ClimateMachine.Settings.degree != "(-1,-1)"
        (polyorder_horiz, polyorder_vert) = parse_tuple(ClimateMachine.Settings.degree)
    end

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
        polynomialorder = (polyorder_horiz, polyorder_vert),
        meshwarp = meshwarp,
    )

    @info @sprintf(
        """
Establishing single stack configuration for %s
    precision               = %s
    horiz polynomial order  = %d
    vert polynomial order   = %d
    domain_min              = %.2f m x%.2f m x%.2f m
    domain_max              = %.2f m x%.2f m x%.2f m
    # vert elems            = %d
    MPI ranks               = %d
    min(Δ_horiz)            = %.2f m
    min(Δ_vert)             = %.2f m""",
        name,
        FT,
        polyorder_horiz,
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
        (polyorder_horiz, polyorder_vert),
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
