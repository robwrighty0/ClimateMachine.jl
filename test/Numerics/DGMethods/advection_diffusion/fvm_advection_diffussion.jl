using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using LinearAlgebra
using Printf
using Test
import ClimateMachine.VTK: writevtk, writepvtu
import ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("advection_diffusion_model.jl")

struct Pseudo1D{n1, n2, n3, α, β, μ, δ} <: AdvectionDiffusionProblem end

function init_velocity_diffusion!(
    ::Pseudo1D{n1, n2, n3, α, β},
    aux::Vars,
    geom::LocalGeometry,
) where {n1, n2, n3, α, β}
    # Direction of flow is n1 (resp n2 or n3) with magnitude α
    aux.advection.u = hcat(α * n1, α * n2, α * n3)

    # diffusion of strength β in the n1 and n2 directions
    aux.diffusion.D = hcat(β * n1 * n1', β * n2 * n2', β * n3 * n3')
end

function initial_condition!(
    ::Pseudo1D{n1, n2, n3, α, β, μ, δ},
    state,
    aux,
    localgeo,
    t,
) where {n1, n2, n3, α, β, μ, δ}
    ξn1 = dot(n1, localgeo.coord)
    ξn2 = dot(n2, localgeo.coord)
    ξn3 = dot(n3, localgeo.coord)
    ρ1 = exp(-(ξn1 - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
    ρ2 = exp(-(ξn2 - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
    ρ3 = exp(-(ξn3 - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
    state.ρ = (ρ1, ρ2, ρ3)
end

Dirichlet_data!(P::Pseudo1D, x...) = initial_condition!(P, x...)

function Neumann_data!(
    ::Pseudo1D{n1, n2, n3, α, β, μ, δ},
    ∇state,
    aux,
    x,
    t,
) where {n1, n2, n3, α, β, μ, δ}
    ξn1 = dot(n1, x)
    ξn2 = dot(n2, x)
    ξn3 = dot(n3, x)
    ∇ρ1 =
        -(
            2n1 * (ξn1 - μ - α * t) / (4 * β * (δ + t)) *
            exp(-(ξn1 - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
        )
    ∇ρ2 =
        -(
            2n2 * (ξn2 - μ - α * t) / (4 * β * (δ + t)) *
            exp(-(ξn2 - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
        )
    ∇ρ3 =
        -(
            2n3 * (ξn3 - μ - α * t) / (4 * β * (δ + t)) *
            exp(-(ξn3 - μ - α * t)^2 / (4 * β * (δ + t))) / sqrt(1 + t / δ)
        )
    ∇state.ρ = hcat(∇ρ1, ∇ρ2, ∇ρ3)
end

function do_output(mpicomm, vtkdir, vtkstep, dgfvm, Q, Qe, model, testname)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    exactnames = statenames .* "_exact"

    writevtk(filename, Q, dgfvm, statenames, Qe, exactnames)

    ## generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(
            pvtuprefix,
            prefixes,
            (statenames..., exactnames...),
            eltype(Q),
        )

        @info "Done writing VTK: $pvtuprefix"
    end
end


function test_run(mpicomm, dim, polynomialorders, level, ArrayType, FT, vtkdir)

    n_hd =
        dim == 2 ? SVector{3, FT}(1, 0, 0) :
        SVector{3, FT}(1 / sqrt(2), 1 / sqrt(2), 0)

    n_vd = dim == 2 ? SVector{3, FT}(0, 1, 0) : SVector{3, FT}(0, 0, 1)

    n_dg =
        dim == 2 ? SVector{3, FT}(1 / sqrt(2), 1 / sqrt(2), 0) :
        SVector{3, FT}(1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3))

    α = FT(1)
    β = FT(1 // 100)
    μ = FT(-1 // 2)
    δ = FT(1 // 10)

    # Grid/topology information
    base_num_elem = 4
    Ne = 2^(level - 1) * base_num_elem
    N = polynomialorders
    L = ntuple(j -> FT(j == dim ? 1 : N[1]) / 4, dim)
    brickrange = ntuple(j -> range(-L[j]; length = Ne + 1, stop = L[j]), dim)
    periodicity = ntuple(j -> false, dim)
    bc = ntuple(j -> (1, 2), dim)

    topl = StackedBrickTopology(
        mpicomm,
        brickrange;
        periodicity = periodicity,
        boundary = bc,
    )

    dt = (α / 4) * L[1] / (Ne * polynomialorders[1]^2)
    timeend = 1
    outputtime = timeend / 10
    @info "time step" dt

    @info @sprintf """Test parameters:
    ArrayType                   = %s
    FloatType                   = %s
    Dimension                   = %s
    Horizontal polynomial order = %s
    Vertical polynomial order   = %s
      """ ArrayType FT dim polynomialorders[1] polynomialorders[end]

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorders,
    )

    # Model being tested
    model = AdvectionDiffusion{dim}(
        Pseudo1D{n_hd, n_vd, n_dg, α, β, μ, δ}(),
        num_equations = 3,
    )

    # Main DG discretization
    dgfvm = DGFVModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = EveryDirection(),
    )

    # Initialize all relevant state arrays and create solvers
    Q = init_ode_state(dgfvm, FT(0))

    eng0 = norm(Q, dims = (1, 3))
    @info @sprintf """Starting
    norm(Q₀) = %.16e""" eng0[1]

    solver = LSRK54CarpenterKennedy(dgfvm, Q; dt = dt, t0 = 0)

    # Set up the information callback
    starttime = Ref(Dates.now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = Dates.now()
        else
            energy = norm(Q)
            @info @sprintf(
                """Update
                simtime = %.16e
                runtime = %s
                norm(Q) = %.16e""",
                gettime(solver),
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end
    callbacks = (cbinfo,)
    if ~isnothing(vtkdir)
        # create vtk dir
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(
            mpicomm,
            vtkdir,
            vtkstep,
            dgfvm,
            Q,
            Q,
            model,
            "advection_diffusion",
        )

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dgfvm, gettime(solver))
            do_output(
                mpicomm,
                vtkdir,
                vtkstep,
                dgfvm,
                Q,
                Qe,
                model,
                "advection_diffusion",
            )
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(Q, solver; timeend = timeend, callbacks = callbacks)

    # Reference solution
    engf = norm(Q, dims = (1, 3))
    Q_ref = init_ode_state(dgfvm, FT(timeend))

    engfe = norm(Q_ref, dims = (1, 3))
    errf = norm(Q_ref .- Q, dims = (1, 3))

    metrics = @. (engf, engf / eng0, engf - eng0, errf, errf / engfe)

    @info @sprintf """Finished
    Horizontal field:
      norm(Q)                 = %.16e
      norm(Q) / norm(Q₀)      = %.16e
      norm(Q) - norm(Q₀)      = %.16e
      norm(Q - Qe)            = %.16e
      norm(Q - Qe) / norm(Qe) = %.16e
    Vertical field:
      norm(Q)                 = %.16e
      norm(Q) / norm(Q₀)      = %.16e
      norm(Q) - norm(Q₀)      = %.16e
      norm(Q - Qe)            = %.16e
      norm(Q - Qe) / norm(Qe) = %.16e
    Diagonal field:
      norm(Q)                 = %.16e
      norm(Q) / norm(Q₀)      = %.16e
      norm(Q) - norm(Q₀)      = %.16e
      norm(Q - Qe)            = %.16e
      norm(Q - Qe) / norm(Qe) = %.16e
      """ first.(metrics)... ntuple(f -> metrics[f][2], 5)... last.(metrics)...

    return Tuple(errf)
end

"""
    main()

Run this test problem
"""
function main()

    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    # Dictionary keys: dim, level, and FT
    expected_result = Dict()
    expected_result[2, 1, Float64] =
        (2.3391871809628567e-02, 5.0738761087541523e-02, 4.1857480220018339e-02)
    expected_result[2, 2, Float64] =
        (2.0332783913617892e-03, 3.3669399006499491e-02, 2.3904204301839819e-02)
    expected_result[2, 3, Float64] =
        (2.6572168347086839e-05, 2.0002110124502395e-02, 1.2790993871268749e-02)
    expected_result[2, 4, Float64] =
        (1.9890000550154039e-07, 1.0929144069594809e-02, 6.5110938897763176e-03)

    expected_result[3, 1, Float64] =
        (8.7378337431297422e-03, 7.1755444068009461e-02, 4.3733196512722658e-02)
    expected_result[3, 2, Float64] =
        (6.2510740807095622e-04, 4.7615720711942776e-02, 2.3986017606198885e-02)
    expected_result[3, 3, Float64] =
        (3.4995405318038341e-05, 2.8287255414151381e-02, 1.2639742577376040e-02)
    expected_result[3, 4, Float64] =
        (1.4362091045094841e-06, 1.5456143768350493e-02, 6.3677406803847331e-03)

    expected_result[2, 1, Float32] =
        (2.3391991853713989e-02, 5.0738703459501266e-02, 4.1857466101646423e-02)
    expected_result[2, 2, Float32] =
        (2.0331495907157660e-03, 3.3669382333755493e-02, 2.3904176428914070e-02)
    expected_result[2, 3, Float32] =
        (2.6557327146292664e-05, 2.0002063363790512e-02, 1.2790882028639317e-02)

    expected_result[3, 1, Float32] =
        (8.7377587333321571e-03, 7.1755334734916687e-02, 4.3733172118663788e-02)
    expected_result[3, 2, Float32] =
        (6.2518188497051597e-04, 4.7615684568881989e-02, 2.3986009880900383e-02)
    expected_result[3, 3, Float32] =
        (3.5005086829187348e-05, 2.8287241235375404e-02, 1.2639495544135571e-02)


    @testset "Variable degree DG: advection diffusion model" begin
        for FT in (Float32, Float64)
            numlevels =
                integration_testing ||
                ClimateMachine.Settings.integration_testing ?
                (FT == Float64 ? 4 : 3) : 1
            for dim in 2:3
                polynomialorders = (4, 0)
                result = Dict()
                for level in 1:numlevels
                    vtkdir =
                        output ?
                        "vtk_advection" *
                        "_poly$(polynomialorders)" *
                        "_dim$(dim)_$(ArrayType)_$(FT)" *
                        "_level$(level)" :
                        nothing
                    result[level] = test_run(
                        mpicomm,
                        dim,
                        polynomialorders,
                        level,
                        ArrayType,
                        FT,
                        vtkdir,
                    )
                    @test all(
                        result[level] .≈ FT.(expected_result[dim, level, FT]),
                    )
                end
                @info begin
                    msg = ""
                    for l in 1:(numlevels - 1)
                        rate = @. log2(result[l]) - log2(result[l + 1])
                        msg *= @sprintf(
                            "\n  rates for level %d Horizontal = %e",
                            l,
                            rate[1]
                        )
                        msg *= @sprintf(", Vertical = %e", rate[2])
                        msg *= @sprintf(", Diagonal = %e\n", rate[3])
                    end
                    msg
                end
            end
        end
    end
end

main()
