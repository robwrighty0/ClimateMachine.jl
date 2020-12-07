# AtmosTurbulenceStats
#
# Computes average kinetic energy and dissipation.

using ..Atmos
using ..Mesh.Topologies
using ..Mesh.Grids

struct TurbulenceStatsParams <: DiagnosticsGroupParams
    nor::Float64
    iter::Float64
end

"""
    setup_atmos_turbulence_stats(
        ::ClimateMachineConfigType,
        interval::String,
        out_prefix::String,
        nor::Float64,
        iter::Float64;
        writer = NetCDFWriter(),
        interpol = nothing,
    )

Create the "AtmosTurbulenceStats" `DiagnosticsGroup` which contains the
following diagnostic variables:

- E_k: volumetrically-averaged dimensionless kinetic energy
- dE: volumetrically-averaged kinetic energy dissipation

These are scalar variables computed on the DG grid (`interpol` may _not_
be specified), output on the (unlimited) `time` dimension at the specified
`interval`.
"""
function setup_atmos_turbulence_stats(
    ::ClimateMachineConfigType,
    interval::String,
    out_prefix::String,
    nor::Float64,
    iter::Float64;
    writer = NetCDFWriter(),
    interpol = nothing,
)
    @assert isnothing(interpol)

    return DiagnosticsGroup(
        "AtmosTurbulenceStats",
        Diagnostics.atmos_turbulence_stats_init,
        Diagnostics.atmos_turbulence_stats_collect,
        Diagnostics.atmos_turbulence_stats_fini,
        interval,
        out_prefix,
        writer,
        interpol,
        TurbulenceStatsParams(nor, iter),
    )
end

# store average kinetic energy and dissipation
Base.@kwdef mutable struct AtmosTurbulenceStats{FT}
    E_k::FT = 0.0
end
const AtmosTurbulenceStatsCollected = AtmosTurbulenceStats()

function atmos_turbulence_stats_init(dgngrp, currtime)
    FT = eltype(Settings.Q)
    mpicomm = Settings.mpicomm
    mpirank = MPI.Comm_rank(mpicomm)

    if mpirank == 0
        # outputs are scalars
        dims = OrderedDict()

        # set up the variables we're going to be writing
        vars = OrderedDict(
            "E_k" => ((), FT, Variables["E_k"].attrib),
            "dE" => ((), FT, Variables["dE"].attrib),
        )

        # create the output file
        dprefix = @sprintf(
            "%s_%s_%s",
            dgngrp.out_prefix,
            dgngrp.name,
            Settings.starttime,
        )
        dfilename = joinpath(Settings.output_dir, dprefix)
        init_data(dgngrp.writer, dfilename, dims, vars)
    end

    return nothing
end

function average_kinetic_energy_and_dissipation(
    Q,
    vgeo,
    E_0,
    nor,
    iter,
    mpicomm,
)
    u₀ = Q.ρu ./ Q.ρ
    u_0 = u₀[:, 1, :] ./ nor
    v_0 = u₀[:, 2, :] ./ nor
    w_0 = u₀[:, 3, :] ./ nor

    M = vgeo[:, Grids._M, 1:size(u_0, 2)]
    SM = sum(M)

    E_k = 0.5 * sum((u_0 .^ 2 .+ v_0 .^ 2 .+ w_0 .^ 2) .* M) / SM
    E_k = MPI.Reduce(E_k, +, 0, mpicomm)

    mpirank = MPI.Comm_rank(mpicomm)
    nranks = MPI.Comm_size(mpicomm)
    if mpirank == 0
        E_k /= nranks
        dE = -(E_k - E_0) / iter
        return (E_k, dE)
    end

    return (0.0, 0.0)
end

function atmos_turbulence_stats_collect(dgngrp, currtime)
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    mpirank = MPI.Comm_rank(mpicomm)
    nranks = MPI.Comm_size(mpicomm)

    E_k, dE = average_kinetic_energy_and_dissipation(
        Q,
        dg.grid.vgeo,
        AtmosTurbulenceStatsCollected.E_k,
        dgngrp.params.nor,
        dgngrp.params.iter,
        mpicomm,
    )

    if mpirank == 0
        AtmosTurbulenceStatsCollected.E_k = E_k

        varvals = OrderedDict("E_k" => E_k, "dE" => dE)

        # write output
        append_data(dgngrp.writer, varvals, currtime)
    end

end

function atmos_turbulence_stats_fini(dgngrp, currtime) end
