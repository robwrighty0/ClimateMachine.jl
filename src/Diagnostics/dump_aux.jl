"""
    setup_dump_aux_diagnostics(
        ::ClimateMachineConfigType,
        interval::String,
        out_prefix::String;
        writer = NetCDFWriter(),
        interpol = nothing,
    )

Create the "DumpAux" `DiagnosticsGroup` which contains all the
auxiliary state variables. These are output on `x`, `y`, `z` or `lat`,
`long`, `level` dimensions of an interpolated grid (`interpol` _must_
be specified) as well as a `time` dimension at the specified `interval`.
"""
function setup_dump_aux_diagnostics(
    ::ClimateMachineConfigType,
    interval::String,
    out_prefix::String;
    writer = NetCDFWriter(),
    interpol = nothing,
)
    # TODO: remove this
    @assert !isnothing(interpol)

    return DiagnosticsGroup(
        "DumpAux",
        Diagnostics.dump_aux_init,
        Diagnostics.dump_aux_collect,
        Diagnostics.dump_aux_fini,
        interval,
        out_prefix,
        writer,
        interpol,
    )
end

dump_aux_init(dgngrp, currtime) = dump_init(dgngrp, currtime, Auxiliary())

function dump_aux_collect(dgngrp, currtime)
    interpol = dgngrp.interpol
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    FT = eltype(Q.data)
    bl = dg.balance_law
    mpirank = MPI.Comm_rank(mpicomm)

    iaux = similar(
        dg.state_auxiliary.data,
        interpol.Npl,
        number_states(bl, Auxiliary()),
    )

    interpolate_local!(interpol, dg.state_auxiliary.data, iaux)

    all_aux_data = accumulate_interpolated_data(mpicomm, interpol, iaux)

    if mpirank == 0
        auxnames = flattenednames(vars_state(bl, Auxiliary(), FT))
        varvals = OrderedDict()
        for (vari, varname) in enumerate(auxnames)
            varvals[varname] = all_aux_data[:, :, :, vari]
        end
        append_data(dgngrp.writer, varvals, currtime)
    end

    MPI.Barrier(mpicomm)
    return nothing
end

function dump_aux_fini(dgngrp, currtime) end
