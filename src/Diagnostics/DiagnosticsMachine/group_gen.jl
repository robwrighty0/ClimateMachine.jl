# Given an expression that is either a symbol which is a type name or a
# Union of types, each of which shares the same supertype, return the
# supertype.
parent_type(sym::Symbol) = supertype(getfield(@__MODULE__, sym))
function parent_type(ex::Expr)
    @assert ex.head == :curly && ex.args[1] == :Union
    st = supertype(getfield(@__MODULE__, ex.args[2]))
    for ut in ex.args[3:end]
        otherst = supertype(getfield(@__MODULE__, ut))
        @assert otherst == st
    end
    return st
end
# Return `true` if the specified symbol is a type name that is a subtype
# of `BalanceLaw` and `false` otherwise.
isa_bl(sym::Symbol) = any(
    bl -> endswith(bl, "." * String(sym)),
    map(bl -> String(Symbol(bl)), subtypes(BalanceLaw)),
)
isa_bl(ex) = false

uppers_in(s) = foldl((f, c) -> isuppercase(c) ? f * c : f, s, init = "")

# Generate a `VariableTemplates` defining function.
function generate_vars_fun(varsfunname, DN, DT, vardecls, compdecls)
    quote
        function $varsfunname($DN::$DT, FT)
            @vars begin
                $(vardecls...)
                $(compdecls...)
            end
        end
    end
end

# Generate the `VariableTemplates` defining functions for the specified
# `dvars`.
function generate_vars_funs(
    name,
    config_type,
    params_type,
    init_fun,
    interpolate,
    dvars,
    dvtype_dvars_map,
)

    CT = getfield(ConfigTypes, config_type)
    vars_funs = Any[]

    # Generate `vars_*` functions for each type of diagnostic variable.
    for (dvtype, dvlst) in dvtype_dvars_map
        dvtypename = split(String(Symbol(dvtype)), ".")[end]
        varsfunname = Symbol("vars_", name, "_", uppers_in(dvtypename))

        DT_name_map = Dict() # dispatch type to argument name
        DT_var_map = Dict()  # dispatch type to list of variable definitions

        # Look at each diagnostic variable implementation; the first
        # argument is used for dispatch and is either the `BalanceLaw` or a
        # component of the `BalanceLaw`. Group the diagnostic variables by
        # this dispatch type.
        for dvar in dvlst
            dispatch_arg = first(dv_args(CT(), dvar))
            DN = dispatch_arg[1] # the name
            DT = dispatch_arg[2] # the type
            compname = get(DT_name_map, DT, nothing)
            if isnothing(compname)
                DT_name_map[DT] = DN
            else
                # if a different name is used, we _could_ rewrite the function...
                @assert compname == DN
            end
            push!(
                get!(DT_var_map, DT, Any[]),
                :($(Symbol(dv_name(CT(), dvar)))::FT),
            )
        end

        # Generate a function for each dispatch type.
        for (DT, dvlst) in DT_var_map
            # There should be a function for the `BalanceLaw`, and this may
            # contain component declarations, i.e. calls to the other `vars`
            # functions. Find which calls have to be inserted.
            complst = Any[]
            if isa_bl(DT)
                for (otherDT, compname) in DT_name_map
                    if otherDT != DT
                        push!(
                            complst,
                            :(
                                $(
                                    compname
                                )::$varsfunname(
                                    $(DT_name_map[DT]).$compname,
                                    FT,
                                )
                            ),
                        )
                    end
                end
            else
                # For components of the `BalanceLaw`, add an empty function
                # for the parent of the dispatch type.
                push!(
                    vars_funs,
                    generate_vars_fun(
                        varsfunname,
                        DT_name_map[DT],
                        parent_type(DT),
                        Any[],
                        Any[],
                    ),
                )
            end

            # Add the `vars_` function.
            push!(
                vars_funs,
                generate_vars_fun(
                    varsfunname,
                    DT_name_map[DT],
                    DT,
                    dvlst,
                    complst,
                ),
            )
        end
    end

    return Expr(:block, (vars_funs...))
end

# Generate the common definitions used in many places.
function generate_common_defs()
    quote
        interpol = dgngrp.interpol
        params = dgngrp.params
        mpicomm = Settings.mpicomm
        mpirank = MPI.Comm_rank(mpicomm)
        dg = Settings.dg
        bl = dg.balance_law
        grid = dg.grid
        topology = grid.topology
        N = polynomialorder(grid)
        Nq = N + 1
        Nqk = dimensionality(grid) == 2 ? 1 : Nq
        npoints = Nq * Nq * Nqk
        nrealelem = length(topology.realelems)
        nvertelem = topology.stacksize
        nhorzelem = div(nrealelem, nvertelem)
        Q = Settings.Q
        FT = eltype(Q)
    end
end

# Generate the `dims` dictionary for `Writers.init_data`.
function generate_init_dims(name, interpolate, dvars)
    # For a diagnostics group to be output on the DG grid, we add some
    # "dimensions". For pointwise diagnostics, we add `nodes` and
    # `elements`. For horizontal averages, we add a `z` dimension.
    # TODO: abstract this by using `dv_<something>` so that we aren't
    # checking types here?
    add_dim_ex = quote end
    if interpolate == :NoInterpolation
        add_z_dim_ex = quote end
        if any(dv -> typeof(dv) <: HorizontalAverage, dvars)
            add_z_dim_ex = quote
                dims["z"] = (AtmosCollected.zvals, Dict())
            end
        end
        add_ne_dims_ex = quote end
        if any(dv -> typeof(dv) <: PointwiseDiagnostic, dvars)
            add_ne_dims_ex = quote
                dims["nodes"] = (collect(1:npoints), Dict())
                dims["elements"] = (collect(1:nrealelem), Dict())
            end
        end
        add_dim_ex = quote
            $(add_z_dim_ex)
            $(add_ne_dims_ex)
        end
    end

    quote
        dims = dimensions(interpol)
        if isempty(dims)
            $(add_dim_ex)
        elseif interpol isa InterpolationCubedSphere
            # Adjust `level` on the sphere.
            level_val = dims["level"]
            dims["level"] = (
                level_val[1] .- FT(planet_radius(Settings.param_set)),
                level_val[2],
            )
        end
        dims
    end
end

# Generate the `vars` dictionary for `Writers.init_data`.
function generate_init_vars(config_type, dvars)
    CT = getfield(ConfigTypes, config_type)
    varslst = Any[]
    for dvar in dvars
        rhs = :((dv_dimnames($dvar, dims), FT, $(dv_attrib(CT(), dvar))))
        lhs = :($(dv_name(CT(), dvar)))
        push!(varslst, :($lhs => $rhs))
    end

    quote
        # TODO: add code to filter this based on what's actually in `bl`.
        OrderedDict($(Expr(:tuple, varslst...))...)
    end
end

# Generate `Diagnostics.$(name)_init(...)` which will initialize the
# `DiagnosticsGroup` when called.
function generate_init_fun(
    name,
    config_type,
    params_type,
    init_fun,
    interpolate,
    dvars,
    dvtype_dvars_map,
)
    init_name = Symbol(name, "_init")
    CT = getfield(ConfigTypes, config_type)
    quote
        function $init_name(dgngrp, curr_time)
            $(generate_common_defs())

            $(init_fun)(dgngrp, curr_time)

            if dgngrp.onetime
                collect_onetime(Settings.mpicomm, Settings.dg, Settings.Q)
            end

            if mpirank == 0
                dims = $(generate_init_dims(name, interpolate, dvars))
                vars = $(generate_init_vars(config_type, dvars))

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
    end
end

# Generate code snippet for copying arrays to the CPU if needed. Ideally,
# this will be removed when diagnostics are made to run on GPU.
function generate_array_copies()
    quote
        # get needed arrays onto the CPU
        if array_device(Q) isa CPU
            state_data = Q.realdata
            gradflux_data = dg.state_gradient_flux.realdata
            aux_data = dg.state_auxiliary.realdata
            vgeo = grid.vgeo
        else
            state_data = Array(Q.realdata)
            gradflux_data = Array(dg.state_gradient_flux.realdata)
            aux_data = Array(dg.state_auxiliary.realdata)
            vgeo = Array(grid.vgeo)
        end
    end
end

# Generate code to create the necessary arrays for the diagnostics
# variables.
function generate_create_vars_arrays(name, config_type, dvtype_dvars_map)
    CT = getfield(ConfigTypes, config_type)
    cva_exs = []
    for (dvtype, dvlst) in dvtype_dvars_map
        dvt_short = uppers_in(split(String(Symbol(dvtype)), ".")[end])
        arr_name = Symbol("vars_", dvt_short, "_array")
        varsfunname = Symbol("vars_", name, "_", dvt_short)
        npoints = dv_dg_points_range(CT(), dvtype)
        nelems = dv_dg_elems_range(CT(), dvtype)
        cva_ex = quote
            nvars = varsize($(varsfunname)(bl, FT))
            $(arr_name) = Array{FT}(undef, $(npoints), nvars, $(nelems))
        end
        push!(cva_exs, cva_ex)
    end
    return Expr(:block, (cva_exs...))
end

# Generate calls to the implementations for the `DiagnosticVar`s in this
# group and store the results.
function generate_collect_calls(name, config_type, dvtype_dvars_map)
    CT = getfield(ConfigTypes, config_type)
    cc_exs = []
    for (dvtype, dvlst) in dvtype_dvars_map
        dvt_short = uppers_in(split(String(Symbol(dvtype)), ".")[end])
        vars_name = Symbol("vars_", dvt_short)
        for dvar in dvlst
            # TODO: call_ex = 
            cc_ex = quote
                dv_op(
                    $(CT()),
                    $(dvtype),
                    getproperty($(vars_name), $(Symbol(dv_name(CT(), dvar)))),
                    MH,
                )
            end
            push!(cc_exs, cc_ex)
        end
    end
    println(cc_exs)

    return Expr(:block, (cc_exs...))
end

# Generate the nested loops to traverse the DG grid within which we extract
# the various states and then generate the individual collection calls.
function generate_dg_collection(name, config_type, dvtype_dvars_map)
    CT = getfield(ConfigTypes, config_type)
    gv_exs = []
    for (dvtype, dvlst) in dvtype_dvars_map
        dvt_short = uppers_in(split(String(Symbol(dvtype)), ".")[end])
        vars_name = Symbol("vars_", dvt_short)
        arr_name = Symbol("vars_", dvt_short, "_array")
        varsfunname = Symbol("vars_", name, "_", dvt_short)
        pt = dv_dg_points_index(CT(), dvtype)
        elem = dv_dg_elems_index(CT(), dvtype)
        gv_ex = quote
            $(vars_name) = Vars{$(varsfunname)(bl, FT)}(
                view($(arr_name), $(pt), :, $(elem))
            )
        end
        push!(gv_exs, gv_ex)
    end
    quote
        for eh in 1:nhorzelem, ev in 1:nvertelem
            e = ev + (eh - 1) * nvertelem
            for k in 1:Nqk, j in 1:Nq, i in 1:Nq
                ijk = i + Nq * ((j - 1) + Nq * (k - 1))
                evk = Nqk * (ev - 1) + k
                MH = vgeo[ijk, grid.MHid, e]
                z = AtmosCollected.zvals[evk]
                states = States(
                    extract_state(bl, state_data, ijk, e, Prognostic()),
                    extract_state(
                        bl,
                        gradflux_data,
                        ijk,
                        e,
                        GradientFlux(),
                    ),
                    extract_state(bl, aux_data, ijk, e, Auxiliary()),
                )
                $(gv_exs...)
                $(generate_collect_calls(name, config_type, dvtype_dvars_map))
            end
        end
    end
end

function generate_interpolation(dvars)
    if interpolate != :NoInterpolation
        quote
            all_state_data = nothing
            all_gradflux_data = nothing
            all_aux_data = nothing

            if interpolate && !isempty(dvars_ig)
                istate_array = similar(
                    Q.realdata,
                    interpol.Npl,
                    number_states(bl, Prognostic()),
                )
                interpolate_local!(interpol, Q.realdata, istate_array)
                igradflux_array = similar(
                    Q.realdata,
                    interpol.Npl,
                    number_states(bl, GradientFlux()),
                )
                interpolate_local!(
                    interpol,
                    dg.state_gradient_flux.realdata,
                    igradflux_array,
                )
                iaux_array = similar(
                    Q.realdata,
                    interpol.Npl,
                    number_states(bl, Auxiliary()),
                )
                interpolate_local!(
                    interpol,
                    dg.state_auxiliary.realdata,
                    iaux_array,
                )

                _ρu, _ρv, _ρw = 2, 3, 4
                project_cubed_sphere!(interpol, istate_array, (_ρu, _ρv, _ρw))

                # FIXME: accumulating to rank 0 is not scalable
                all_state_data = accumulate_interpolated_data(
                    mpicomm,
                    interpol,
                    istate_array,
                )
                all_gradflux_data = accumulate_interpolated_data(
                    mpicomm,
                    interpol,
                    igradflux_array,
                )
                all_aux_data =
                    accumulate_interpolated_data(mpicomm, interpol, iaux_array)
            end
        end
    else
        quote end
    end
end

# Generate `Diagnostics.$(name)_collect(...)` which when called,
# performs a collection of all the diagnostic variables in the group
# and writes them out.
function generate_collect_fun(
    name,
    config_type,
    params_type,
    init_fun,
    interpolate,
    dvars,
    dvtype_dvars_map,
)
    collect_name = Symbol(name, "_collect")
    CT = getfield(ConfigTypes, config_type)
    quote
        function $collect_name(dgngrp, curr_time)
            $(generate_common_defs())
            $(generate_array_copies())
            $(generate_create_vars_arrays(name, config_type, dvtype_dvars_map))

            # Traverse the DG grid and collect diagnostics as needed.
            $(generate_dg_collection(name, config_type, dvtype_dvars_map))

            #=
            # Interpolate and accumulate if needed.
            $(generate_interpolation(dvars))

            # Traverse the interpolated grid and collect diagnostics if needed.
            $(generate_ig_collection(dvars))
            if interpolate
                ivars_array = similar(Q.realdata, interpol.Npl, n_grp_vars)
                interpolate_local!(interpol, vars_array, ivars_array)
                all_ivars_data = accumulate_interpolated_data(
                    mpicomm,
                    interpol,
                    ivars_array,
                )
            end

            # TODO: density averaging.

            if mpirank == 0
                dims = dimensions(interpol)
                (nx, ny, nz) = map(k -> dims[k][1], collect(keys(dims)))
                for x in 1:nx, y in 1:ny, z in 1:nz
                    istate = Vars{vars_state(bl, Prognostic(), FT)}(view(
                        all_state_data,
                        x,
                        y,
                        z,
                        :,
                    ))
                    igradflux = Vars{vars_state(bl, GradientFlux(), FT)}(view(
                        all_gradflux_data,
                        x,
                        y,
                        z,
                        :,
                    ))
                    iaux = Vars{vars_state(bl, Auxiliary(), FT)}(view(
                        all_aux_data,
                        x,
                        y,
                        z,
                        :,
                    ))
                    states = States(istate, igradflux, iaux)
                    vars = Vars{grp_vars}(view(ivars_array, x, y, z, :))
                    $(generate_collect_calls(name, config_type, dvars_ig))
                end

                varvals = OrderedDict()
                varnames = map(
                    s -> startswith(s, "moisture.") ? s[10:end] : s, # XXX: FIXME
                    flattenednames(grp_vars),
                )
                for (vari, varname) in enumerate(varnames)
                    varvals[varname] = vars_array[
                        ntuple(_ -> Colon(), ndims(vars_array))...,
                        vari,
                    ] # XXX: FIXME
                end
                append_data(dgngrp.writer, varvals, curr_time)
            end
            =#

            MPI.Barrier(mpicomm)
            return nothing
        end
    end
end

# Generate `Diagnostics.$(name)_fini(...)`, which does nothing right now.
function generate_fini_fun(name, args...)
    fini_name = Symbol(name, "_fini")
    quote
        function $fini_name(dgngrp, curr_time) end
    end
end

# Generate `setup_$(name)(...)` which will create the `DiagnosticsGroup`
# for $name when called.
function generate_setup_fun(
    name,
    config_type,
    params_type,
    init_fun,
    interpolate,
    dvars,
    dvtype_dvars_map,
)
    setupfun = Symbol("setup_", name)
    initfun = Symbol(name, "_init")
    collectfun = Symbol(name, "_collect")
    finifun = Symbol(name, "_fini")

    no_intrp_err = quote end
    some_intrp_err = quote end
    if interpolate != :NoInterpolation
        some_intrp_err = quote
            throw(ArgumentError(
                "$name specifies interpolation, but no " *
                "`InterpolationTopology` has been provided.",
            ))
        end
    else
        no_intrp_err = quote
            @warn "$(name) does not specify interpolation, but an " *
                  "`InterpolationTopology` has been provided; ignoring."
            interpol = nothing
        end
    end
    quote
        function $setupfun(
            ::$config_type,
            params::$params_type,
            interval::String,
            out_prefix::String,
            writer = NetCDFWriter(),
            interpol = nothing,
        ) where {
            $config_type <: ClimateMachineConfigType,
            $params_type <: Union{Nothing, DiagnosticsGroupParams},
        }
            if isnothing(interpol)
                $(some_intrp_err)
            else
                $(no_intrp_err)
            end

            return DiagnosticsGroup(
                $(name),
                $(initfun),
                $(collectfun),
                $(finifun),
                interval,
                out_prefix,
                writer,
                interpol,
                $(interpolate != :InterpolateAfterCollection),
                params,
            )
        end
    end
end
