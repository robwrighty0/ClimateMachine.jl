import KernelAbstractions: CPU

using ..Mesh.Grids
using ..Mesh.Elements: interpolationmatrix
using ..MPIStateArrays
using ..DGMethods
using ..TicToc

"""
    writevtk(prefix, Q::MPIStateArray, dg::DGModel [, fieldnames];
             number_sample_points = 0)

Write a vtk file for all the fields in the state array `Q` using geometry and
connectivity information from `dg.grid`. The filename will start with `prefix`
which may also contain a directory path. The names used for each of the fields
in the vtk file can be specified through the collection of strings `fieldnames`;
if not specified the fields names will be `"Q1"` through `"Qk"` where `k` is the
number of states in `Q`, i.e., `k = size(Q,2)`.

If `number_sample_points > 0` then the fields are sampled on an equally spaced,
tensor-product grid of points with 'number_sample_points' in each direction and
the output VTK element type is set to by a VTK lagrange type.

When `number_sample_points == 0` the raw nodal values are saved, and linear VTK
elements are used connecting the degree of freedom boxes.
"""
function writevtk(
    prefix,
    Q::MPIStateArray,
    dg::DGModel,
    fieldnames = nothing;
    number_sample_points = 0,
)
    vgeo = dg.grid.vgeo
    device = array_device(Q)
    (h_vgeo, h_Q) = device isa CPU ? (vgeo, Q.data) : (Array(vgeo), Array(Q))
    writevtk_helper(
        prefix,
        h_vgeo,
        h_Q,
        dg.grid,
        fieldnames;
        number_sample_points = number_sample_points,
    )
    return nothing
end

"""
    writevtk(prefix, Q::MPIStateArray, dg::DGModel, fieldnames,
             state_auxiliary::MPIStateArray, auxfieldnames;
             number_sample_points = 0)

Write a vtk file for all the fields in the state array `Q` and auxiliary state
`state_auxiliary` using geometry and connectivity information from `dg.grid`. The
filename will start with `prefix` which may also contain a directory path. The
names used for each of the fields in the vtk file can be specified through the
collection of strings `fieldnames` and `auxfieldnames`.

If `fieldnames === nothing` then the fields names will be `"Q1"` through `"Qk"`
where `k` is the number of states in `Q`, i.e., `k = size(Q,2)`.

If `auxfieldnames === nothing` then the fields names will be `"aux1"` through
`"auxk"` where `k` is the number of states in `state_auxiliary`, i.e., `k =
size(state_auxiliary,2)`.

If `number_sample_points > 0` then the fields are sampled on an equally spaced,
tensor-product grid of points with 'number_sample_points' in each direction and
the output VTK element type is set to by a VTK lagrange type.

When `number_sample_points == 0` the raw nodal values are saved, and linear VTK
elements are used connecting the degree of freedom boxes.
"""
function writevtk(
    prefix,
    Q::MPIStateArray,
    dg::DGModel,
    fieldnames,
    state_auxiliary,
    auxfieldnames;
    number_sample_points = 0,
)
    vgeo = dg.grid.vgeo
    device = array_device(Q)
    (h_vgeo, h_Q, h_aux) =
        device isa CPU ? (vgeo, Q.data, state_auxiliary.data) :
        (Array(vgeo), Array(Q), Array(state_auxiliary))
    writevtk_helper(
        prefix,
        h_vgeo,
        h_Q,
        dg.grid,
        fieldnames,
        h_aux,
        auxfieldnames;
        number_sample_points = number_sample_points,
    )
    return nothing
end

"""
    writevtk_helper(prefix, vgeo::Array, Q::Array, grid, fieldnames)

Internal helper function for `writevtk`
"""
function writevtk_helper(
    prefix,
    vgeo::Array,
    Q::Array,
    grid,
    fieldnames,
    state_auxiliary = nothing,
    auxfieldnames = nothing;
    number_sample_points,
)
    @assert number_sample_points >= 0

    dim = dimensionality(grid)
    N = polynomialorders(grid)
    Nq = N .+ 1

    nelem = size(Q)[end]

    X = grid.x_vtk[1:dim]
    fields = ntuple(j -> (@view Q[:, j, :]), size(Q, 2))

    auxfields =
        isnothing(state_auxiliary) ? () :
        (
            auxfields = ntuple(
                j -> (@view state_auxiliary[:, j, :]),
                size(state_auxiliary, 2),
            )
        )

    # If any dimension are N = 0 we mirror these out to the boundaries for viz
    # purposed
    if any(N .== 0)
        Nq_N1 = max.(Nq, 2)
        Np_N1 = prod(Nq_N1)
        ind = ntuple(i -> N[i] == 0 ? [1, 1] : Colon(), dim)
        fields = ntuple(length(fields)) do i
            fld = reshape(fields[i], Nq..., nelem)
            fld2 = @view fld[ind..., :]
            reshape(fld2, (Np_N1, nelem))
        end
        auxfields = ntuple(length(auxfields)) do i
            fld = reshape(auxfields[i], Nq..., nelem)
            fld2 = @view fld[ind..., :]
            reshape(fld2, (Np_N1, nelem))
        end
        Nq = Nq_N1
    end

    # Interpolate to an equally spaced grid if necessary
    if number_sample_points > 0
        FT = eltype(Q)
        # If any dimension are N = 0 we manual set (-1, 1) grids
        ξ = ntuple(
            i -> N[i] == 0 ? FT.([-1, 1]) : referencepoints(grid)[i],
            dim,
        )
        ξdst = range(FT(-1); length = number_sample_points, stop = 1)
        I1d = ntuple(i -> interpolationmatrix(ξ[dim - i + 1], ξdst), dim)
        I = kron(I1d...)
        fields = ntuple(i -> I * fields[i], length(fields))
        auxfields = ntuple(i -> I * auxfields[i], length(auxfields))
        X = ntuple(i -> I * X[i], length(X))
        Nq = ntuple(j -> number_sample_points, dim)
    end

    X = ntuple(i -> reshape(X[i], Nq..., nelem), length(X))
    fields = ntuple(i -> reshape(fields[i], Nq..., nelem), length(fields))
    auxfields =
        ntuple(i -> reshape(auxfields[i], Nq..., nelem), length(auxfields))

    if fieldnames === nothing
        fields = ntuple(i -> ("Q$i", fields[i]), length(fields))
    else
        fields = ntuple(i -> (fieldnames[i], fields[i]), length(fields))
    end

    if auxfieldnames === nothing
        auxfields = ntuple(i -> ("aux$i", auxfields[i]), length(auxfields))
    else
        auxfields =
            ntuple(i -> (auxfieldnames[i], auxfields[i]), length(auxfields))
    end

    fields = (fields..., auxfields...)
    if number_sample_points > 0
        return writemesh_highorder(
            prefix,
            X...;
            fields = fields,
            realelems = grid.topology.realelems,
        )
    else
        return writemesh_raw(
            prefix,
            X...;
            fields = fields,
            realelems = grid.topology.realelems,
        )
    end
end
