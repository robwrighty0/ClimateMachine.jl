# Helpers to gather and store some information useful to multiple diagnostics
# groups.
#

Base.@kwdef mutable struct AtmosCollectedDiagnostics
    onetime_done::Bool = false
    zvals::Union{Nothing, Array} = nothing
    MH_z::Union{Nothing, Array} = nothing
end
const AtmosCollected = AtmosCollectedDiagnostics()

function atmos_collect_onetime(mpicomm, dg, Q)
    if !AtmosCollected.onetime_done
        FT = eltype(Q)
        grid = dg.grid
        topology = grid.topology
        # XXX: Needs updating for multiple polynomial orders
        N = polynomialorders(grid)
        # Currently only support single polynomial order
        @assert all(N[1] .== N)
        N = N[1]
        Nq = N + 1
        Nqk = dimensionality(grid) == 2 ? 1 : Nq
        nrealelem = length(topology.realelems)
        nvertelem = topology.stacksize
        nhorzelem = div(nrealelem, nvertelem)

        vgeo = array_device(Q) isa CPU ? grid.vgeo : Array(grid.vgeo)

        AtmosCollected.zvals = zeros(FT, Nqk * nvertelem)
        AtmosCollected.MH_z = zeros(FT, Nqk * nvertelem)

        @visitQ nhorzelem nvertelem Nqk Nq begin
            evk = Nqk * (ev - 1) + k
            z = vgeo[ijk, grid.x3id, e]
            MH = vgeo[ijk, grid.MHid, e]
            AtmosCollected.zvals[evk] = z
            AtmosCollected.MH_z[evk] += MH
        end

        # compute the full number of points on a slab
        MPI.Allreduce!(AtmosCollected.MH_z, +, mpicomm)

        AtmosCollected.onetime_done = true
    end

    return nothing
end
