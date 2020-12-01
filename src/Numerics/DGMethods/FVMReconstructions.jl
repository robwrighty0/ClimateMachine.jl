module FVMReconstructions

"""
    AbstractFVMReconstruction

Supertype for FVM reconstructions.

Concrete types must provide implementions of

    - `width(recon)` which returns the width of the
       reconstruction. Total number of points used in reconstruction of top and
       bottom states is `2width(recon) + 1`
"""
abstract type AbstractFVMReconstruction end

"""
    width(recon::AbstractFVMReconstruction)

Returns the width of the stencil need for the FVM reconstruction `recon`. Total
number of values used in the reconstruction are `2width(recon) + 1`
"""
width(recon::AbstractFVMReconstruction) = throw(MethodError(width, (recon,)))

"""
    reconstruction!(
        recon::AbstractFVMReconstruction,
        state_top,
        state_bottom,
        cell_state,
        cell_weights,
    )

Perform the finite volume reconstruction for the top and bottom states using the
tuple of `cell_state` values using the `cell_weights`.
"""
reconstruction!(
    recon::AbstractFVMReconstruction,
    state_top,
    state_bottom,
    cell_state::NTuple,
    cell_weights,
) = throw(MethodError(
    reconstruction!,
    (recon, state_top, state_bottom, cell_state, cell_weights),
))

"""
    FVMCellCentered <: AbstractFVMReconstruction

Concrete reconstruction type for cell centered finite volume methods (e.g.,
constants)
"""
struct FVMCellCentered <: AbstractFVMReconstruction end
width(::FVMCellCentered) = 0
function reconstruction!(
    ::FVMCellCentered,
    state_top,
    state_bottom,
    cell_state::NTuple{1},
    _,
) where {FT}
    state_top .= cell_state[1]
    state_bottom .= cell_state[1]
end

end
