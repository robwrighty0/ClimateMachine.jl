using LinearAlgebra
export vert_fvm_interface_tendency!
"""
    weno_reconstruction!(
        state_primitive_top::AbstractArray{FT},
        state_primitive_bottom::AbstractArray{FT},
        cell_states_primitive::NTuple{5, AbstractArray{FT}},
        cell_weights::SVector{5, FT},
    )

Fifth order WENO reconstruction on nonuniform grids
Implemented based on
Wang, Rong, Hui Feng, and Raymond J. Spiteri.
"Observations on the fifth-order WENO method with non-uniform meshes."
Applied Mathematics and Computation 196.1 (2008): 433-447.

size(Δh) = 5
size(u) = (num_state_primitive, 5)
construct left/right face states of cell[3]
h1     h2     h3     h4      h5
|--i-2--|--i-1--|--i--|--i+1--|--i+2--|
without hat : i - 1/2
with hat    : i + 1/2
r = 0, 1, 2 cells to the left, => P_r^{i}
I_{i-r}, I_{i-r+1}, I_{i-r+2}
P_r(x) =  ∑_{j=0}^{2} C_{rj}(x) u_{i - r + j}  (use these 3 cell averaged values)
C_{rj}(x) = B_{rj}(x) h_{3-r+j}                (i = 3)
C''_{rj}(x) = B''_{rj}(x) h_{3-r+j}                (i = 3)
P''_r(x)    =  ∑_{j=0}^{2} C''_{rj}(x) u_{i - r + j}  (use these 3 cell averaged values)
=  ∑_{j=0}^{2} B''_{rj}(x) h_{3-r+j}  u_{i - r + j}  (use these 3 cell averaged values)

"""
function weno_reconstruction!(
    state_primitive_top::AbstractArray{FT},
    state_primitive_bottom::AbstractArray{FT},
    cell_states_primitive::NTuple{5, AbstractArray{FT}},
    cell_weights::SVector{5, FT},
) where {FT}


    num_state_primitive = length(state_primitive_top)
    h1, h2, h3, h4, h5 = cell_weights

    b̂ = similar(state_primitive_top, 3, 3)
    b̂[3, 3] = 1 / (h1 + h2 + h3) + 1 / (h2 + h3) + 1 / h3
    b̂[3, 2] = b̂[3, 3] - (h1 + h2 + h3) * (h2 + h3) / ((h1 + h2) * h2 * h3)
    b̂[3, 1] = b̂[3, 2] + (h1 + h2 + h3) * h3 / (h1 * h2 * (h2 + h3))
    b̂[2, 3] = (h2 + h3) * h3 / ((h2 + h3 + h4) * (h3 + h4) * h4)
    b̂[2, 2] = b̂[2, 3] + 1 / (h2 + h3) + 1 / h3 - 1 / h4
    b̂[2, 1] = b̂[2, 2] - ((h2 + h3) * h4) / (h2 * h3 * (h3 + h4))
    b̂[1, 3] = -(h3 * h4) / ((h3 + h4 + h5) * (h4 + h5) * h5)
    b̂[1, 2] = b̂[1, 3] + h3 * (h4 + h5) / ((h3 + h4) * h4 * h5)
    b̂[1, 1] = b̂[1, 2] + 1 / h3 - 1 / h4 - 1 / (h4 + h5)

    b = similar(state_primitive_top, 3, 3)
    b[3, 3] = 1 / (h5 + h4 + h3) + 1 / (h4 + h3) + 1 / h3
    b[3, 2] = b[3, 3] - (h5 + h4 + h3) * (h4 + h3) / ((h5 + h4) * h4 * h3)
    b[3, 1] = b[3, 2] + (h5 + h4 + h3) * h3 / (h5 * h4 * (h4 + h3))
    b[2, 3] = (h4 + h3) * h3 / ((h4 + h3 + h2) * (h3 + h2) * h2)
    b[2, 2] = b[2, 3] + 1 / (h4 + h3) + 1 / h3 - 1 / h2
    b[2, 1] = b[2, 2] - ((h4 + h3) * h2) / (h4 * h3 * (h3 + h2))
    b[1, 3] = -(h3 * h2) / ((h3 + h2 + h1) * (h2 + h1) * h1)
    b[1, 2] = b[1, 3] + h3 * (h2 + h1) / ((h3 + h2) * h2 * h1)
    b[1, 1] = b[1, 2] + 1 / h3 - 1 / h2 - 1 / (h2 + h1)


    # at i - 1/2, i + 1/2
    P = similar(state_primitive_top, num_state_primitive, 2, 3)
    P .= 0
    @unroll for r in 0:2
        @unroll for j in 0:2
            P[:, 1, 3 - r] +=
                b[r + 1, j + 1] *
                cell_weights[3 + r - j] *
                cell_states_primitive[3 + r - j]
            P[:, 2, r + 1] +=
                b̂[r + 1, j + 1] *
                cell_weights[3 - r + j] *
                cell_states_primitive[3 - r + j]
        end
    end


    # build the second derivative part in smoothness measure
    d2B = similar(state_primitive_top, 3, 3)
    d2B .= 0
    @unroll for r in 0:2
        d2B[r + 1, 3] =
            6 / (
                (
                    cell_weights[3 - r] +
                    cell_weights[4 - r] +
                    cell_weights[5 - r]
                ) *
                (cell_weights[4 - r] + cell_weights[5 - r]) *
                cell_weights[5 - r]
            )
        d2B[r + 1, 2] =
            d2B[r + 1, 3] -
            6 / (
                (cell_weights[3 - r] + cell_weights[4 - r]) *
                cell_weights[4 - r] *
                cell_weights[5 - r]
            )
        d2B[r + 1, 1] =
            d2B[r + 1, 2] +
            6 / (
                cell_weights[3 - r] *
                cell_weights[4 - r] *
                (cell_weights[4 - r] + cell_weights[5 - r])
            )
    end

    d2P = similar(state_primitive_top, num_state_primitive, 3)
    d2P .= 0
    @unroll for r in 0:2
        @unroll for j in 0:2
            d2P[:, r + 1] +=
                d2B[r + 1, j + 1] *
                cell_weights[3 - r + j] *
                cell_states_primitive[3 - r + j]
        end
    end

    IS2 = h3^4 * d2P .^ 2

    # build the first derivative part in smoothness measure

    d1B = similar(state_primitive_top, 3, 3, 3) # xi-1/2 xi, xi+1/2; r, j
    d1B[1, 3, 3] = 2 * (h1 + 2 * h2) / ((h1 + h2 + h3) * (h2 + h3) * h3)
    d1B[1, 3, 2] = d1B[1, 3, 3] - 2 * (h1 + 2 * h2 - h3) / ((h1 + h2) * h2 * h3)
    d1B[1, 3, 1] = d1B[1, 3, 2] + 2 * (h1 + h2 - h3) / (h1 * h2 * (h2 + h3))

    d1B[1, 2, 3] = 2 * (h2 - h3) / ((h2 + h3 + h4) * (h3 + h4) * h4)
    d1B[1, 2, 2] = d1B[1, 2, 3] - 2 * (h2 - h3 - h4) / ((h2 + h3) * h3 * h4)
    d1B[1, 2, 1] = d1B[1, 2, 2] + 2 * (h2 - 2 * h3 - h4) / (h2 * h3 * (h3 + h4))

    # bug in the paper
    d1B[1, 1, 3] = -2 * (2 * h3 + h4) / ((h3 + h4 + h5) * (h4 + h5) * h5)
    d1B[1, 1, 2] = d1B[1, 1, 3] + 2 * (2 * h3 + h4 + h5) / ((h3 + h4) * h4 * h5)
    d1B[1, 1, 1] =
        d1B[1, 1, 2] - 2 * (2 * h3 + 2 * h4 + h5) / (h3 * h4 * (h4 + h5))

    @unroll for r in 0:2
        @unroll for j in 0:2
            d1B[2, r + 1, j + 1] =
                d1B[1, r + 1, j + 1] + FT(0.5) * h3 * d2B[r + 1, j + 1]
            d1B[3, r + 1, j + 1] = d1B[1, r + 1, j + 1] + h3 * d2B[r + 1, j + 1]
        end
    end
    d1P = similar(state_primitive_top, num_state_primitive, 3, 3)   # xi-1/2 xi, xi+1/2; r
    d1P .= 0
    @unroll for i in 1:3
        @unroll for r in 0:2
            @unroll for j in 0:2
                d1P[:, i, r + 1] +=
                    d1B[i, r + 1, j + 1] *
                    cell_weights[3 - r + j] *
                    cell_states_primitive[3 - r + j]
            end
        end
    end

    fact = d1P[:, 1, :] .^ 2 + 4 * d1P[:, 2, :] .^ 2 + d1P[:, 3, :] .^ 2
    IS1 = h3^2 * fact / 6


    IS = IS1 + IS2

    # high order test
    # IS .= 1

    d = similar(state_primitive_top, 3)
    d[3] =
        (h3 + h4) * (h3 + h4 + h5) /
        ((h1 + h2 + h3 + h4) * (h1 + h2 + h3 + h4 + h5))
    d[2] =
        (h1 + h2) * (h3 + h4 + h5) * (h1 + 2 * h2 + 2 * h3 + 2 * h4 + h5) /
        ((h1 + h2 + h3 + h4) * (h2 + h3 + h4 + h5) * (h1 + h2 + h3 + h4 + h5))
    d[1] = h2 * (h1 + h2) / ((h2 + h3 + h4 + h5) * (h1 + h2 + h3 + h4 + h5))

    d̂ = similar(state_primitive_top, 3)
    d̂[3] = h4 * (h4 + h5) / ((h1 + h2 + h3 + h4) * (h1 + h2 + h3 + h4 + h5))
    d̂[2] =
        (h1 + h2 + h3) * (h4 + h5) * (h1 + 2 * h2 + 2 * h3 + 2 * h4 + h5) /
        ((h1 + h2 + h3 + h4) * (h2 + h3 + h4 + h5) * (h1 + h2 + h3 + h4 + h5))
    d̂[1] =
        (h2 + h3) * (h1 + h2 + h3) /
        ((h2 + h3 + h4 + h5) * (h1 + h2 + h3 + h4 + h5))

    ϵ = FT(1.0e-6)
    α = d' ./ (ϵ .+ IS) .^ 2
    α̂ = d̂' ./ (ϵ .+ IS) .^ 2

    w = α ./ sum(α, dims = 2)
    ŵ = α̂ ./ sum(α̂, dims = 2)


    # at  i - 1/2,  i + 1/2
    @unroll for i in 1:num_state_primitive
        state_primitive_bottom[i] += w[i, :]' * P[i, 1, :]
        state_primitive_top[i] += ŵ[i, :]' * P[i, 2, :]
    end
end

"""
    weno_reconstruction!(
        state_primitive_top::AbstractArray{FT},
        state_primitive_bottom::AbstractArray{FT},
        cell_states_primitive::NTuple{3, AbstractArray{FT}},
        cell_weights::SVector{3, FT},
    )

Third order WENO reconstruction on nonuniform grids
Implemented based on
Cravero, Isabella, and Matteo Semplice.
"On the accuracy of WENO and CWENO reconstructions of third order on nonuniform meshes."
Journal of Scientific Computing 67.3 (2016): 1219-1246.

"""
function weno_reconstruction!(
    state_primitive_top::AbstractArray{FT},
    state_primitive_bottom::AbstractArray{FT},
    cell_states_primitive::NTuple{3, AbstractArray{FT}},
    cell_weights::SVector{3, FT},
) where {FT}

    num_state_primitive = length(state_primitive_top)
    h1, h2, h3 = cell_weights


    β, γ = h1 / h2, h3 / h2
    C⁺_l, C⁺_r = γ / (1 + β + γ), (1 + β) / (1 + β + γ)
    C⁻_l, C⁻_r = (1 + β) / (1 + β + γ), γ / (1 + β + γ)

    dP = similar(state_primitive_top, num_state_primitive, 2)
    dP[:, 1] =
        2 * (cell_states_primitive[2] - cell_states_primitive[1]) / (h1 + h2)
    dP[:, 2] =
        2 * (cell_states_primitive[3] - cell_states_primitive[2]) / (h2 + h3)


    # at i - 1/2, i + 1/2, r = 0, 1
    P = similar(state_primitive_top, num_state_primitive, 2, 2)
    @unroll for r in 0:1
        P[:, 1, r + 1] = cell_states_primitive[2] - dP[:, r + 1] * h2 / 2
        P[:, 2, r + 1] = cell_states_primitive[2] + dP[:, r + 1] * h2 / 2
    end

    # IS = int h2 *P'^2 dx, P  is a linear function
    IS = h2^2 * dP .^ 2

    # high order test
    # IS .= 1.0

    d = [(1 + γ) / (1 + β + γ); β / (1 + β + γ)]
    d̂ = [γ / (1 + β + γ); (1 + β) / (1 + β + γ)]

    ϵ = FT(1.0e-6)
    α = d' ./ (ϵ .+ IS) .^ 2
    α̂ = d̂' ./ (ϵ .+ IS) .^ 2

    w = α ./ sum(α, dims = 2)
    ŵ = α̂ ./ sum(α̂, dims = 2)

    # at  i - 1/2,  i + 1/2
    @unroll for i in 1:num_state_primitive
        state_primitive_bottom[i] += w[i, :]' * P[i, 1, :]
        state_primitive_top[i] += ŵ[i, :]' * P[i, 2, :]
    end

end




"""
    function limiter(
        Δ⁺::AbstractArray{FT, 1},
        Δ⁻::AbstractArray{FT, 1},
        ::Val{num_state},
    )

Classical Second order FV reconstruction on nonuniform grids
Van Leer Limiter is used
Implemented based on https://en.wikipedia.org/wiki/Flux_limiter
"""
function limiter(
    Δ⁺::AbstractArray{FT, 1},
    Δ⁻::AbstractArray{FT, 1},
    ::Val{num_state},
) where {FT, num_state}
    Δ = similar(Δ⁺, num_state)
    Δ .= 0
    @unroll for s in 1:num_state
        if Δ⁺[s] * Δ⁻[s] > 0
            Δ[s] = FT(2) * Δ⁺[s] * Δ⁻[s] / (Δ⁺[s] + Δ⁻[s])
        end
    end
    return Δ
end
function fv_reconstruction!(
    state_primitive_top::AbstractArray{FT},
    state_primitive_bottom::AbstractArray{FT},
    cell_states_primitive::NTuple{3, AbstractArray{FT}},
    cell_weights::SVector{3, FT},
) where {FT}
    num_state_primitive = length(state_primitive_top)
    Δz⁻, Δz, Δz⁺ = cell_weights

    Δu⁺ = (cell_states_primitive[3] .- cell_states_primitive[2])
    Δu⁻ = (cell_states_primitive[2] .- cell_states_primitive[1])

    ∂state =
        2 *
        limiter(Δu⁺ / (Δz⁺ + Δz), Δu⁻ / (Δz⁻ + Δz), Val(num_state_primitive))

    state_primitive_top .= cell_states_primitive[2] .+ ∂state * Δz / 2
    state_primitive_bottom .= cell_states_primitive[2] .- ∂state * Δz / 2
end


"""
    const_reconstruction!(
        state_primitive_top::AbstractArray{FT},
        state_primitive_bottom::AbstractArray{FT},
        cell_states_primitive::NTuple{1, AbstractArray{FT}},
        cell_weights::SVector{1, FT},
    ) where {FT}

First order (constant) FV reconstruction on nonuniform grids
Mainly used for Boundary conditions
"""
function const_reconstruction!(
    state_primitive_top::AbstractArray{FT},
    state_primitive_bottom::AbstractArray{FT},
    cell_states_primitive::NTuple{1, AbstractArray{FT}},
    cell_weights::SVector{1, FT},
) where {FT}
    state_primitive_top .= cell_states_primitive[1]
    state_primitive_bottom .= cell_states_primitive[1]
end




"""
    prognostic_to_primitive!(bl, state_prim, state_prog)

Compute `state_prim` from `state_prog`. By default,
we use the identity.
"""
function prognostic_to_primitive!(bl, state_prim, state_prog, state_aux)
    state_prog_arr = parent(state_prog)
    state_prim_arr = parent(state_prim)
    state_prim_arr .= state_prog_arr
end

"""
    primitive_to_prognostic!(bl, state_prog, state_prim)

Compute `state_prog` from `state_prim`. By default,
we use the identity.
"""
function primitive_to_prognostic!(bl, state_prog, state_prim, state_aux)
    state_prog_arr = parent(state_prog)
    state_prim_arr = parent(state_prim)
    state_prog_arr .= state_prim_arr
end




@doc """
    function vert_fvm_interface_tendency!(
        balance_law::BalanceLaw,
        ::Val{info},
        ::Val{nvertelem},
        ::Val{periodicstack},
        ::VerticalDirection,
        numerical_flux_first_order,
        tendency,
        state_prognostic,
        state_auxiliary,
        vgeo,
        sgeo,
        t,
        vmap⁻,
        vmap⁺,
        elemtobndy,
        elems,
        α,
    )

Compute kernel for evaluating the interface tendencies using vertical FVM
reconstructions with a DG method of the form:

∫ₑ ψ⋅ ∂q/∂t dx - ∫ₑ ∇ψ⋅(Fⁱⁿᵛ + Fᵛⁱˢᶜ) dx + ∮ₑ n̂ ψ⋅(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆) dS,

or equivalently in matrix form:

dQ/dt = M⁻¹(MS + DᵀM(Fⁱⁿᵛ + Fᵛⁱˢᶜ) + ∑ᶠ LᵀMf(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆)).

This kernel computes the surface terms: M⁻¹ ∑ᶠ LᵀMf(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆)), where M
is the mass matrix, Mf is the face mass matrix, L is an interpolator from
volume to face, and Fⁱⁿᵛ⋆, Fᵛⁱˢᶜ⋆ are the numerical fluxes for the inviscid
and viscous fluxes, respectively.

A finite volume reconstruction is used to construction `Fⁱⁿᵛ⋆`

!!! note
    Currently `Fᵛⁱˢᶜ⋆` is not supported
""" vert_fvm_interface_tendency!
@kernel function vert_fvm_interface_tendency!(
    balance_law::BalanceLaw,
    ::Val{info},
    ::Val{nvertelem},
    ::Val{periodicstack},
    ::VerticalDirection,
    numerical_flux_first_order,
    numerical_flux_second_order,
    tendency,
    state_prognostic,
    state_gradient_flux,
    state_auxiliary,
    vgeo,
    sgeo,
    t,
    elemtobndy,
    elems,
    α,
) where {info, nvertelem, periodicstack}
    @uniform begin
        dim = info.dim
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        num_state_hyperdiffusion = number_states(balance_law, Hyperdiffusive())
        @assert num_state_hyperdiffusion == 0

        vsp = Vars{vars_state(balance_law, Prognostic(), FT)}
        vsa = Vars{vars_state(balance_law, Auxiliary(), FT)}

        nface = info.nface
        Np = info.Np
        Nqk = info.Nqk # can only be 1 for the FVM method!
        @assert Nqk == 1

        # We only have the vertical faces
        faces = (nface - 1):nface

        # +/- indicate top/bottom elements
        # top and bottom indicate face states

        local_first_order_flux_bottom =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_first_order_flux_top =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_second_order_flux_bottom =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_second_order_flux_top =
            MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic⁺ = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic⁻ = MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_primitive = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_primitive⁺ = MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_primitive⁻ = MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_auxiliary⁺ = MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_auxiliary⁻ = MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_state_gradient_flux⁻ =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_gradient_flux =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_gradient_flux⁺ =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)

        local_state_hyperdiffusion⁻ =
            MArray{Tuple{num_state_hyperdiffusion}, FT}(undef)
        local_state_hyperdiffusion =
            MArray{Tuple{num_state_hyperdiffusion}, FT}(undef)
        local_state_hyperdiffusion⁺ =
            MArray{Tuple{num_state_hyperdiffusion}, FT}(undef)

        local_state_primitive_bottom =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_primitive_top =
            MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_prognostic_bottom =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic_top =
            MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_auxiliary_bottom =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_auxiliary_top =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)

        local_state_prognostic⁻_top =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_prognostic⁺_bottom =
            MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_auxiliary⁻_top =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_auxiliary⁺_bottom =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)


        local_state_prognostic_boundary =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_auxiliary_boundary =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)
        local_state_gradient_flux_boundary =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)

        # XXX: will revisit this later for FVM
        fill!(local_state_prognostic_boundary, NaN)
        fill!(local_state_auxiliary_boundary, NaN)
        fill!(local_state_gradient_flux_boundary, NaN)

        # The remainder model needs to know which direction of face the model is
        # being evaluated for. In this case we only have `VerticalDirection()`
        # faces
        face_direction = (VerticalDirection())
    end

    # Get the horizontal group IDs
    grp_H = @index(Group, Linear)

    # Determine the index for the element at the bottom of the stack
    eHI = (grp_H - 1) * nvertelem + 1

    # Compute bottom stack element index minus one (so we can add vert element
    # number directly)
    eH = elems[eHI] - 1

    # Which degree of freedom do we handle in the element
    n = @index(Local, Linear)

    # Loads the data for a given element
    function load_data!(
        local_state_prognostic,
        local_state_auxiliary,
        local_state_gradient_flux,
        e,
    )
        @unroll for s in 1:num_state_prognostic
            local_state_prognostic[s] = state_prognostic[n, s, e]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end

        @unroll for s in 1:num_state_gradient_flux
            local_state_gradient_flux[s] = state_gradient_flux[n, s, e]
        end
    end

    # We need to compute the first element we handles bottom flux (only for nonperiodic boundary condition)
    # elements will just copied from the prior element)
    @inbounds begin
        eV = 1
        # the first element
        e = eH + eV

        # bottom face
        f_bottom = faces[1]
        # surface mass
        sM_bottom = sgeo[_sM, n, f_bottom, e]
        cw = vgeo[n, _M, e]
        vMI = vgeo[n, _MI, e]

        # outward normal for this the bottom face
        normal_vector_bottom = SVector(
            sgeo[_n1, n, f_bottom, e],
            sgeo[_n2, n, f_bottom, e],
            sgeo[_n3, n, f_bottom, e],
        )

        if periodicstack
            e⁻ = eH + nvertelem
            e⁺ = e + 1
            cw⁻ = vgeo[n, _M, e⁻]
            cw⁺ = vgeo[n, _M, e⁺]

            load_data!(
                local_state_prognostic⁻,
                local_state_auxiliary⁻,
                local_state_gradient_flux⁻,
                e⁻,
            )
            load_data!(
                local_state_prognostic,
                local_state_auxiliary,
                local_state_gradient_flux,
                e,
            )
            load_data!(
                local_state_prognostic⁺,
                local_state_auxiliary⁺,
                local_state_gradient_flux⁺,
                e⁺,
            )

            prognostic_to_primitive!(
                balance_law,
                Vars{vsp}(local_state_primitive⁻),
                Vars{vsp}(local_state_prognostic⁻),
                Vars{vsa}(local_state_auxiliary⁻),
            )
            prognostic_to_primitive!(
                balance_law,
                Vars{vsp}(local_state_primitive),
                Vars{vsp}(local_state_prognostic),
                Vars{vsa}(local_state_auxiliary),
            )
            prognostic_to_primitive!(
                balance_law,
                Vars{vsp}(local_state_primitive⁺),
                Vars{vsp}(local_state_prognostic⁺),
                Vars{vsa}(local_state_auxiliary⁺),
            )

            cell_states_primitive = (
                local_state_primitive⁻,
                local_state_primitive,
                local_state_primitive⁺,
            )

            cell_weights = SVector(cw⁻, cw, cw⁺)

            fv_reconstruction!(
                local_state_primitive_top,
                local_state_primitive_bottom,
                cell_states_primitive,
                cell_weights,
            )

            # TODO
            local_state_auxiliary_bottom .= local_state_auxiliary
            local_state_auxiliary_top .= local_state_auxiliary

            primitive_to_prognostic!(
                balance_law,
                Vars{vsp}(local_state_prognostic_bottom),
                Vars{vsp}(local_state_primitive_bottom),
                Vars{vsa}(local_state_auxiliary_bottom),
            )

            primitive_to_prognostic!(
                balance_law,
                Vars{vsp}(local_state_prognostic_top),
                Vars{vsp}(local_state_primitive_top),
                Vars{vsa}(local_state_auxiliary_top),
            )

            # this is used for the stack top element
            local_state_prognostic⁺_bottom .= local_state_prognostic_bottom
            local_state_auxiliary⁺_bottom .= local_state_auxiliary_bottom
            local_state_prognostic⁻_top .= local_state_prognostic_top
            local_state_auxiliary⁻_top .= local_state_auxiliary_top

        else

            e⁺ = e + 1
            load_data!(
                local_state_prognostic,
                local_state_auxiliary,
                local_state_gradient_flux,
                e,
            )
            load_data!(
                local_state_prognostic⁺,
                local_state_auxiliary⁺,
                local_state_gradient_flux⁺,
                e⁺,
            )

            prognostic_to_primitive!(
                balance_law,
                Vars{vsp}(local_state_primitive),
                Vars{vsp}(local_state_prognostic),
                Vars{vsa}(local_state_auxiliary),
            )
            cell_states_primitive = (local_state_primitive,)
            cell_weights = SVector(cw)

            const_reconstruction!(
                local_state_primitive_top,
                local_state_primitive_bottom,
                cell_states_primitive,
                cell_weights,
            )

            # TODO
            local_state_auxiliary_bottom .= local_state_auxiliary
            local_state_auxiliary_top .= local_state_auxiliary

            primitive_to_prognostic!(
                balance_law,
                Vars{vsp}(local_state_prognostic_bottom),
                Vars{vsp}(local_state_primitive_bottom),
                Vars{vsa}(local_state_auxiliary_bottom),
            )

            primitive_to_prognostic!(
                balance_law,
                Vars{vsp}(local_state_prognostic_top),
                Vars{vsp}(local_state_primitive_top),
                Vars{vsa}(local_state_auxiliary_top),
            )

            fill!(
                local_first_order_flux_bottom,
                -zero(eltype(local_first_order_flux_bottom)),
            )
            fill!(
                local_second_order_flux_bottom,
                -zero(eltype(local_second_order_flux_bottom)),
            )
            # TODO
            bctag = elemtobndy[f_bottom, e]
            numerical_boundary_flux_first_order!(
                numerical_flux_first_order,
                bctag,
                balance_law,
                local_first_order_flux_bottom,
                normal_vector_bottom,
                local_state_prognostic_bottom,
                local_state_auxiliary_bottom,
                # TODO 
                local_state_prognostic_bottom,
                local_state_auxiliary_bottom,
                t,
                face_direction,
                local_state_prognostic_boundary,
                local_state_auxiliary_boundary,
            )

            numerical_boundary_flux_second_order!(
                numerical_flux_second_order,
                bctag,
                balance_law,
                local_second_order_flux_bottom,
                normal_vector_bottom,
                #
                local_state_prognostic_bottom,
                local_state_gradient_flux,
                local_state_hyperdiffusion,
                local_state_auxiliary_bottom,
                #
                local_state_prognostic_bottom,
                local_state_gradient_flux,
                local_state_hyperdiffusion,
                local_state_auxiliary_bottom,
                t,
                local_state_prognostic_boundary,
                local_state_gradient_flux_boundary,
                local_state_auxiliary_boundary,
            )

            @unroll for s in 1:num_state_prognostic
                tendency[n, s, e] -=
                    α *
                    vMI *
                    sM_bottom *
                    (
                        local_first_order_flux_bottom[s] +
                        local_second_order_flux_bottom[s]
                    )
            end

            #TODO
            local_state_prognostic⁻_top .= local_state_prognostic_top
            local_state_auxiliary⁻_top .= local_state_auxiliary_top

        end
    end

    # Loop up the vertical stack to update the minus side element (we have the
    # bottom flux from the previous element, so only need to calculate the top
    # flux)
    @inbounds for eV in 2:(nvertelem - 1)
        e = eH + eV
        e⁻ = e - 1
        e⁺ = e + 1

        # volume mass inverse
        vMI = vgeo[n, _MI, e]
        cw⁻, cw, cw⁺ = vgeo[n, _M, e⁻], vgeo[n, _M, e], vgeo[n, _M, e⁺]

        # Compute the top face numerical flux
        # The minus side is the bottom element
        # The plus side is the top element
        f_bottom = faces[1]
        # surface mass
        sM_bottom = sgeo[_sM, n, f_bottom, e]

        # outward normal with respect to the element
        normal_vector_bottom = SVector(
            sgeo[_n1, n, f_bottom, e],
            sgeo[_n2, n, f_bottom, e],
            sgeo[_n3, n, f_bottom, e],
        )


        # Load plus side data (minus data is already set)
        local_state_prognostic⁻ .= local_state_prognostic
        local_state_auxiliary⁻ .= local_state_auxiliary
        local_state_gradient_flux⁻ .= local_state_gradient_flux
        local_state_prognostic .= local_state_prognostic⁺
        local_state_auxiliary .= local_state_auxiliary⁺
        local_state_gradient_flux .= local_state_gradient_flux⁺
        load_data!(
            local_state_prognostic⁺,
            local_state_auxiliary⁺,
            local_state_gradient_flux⁺,
            e⁺,
        )


        prognostic_to_primitive!(
            balance_law,
            Vars{vsp}(local_state_primitive⁻),
            Vars{vsp}(local_state_prognostic⁻),
            Vars{vsa}(local_state_auxiliary⁻),
        )

        prognostic_to_primitive!(
            balance_law,
            Vars{vsp}(local_state_primitive),
            Vars{vsp}(local_state_prognostic),
            Vars{vsa}(local_state_auxiliary),
        )

        prognostic_to_primitive!(
            balance_law,
            Vars{vsp}(local_state_primitive⁺),
            Vars{vsp}(local_state_prognostic⁺),
            Vars{vsa}(local_state_auxiliary⁺),
        )


        cell_states_primitive = (
            local_state_primitive⁻,
            local_state_primitive,
            local_state_primitive⁺,
        )

        cell_weights = SVector(cw⁻, cw, cw⁺)

        fv_reconstruction!(
            local_state_primitive_top,
            local_state_primitive_bottom,
            cell_states_primitive,
            cell_weights,
        )

        # TODO
        local_state_auxiliary_bottom .= local_state_auxiliary
        local_state_auxiliary_top .= local_state_auxiliary

        primitive_to_prognostic!(
            balance_law,
            Vars{vsp}(local_state_prognostic_bottom),
            Vars{vsp}(local_state_primitive_bottom),
            Vars{vsa}(local_state_auxiliary_bottom),
        )

        primitive_to_prognostic!(
            balance_law,
            Vars{vsp}(local_state_prognostic_top),
            Vars{vsp}(local_state_primitive_top),
            Vars{vsa}(local_state_auxiliary_top),
        )

        ###  TODO HYDROSTATIC BALANCE RECONSTRUCTION

        # compute the flux
        fill!(
            local_first_order_flux_bottom,
            -zero(eltype(local_first_order_flux_bottom)),
        )
        fill!(
            local_second_order_flux_bottom,
            -zero(eltype(local_second_order_flux_bottom)),
        )

        numerical_flux_first_order!(
            numerical_flux_first_order,
            balance_law,
            local_first_order_flux_bottom,
            normal_vector_bottom,
            local_state_prognostic_bottom,
            local_state_auxiliary_bottom,
            local_state_prognostic⁻_top,
            local_state_auxiliary⁻_top,
            t,
            face_direction,
        )

        numerical_flux_second_order!(
            numerical_flux_second_order,
            balance_law,
            local_second_order_flux_bottom,
            normal_vector_bottom,
            #
            # local_state_prognostic_bottom,
            local_state_prognostic,
            local_state_gradient_flux,
            local_state_hyperdiffusion,
            # local_state_auxiliary_bottom,
            local_state_auxiliary,
            #
            # local_state_prognostic⁻_top,
            local_state_prognostic⁻,
            local_state_gradient_flux⁻,
            local_state_hyperdiffusion⁻,
            # local_state_auxiliary⁻_top,
            local_state_auxiliary⁻,
            t,
        )

        # Update RHS (in outer face loop): M⁻¹ Mfᵀ(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆))
        # TODO: This isn't correct:
        # FIXME: Should we pretch these?
        @unroll for s in 1:num_state_prognostic
            tendency[n, s, e⁻] +=
                α *
                vMI *
                sM_bottom *
                (
                    local_first_order_flux_bottom[s] +
                    local_second_order_flux_bottom[s]
                )
        end
        @unroll for s in 1:num_state_prognostic
            tendency[n, s, e] -=
                α *
                vMI *
                sM_bottom *
                (
                    local_first_order_flux_bottom[s] +
                    local_second_order_flux_bottom[s]
                )
        end

        local_state_prognostic⁻_top .= local_state_prognostic_top
        local_state_auxiliary⁻_top .= local_state_auxiliary_top
    end

    # The top element
    @inbounds begin
        eV = nvertelem
        # the first element
        e = eH + eV
        # bottom face
        f_bottom, f_top = faces[1], faces[2]

        # surface mass
        sM_bottom, sM_top = sgeo[_sM, n, f_bottom, e], sgeo[_sM, n, f_top, e]
        cw = vgeo[n, _M, e]
        vMI = vgeo[n, _MI, e]
        # outward normal for this face
        # outward normal for this the bottom face
        normal_vector_bottom = SVector(
            sgeo[_n1, n, f_bottom, e],
            sgeo[_n2, n, f_bottom, e],
            sgeo[_n3, n, f_bottom, e],
        )

        # outward normal for this the bottom face
        normal_vector_top = SVector(
            sgeo[_n1, n, f_top, e],
            sgeo[_n2, n, f_top, e],
            sgeo[_n3, n, f_top, e],
        )

        if periodicstack

            e⁻ = e - 1
            e⁺ = eH + 1
            cw⁻ = vgeo[n, _M, e⁻]
            cw⁺ = vgeo[n, _M, e⁺]

            local_state_prognostic⁻ .= local_state_prognostic
            local_state_auxiliary⁻ .= local_state_auxiliary
            local_state_gradient_flux⁻ .= local_state_gradient_flux
            local_state_prognostic .= local_state_prognostic⁺
            local_state_auxiliary .= local_state_auxiliary⁺
            local_state_gradient_flux .= local_state_gradient_flux⁺
            load_data!(
                local_state_prognostic⁺,
                local_state_auxiliary⁺,
                local_state_gradient_flux⁺,
                e⁺,
            )

            prognostic_to_primitive!(
                balance_law,
                Vars{vsp}(local_state_primitive⁻),
                Vars{vsp}(local_state_prognostic⁻),
                Vars{vsa}(local_state_auxiliary⁻),
            )
            prognostic_to_primitive!(
                balance_law,
                Vars{vsp}(local_state_primitive),
                Vars{vsp}(local_state_prognostic),
                Vars{vsa}(local_state_auxiliary),
            )
            prognostic_to_primitive!(
                balance_law,
                Vars{vsp}(local_state_primitive⁺),
                Vars{vsp}(local_state_prognostic⁺),
                Vars{vsa}(local_state_auxiliary⁺),
            )

            cell_states_primitive = (
                local_state_primitive⁻,
                local_state_primitive,
                local_state_primitive⁺,
            )

            cell_weights = SVector(cw⁻, cw, cw⁺)

            fv_reconstruction!(
                local_state_primitive_top,
                local_state_primitive_bottom,
                cell_states_primitive,
                cell_weights,
            )

            # TODO
            local_state_auxiliary_bottom .= local_state_auxiliary
            local_state_auxiliary_top .= local_state_auxiliary

            primitive_to_prognostic!(
                balance_law,
                Vars{vsp}(local_state_prognostic_bottom),
                Vars{vsp}(local_state_primitive_bottom),
                Vars{vsa}(local_state_auxiliary_bottom),
            )

            primitive_to_prognostic!(
                balance_law,
                Vars{vsp}(local_state_prognostic_top),
                Vars{vsp}(local_state_primitive_top),
                Vars{vsa}(local_state_auxiliary_top),
            )

            # compute the bottom flux
            fill!(
                local_first_order_flux_bottom,
                -zero(eltype(local_first_order_flux_bottom)),
            )
            fill!(
                local_second_order_flux_bottom,
                -zero(eltype(local_second_order_flux_bottom)),
            )

            numerical_flux_first_order!(
                numerical_flux_first_order,
                balance_law,
                local_first_order_flux_bottom,
                normal_vector_bottom,
                local_state_prognostic_bottom,
                local_state_auxiliary_bottom,
                local_state_prognostic⁻_top,
                local_state_auxiliary⁻_top,
                t,
                face_direction,
            )

            numerical_flux_second_order!(
                numerical_flux_second_order,
                balance_law,
                local_second_order_flux_bottom,
                normal_vector_bottom,
                #
                # local_state_prognostic_bottom,
                local_state_prognostic,
                local_state_gradient_flux,
                local_state_hyperdiffusion,
                # local_state_auxiliary_bottom,
                local_state_auxiliary,
                #
                # local_state_prognostic⁻_top,
                local_state_prognostic⁻,
                local_state_gradient_flux⁻,
                local_state_hyperdiffusion⁻,
                # local_state_auxiliary⁻_top,
                local_state_auxiliary⁻,
                t,
            )

            # Update RHS (in outer face loop): M⁻¹ Mfᵀ(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆))
            # TODO: This isn't correct:
            # FIXME: Should we pretch these?
            @unroll for s in 1:num_state_prognostic
                tendency[n, s, e⁻] +=
                    α *
                    vMI *
                    sM_bottom *
                    (
                        local_first_order_flux_bottom[s] +
                        local_second_order_flux_bottom[s]
                    )
            end
            @unroll for s in 1:num_state_prognostic
                tendency[n, s, e] -=
                    α *
                    vMI *
                    sM_bottom *
                    (
                        local_first_order_flux_bottom[s] +
                        local_second_order_flux_bottom[s]
                    )
            end

            # compute the top flux
            fill!(
                local_first_order_flux_top,
                -zero(eltype(local_first_order_flux_top)),
            )
            fill!(
                local_second_order_flux_top,
                -zero(eltype(local_second_order_flux_top)),
            )

            numerical_flux_first_order!(
                numerical_flux_first_order,
                balance_law,
                local_first_order_flux_top,
                normal_vector_top,
                local_state_prognostic_top,
                local_state_auxiliary_top,
                local_state_prognostic⁺_bottom,
                local_state_auxiliary⁺_bottom,
                t,
                face_direction,
            )

            numerical_flux_second_order!(
                numerical_flux_second_order,
                balance_law,
                local_second_order_flux_top,
                normal_vector_top,
                #
                # local_state_prognostic_top,
                local_state_prognostic,
                local_state_gradient_flux,
                local_state_hyperdiffusion,
                # local_state_auxiliary_top,
                local_state_auxiliary,
                #
                # local_state_prognostic⁺_bottom,
                local_state_prognostic⁺,
                local_state_gradient_flux⁺,
                local_state_hyperdiffusion⁺,
                # local_state_auxiliary⁺_bottom,
                local_state_auxiliary⁺,
                t,
            )

            # Update RHS (in outer face loop): M⁻¹ Mfᵀ(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆))

            @unroll for s in 1:num_state_prognostic
                tendency[n, s, e] -=
                    α *
                    vMI *
                    sM_top *
                    (
                        local_first_order_flux_top[s] +
                        local_second_order_flux_top[s]
                    )
            end
            @unroll for s in 1:num_state_prognostic
                tendency[n, s, e⁺] +=
                    α *
                    vMI *
                    sM_top *
                    (
                        local_first_order_flux_top[s] +
                        local_second_order_flux_top[s]
                    )
            end


        else
            e⁻ = e - 1

            local_state_gradient_flux⁻ .= local_state_gradient_flux
            local_state_prognostic .= local_state_prognostic⁺
            local_state_auxiliary .= local_state_auxiliary⁺
            local_state_gradient_flux .= local_state_gradient_flux⁺

            prognostic_to_primitive!(
                balance_law,
                Vars{vsp}(local_state_primitive),
                Vars{vsp}(local_state_prognostic),
                Vars{vsa}(local_state_auxiliary),
            )
            cell_states_primitive = (local_state_primitive,)
            cell_weights = SVector(cw)

            const_reconstruction!(
                local_state_primitive_top,
                local_state_primitive_bottom,
                cell_states_primitive,
                cell_weights,
            )

            primitive_to_prognostic!(
                balance_law,
                Vars{vsp}(local_state_prognostic_bottom),
                Vars{vsp}(local_state_primitive_bottom),
                Vars{vsa}(local_state_auxiliary_bottom),
            )

            primitive_to_prognostic!(
                balance_law,
                Vars{vsp}(local_state_prognostic_top),
                Vars{vsp}(local_state_primitive_top),
                Vars{vsa}(local_state_auxiliary_top),
            )

            # bottom flux
            fill!(
                local_first_order_flux_bottom,
                -zero(eltype(local_first_order_flux_bottom)),
            )
            fill!(
                local_second_order_flux_bottom,
                -zero(eltype(local_second_order_flux_bottom)),
            )

            numerical_flux_first_order!(
                numerical_flux_first_order,
                balance_law,
                local_first_order_flux_bottom,
                normal_vector_bottom,
                local_state_prognostic_bottom,
                local_state_auxiliary_bottom,
                local_state_prognostic⁻_top,
                local_state_auxiliary⁻_top,
                t,
                face_direction,
            )

            numerical_flux_second_order!(
                numerical_flux_second_order,
                balance_law,
                local_second_order_flux_top,
                normal_vector_bottom,
                #
                local_state_prognostic_bottom,
                local_state_gradient_flux,
                local_state_hyperdiffusion,
                local_state_auxiliary_bottom,
                #
                local_state_prognostic⁻_top,
                local_state_gradient_flux⁻,
                local_state_hyperdiffusion⁻,
                local_state_auxiliary⁻_top,
                t,
            )




            # Update RHS (in outer face loop): M⁻¹ Mfᵀ(Fⁱⁿᵛ⋆ + Fᵛⁱˢᶜ⋆))
            # TODO: This isn't correct:
            # FIXME: Should we pretch these?
            @unroll for s in 1:num_state_prognostic
                tendency[n, s, e] -=
                    α *
                    vMI *
                    sM_bottom *
                    (
                        local_first_order_flux_bottom[s] +
                        local_second_order_flux_bottom[s]
                    )
            end
            @unroll for s in 1:num_state_prognostic
                tendency[n, s, e⁻] +=
                    α *
                    vMI *
                    sM_bottom *
                    (
                        local_first_order_flux_bottom[s] +
                        local_second_order_flux_bottom[s]
                    )
            end


            bctag = elemtobndy[f_top, e]

            fill!(
                local_first_order_flux_top,
                -zero(eltype(local_first_order_flux_top)),
            )
            fill!(
                local_second_order_flux_top,
                -zero(eltype(local_second_order_flux_top)),
            )

            numerical_boundary_flux_first_order!(
                numerical_flux_first_order,
                bctag,
                balance_law,
                local_first_order_flux_top,
                normal_vector_top,
                local_state_prognostic_top,
                local_state_auxiliary_top,
                #TODO
                local_state_prognostic_top,
                local_state_auxiliary_top,
                t,
                face_direction,
                local_state_prognostic_boundary,
                local_state_auxiliary_boundary,
            )

            numerical_boundary_flux_second_order!(
                numerical_flux_second_order,
                bctag,
                balance_law,
                local_second_order_flux_bottom,
                normal_vector_top,
                #
                local_state_prognostic_top,
                local_state_gradient_flux,
                local_state_hyperdiffusion,
                local_state_auxiliary_top,
                #
                local_state_prognostic_top,
                local_state_gradient_flux,
                local_state_hyperdiffusion,
                local_state_auxiliary_top,
                #
                t,
                local_state_prognostic_boundary,
                local_state_gradient_flux_boundary,
                local_state_auxiliary_boundary,
            )

            @unroll for s in 1:num_state_prognostic
                tendency[n, s, e] -=
                    α *
                    vMI *
                    sM_top *
                    (
                        local_first_order_flux_top[s] +
                        local_second_order_flux_top[s]
                    )
            end


        end
    end
end
