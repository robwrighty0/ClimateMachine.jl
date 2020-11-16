module NumericalFluxes

export NumericalFlux,
    RusanovNumericalFlux,
    RoeNumericalFlux,
    HLLCNumericalFlux,
    CentralNumericalFlux


using StaticArrays, LinearAlgebra
using ClimateMachine.VariableTemplates
using KernelAbstractions.Extras: @unroll
using ...BalanceLaws
import ...BalanceLaws:
    vars_state,
    boundary_state!,
    wavespeed,
    flux!,
    compute_gradient_flux!,
    compute_gradient_argument!,
    transform_post_gradient_laplacian!

"""
    CentralNumericalFlux{O} <: NumericalFlux{O}

The central numerical flux for
 - gradient terms (`O<:Gradient`)
 - nondiffusive terms (`O<:FirstOrder`)
 - diffusive terms (`O<:SecondOrder`)

Requires a `flux!` method for the balance law.
"""
struct CentralNumericalFlux{O} <: NumericalFlux{O} end

abstract type DivNumericalPenalty <: AbstractFluxType end

function numerical_flux!(
    ::CentralNumericalFlux{Gradient},
    balance_law::BalanceLaw,
    transform_gradient::MMatrix,
    normal_vector::SVector,
    state_gradient⁻::Vars{T},
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_gradient⁺::Vars{T},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
) where {T, S, A}

    transform_gradient .=
        normal_vector .*
        (parent(state_gradient⁺) .+ parent(state_gradient⁻))' ./ 2
end

function numerical_boundary_flux!(
    numerical_flux::CentralNumericalFlux{Gradient},
    bctype,
    balance_law::BalanceLaw,
    transform_gradient::MMatrix,
    normal_vector::SVector,
    state_gradient⁻::Vars{T},
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_gradient⁺::Vars{T},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {D, T, S, A}
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        aux1⁻,
    )

    compute_gradient_argument!(
        balance_law,
        state_gradient⁺,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
    )
    transform_gradient .= normal_vector .* parent(state_gradient⁺)'
end

"""
    NumericalFlux{AbstractFluxType}
    NumericalFlux{FirstOrder}
    NumericalFlux{SecondOrder}

Any `N <: NumericalFlux{FirstOrder}` should define the a method for

    numerical_flux!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux, normal_vector⁻,
        Q⁻, Qaux⁻,
        Q⁺, Qaux⁺, t)

where
- `flux` is the numerical flux array
- `normal_vector⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux, normal_vector⁻,
        Q⁻, Qaux⁻,
        Q⁺, Qaux⁺,
        bctype, t)

    NumericalFlux{SecondOrder}

Any `N <: NumericalFlux{SecondOrder}` should define the a method for

    numerical_flux!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux, normal_vector⁻,
        Q⁻, Qstate_gradient_flux⁻, Qaux⁻,
        Q⁺, Qstate_gradient_flux⁺, Qaux⁺,
        t)

where
- `flux` is the numerical flux array
- `normal_vector⁻` is the unit normal
- `Q⁻`/`Q⁺` are the minus/positive state arrays
- `Qstate_gradient_flux⁻`/`Qstate_gradient_flux⁺` are the minus/positive diffusive state arrays
- `Qstate_gradient_flux⁻`/`Qstate_gradient_flux⁺` are the minus/positive auxiliary state arrays
- `t` is the time

An optional method can also be defined for

    numerical_boundary_flux!(
        numerical_flux::N,
        balance_law::BalanceLaw,
        flux, normal_vector⁻,
        Q⁻, Qstate_gradient_flux⁻, Qaux⁻,
        Q⁺, Qstate_gradient_flux⁺, Qaux⁺,
        bctype,
        t)
"""
abstract type NumericalFlux{FT <: AbstractFluxType} end

function numerical_flux! end

function numerical_boundary_flux!(
    numerical_flux::NumericalFlux{FirstOrder},
    bctype,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
    state1⁻::Vars{S},
    aux1⁻::Vars{A},
) where {S, A}

    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        aux1⁻,
    )

    numerical_flux!(
        numerical_flux,
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
end


"""
    RusanovNumericalFlux <: NumericalFlux{FirstOrder}

The RusanovNumericalFlux (aka local Lax-Friedrichs) numerical flux.

# Usage

    RusanovNumericalFlux()

Requires a `flux!` and `wavespeed` method for the balance law.
"""
struct RusanovNumericalFlux <: NumericalFlux{FirstOrder} end

update_penalty!(::RusanovNumericalFlux, ::BalanceLaw, _...) = nothing

function numerical_flux!(
    numerical_flux::RusanovNumericalFlux,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    numerical_flux!(
        CentralNumericalFlux{FirstOrder}(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    fluxᵀn = parent(fluxᵀn)
    wavespeed⁻ = wavespeed(
        balance_law,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )
    wavespeed⁺ = wavespeed(
        balance_law,
        normal_vector,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    max_wavespeed = max.(wavespeed⁻, wavespeed⁺)
    penalty =
        max_wavespeed .* (parent(state_prognostic⁻) - parent(state_prognostic⁺))

    # TODO: should this operate on ΔQ or penalty?
    update_penalty!(
        numerical_flux,
        balance_law,
        normal_vector,
        max_wavespeed,
        Vars{S}(penalty),
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
    )

    fluxᵀn .+= penalty / 2
end

function numerical_flux!(
    nf::CentralNumericalFlux{FirstOrder},
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux⁻ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    flux_first_order!(
        balance_law,
        Grad{S}(flux⁻),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )

    flux⁺ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁺, -zero(FT))
    flux_first_order!(
        balance_law,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    fluxᵀn .+= (flux⁻ + flux⁺)' * (normal_vector / 2)
end

"""
    RoeNumericalFlux <: NumericalFlux{FirstOrder}

A numerical flux based on the approximate Riemann solver of Roe

# Usage

    RoeNumericalFlux()

Requires a custom implementation for the balance law.
"""
struct RoeNumericalFlux <: NumericalFlux{FirstOrder} end

"""
    HLLCNumericalFlux <: NumericalFlux{FirstOrder}

A numerical flux based on the approximate Riemann solver of the
HLLC method. The HLLC flux is a modification of the Harten, Lax, van-Leer
(HLL) flux, where an additional contact property is introduced in order
to restore missing rarefraction waves. The HLLC flux requires
model-specific information, hence it requires a custom implementation
based on the underlying balance law.

# Usage

    HLLCNumericalFlux()

Requires a custom implementation for the balance law.

    @book{toro2013riemann,
        title={Riemann solvers and numerical methods for fluid dynamics: a practical introduction},
        author={Toro, Eleuterio F},
        year={2013},
        publisher={Springer Science & Business Media}
    }
"""
struct HLLCNumericalFlux <: NumericalFlux{FirstOrder} end

function numerical_flux!(
    ::CentralNumericalFlux{SecondOrder},
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector⁻::SVector,
    state_prognostic⁻::Vars{S},
    state_gradient_flux⁻::Vars{D},
    state_hyperdiffusive⁻::Vars{HD},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_gradient_flux⁺::Vars{D},
    state_hyperdiffusive⁺::Vars{HD},
    state_auxiliary⁺::Vars{A},
    t,
) where {S, D, HD, A}

    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux⁻ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    flux_second_order!(
        balance_law,
        Grad{S}(flux⁻),
        state_prognostic⁻,
        state_gradient_flux⁻,
        state_hyperdiffusive⁻,
        state_auxiliary⁻,
        t,
    )

    flux⁺ = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux⁺, -zero(FT))
    flux_second_order!(
        balance_law,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        t,
    )

    fluxᵀn .+= (flux⁻ + flux⁺)' * (normal_vector⁻ / 2)
end

function numerical_flux!(
    ::CentralNumericalFlux{Divergence},
    balance_law::BalanceLaw,
    div_penalty::Vars{GL},
    normal_vector::SVector,
    grad⁻::Grad{GL},
    grad⁺::Grad{GL},
) where {GL}
    parent(div_penalty) .=
        (parent(grad⁺) .- parent(grad⁻))' * (normal_vector / 2)
end

function numerical_boundary_flux!(
    numerical_flux::CentralNumericalFlux{Divergence},
    bctype,
    balance_law::BalanceLaw,
    div_penalty::Vars{GL},
    normal_vector::SVector,
    grad⁻::Grad{GL},
    grad⁺::Grad{GL},
) where {GL}
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        grad⁺,
        normal_vector,
        grad⁻,
    )
    numerical_flux!(
        numerical_flux,
        balance_law,
        div_penalty,
        normal_vector,
        grad⁻,
        grad⁺,
    )
end

abstract type GradNumericalFlux end
struct CentralNumericalFlux{FT <: HigherOrder} <: NumericalFlux{FT} end

function numerical_flux!(
    ::CentralNumericalFlux{HigherOrder},
    balance_law::BalanceLaw,
    hyperdiff::Vars{HD},
    normal_vector::SVector,
    lap⁻::Vars{GL},
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    lap⁺::Vars{GL},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
) where {HD, GL, S, A}
    G = normal_vector .* (parent(lap⁻) .+ parent(lap⁺))' ./ 2
    transform_post_gradient_laplacian!(
        balance_law,
        hyperdiff,
        Grad{GL}(G),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
    )
end

function numerical_boundary_flux!(
    numerical_flux::CentralNumericalFlux{HigherOrder},
    bctype,
    balance_law::BalanceLaw,
    hyperdiff::Vars{HD},
    normal_vector::SVector,
    lap⁻::Vars{GL},
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    lap⁺::Vars{GL},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
) where {HD, GL, S, A}
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_auxiliary⁺,
        lap⁺,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        lap⁻,
        t,
    )
    numerical_flux!(
        numerical_flux,
        balance_law,
        hyperdiff,
        normal_vector,
        lap⁻,
        state_prognostic⁻,
        state_auxiliary⁻,
        lap⁺,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
    )
end

numerical_boundary_flux!(
    numerical_flux::CentralNumericalFlux{SecondOrder},
    bctype,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_gradient_flux⁻::Vars{D},
    state_hyperdiffusive⁻::Vars{HD},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_gradient_flux⁺::Vars{D},
    state_hyperdiffusive⁺::Vars{HD},
    state_auxiliary⁺::Vars{A},
    t,
    state1⁻::Vars{S},
    diff1⁻::Vars{D},
    aux1⁻::Vars{A},
) where {S, D, HD, A} = normal_boundary_flux!(
    numerical_flux,
    bctype,
    balance_law,
    fluxᵀn,
    normal_vector,
    state_prognostic⁻,
    state_gradient_flux⁻,
    state_hyperdiffusive⁻,
    state_auxiliary⁻,
    state_prognostic⁺,
    state_gradient_flux⁺,
    state_hyperdiffusive⁺,
    state_auxiliary⁺,
    t,
    state1⁻,
    diff1⁻,
    aux1⁻,
)

function normal_boundary_flux!(
    numerical_flux::NumericalFlux{O <: SecondOrder},
    bctype,
    balance_law::BalanceLaw,
    fluxᵀn::Vars{S},
    normal_vector,
    state_prognostic⁻,
    state_gradient_flux⁻,
    state_hyperdiffusive⁻,
    state_auxiliary⁻,
    state_prognostic⁺,
    state_gradient_flux⁺,
    state_hyperdiffusive⁺,
    state_auxiliary⁺,
    t,
    state1⁻,
    diff1⁻,
    aux1⁻,
) where {S}
    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    fluxᵀn = parent(fluxᵀn)

    flux = similar(fluxᵀn, Size(3, num_state_prognostic))
    fill!(flux, -zero(FT))
    boundary_flux!(
        numerical_flux,
        bctype,
        balance_law,
        Grad{S}(flux),
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_gradient_flux⁻,
        state_hyperdiffusive⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        diff1⁻,
        aux1⁻,
    )

    fluxᵀn .+= flux' * normal_vector
end

# This is the function that my be overloaded for flux-based BCs
function boundary_flux!(
    numerical_flux::NumericalFlux{O <: SecondOrder},
    bctype,
    balance_law,
    flux,
    state_prognostic⁺,
    state_gradient_flux⁺,
    state_hyperdiffusive⁺,
    state_auxiliary⁺,
    normal_vector,
    state_prognostic⁻,
    state_gradient_flux⁻,
    state_hyperdiffusive⁻,
    state_auxiliary⁻,
    t,
    state1⁻,
    diff1⁻,
    aux1⁻,
) where {O}
    boundary_state!(
        numerical_flux,
        bctype,
        balance_law,
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_auxiliary⁺,
        normal_vector,
        state_prognostic⁻,
        state_gradient_flux⁻,
        state_auxiliary⁻,
        t,
        state1⁻,
        diff1⁻,
        aux1⁻,
    )
    flux!(
        O,
        balance_law,
        flux,
        state_prognostic⁺,
        state_gradient_flux⁺,
        state_hyperdiffusive⁺,
        state_auxiliary⁺,
        t,
    )
end

end
