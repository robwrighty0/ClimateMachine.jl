"""
    PhaseEquil{FT} <: ThermodynamicState

A thermodynamic state assuming thermodynamic equilibrium (therefore, saturation adjustment
may be needed).

# Constructors

    PhaseEquil(e_int, ρ, q_tot)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhaseEquil{FT,PS<:APS{FT}} <: ThermodynamicState{FT}
  "parameter set (e.g., planet parameters)"
  param_set::PS
  "internal energy"
  e_int::FT
  "density of air (potentially moist)"
  ρ::FT
  "total specific humidity"
  q_tot::FT
  "temperature: computed via [`saturation_adjustment`](@ref)"
  T::FT
end
function PhaseEquil{FT}(;
    ρ::Union{FT,Nothing}=nothing,
    e_int::Union{FT,Nothing}=nothing,
    q_tot::Union{FT,Nothing}=nothing,
    T::Union{FT,Nothing}=nothing,
    θ_liq_ice::Union{FT,Nothing}=nothing,
    p::Union{FT,Nothing}=nothing,
    maxiter::Union{Int,Nothing}=nothing,
    tol::Union{FT,Nothing}=nothing,
    sat_adjust::Function=saturation_adjustment,
    param_set::PS=MTPS{FT}()
    ) where {FT<:Real,PS}

    @assert q_tot ≠ nothing
    _FT = FT
    if e_int ≠ nothing && ρ ≠ nothing

        @assert T === nothing
        @assert θ_liq_ice === nothing
        @assert p === nothing
        maxiter === nothing && (maxiter = 3)
        tol === nothing && (tol = _FT(1e-1))

        # TODO: Remove these safety nets, or at least add warnings
        # waiting on fix: github.com/vchuravy/GPUifyLoops.jl/issues/104
        q_tot = clamp(q_tot, _FT(0), _FT(1))
        T = sat_adjust(e_int, ρ, q_tot, maxiter, tol, param_set)

    elseif θ_liq_ice ≠ nothing && ρ ≠ nothing

        @assert e_int === nothing
        @assert p === nothing
        @assert T === nothing
        maxiter === nothing && (maxiter = 30)
        tol === nothing && (tol = _FT(1e-1))

        # TODO: expose which numerical method to use for sat adjustment?
        T = saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice, ρ, q_tot, maxiter, tol, param_set)
        q_pt = PhasePartition_equil(T, ρ, q_tot, param_set)
        e_int = internal_energy(T, q_pt, param_set)

    elseif T ≠ nothing && p ≠ nothing

        @assert θ_liq_ice === nothing
        @assert e_int === nothing
        @assert ρ === nothing
        @assert maxiter === nothing
        @assert tol === nothing

        ρ = air_density(T, p, PhasePartition(q_tot), param_set)
        q = PhasePartition_equil(T, ρ, q_tot, param_set)
        e_int = internal_energy(T, q, param_set)

    elseif θ_liq_ice ≠ nothing && p ≠ nothing

        @assert T === nothing
        @assert e_int === nothing
        @assert ρ === nothing
        maxiter === nothing && (maxiter = 30)
        tol === nothing && (tol = _FT(1e-1))

        # TODO: expose which numerical method to use for sat adjustment?
        T = saturation_adjustment_q_tot_θ_liq_ice_given_pressure(θ_liq_ice, p, q_tot, maxiter, tol, param_set)
        ρ = air_density(T, p, PhasePartition(q_tot), param_set)
        q = PhasePartition_equil(T, ρ, q_tot, param_set)
        e_int = internal_energy(T, q, param_set)
        q_tot = q.tot

    else
        throw(ArgumentError("PhaseEquil kwarg combination incorrect."))
    end
    return PhaseEquil{_FT,PS}(param_set, e_int, ρ, q_tot, T)
end
PhaseEquil(; param_set::PS=MTPS{FT}(), kwargs...) where {FT,PS} =
  PhaseEquil{FT}(;param_set=param_set, kwargs...)


"""
    PhaseDry{FT} <: ThermodynamicState

A dry thermodynamic state (`q_tot = 0`).

# Constructors

    PhaseDry(e_int, ρ)

# Fields

$(DocStringExtensions.FIELDS)
"""
struct PhaseDry{FT,PS<:APS{FT}} <: ThermodynamicState{FT}
  "parameter set (e.g., planet parameters)"
  param_set::PS
  "internal energy"
  e_int::FT
  "density of dry air"
  ρ::FT
end
function PhaseDry{FT}(;e_int::Union{FT,Nothing}=nothing,
                       ρ::Union{FT,Nothing}=nothing,
                       p::Union{FT,Nothing}=nothing,
                       T::Union{FT,Nothing}=nothing,
                       param_set::PS=MTPS{FT}()
                       ) where {FT<:AbstractFloat,PS}
    if e_int ≠ nothing && ρ ≠ nothing
        _FT = eltype(e_int)
        @assert p === nothing
        @assert T === nothing
    elseif p ≠ nothing && T ≠ nothing
        _FT = eltype(T)
        @assert e_int === nothing
        @assert ρ === nothing
        e_int = internal_energy(T, param_set)
        ρ = air_density(T, p, param_set)
    else
        throw(ArgumentError("PhaseDry kwarg combination incorrect."))
    end

  return PhaseDry{_FT,PS}(param_set, e_int, ρ)
end
PhaseDry(;kwargs...) = PhaseDry{eltype(values(kwargs))}(;kwargs...)
