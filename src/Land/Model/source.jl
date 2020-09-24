#### Land sources

export FreezeThaw

function land_source!(
    f::Function,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    f(land, source, state, diffusive, aux, t, direction)
end

function land_source!(
    ::Nothing,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) end


abstract type Source end

abstract type PostTendencySource end

"""
    FreezeThaw <: PostTendencySource
The function which computes the freeze/thaw source term for Richard's equation.
"""
Base.@kwdef struct FreezeThaw <: PostTendencySource
    "Freeze thaw timescale function"
    τft::Function = nothing
end


function land_post_tendency_source!(
    source_type::FreezeThaw,
    land::LandModel,
    tendency,
    state,
    aux,
    t,
)
    FT = eltype(state)
    _ρliq = FT(ρ_cloud_liq(land.param_set))
    _ρice = FT(ρ_cloud_ice(land.param_set))
    _Tfreeze = FT(T_freeze(land.param_set))

    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, t)
    θ_l = volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)


    # We've piggybacked on the tendency calculation to store this
    ∇κ∇T = tendency.soil.heat.∇κ∇T

    τft = source_type.τft(land, ∇κ∇T, state, aux) # we also would like the option of using a constant timescale for a test
    # plus the default should be to run without this.



    # Zero this out so we don't really compute a tendency for it (need to
    # incrementally added time stepping methods since we want to tendency to be 
    # ∇κ∇T for a single tendency without previous values)
    tendency.soil.heat.∇κ∇T = 0

    F_T =
        1.0 / τft * (
            _ρliq * θ_l * heaviside(_Tfreeze - T) -
            _ρice * θ_i * heaviside(T - _Tfreeze)
        )

    tendency.soil.water.ϑ_l -= F_T / _ρliq
    tendency.soil.water.θ_i += F_T / _ρice
end



# sources are applied additively

@generated function land_source!(
    stuple::Tuple,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    N = fieldcount(stuple)
    return quote
        Base.Cartesian.@nexprs $N i -> if stuple[i] isa Source
            land_source!(
                stuple[i],
                land,
                source,
                state,
                diffusive,
                aux,
                t,
                direction,
            )
        end
        return nothing
    end
end

@generated function land_post_tendency_source!(
    stuple::Tuple,
    land::LandModel,
    tendency::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    N = fieldcount(stuple)
    return quote
        Base.Cartesian.@nexprs $N i -> if stuple[i] isa PostTendencySource
            land_post_tendency_source!(stuple[i], land, tendency, state, aux, t)
        end
        return nothing
    end
end
