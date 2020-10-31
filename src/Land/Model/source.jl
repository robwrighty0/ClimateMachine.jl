#### Land sources

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


abstract type Source{FT <: AbstractFloat} end

abstract type PostTendencySource{FT <: AbstractFloat} end


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
