#### Land sources

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

function post_tendency_source!(land::LandModel, tendency, state, aux, t)
    # We've piggybacked on the tendency calculation to store this
    ∇κ∇T = tendency.soil.heat.∇κ∇T

    # Zero this out so we don't really compute a tendency for it (need to
    # incrementally added time stepping methods since we want to tendency to be 
    # ∇κ∇T for a single tendency without previous values)
    tendency.soil.heat.∇κ∇T = 0

    F_T = 0 # TODO: fill me!

    # XXX: CHECK ALL THIS!
    ρe_int_l = volumetric_internal_energy_liq(aux.soil.heat.T, land.param_set)
    tendency.soil.water.ϑ_l += F_T / ρe_int_l
    tendency.soil.water.θ_i -= F_T / soil.heat.ρe_int
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
        Base.Cartesian.@nexprs $N i -> land_source!(
            stuple[i],
            land,
            source,
            state,
            diffusive,
            aux,
            t,
            direction,
        )
        return nothing
    end
end
