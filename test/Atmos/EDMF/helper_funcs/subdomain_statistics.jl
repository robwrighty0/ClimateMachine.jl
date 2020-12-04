#### Subdomain statistics

compute_subdomain_statistics(atmos::AtmosEquations, state, aux, t) =
    compute_subdomain_statistics(
        atmos,
        state,
        aux,
        t,
        atmos.turbconv.micro_phys.statistical_model,
    )

"""
    compute_subdomain_statistics(
        atmos::AtmosEquations{FT},
        state::Vars,
        aux::Vars,
        t::Real,
        statistical_model::SubdomainMean,
    ) where {FT}

Returns a cloud fraction and cloudy and dry thermodynamic
states in the subdomain.
"""
function compute_subdomain_statistics(
    atmos::AtmosEquations{FT},
    state::Vars,
    aux::Vars,
    t::Real,
    statistical_model::SubdomainMean,
) where {FT}
    ts_en = recover_thermo_state_en(atmos, state, aux)
    cloud_frac = has_condensate(ts_en) ? FT(1) : FT(0)
    dry = ts_en
    cloudy = ts_en
    return (dry = dry, cloudy = cloudy, cloud_frac = cloud_frac)
end
