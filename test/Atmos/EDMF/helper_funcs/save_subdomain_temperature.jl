# Convenience wrapper
save_subdomain_temperature!(m, state, aux) =
    save_subdomain_temperature!(m, m.moisture, state, aux)

using KernelAbstractions: @print

"""
    save_subdomain_temperature!(
        atmos::AtmosEquations,
        moist::EquilMoist,
        state::Vars,
        aux::Vars,
    )

Updates the subdomain sensible temperature, given:
 - `atmos`, the `AtmosEquations`
 - `moist`, an `EquilMoist` model
 - `state`, state variables
 - `aux`, auxiliary variables
"""
function save_subdomain_temperature!(
    atmos::AtmosEquations,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    N_up = n_updrafts(atmos.turbconv)
    ts = recover_thermo_state(atmos, state, aux)
    ts_up = new_thermo_state_up(atmos, state, aux, ts)
    ts_en = new_thermo_state_en(atmos, state, aux, ts)

    @unroll_map(N_up) do i
        aux.turbconv.updraft[i].T = air_temperature(ts_up[i])
    end
    aux.turbconv.environment.T = air_temperature(ts_en)
    return nothing
end
