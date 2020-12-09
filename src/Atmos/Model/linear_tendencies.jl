##### Mass tendencies

#####
##### First order fluxes
#####

function flux(::Advect{Mass}, m::AtmosLinearModel, state, aux, t, ts, direction)
    return state.ρu
end


##### Momentum tendencies

using CLIMAParameters.Planet: Omega

#####
##### First order fluxes
#####

struct LinearPressureGradient{PV <: Momentum} <: TendencyDef{Flux{FirstOrder}, PV} end

function flux(::LinearPressureGradient{Momentum}, lm::AtmosLinearModel, state, aux, t, ts, direction)
    pad = (state.ρu .* (state.ρu / state.ρ)') * 0
    pL = linearized_pressure(
        lm.atmos.moisture,
        lm.atmos.param_set,
        lm.atmos.orientation,
        state,
        aux,
    )
    return pad + pL * I
end

#####
##### Sources (Momentum)
#####
function source(
    s::Gravity{Momentum},
    lm::AtmosAcousticGravityLinearModel,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    if direction isa VerticalDirection || direction isa EveryDirection
        ∇Φ = ∇gravitational_potential(lm.atmos.orientation, aux)
        return -state.ρ * ∇Φ
    end
    FT = eltype(state)
    return SVector{3,FT}(0,0,0)
end

##### Energy tendencies

#####
##### First order fluxes
#####

struct LinearEnergyFlux{PV <: Energy} <: TendencyDef{Flux{FirstOrder}, PV} end

function flux(::LinearEnergyFlux{Energy}, m, state, aux, t, ts, direction)
    ref = aux.ref_state
    return ((ref.ρe + ref.p) / ref.ρ - e_pot) * state.ρu
end

prognostic_vars(m::AtmosLinearModel) = (
    Mass(),
    Momentum(),
    Energy(),
)
