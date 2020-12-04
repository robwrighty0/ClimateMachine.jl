export AbstractRadiationEquations, NoRadiation

abstract type AbstractRadiationEquations <: AbstractAtmosComponent end

vars_state(::AbstractRadiationEquations, ::AbstractStateType, FT) = @vars()

function atmos_nodal_update_auxiliary_state!(
    ::AbstractRadiationEquations,
    ::AtmosEquations,
    state::Vars,
    aux::Vars,
    t::Real,
) end
function integral_set_auxiliary_state!(::AbstractRadiationEquations, integ::Vars, aux::Vars) end
function integral_load_auxiliary_state!(
    ::AbstractRadiationEquations,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function reverse_integral_set_auxiliary_state!(
    ::AbstractRadiationEquations,
    integ::Vars,
    aux::Vars,
) end
function reverse_integral_load_auxiliary_state!(
    ::AbstractRadiationEquations,
    integ::Vars,
    state::Vars,
    aux::Vars,
) end
function flux_radiation!(
    ::AbstractRadiationEquations,
    atmos::AtmosEquations,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end

struct NoRadiation <: AbstractRadiationEquations end
