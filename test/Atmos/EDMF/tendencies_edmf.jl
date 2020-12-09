#####
##### Tendency types
#####

##### First order fluxes
struct SGSFlux{PV <: Union{Momentum, Energy, TotalMoisture}} <:
       TendencyDef{Flux{SecondOrder}, PV} end

##### Second order fluxes

##### Sources

const EnvVars = Union{en_ρatke,
    en_ρaθ_liq_cv,
    en_ρaq_tot_cv,
    en_ρaθ_liq_q_tot_cv}

const CVEnvVars = Union{en_ρaθ_liq_cv,
    en_ρaq_tot_cv,
    en_ρaθ_liq_q_tot_cv}

const UpVars = Union{UP_ρa,
    UP_ρaw,
    UP_ρaθ_liq,
    UP_ρaq_tot}

const EntrDetrVars = Union{UpVars, EnvVars}

struct EntrSource{PV <: EntrDetrVars} <: TendencyDef{Source, PV} end
struct DetrSource{PV <: EntrDetrVars} <: TendencyDef{Source, PV} end
struct BuoySource{PV <: Union{en_ρatke, UP_ρaw}} <: TendencyDef{Source, PV} end
struct BuoyPressSource{PV <: UP_ρaw} <: TendencyDef{Source, PV} end
struct TurbEntrSource{PV <: EnvVars} <: TendencyDef{Source, PV} end
struct PressSource{PV <: en_ρatke} <: TendencyDef{Source, PV} end
struct ShearSource{PV <: en_ρatke} <: TendencyDef{Source, PV} end
struct DissSource{PV <: EnvVars} <: TendencyDef{Source, PV} end
struct GradProdSource{PV <: CVEnvVars} <: TendencyDef{Source, PV} end

#####
##### Tendency definitions
#####


##### First order fluxes
##### Second order fluxes

##### Sources

function compute_entr_detr_params(m, state, aux, t, ts_all, direction, diffusive)
    EΔ_up = ntuple(N_up) do i
        entr_detr(m, m.turbconv.entr_detr, state, aux, t, ts, env, i)
    end

    E_dyn, Δ_dyn, E_trb = ntuple(i -> map(x -> x[i], EΔ_up), 3)
    return (E_dyn=E_dyn, Δ_dyn=Δ_dyn, E_trb=E_trb)
end


function source(s::EntrSource{en_ρatke}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::EntrSource{en_ρaθ_liq_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::EntrSource{en_ρaq_tot_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::EntrSource{en_ρaθ_liq_q_tot_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::EntrSource{UP_ρa}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::EntrSource{UP_ρaw}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::EntrSource{UP_ρaθ_liq}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::EntrSource{UP_ρaq_tot}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::DetrSource{en_ρatke}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::DetrSource{en_ρaθ_liq_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::DetrSource{en_ρaq_tot_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::DetrSource{en_ρaθ_liq_q_tot_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::DetrSource{UP_ρa}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::DetrSource{UP_ρaw}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::DetrSource{UP_ρaθ_liq}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::DetrSource{UP_ρaq_tot}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end

function source(s::TurbEntrSource{en_ρatke}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::TurbEntrSource{en_ρaθ_liq_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::TurbEntrSource{en_ρaq_tot_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::TurbEntrSource{en_ρaθ_liq_q_tot_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end

function source(s::DissSource{en_ρatke}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::DissSource{en_ρaθ_liq_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::DissSource{en_ρaq_tot_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::DissSource{en_ρaθ_liq_q_tot_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end

function source(s::GradProdSource{en_ρaθ_liq_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::GradProdSource{en_ρaq_tot_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::GradProdSource{en_ρaθ_liq_q_tot_cv}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end

function source(s::BuoySource{en_ρatke}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::BuoySource{UP_ρaw}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end

function source(s::BuoyPressSource{UP_ρaw}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::PressSource{en_ρatke}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end
function source(s::ShearSource{en_ρatke}, args...)
    (m, state, aux, t, ts, direction, diffusive) = args
end

