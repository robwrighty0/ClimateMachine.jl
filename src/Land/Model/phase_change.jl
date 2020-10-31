
export Constantτ_FreezeThaw, Variableτ_FreezeThaw, Variableτ_FreezeThawApprox

"""
    Variableτ_FreezeThawApprox <: Source

The function which computes the freeze/thaw source term for Richard's equation,
assuming the timescale is a function of how quickly the volumetric 
internal energy can change, the amount of energy required to freeze or melt
the appropriate amount of water, and other constants.

In this function, we approximate the time derivative of the volumetric
internal energy.
"""
Base.@kwdef struct Variableτ_FreezeThawApprox{FT} <: Source{FT}
    "Timestep"
    Δt::FT = FT(NaN)
    "Timescale for temperature changes"
    τLTE::FT = FT(NaN)
end


"""
    Constantτ_FreezeThaw <: Source

The function which computes the freeze/thaw source term for Richard's equation,
assuming the timescale is the maximum of the thermal timescale and the timestep.
"""
Base.@kwdef struct Constantτ_FreezeThaw{FT} <: Source{FT} 
    "Timestep"
    Δt::FT = FT(NaN)
    "Timescale for temperature changes"
    τLTE::FT = FT(NaN)
end


"""
        Variableτ_FreezeThaw <: PostTendencySource

The function which computes the freeze/thaw source term for Richard's equation,
assuming the timescale is a function of how quickly the volumetric 
internal energy can change, the amount of energy required to freeze or melt
the appropriate amount of water, and other constants.
"""
Base.@kwdef struct Variableτ_FreezeThaw{FT} <: PostTendencySource{FT} 
    "Timestep"
    Δt::FT = FT(NaN)
    "Timescale for temperature changes"
    τLTE::FT = FT(NaN)
end


function initialize_defaults(phase_change_source::Nothing, state::Vars)
end

function initialize_defaults(phase_change_source::Source, state::Vars)
end

function initialize_defaults(phase_change_source::PostTendencySource, state::Vars)
        state.soil.phase_change_source.∇κ∇T = eltype(state)(0)

end


#the user picks a source term from the above, or none at all
vars_state(phase_change_source::Nothing, st::Prognostic, FT) = @vars()
vars_state(phase_change_source::Source, st::Prognostic, FT) = @vars()
vars_state(phase_change_source::PostTendencySource, st::Prognostic, FT) = @vars(∇κ∇T::FT)


function flux_second_order!(
    land::LandModel,
    soil::SoilModel,
    phase_change::PostTendencySource,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux.soil.phase_change_source.∇κ∇T += -diffusive.soil.heat.κ∇T
end



function flux_second_order!(
    land::LandModel,
    soil::SoilModel,
    phase_change::Source,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
end


function flux_second_order!(
    land::LandModel,
    soil::SoilModel,
    phase_change::Nothing,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
end

function heaviside(x::FT) where {FT}
    if x> FT(0)
        output = FT(1)
    else
        output = FT(0)
    end
    return output
end


function land_source!(
    source_type::Constantτ_FreezeThaw,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)
    _ρliq = FT(ρ_cloud_liq(land.param_set))
    _ρice = FT(ρ_cloud_ice(land.param_set))
    _Tfreeze = FT(T_freeze(land.param_set))
    _LH_f0 = FT(LH_f0(land.param_set))
    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, t)
    eff_porosity = land.soil.param_functions.porosity - θ_i
    θ_l = volumetric_liquid_fraction(ϑ_l, eff_porosity)
    
    T = get_temperature(land.soil.heat, aux, t)
    τft = max(source_type.Δt, source_type.τLTE)
    m = land.soil.water.hydraulics.m
    α = land.soil.water.hydraulics.α
    n = land.soil.water.hydraulics.n
    
    ψ = _LH_f0/FT(9.8)/_Tfreeze*(T-_Tfreeze)
    θstar = land.soil.param_functions.porosity*(FT(1)+(α*abs(ψ))^n)^(-m)
    
    freeze_thaw = 1.0/τft *(_ρliq*(θ_l-θstar)*heaviside(_Tfreeze - T)*heaviside(θ_l-θstar) -
                            _ρice*θ_i*heaviside(T - _Tfreeze))
    source.soil.water.ϑ_l -= freeze_thaw/_ρliq
    source.soil.water.θ_i += freeze_thaw/_ρice
end




function land_source!(
    source_type::Variableτ_FreezeThawApprox,
    land::LandModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)
    _ρliq = FT(ρ_cloud_liq(land.param_set))
    _ρice = FT(ρ_cloud_ice(land.param_set))
    _Tfreeze = FT(T_freeze(land.param_set))
    _LH_f0 = FT(LH_f0(land.param_set))

    T = get_temperature(land.soil.heat,aux,t)

    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, t)
    eff_porosity = land.soil.param_functions.porosity - θ_i
    θ_l = volumetric_liquid_fraction(ϑ_l, eff_porosity)
    ρc_ds = land.soil.param_functions.ρc_ds
    ρc = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, land.param_set)

    m = land.soil.water.hydraulics.m
    α = land.soil.water.hydraulics.α
    n = land.soil.water.hydraulics.n
    
    ψ = _LH_f0/FT(9.8)/_Tfreeze*(T-_Tfreeze)
    θstar = land.soil.param_functions.porosity*(FT(1)+(α*abs(ψ))^n)^(-m)
    
    τ_pc = source_type.τLTE*abs(FT(_ρliq*_LH_f0/ρc*abs((θ_l-θstar)))/(T-_Tfreeze))
    τft = max(source_type.Δt, source_type.τLTE, τ_pc)

    freeze_thaw = 1.0/τft *(_ρliq*(θ_l-θstar)*heaviside(_Tfreeze - T)*heaviside(θ_l-θstar) -
                            _ρice*θ_i*heaviside(T - _Tfreeze))
    source.soil.water.ϑ_l -= freeze_thaw/_ρliq
    source.soil.water.θ_i += freeze_thaw/_ρice
end


function land_post_tendency_source!(
    source_type::Variableτ_FreezeThaw,
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
    _LH_f0 = FT(LH_f0(land.param_set))

    T = get_temperature(land.soil.heat,aux,t)

    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, t)
    eff_porosity = land.soil.param_functions.porosity - θ_i
    θ_l = volumetric_liquid_fraction(ϑ_l, eff_porosity)
    ρc_ds = land.soil.param_functions.ρc_ds
    ρc = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, land.param_set)

    ∇κ∇T = tendency.soil.phase_change_source.∇κ∇T
    tendency.soil.phase_change_source.∇κ∇T = FT(0.0)
    Idot = ∇κ∇T
    m = land.soil.water.hydraulics.m
    α = land.soil.water.hydraulics.α
    n = land.soil.water.hydraulics.n
    
    ψ = _LH_f0/FT(9.8)/_Tfreeze*(T-_Tfreeze)
    θstar = land.soil.param_functions.porosity*(FT(1)+(α*abs(ψ))^n)^(-m)

    τ_pc = abs(_ρliq*_LH_f0*(θ_l-θstar)/Idot)
    τ_pc_est = source_type.τLTE*abs(FT(_ρliq*_LH_f0/ρc*abs((θ_l-θstar)))/(T-_Tfreeze))
    # Do we need to deal with cases where there is no flux (not so hard in a test case to set up)?
#    if heaviside(_Tfreeze - T)*heaviside(θ_l-θstar) > 0
#        @printf("%lf %lf %lf %le %le %le\n ", t, T, aux.z, source_type.τLTE, τ_pc/source_type.τLTE, τ_pc_est/source_type.τLTE)
#    end
    
    τft = max(source_type.Δt, source_type.τLTE, τ_pc)
    
    freeze_thaw = 1.0/τft *(_ρliq*(θ_l-θstar)*heaviside(_Tfreeze - T)*heaviside(θ_l-θstar) -
                            _ρice*θ_i*heaviside(T - _Tfreeze))
    tendency.soil.water.ϑ_l -= freeze_thaw/_ρliq
    tendency.soil.water.θ_i += freeze_thaw/_ρice
end
