#### Land sources

export Constantτ_FreezeThaw, Variableτ_FreezeThaw, Variableτ_FreezeThawApprox

function heaviside(x::FT) where {FT}
    if x> FT(0)
        output = FT(1)
    else
        output = FT(0)
    end
    return output
end


# Example of how to specify a source
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


abstract type Source{FT <: AbstractFloat} end

abstract type PostTendencySource{FT <: AbstractFloat} end


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

# or this
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

    ∇κ∇T = tendency.soil.heat.∇κ∇T
    tendency.soil.heat.∇κ∇T = FT(0.0)
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


#function land_post_tendency_source!(
#    source_type::FreezeThawGCCE,
#    land::LandModel,
#    tendency,
#    state,
#    aux,
#    t,
#)
#    FT = eltype(state)
#    _LH_f0 = FT(LH_f0(land.param_set))
#    _ρliq = FT(ρ_cloud_liq(land.param_set))
#    _ρice = FT(ρ_cloud_ice(land.param_set))
#    _Tfreeze = FT(T_freeze(land.param_set))#
#
#    
#    ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, t)
#    eff_porosity = land.soil.param_functions.porosity - θ_i
#    θ_l = volumetric_liquid_fraction(ϑ_l, eff_porosity)
#
#    
#    T = get_temperature(land.soil.heat,aux,t)
#    m_w = (_ρliq * θ_l * heaviside(_Tfreeze - T))# +
#           #_ρice * θ_i * heaviside(T - _Tfreeze)
#           #)
#    ∇κ∇T = tendency.soil.heat.∇κ∇T
#    ρc_ds = land.soil.param_functions.ρc_ds
#    ρc_s = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, land.param_set)
#    Tdot = ∇κ∇T/ρc_s
#    
#    # Zero this out so we don't really compute a tendency for it (need to
#    # incrementally added time stepping methods since we want to tendency to be 
#    # ∇κ∇T for a single tendency without previous values)
#    tendency.soil.heat.∇κ∇T = FT(0)
#
#    S = effective_saturation(land.soil.param_functions.porosity, θ_l)
#    m = land.soil.water.hydraulics.m
#    α = land.soil.water.hydraulics.α
#    n = land.soil.water.hydraulics.n
#    ψ0 = matric_potential(land.soil.water.hydraulics, S)
#
#    dSdψ = m*n*α^n*abs(ψ0)^(n-FT(1))*
#        (FT(1)+(α*abs(ψ0))^n)^(-m-FT(1))
#
#    g = FT(9.8)
#    ν = land.soil.param_functions.porosity
#    
#    if m_w > eps(FT)
#        F_T = (_LH_f0*ν*dSdψ*Tdot/g/_Tfreeze/(FT(1)+dSdψ*_LH_f0^FT(2)*ν*_ρliq/g/_Tfreeze/ρc_s))*source_type.factor
#    else
#        F_T = FT(0.0)
#    end
#    tendency.soil.water.ϑ_l += F_T
#    tendency.soil.water.θ_i -= F_T*_ρliq/_ρice
#
#end



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
