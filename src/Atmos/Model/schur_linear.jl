import ...DGMethods: SchurComplement,
                        schur_init_aux!,
                        schur_update_aux!,
                        schur_init_state!,
                        schur_lhs_conservative!,
                        schur_lhs_nonconservative!,
                        schur_rhs_conservative!,
                        schur_rhs_nonconservative!,
                        schur_update_conservative!,
                        schur_update_nonconservative!,
                        schur_lhs_boundary_state!,
                        schur_gradient_boundary_state!,
                        schur_update_boundary_state!,
                        schur_rhs_boundary_state!

using ...DGMethods: SchurAuxiliary, SchurAuxiliaryGradient

import ...BalanceLaws: vars_state

using CLIMAParameters.Planet: kappa_d

export AtmosAcousticLinearSchurComplement,
       AtmosAcousticGravityLinearSchurComplement 

struct AtmosAcousticLinearSchurComplement <: SchurComplement end

function vars_state(::AtmosAcousticLinearSchurComplement, ::SchurAuxiliary, FT)
  @vars begin
    h0::FT
    Φ::FT # FIXME: get this from aux directly
    ∇h0::SVector{3, FT}
  end
end
vars_state(::AtmosAcousticLinearSchurComplement, ::SchurAuxiliaryGradient, FT) = @vars(h0::FT)

function schur_init_aux!(::AtmosAcousticLinearSchurComplement,
                         lm, schur_aux, aux, geom)
  Φ = gravitational_potential(lm.atmos.orientation, aux)
  schur_aux.h0 = (aux.ref_state.ρe + aux.ref_state.p) / aux.ref_state.ρ - Φ
  schur_aux.Φ = Φ
end

function schur_update_aux!(::AtmosAcousticLinearSchurComplement,
                         lm, schur_aux, state, aux)
end

function schur_init_state!(::AtmosAcousticLinearSchurComplement,
                           lm,
                           schur_state, schur_aux, state_rhs, aux)
  Φ = schur_aux.Φ
  param_set = lm.atmos.param_set
  γ = 1 / (1 - kappa_d(param_set))
  schur_state.p = (γ - 1) * (state_rhs.ρe - state_rhs.ρ * (Φ - R_d(param_set) * T_0(param_set) / (γ - 1)))
end

function schur_lhs_conservative!(::AtmosAcousticLinearSchurComplement, 
                                 lm, lhs_flux, schur_state, schur_grad, schur_aux, α)
  lhs_flux.p = -α * schur_grad.∇p
end
function schur_lhs_nonconservative!(::AtmosAcousticLinearSchurComplement,
                                    lm, lhs_state, schur_state, schur_grad, schur_aux, α)
  Φ = schur_aux.Φ
  param_set = lm.atmos.param_set
  γ = 1 / (1 - kappa_d(param_set))
  p = schur_state.p
  h0 = schur_aux.h0
  ∇h0 = schur_aux.∇h0
  ∇p = schur_grad.∇p
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)
  lhs_state.p = p / ((γ - 1) * α * (h0 - Δp - Φ)) - α * ∇h0' * ∇p / (h0 - Δp - Φ)
end
function schur_gradient_boundary_state!(::AtmosAcousticLinearSchurComplement, 
                                   balance_law,
                                   schur_state⁺, n,
                                   schur_state⁻, bctype
                                  )
end
function schur_lhs_boundary_state!(::AtmosAcousticLinearSchurComplement, 
                                   balance_law,
                                   schur_state⁺, schur_grad⁺, schur_aux⁺, n,
                                   schur_state⁻, schur_grad⁻, schur_aux⁻, bctype
                                  )
  schur_grad⁺.∇p = schur_grad⁻.∇p - 2 * dot(schur_grad⁻.∇p, n) * n
end

function schur_rhs_conservative!(::AtmosAcousticLinearSchurComplement, 
                                 balance_law,
                                 rhs_flux, state, schur_aux, α)
  rhs_flux.p = -state.ρu
end
function schur_rhs_nonconservative!(::AtmosAcousticLinearSchurComplement, 
                                    lm,
                                    rhs_state, state, schur_aux, α)
  Φ = schur_aux.Φ
  param_set = lm.atmos.param_set
  γ = 1 / (1 - kappa_d(param_set))
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  h0 = schur_aux.h0
  ∇h0 = schur_aux.∇h0
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)
  
  rhs_state.p = (ρe - (Φ + Δp) * ρ) / (α * (h0 - Δp - Φ)) - ∇h0' * ρu / (h0 - Δp - Φ)
end
function schur_rhs_boundary_state!(::AtmosAcousticLinearSchurComplement, 
                                   balance_law,
                                   state⁺, schur_aux⁺, n,
                                   state⁻, schur_aux⁻, bctype
                                  )
  state⁺.ρu = state⁻.ρu - 2 * dot(state⁻.ρu, n) * n
end

function schur_update_conservative!(::AtmosAcousticLinearSchurComplement, 
                                    balance_law,
                                    lhs_flux, schur_state, schur_grad, schur_aux, state_rhs, α)
  p = schur_state.p
  h0 = schur_aux.h0
  ∇p = schur_grad.∇p
  
  lhs_flux.ρ = -α * (state_rhs.ρu - α * ∇p)
  lhs_flux.ρu -= α * p * I
  lhs_flux.ρe = -α * (h0 * (state_rhs.ρu - α * ∇p))
end
function schur_update_nonconservative!(::AtmosAcousticLinearSchurComplement, 
                                       balance_law,
                                       lhs_state,
                                       schur_state, schur_grad, schur_aux, state_rhs, α)
  lhs_state.ρ = state_rhs.ρ
  lhs_state.ρu = state_rhs.ρu
  lhs_state.ρe = state_rhs.ρe
end

function schur_update_boundary_state!(::AtmosAcousticLinearSchurComplement, 
                                   balance_law,
                                   schur_state⁺, schur_grad⁺, schur_aux⁺, state_rhs⁺, n,
                                   schur_state⁻, schur_grad⁻, schur_aux⁻, state_rhs⁻, bctype
                                  )
  schur_grad⁺.∇p = schur_grad⁻.∇p - 2 * dot(schur_grad⁻.∇p, n) * n
  state_rhs⁺.ρu = state_rhs⁻.ρu - 2 * dot(state_rhs⁻.ρu, n) * n
end

# GRAVITY
struct AtmosAcousticGravityLinearSchurComplement <: SchurComplement end

function vars_state(::AtmosAcousticGravityLinearSchurComplement, ::SchurAuxiliary, FT)
  @vars begin
    h0::FT
    ρ0::FT # Testing
    ρe0::FT # Testing
    T0::FT # Testing
    p0::FT # Testing
    Φ::FT # FIXME: get this from aux directly
    g0::FT # Do we need this ?
    ∇Φ::SVector{3, FT} # FIXME: get this from aux directly
    ∇h0::SVector{3, FT}
  end
end
vars_state(::AtmosAcousticGravityLinearSchurComplement, ::SchurAuxiliaryGradient, FT) = @vars(h0::FT)

function schur_init_aux!(::AtmosAcousticGravityLinearSchurComplement,
                         lm, schur_aux, aux, geom)
  Φ = gravitational_potential(lm.atmos, aux)
  ∇Φ = ∇gravitational_potential(lm.atmos, aux)
  schur_aux.h0 = (aux.ref_state.ρe + aux.ref_state.p) / aux.ref_state.ρ
  schur_aux.Φ = Φ
  schur_aux.ρ0 = aux.ref_state.ρ
  schur_aux.T0 = aux.ref_state.T
  schur_aux.p0 = aux.ref_state.p
  schur_aux.ρe0 = aux.ref_state.ρe
  schur_aux.∇Φ = ∇Φ
end

function schur_update_aux!(::AtmosAcousticGravityLinearSchurComplement,
                         lm, schur_aux, state, aux)
  schur_aux.g0 = schur_aux.h0 * state.ρ - state.ρe
end

function schur_init_state!(::AtmosAcousticGravityLinearSchurComplement,
                           lm,
                           schur_state, schur_aux, state_rhs, aux)
  Φ = schur_aux.Φ
  param_set = lm.atmos.param_set
  γ = 1 / (1 - kappa_d(param_set))
  schur_state.p = (γ - 1) * (state_rhs.ρe - state_rhs.ρ * (Φ - R_d(param_set) * T_0(param_set) / (γ - 1)))
  #schur_state.p = aux.ref_state.p
end

function schur_lhs_conservative!(::AtmosAcousticGravityLinearSchurComplement, 
                                 lm, lhs_flux, schur_state, schur_grad, schur_aux, α)
  FT = eltype(schur_state)
  Φ = schur_aux.Φ
  param_set = lm.atmos.param_set
  γ::FT = 1 / (1 - kappa_d(param_set))
  _grav::FT = grav(lm.atmos.param_set)
  p = schur_state.p
  h0 = schur_aux.h0
  ∇h0 = schur_aux.∇h0
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)
  
  #k = vertical_unit_vector(lm.atmos, schur_aux)
  k = schur_aux.∇Φ / FT(grav(param_set))
  #k = SVector{3, FT}(0, 0, 0)

  A = I + α ^ 2 * _grav / (h0 - Φ - Δp) * k * ∇h0'
  lhs_flux.p = -α * (A \ (schur_grad.∇p + _grav * k / ((h0 - Φ - Δp) * (γ - 1)) * p))
  #lhs_flux.p = -α * (A \ (schur_grad.∇p))# - _grav * k * p / (R_d(param_set) * schur_aux.T0)))
  #lhs_flux.p = -α * schur_grad.∇p
  #@show α, schur_grad.∇p
  #error("hey")
  #lhs_flux.p = SVector{3, FT}(0, 0, 0)
end
function schur_lhs_nonconservative!(::AtmosAcousticGravityLinearSchurComplement,
                                    lm, lhs_state, schur_state, schur_grad, schur_aux, α)
  FT = eltype(schur_state)
  Φ = schur_aux.Φ
  param_set = lm.atmos.param_set
  γ::FT = 1 / (1 - kappa_d(param_set))
  p = schur_state.p
  h0 = schur_aux.h0
  ∇h0 = schur_aux.∇h0
  ∇p = schur_grad.∇p
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)
  
  #k = vertical_unit_vector(lm.atmos, schur_aux)
  k = schur_aux.∇Φ / FT(grav(param_set))
  #k = SVector{3, FT}(0, 0, 0)

  _grav::FT = grav(lm.atmos.param_set)
  A = I + α ^ 2 * _grav / (h0 - Φ - Δp) * k * ∇h0'

  lhs_state.p = p / ((γ - 1) * α * (h0 - Δp - Φ)) #-
                #α * ∇h0' / (h0 - Δp - Φ) * (A \ (∇p + _grav * k / ((h0 - Φ -Δp) * (γ - 1)) * p)) 
  #lhs_state.p = 0
end
function schur_gradient_boundary_state!(::AtmosAcousticGravityLinearSchurComplement, 
                                   balance_law,
                                   schur_state⁺, n,
                                   schur_state⁻, bctype
                                  )
end
function schur_lhs_boundary_state!(::AtmosAcousticGravityLinearSchurComplement, 
                                   lm,
                                   schur_state⁺, schur_grad⁺, schur_aux⁺, n,
                                   schur_state⁻, schur_grad⁻, schur_aux⁻, bctype
                                  )
  FT = eltype(schur_state⁻)
  param_set = lm.atmos.param_set
  #schur_grad⁺.∇p = schur_grad⁻.∇p - 2 * dot(schur_grad⁻.∇p, n) * n
  #lhs_state.ρu = A \ (ρu -
  #                    α * ∇p -
  #                    α * _grav * k / (h0 - Φ - Δp) * (h0 * ρ - ρe) -
  #                    α * _grav * k / ((h0 - Φ - Δp) * (γ - 1)) * p)

  h0 = schur_aux⁻.h0
  g0 = schur_aux⁻.g0
  Φ = schur_aux⁻.Φ
  
  γ::FT = 1 / (1 - kappa_d(param_set))
  _grav::FT = grav(lm.atmos.param_set)
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)
  
  k = schur_aux⁻.∇Φ / FT(grav(param_set))
  
  #p = schur_state⁻.p
  p = schur_aux⁻.p0
  
  β = (_grav * k / (h0 - Φ - Δp) * g0 +
       _grav * k / ((h0 - Φ - Δp) * (γ - 1)) * p)
  schur_grad⁺.∇p = schur_grad⁻.∇p - 2 * dot(schur_grad⁻.∇p + β, n) * n
end

function schur_rhs_conservative!(::AtmosAcousticGravityLinearSchurComplement, 
                                 lm,
                                 rhs_flux, state, schur_aux, α)
  FT = eltype(schur_aux)
  Φ = schur_aux.Φ
  param_set = lm.atmos.param_set
  γ::FT = 1 / (1 - kappa_d(param_set))
  _grav::FT = grav(lm.atmos.param_set)
  h0 = schur_aux.h0
  ∇h0 = schur_aux.∇h0
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)
  
  #k = vertical_unit_vector(lm.atmos, schur_aux)
  k = schur_aux.∇Φ / FT(grav(param_set))
  #k = SVector{3, FT}(0, 0, 0)

  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe

  A = I + α ^ 2 * _grav / (h0 - Φ - Δp) * k * ∇h0'
  rhs_flux.p = -A \ (ρu - α * _grav * k / (h0 - Φ - Δp) * (h0 * ρ - ρe))
  #rhs_flux.p = SVector{3, FT}(0, 0, 0)
  #rhs_flux.p = α * _grav * k * schur_aux.ρ0
end
function schur_rhs_nonconservative!(::AtmosAcousticGravityLinearSchurComplement, 
                                    lm,
                                    rhs_state, state, schur_aux, α)
  FT = eltype(schur_aux)
  Φ = schur_aux.Φ
  param_set = lm.atmos.param_set
  γ::FT = 1 / (1 - kappa_d(param_set))
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  h0 = schur_aux.h0
  ∇h0 = schur_aux.∇h0
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)
  
  #k = vertical_unit_vector(lm.atmos, schur_aux)
  k = schur_aux.∇Φ / FT(grav(param_set))
  #k = SVector{3, FT}(0, 0, 0)


  _grav::FT = grav(lm.atmos.param_set)
  A = I + α ^ 2 * _grav / (h0 - Φ - Δp) * k * ∇h0'
  
  rhs_state.p = (ρe - (Φ + Δp) * ρ) / (α * (h0 - Δp - Φ))# -
                #∇h0' * (A \ (ρu - α * _grav * k / (h0 - Φ - Δp) * (h0 * ρ - ρe))) / (h0 - Δp - Φ)
  #rhs_state.p = 0
end
function schur_rhs_boundary_state!(::AtmosAcousticGravityLinearSchurComplement, 
                                   balance_law,
                                   state⁺, schur_aux⁺, n,
                                   state⁻, schur_aux⁻, bctype
                                  )
  state⁺.ρu = state⁻.ρu - 2 * dot(state⁻.ρu, n) * n
end

function schur_update_conservative!(::AtmosAcousticGravityLinearSchurComplement, 
                                    lm,
                                    lhs_flux, schur_state, schur_grad, schur_aux, state_rhs, α)
  FT = eltype(schur_state)
  param_set = lm.atmos.param_set
  p = schur_state.p
  h0 = schur_aux.h0
  ∇p = schur_grad.∇p

  Φ = schur_aux.Φ

  #k = vertical_unit_vector(lm.atmos, schur_aux)
  k = schur_aux.∇Φ / FT(grav(param_set))
  #k = SVector{3, FT}(0, 0, 0)


  ∇h0 = schur_aux.∇h0
  _grav::FT = grav(param_set)
  γ::FT = 1 / (1 - kappa_d(param_set))
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)
  
  A = I + α ^ 2 * _grav / (h0 - Φ - Δp) * k * ∇h0'

  p0 = schur_aux.p0

  ρ = state_rhs.ρ
  ρu = state_rhs.ρu
  ρe = state_rhs.ρe
  
  updated_ρu = A \ (ρu -
                    α * ∇p -
                    α * _grav * k / (h0 - Φ - Δp) * (h0 * ρ - ρe) -
                    α * _grav * k / ((h0 - Φ - Δp) * (γ - 1)) * p0)
  
  lhs_flux.ρ = -α * updated_ρu
  #lhs_flux.ρu -= α * p * I
  lhs_flux.ρe = -α * h0 * updated_ρu
end
function schur_update_nonconservative!(::AtmosAcousticGravityLinearSchurComplement, 
                                       lm,
                                       lhs_state,
                                       schur_state, schur_grad, schur_aux, state_rhs, α)
 
  FT = eltype(schur_state)
  param_set = lm.atmos.param_set
  Φ = schur_aux.Φ
  
  #k = vertical_unit_vector(lm.atmos, schur_aux)
  k = schur_aux.∇Φ / FT(grav(param_set))
  #k = SVector{3, FT}(0, 0, 0)


  ∇h0 = schur_aux.∇h0
  h0 = schur_aux.h0
  _grav::FT = grav(param_set)
  
  γ::FT = 1 / (1 - kappa_d(param_set))
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)

  A = I + α ^ 2 * _grav / (h0 - Φ - Δp) * k * ∇h0'
  
  ρ = state_rhs.ρ
  ρu = state_rhs.ρu
  ρe = state_rhs.ρe
  ∇p = schur_grad.∇p
  p = schur_state.p
  p0 = schur_aux.p0

  lhs_state.ρ = ρ
  #lhs_state.ρ = schur_aux.ρ0
  #lhs_state.ρu = ρu - α * _grav * k / (h0 - Φ - Δp) * (h0 * ρ - ρe) -
  #                    α * _grav * k / ((h0 - Φ - Δp) * (γ - 1)) * p
  lhs_state.ρu =  (ρu -
                      α * ∇p -
                      α * _grav * k / (h0 - Φ - Δp) * (h0 * ρ - ρe) -
                      α * _grav * k / ((h0 - Φ - Δp) * (γ - 1)) * p0)
  #lhs_state.ρu = SVector{3, FT}(0, 0, 0)
  lhs_state.ρe = ρe
  #lhs_state.ρe = schur_aux.ρe0
end

function schur_update_boundary_state!(::AtmosAcousticGravityLinearSchurComplement, 
                                   lm,
                                   schur_state⁺, schur_grad⁺, schur_aux⁺, state_rhs⁺, n,
                                   schur_state⁻, schur_grad⁻, schur_aux⁻, state_rhs⁻, bctype
                                  )
  state_rhs⁺.ρu = state_rhs⁻.ρu - 2 * dot(state_rhs⁻.ρu, n) * n
  
  #schur_grad⁺.∇p = schur_grad⁻.∇p - 2 * dot(schur_grad⁻.∇p, n) * n
  FT = eltype(schur_state⁻)
  param_set = lm.atmos.param_set
  #schur_grad⁺.∇p = schur_grad⁻.∇p - 2 * dot(schur_grad⁻.∇p, n) * n
  #lhs_state.ρu = A \ (ρu -
  #                    α * ∇p -
  #                    α * _grav * k / (h0 - Φ - Δp) * (h0 * ρ - ρe) -
  #                    α * _grav * k / ((h0 - Φ - Δp) * (γ - 1)) * p)

  h0 = schur_aux⁻.h0
  g0 = schur_aux⁻.g0
  #g0 = schur_aux⁻.h0 * state_rhs⁻.ρ - state_rhs⁻.ρe
  Φ = schur_aux⁻.Φ
  
  γ::FT = 1 / (1 - kappa_d(param_set))
  _grav::FT = grav(lm.atmos.param_set)
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)
  
  k = schur_aux⁻.∇Φ / FT(grav(param_set))
  
  p = schur_state⁻.p
  p0 = schur_aux⁻.p0
  
  β = (_grav * k / (h0 - Φ - Δp) * g0 +
       _grav * k / ((h0 - Φ - Δp) * (γ - 1)) * p0)
  schur_grad⁺.∇p = schur_grad⁻.∇p - 2 * dot(schur_grad⁻.∇p + β, n) * n
end