# [AtmosEquations](@id AtmosEquations-docs)

```@meta
CurrentModule = ClimateMachine
```

## AtmosProblem

```@docs
ClimateMachine.Atmos.AtmosProblem
```

## AtmosEquations balance law

```@docs
ClimateMachine.Atmos.AtmosEquations
```

## AtmosEquations methods

```@docs
ClimateMachine.BalanceLaws.flux_first_order!(atmos::AtmosEquations, flux::Grad, state::Vars, aux::Vars, t::Real, direction)
ClimateMachine.BalanceLaws.flux_second_order!(atmos::AtmosEquations, flux::Grad, state::Vars, diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real)
ClimateMachine.BalanceLaws.init_state_auxiliary!(atmos::AtmosEquations, state_auxiliary::MPIStateArray, grid, direction)
ClimateMachine.BalanceLaws.source!(atmos::AtmosEquations, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real, direction)
ClimateMachine.BalanceLaws.init_state_prognostic!(atmos::AtmosEquations, state::Vars, aux::Vars, coords, t, args...)
```

## Reference states

```@docs
ClimateMachine.Atmos.HydrostaticState
ClimateMachine.Atmos.InitStateBC
ClimateMachine.Atmos.ReferenceState
ClimateMachine.Atmos.NoReferenceState
```

## Thermodynamics

```@docs
ClimateMachine.Atmos.recover_thermo_state
ClimateMachine.Atmos.new_thermo_state
```

## Moisture

```@docs
ClimateMachine.Atmos.DryEquations
ClimateMachine.Atmos.EquilMoist
ClimateMachine.Atmos.NonEquilMoist
ClimateMachine.Atmos.NoPrecipitation
ClimateMachine.Atmos.Rain
```

## Stabilization

```@docs
ClimateMachine.Atmos.RayleighSponge
```

## BCs

```@docs
ClimateMachine.Atmos.AtmosBC
ClimateMachine.Atmos.DragLaw
ClimateMachine.Atmos.Impermeable
ClimateMachine.Atmos.PrescribedMoistureFlux
ClimateMachine.Atmos.BulkFormulaMoisture
ClimateMachine.Atmos.FreeSlip
ClimateMachine.Atmos.PrescribedTemperature
ClimateMachine.Atmos.PrescribedEnergyFlux
ClimateMachine.Atmos.BulkFormulaEnergy
ClimateMachine.Atmos.ImpermeableTracer
ClimateMachine.Atmos.Impenetrable
ClimateMachine.Atmos.Insulating
ClimateMachine.Atmos.NoSlip
ClimateMachine.Atmos.average_density
```

## Sources

```@docs
ClimateMachine.Atmos.RemovePrecipitation
ClimateMachine.Atmos.CreateClouds
```
