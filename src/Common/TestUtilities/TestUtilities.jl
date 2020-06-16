"""
    TestUtilities
Convenience functions for experiment setups to enable detection
of unphysical parameters at intermediate evaluation points within
a ClimateMachine simulation. Exploits current vars-aware MPIStateArray 
and callback handling system. Returns a callback instruction which
must be passed to the experiment. 
"""
module TestUtilities

export build_callback_conservation

using LinearAlgebra
using ..Mesh.Grids
using ..GenericCallbacks
using ..MPIStateArrays
using ..VariableTemplates

function build_callback_conservation(solver_config, 
                                     driver_config,
                                     check_frequency=3000; 
                                     tol=FT(0)) where FT
    # State variable
    Q = solver_config.Q
    # Volume geometry information
    vgeo = driver_config.grid.vgeo
    M = vgeo[:, Grids._M, :]
    # Unpack prognostic vars
    ρ₀ = Q.ρ
    ρe₀ = Q.ρe
    # DG variable sums
    Σρ₀ = sum(ρ₀ .* M)
    Σρe₀ = sum(ρe₀ .* M)
    callback =
        GenericCallbacks.EveryXSimulationSteps() do (init = false)
            Q = solver_config.Q
            δρ = (sum(Q.ρ .* M) - Σρ₀) / Σρ₀
            δρe = (sum(Q.ρe .* M) .- Σρe₀) ./ Σρe₀
            @show (abs(δρ))
            @show (abs(δρe))
            @test (abs(δρ) <= 0.0001)
            @test (abs(δρe) <= 0.0025)
            nothing
        end
    return callback
end

end
