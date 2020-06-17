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
using Printf
using Test

using ..DGMethods: BalanceLaw
using ..Atmos: DryModel
using ..Mesh.Grids
using ..GenericCallbacks
using ..MPIStateArrays
using ..VariableTemplates
using ..PhysicsTests

function build_callback_conservation(solver_config, 
                                     driver_config,
                                     frequency=10000,
                                     tolerance=0)
    # Unpack balance law
    balance_law = solver_config.dg.balance_law
    # State variable
    Q = solver_config.Q
    FT = eltype(Q)
    # Volume geometry information
    vgeo = driver_config.grid.vgeo
    M = vgeo[:, Grids._M, :]
    # Unpack prognostic vars
    ρ₀ = Q.ρ
    ρu⃗₀ = Q.ρu
    ρu₀ = view(ρu⃗₀,:,1,:)
    ρv₀ = view(ρu⃗₀,:,2,:)
    ρw₀ = view(ρu⃗₀,:,3,:)
    ρe₀ = Q.ρe
    if balance_law.moisture isa DryModel
        ρq_tot₀ = zeros(similar(ρ₀))
    else
        ρq_tot₀ = Q[:,6,:]
    end
    # DG variable sums
    Σρ₀ = sum(ρ₀ .* M)
    Σρu₀ = sum(ρu₀ .* M)
    Σρv₀ = sum(ρv₀ .* M)
    Σρw₀ = sum(ρw₀ .* M)
    Σρe₀ = sum(ρe₀ .* M)
    Σρe₀ = sum(ρe₀ .* M)
    Σρq_tot₀ = sum(ρq_tot₀ .* M)

    callback =
        GenericCallbacks.EveryXSimulationSteps(frequency) do (init = false)
            Q = solver_config.Q
            ρ = Q.ρ
            ρu⃗ = Q.ρu
            ρu = view(ρu⃗₀,:,1,:)
            ρv = view(ρu⃗₀,:,2,:)
            ρw = view(ρu⃗₀,:,3,:)
            ρe = Q.ρe
            if balance_law.moisture isa DryModel
                ρq_tot = zeros(similar(ρ₀))
            else
                ρq_tot = Q[:,6,:]
            end
            δρ = (sum(ρ .* M) - Σρ₀) / Σρ₀ * 100
            δρu = (sum(ρu .* M) - Σρu₀)
            δρv = (sum(ρv .* M) - Σρv₀)
            δρw = (sum(ρw .* M) - Σρw₀)
            δρe = (sum(ρe .* M) .- Σρe₀) ./ Σρe₀ * 100
            δρq_tot = (sum(ρq_tot .* M) .- Σρq_tot₀) ./ (eps(FT) + Σρq_tot₀) * 100
            # Output Display
            @info @sprintf(
                """ 
                ΔPrognostics 
                ΔMass        (percent)  = %.4e 
                ΔMomentum[x] (absolute) = %.4e
                ΔMomentum[y] (absolute) = %.4e 
                ΔMomentum[z] (absolute) = %.4e
                ΔEnergy      (percent)  = %.4e
                ΔMoisture    (percent)  = %.4e
                """, 
                δρ,δρu,δρv,δρw,δρe,δρq_tot
            )
            nothing
        end
    return callback
end

end
