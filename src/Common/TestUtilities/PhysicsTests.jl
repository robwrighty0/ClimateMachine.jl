module PhysicsTests

using DocStringExtensions
using ClimateMachine.MPIStateArrays

export PhysicsTest, CheckMassConservation, CheckMomentumConservation, CheckMoistureConservation, test_conservation

abstract type PhysicsTest end
"""
    test_conservation(::PhysicsTest)
Function stub for conservation checks
"""
function test_conservation(::PhysicsTest)
    return nothing
end

"""
    struct CheckMassConservation <: PhysicsTest
# Fields
$(DocStringExtensions.FIELDS)
"""
struct CheckMassConservation <: PhysicsTest
    "Initial State"
    Q₀::MPIStateArray
    "Final State"
    Qₑ::MPIStateArray
    "Variables"
end

function test_conservation(X::CheckMassConservation, bl)
    ρₑ = norm(X.Qₑ[:,1,:])
    ρ₀ = norm(Q₀[:,1,:])
    Δρ_rel = (ρₑ - ρ₀) / ρ₀
end

"""
    struct CheckMassConservation <: PhysicsTest
# Fields
$(DocStringExtensions.FIELDS)
"""
struct CheckMomentumConservation <: PhysicsTest
    "Initial State"
    Q₀::MPIStateArray
    "Final State"
    Qₑ::MPIStateArray
end
function test_conservation(X::CheckMomentumConservation, bl)
    ρₑ = norm(X.Qₑ[:,5,:])
    ρ₀ = norm(Q₀[:,5,:])
    Δρe_rel = (ρeₑ - ρe₀) / ρe₀
end

"""
    struct CheckConservation <: PhysicsTest
# Fields
$(DocStringExtensions.FIELDS)
"""
struct CheckMoistureConservation <: PhysicsTest
    "Initial State"
    Q₀::MPIStateArray
    "Final State"
    Qₑ::MPIStateArray
end
function test_conservation(X::CheckMoistureConservation, bl)
    #=
    if typeof(bl.moisture) == EquilMoist
        ρeₑ = norm(solver_config.Q[:,5,:])
        ρe₀ = norm(Qe[:,5,:])
        Δρe_rel = (ρeₑ - ρe₀) / ρe₀
    else
        
    end
    =# 
end


end
