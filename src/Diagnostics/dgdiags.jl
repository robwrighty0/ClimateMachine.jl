using ..Atmos
using ..Atmos: MoistureModel

# Helpers to gather the variables from the model's on-the-fly calculations in DG across the DG grid

function vars_dgdiags(atmos::AtmosModel, FT)
    @vars begin
        Ω_dg₁::FT
        Ω_dg₂::FT
        Ω_dg₃::FT

        moisture::vars_dgdiags(atmos.moisture, FT)
    end
end
vars_dgdiags(::MoistureModel, FT) = @vars()
function vars_dgdiags(m::Union{EquilMoist, NonEquilMoist}, FT)
    @vars begin end
end
num_dgdiags(bl, FT) = varsize(vars_dgdiags(bl, FT))
dgdiags_vars(bl, array) = Vars{vars_dgdiags(bl, eltype(array))}(array)
