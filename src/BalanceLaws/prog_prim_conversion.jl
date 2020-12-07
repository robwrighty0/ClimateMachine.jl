# Add `Primitive` type to BalanceLaws, AtmosModel

# Vars wrapper
function prognostic_to_primitive!(
    prim::AbstractArray,
    prog::AbstractArray,
    bl::BalanceLaw,
    aux::AbstractArray,
)
    FT = eltype(prim)
    prognostic_to_primitive!(
        Vars{vars_state(bl, Primitive(), FT)}(prim),
        Vars{vars_state(bl, Prognostic(), FT)}(prog),
        bl,
        Vars{vars_state(bl, Auxiliary(), FT)}(aux),
    )
end
function primitive_to_prognostic!(
    prog::AbstractArray,
    prim::AbstractArray,
    bl::BalanceLaw,
    aux::AbstractArray,
)
    FT = eltype(prog)
    primitive_to_prognostic!(
        Vars{vars_state(bl, Prognostic(), FT)}(prog),
        Vars{vars_state(bl, Primitive(), FT)}(prim),
        bl,
        Vars{vars_state(bl, Auxiliary(), FT)}(aux),
    )
end

# By default the primitive is the prognostic
vars_state(bl::BalanceLaw, ::Primitive, FT) = vars_state(bl, Prognostic(), FT)

function prognostic_to_primitive!(prim::Vars, prog::Vars, bl, aux)
    prim_arr = parent(prim)
    prog_arr = parent(prog)
    prim_arr .= prog_arr
end
function primitive_to_prognostic!(prog::Vars, prim::Vars, bl, aux)
    prim_arr = parent(prim)
    prog_arr = parent(prog)
    prog_arr .= prim_arr
end
