##### Show tendencies

export show_tendencies

using PrettyTables

first_param(t::TendencyDef{TT, PV}) where {TT, PV} = PV

function format_tend(tend_type, include_params)
    t = "$(nameof(typeof(tend_type)))"
    return include_params ? t * "{$(first_param(tend_type))}" : t
end

format_tends(tend_types, include_params) =
    "(" * join(format_tend.(tend_types, include_params), ", ") * ")"

# Filter tendencies that match the prognostic variable:
full_eq_tends(pv::PV, td::TendencyDef{TT, PV}) where {PV,TT} = td
full_eq_tends(pv::PVA, td::TendencyDef{TT, PVB}) where {PVA,PVB,TT} = ()
full_eq_tends(pv::PV, ::Tuple{}) where {PV} = ()

"""
    full_eq_tends(bl::BalanceLaw, tend_type::AbstractTendencyType)

The full column-vector (Tuple) of
`TendencyDef`s, of length
`length(all_prognostic_vars(bl))`
given the balance law.
"""
function full_eq_tends(bl::BalanceLaw, tend_type::AbstractTendencyType)
  eqt = eq_tends(bl, tend_type)
  @show eqt
  return map(all_prognostic_vars(bl)) do pv
    filter(x->x≠(), full_eq_tends.(Ref(pv), eqt))
  end
end

"""
    show_tendencies(bl; include_params = false)

Show a table of the tendencies for each
prognostic variable for the given balance law.

Using `include_params = true` will include the type
parameters for each of the tendency fluxes
and sources.

Requires definitions for
 - [`prognostic_vars`](@ref)
 - [`eq_tends`](@ref)
for the balance law.
"""
function show_tendencies(bl; include_params = false)
    ip = include_params
    prog_vars = all_prognostic_vars(bl)
    if !isempty(prog_vars)
        header = [
            "Equation" "Flux{FirstOrder}" "Flux{SecondOrder}" "Source"
            "(Y_i)" "(F_1)" "(F_2)" "(S)"
        ]
        eqs = collect((typeof.(prog_vars)))

        fmt_tends(tt) = format_tends.(full_eq_tends(bl, tt), ip) |> collect

        F1 = fmt_tends(Flux{FirstOrder}())
        F2 = fmt_tends(Flux{SecondOrder}())
        S = fmt_tends(Source())
        data = hcat(eqs, F1, F2, S)
        @warn "This table is temporarily incomplete"
        println("\nPDE: ∂_t Y_i + (∇•F_1(Y))_i + (∇•F_2(Y,G)))_i = (S(Y,G))_i")
        pretty_table(
            data,
            header,
            header_crayon = crayon"yellow bold",
            subheader_crayon = crayon"green bold",
        )
        println("")
    else
        msg = "Defining `prognostic_vars` and\n"
        msg *= "`eq_tends` for $(nameof(typeof(bl))) will\n"
        msg *= "enable printing a table of tendencies."
        @info msg
    end
    return nothing
end
