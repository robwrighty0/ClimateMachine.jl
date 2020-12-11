function numerical_boundary_flux_first_order!(
    numerical_flux_first_order,
    bctag::Int,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    face_direction,
    state_prognostic_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
)
    bcs = boundary_conditions(balance_law)
    # TODO: there is probably a better way to unroll this loop
    Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
        bc = bcs[d]
        numerical_boundary_flux_first_order!(
            numerical_flux_first_order,
            bc,
            balance_law,
            Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
            SVector(normal_vector),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
            t,
            face_direction,
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                state_prognostic_bottom1,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                state_auxiliary_bottom1,
            ),
        )
    end
end

function numerical_boundary_flux_second_order!(
    numerical_flux_second_order,
    bctag::Int,
    balance_law,
    flux::AbstractArray,
    normal_vector::AbstractArray,
    state_prognostic⁻::AbstractArray,
    state_gradient_flux⁻::AbstractArray,
    state_hyperdiffusive⁻::AbstractArray,
    state_auxiliary⁻::AbstractArray,
    state_prognostic⁺::AbstractArray,
    state_gradient_flux⁺::AbstractArray,
    state_hyperdiffusive⁺::AbstractArray,
    state_auxiliary⁺::AbstractArray,
    t,
    state_prognostic_bottom1::AbstractArray,
    state_auxiliary_bottom1::AbstractArray,
    state_gradient_flux_bottom1::AbstractArray,
)
    bcs = boundary_conditions(balance_law)
    # TODO: there is probably a better way to unroll this loop
    Base.Cartesian.@nif 7 d -> bctag == d <= length(bcs) d -> begin
        bc = bcs[d]
        numerical_boundary_flux_second_order!(
            numerical_flux_second_order,
            bc,
            balance_law,
            Vars{vars_state(balance_law, Prognostic(), FT)}(flux),
            SVector(normal_vector),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁻),
            Vars{vars_state(balance_law, GradientFlux(), FT)}(
                state_gradient_flux⁻,
            ),
            Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                state_hyperdiffusive⁻,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁻),
            Vars{vars_state(balance_law, Prognostic(), FT)}(state_prognostic⁺),
            Vars{vars_state(balance_law, GradientFlux(), FT)}(
                state_gradient_flux⁺,
            ),
            Vars{vars_state(balance_law, Hyperdiffusive(), FT)}(
                state_hyperdiffusive⁺,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(state_auxiliary⁺),
            t,
            face_direction,
            Vars{vars_state(balance_law, Prognostic(), FT)}(
                state_prognostic_bottom1,
            ),
            Vars{vars_state(balance_law, Auxiliary(), FT)}(
                state_auxiliary_bottom1,
            ),
            Vars{vars_state(balance_law, GradientFlux(), FT)}(
                state_gradient_flux_bottom1,
            ),
        )
    end
end


@kernel function vert_fvm_interface_tendency!(
    balance_law::BalanceLaw,
    ::Val{info},
    ::Val{nvertelem},
    ::Val{periodicstack},
    ::VerticalDirection,
    reconstruction!,
    numerical_flux_first_order,
    numerical_flux_second_order,
    tendency,
    state_prognostic,
    state_gradient_flux,
    state_auxiliary,
    vgeo,
    sgeo,
    t,
    elemtobndy,
    elems,
    α,
) where {info, nvertelem, periodicstack}
    @uniform begin
        dim = info.dim
        FT = eltype(state_prognostic)
        num_state_prognostic = number_states(balance_law, Prognostic())
        num_state_primitive = number_states(balance_law, Primitive())
        num_state_auxiliary = number_states(balance_law, Auxiliary())
        num_state_gradient_flux = number_states(balance_law, GradientFlux())
        num_state_hyperdiffusive = number_states(balance_law, Hyperdiffusive())
        @assert num_state_hyperdiffusive == 0

        vsp = Vars{vars_state(balance_law, Prognostic(), FT)}
        vsa = Vars{vars_state(balance_law, Auxiliary(), FT)}

        nface = info.nface
        Np = info.Np
        Nqk = info.Nqk # can only be 1 for the FVM method!
        @assert Nqk == 1

        # We only have the vertical faces
        faces = (nface - 1):nface

        stencil_width = width(reconstruction!)

        # In the case of stencil_width = 0 we still need two values to evaluate
        # the fluxes, so the minimum stencil diameter is 2
        stencil_diameter = max(2, 2stencil_width + 1)

        # Value in the stencil that corresponds to the top face with respect to
        # face being updated
        stencil_center = max(stencil_width, 1) + 1

        local_first_order_numerical_flux =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_second_order_numerical_flux =
            MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_prognostic = ntuple(Val(stencil_diameter)) do _
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        end

        # 1 → cell i, face i - 1/2
        # 2 → cell i, face i + 1/2
        local_state_face_prognostic = ntuple(Val(stencil_diameter)) do _
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        end

        local_cell_weights = MArray{Tuple{stencil_diameter}, FT}(undef)

        # Two mass matrix inverse corresponding to +/- cells
        vMI = MArray{Tuple{2}, FT}(undef)

        # Storing the value below element when walking up the stack
        # cell i-1, face i - 1/2
        local_state_face_prognostic_neighbor =
            MArray{Tuple{num_state_prognostic}, FT}(undef)

        local_state_face_primitive = ntuple(Val(2)) do _
            MArray{Tuple{num_state_primitive}, FT}(undef)
        end

        local_state_primitive = ntuple(Val(stencil_diameter)) do _
            MArray{Tuple{num_state_primitive}, FT}(undef)
        end

        local_state_auxiliary = ntuple(Val(stencil_diameter)) do _
            MArray{Tuple{num_state_auxiliary}, FT}(undef)
        end

        local_state_gradient_flux = ntuple(Val(stencil_diameter)) do _
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        end

        local_state_hyperdiffusive = ntuple(Val(stencil_diameter)) do
            MArray{Tuple{num_state_hyperdiffusive}, FT}(undef)
        end

        local_state_prognostic_bottom1 =
            MArray{Tuple{num_state_prognostic}, FT}(undef)
        local_state_gradient_flux_bottom1 =
            MArray{Tuple{num_state_gradient_flux}, FT}(undef)
        local_state_auxiliary_bottom1 =
            MArray{Tuple{num_state_auxiliary}, FT}(undef)

        # XXX: will revisit this later for FVM
        fill!(local_state_prognostic_bottom1, NaN)
        fill!(local_state_gradient_flux_bottom1, NaN)
        fill!(local_state_auxiliary_bottom1, NaN)

        # The remainder model needs to know which direction of face the model is
        # being evaluated for. In this case we only have `VerticalDirection()`
        # faces
        face_direction = (VerticalDirection())
    end

    # Get the horizontal group IDs
    grp_H = @index(Group, Linear)

    # Determine the index for the element at the bottom of the stack
    eHI = (grp_H - 1) * nvertelem + 1

    # Compute bottom stack element index minus one (so we can add vert element
    # number directly)
    eH = elems[eHI] - 1

    # Which degree of freedom do we handle in the element
    n = @index(Local, Linear)

    # Loads the data for a given element
    function load_data!(
        local_state_prognostic,
        local_state_auxiliary,
        local_state_gradient_flux,
        e,
    )
        @unroll for s in 1:num_state_prognostic
            local_state_prognostic[s] = state_prognostic[n, s, e]
        end

        @unroll for s in 1:num_state_auxiliary
            local_state_auxiliary[s] = state_auxiliary[n, s, e]
        end

        @unroll for s in 1:num_state_gradient_flux
            local_state_gradient_flux[s] = state_gradient_flux[n, s, e]
        end
    end

    # We need to compute the first element we handles bottom flux (only for nonperiodic boundary condition)
    # elements will just copied from the prior element)
    @inbounds begin
        eV = 1

        # Figure out the data we need
        els = ntuple(Val(stencil_diameter)) do k
            eH + mod1(k - stencil_center - 1, nvertelem)
        end

        # Load all the stencil data
        @unroll for k in 1:stencil_diameter
            load_data!(
                local_state_prognostic[k],
                local_state_auxiliary[k],
                local_state_gradient_flux[k],
                els[k],
            )
            # If local cell weights are NOT _M we need to load _vMI out of sgeo
            local_cell_weights[k] = vgeo[n, _M, els[k]]
        end

        # transform all the data into primitive variables
        @unroll for k in 1:stencil_diameter
            prognostic_to_primitive!(
                balance_law,
                local_state_primitive[k],
                local_state_prognostic[k],
                local_state_auxiliary[k],
            )
        end

        # If we are periodic we reconstruct the top and bottom values for eV
        # then start with eV update in loop below
        if periodicstack
            # Reconstruct the top and bottom values
            reconstruction!(
                local_state_face_primitive[1],
                local_state_face_primitive[2],
                local_state_primitive,
                local_cell_weights,
            )

            # Transform the values back to prognostic state
            @unroll for f in 1:2
                primitive_to_prognostic!(
                    balance_law,
                    local_state_face_prognostic[f],
                    local_state_face_primitive[f],
                    # Use the cell auxiliary data
                    local_state_auxiliary[stencil_center],
                )
            end
        else

            bctag = elemtobndy[faces[1], eH + eV]

            sM = sgeo[_sM, n, faces[1], eH + eV]
            normal = SVector(
                sgeo[_n1, n, face[1], eH + eV],
                sgeo[_n2, n, face[1], eH + eV],
                sgeo[_n3, n, face[1], eH + eV],
            )

            # Reconstruction using only eVs cell value
            reconstruction!(
                local_state_face_primitive[1],
                local_state_face_primitive[2],
                local_state_primitive[stencil_center:stencil_center],
                local_cell_weights[stencil_center:stencil_center],
            )

            # Transform the values back to prognostic state
            @unroll for k in 1:2
                primitive_to_prognostic!(
                    balance_law,
                    local_state_face_prognostic[k],
                    local_state_face_primitive[k],
                    # Use the cell auxiliary data
                    local_state_auxiliary[stencil_center],
                )
            end

            # Fill ghost cell data
            local_state_face_prognostic_neighbor .=
                local_state_face_prognostic[1]
            local_state_auxiliary[stencil_center - 1] .=
                local_state_auxiliary[stencil_center]
            numerical_boundary_flux_first_order!(
                numerical_flux_first_order,
                bctag,
                balance_law,
                local_flux,
                normal_vector,
                local_state_face_prognostic[1],
                local_state_auxiliary[stencil_center],
                local_state_face_prognostic_neighbor,
                local_state_auxiliary[stencil_center - 1],
                t,
                face_direction,
                local_state_prognostic_bottom1,
                local_state_auxiliary_bottom1,
            )

            # Fill / reset ghost cell data
            local_state_prognostic[stencil_center - 1] .=
                local_state_prognostic[stencil_center]
            local_state_gradient_flux[stencil_center - 1] .=
                local_state_gradient_flux[stencil_center]
            local_state_hyperdiffusive[stencil_center - 1] .=
                local_state_hyperdiffusive[stencil_center]
            local_state_auxiliary[stencil_center - 1] .=
                local_state_auxiliary[stencil_center]
            numerical_boundary_flux_second_order!(
                numerical_flux_second_order,
                bctag,
                balance_law,
                local_flux,
                normal_vector,
                local_state_prognostic[stencil_center],
                local_state_gradient_flux[stencil_center],
                local_state_hyperdiffusive[stencil_center],
                local_state_auxiliary[stencil_center],
                local_state_prognostic[stencil_center - 1],
                local_state_gradient_flux[stencil_center - 1],
                local_state_hyperdiffusive[stencil_center - 1],
                local_state_auxiliary[stencil_center - 1],
                t,
                local_state_prognostic_bottom1,
                local_state_gradient_flux_bottom1,
                local_state_auxiliary_bottom1,
            )

            # Compute boundary flux and add it in bottom element of the mesh
            vMI = 1 / local_cell_weights[stencil_center]
            @unroll for s in 1:num_state_prognostic
                tendency[n, s, eH + eV] -= α * sM * vMI * local_flux[s]
            end
        end

        # The rest of the elements in the stack
        # Compute flux and update for face between elements eV and eV - 1
        #    top face of eV - 1
        #    bottom face of eV
        # For the reconstruction arrays `stencil_center - 1` corresponds to `eV - 1`
        # and `stencil_center` corresponds to `eV`
        for eV in 2:nvertelem
            # shift data in storage in order to load new upper element for
            # reconstruction
            # TODO: shift pointers not data?
            @unroll for k in 1:(stencil_diameter - 1)
                local_state_prognostic[k] .= local_state_prognostic[k + 1]
                local_state_auxiliary[k] .= local_state_auxiliary[k + 1]
                local_state_gradient_flux[k] .= local_state_gradient_flux[k + 1]
                local_cell_weights[k] = local_cell_weights[k]
            end

            # Update volume mass inverse as we move up the stack of elements
            vMI[1] = vMI[2]

            # Load surface metrics for the face we will update (top face of `eV`)
            sM = sgeo[_sM, n, faces[2], eH + eV]
            normal = SVector(
                sgeo[_n1, n, face[2], eH + eV],
                sgeo[_n2, n, face[2], eH + eV],
                sgeo[_n3, n, face[2], eH + eV],
            )

            # Reconstruction for eV - 1 was computed in last time through the
            # loop, so we need to store the upper reconstructed values to
            # compute flux for this face
            local_state_face_prognostic_neighbor .=
                local_state_face_prognostic[2]

            # Next data we need to load
            eV_n = eV + stencil_center - 1

            # Assume periodic for now (will mask out below as needed for
            # boundary conditions)
            eV_n = mod1(eV_n, nvertelem)

            # get element number
            e_n = eH + eV_n

            # Load the next cell into the end of the element arrays
            load_data!(
                local_state_prognostic[stencil_diameter],
                local_state_auxiliary[stencil_diameter],
                local_state_gradient_flux[stencil_diameter],
                e_n,
            )

            # Get local volume mass matrix inverse
            local_cell_weights[stencil_diameter] = vgeo[n, _M, e_n]
            vMI[2] = 1 / local_cell_weights[stencil_center]

            # tranform the prognostic data to primitive data
            prognostic_to_primitive!(
                balance_law,
                local_state_primitive[stencil_diameter],
                local_state_prognostic[stencil_diameter],
                local_state_auxiliary[stencil_diameter],
            )

            # Do the reconstruction! for this cell and compute the values at the
            # bottom (1) and top (2) faces of element `eV`
            if periodicstack ||
               stencil_width < eV < nvertelem - stencil_width + 1
                # If we are in the interior or periodic just use the reconstruction
                reconstruction!(
                    local_state_face_primitive[1],
                    local_state_face_primitive[2],
                    local_state_primitive,
                    local_cell_weights,
                )
            elseif eV <= stencil_width
                # Bottom of the element stack requires reconstruct using a
                # subset of the elements
                # Values around stencil center that we need for this reconstruction
                rng = stencil_center .+ ((1 - eV):(eV - 1))
                reconstruction!(
                    local_state_face_primitive[1],
                    local_state_face_primitive[2],
                    local_state_primitive[rng],
                    local_cell_weights[rng],
                )
            elseif eV >= nvertelem - stencil_width + 1
                # Top of the element stack requires reconstruct using a
                # subset of the elements
                rng = stencil_center .+ ((eV - nvertelem):(nvertelem - eV))
                reconstruction!(
                    local_state_face_primitive[1],
                    local_state_face_primitive[2],
                    local_state_primitive[rng],
                    local_cell_weights[rng],
                )
            else
                # We should not hit this
                error("What happened?")
            end

            # Transform reconstructed primitive values to prognostic
            @unroll for k in 1:2
                primitive_to_prognostic!(
                    balance_law,
                    local_state_face_prognostic[k],
                    local_state_face_primitive[k],
                    # Use the cell auxiliary data
                    local_state_auxiliary[stencil_center],
                )
            end

            # TODO: Compute the flux for the bottom face of the element we are
            # considering
            if periodicstack || eV != nvertelem
                numerical_flux_first_order!(
                    numerical_flux_first_order,
                    balance_law,
                    local_flux,
                    normal_vector,
                    local_state_face_prognostic[1],
                    local_state_auxiliary[stencil_center],
                    local_state_face_prognostic_neighbor,
                    local_state_auxiliary[stencil_center - 1],
                    t,
                    face_direction,
                )

                numerical_flux_second_order!(
                    numerical_flux_second_order,
                    balance_law,
                    local_flux,
                    normal_vector,
                    local_state_prognostic[stencil_center],
                    local_state_gradient_flux[stencil_center],
                    local_state_hyperdiffusive[stencil_center],
                    local_state_auxiliary[stencil_center],
                    local_state_prognostic[stencil_center - 1],
                    local_state_gradient_flux[stencil_center - 1],
                    local_state_hyperdiffusive[stencil_center - 1],
                    local_state_auxiliary[stencil_center - 1],
                    t,
                )
            else
                # TODO: Boundary condition
                error()
            end

            # TODO: add flux into the top and bottom elements (will need to be
            # careful later if we use multiple threads per degree of freedom
            # stack to avoid race conditions and applying flux multiple times)
        end
    end
end
