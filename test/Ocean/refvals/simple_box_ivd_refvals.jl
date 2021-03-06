refVals = []
refPrecs = []

#! format: off
# SC ========== Test number 1 reference values and precision match template. =======
# SC ========== /home/jmc/cliMa/cliMa_update/test/Ocean/SplitExplicit/simple_box_ivd.jl test reference values ======================================
# BEGIN SCPRINT
# varr - reference values (from reference run)
# parr - digits match precision (hand edit as needed)
#
# [
#  [ MPIStateArray Name, Field Name, Maximum, Minimum, Mean, Standard Deviation ],
#  [         :                :          :        :      :          :           ],
# ]
varr = [
 [  "oce Q_3D",   "u[1]", -1.66327525125104291881e-01,  1.63223287218068308091e-01,  4.57054199141500999692e-03,  2.21420008866952226778e-02 ],
 [  "oce Q_3D",   "u[2]", -2.16296814964942768489e-01,  2.44105976363460708267e-01, -2.57418981108877616831e-03,  2.49256704872522875938e-02 ],
 [  "oce Q_3D",       :η, -8.06721661019456748321e-01,  3.47844179114868090608e-01, -3.03380420083551127861e-04,  2.92158779168222249023e-01 ],
 [  "oce Q_3D",       :θ,  3.58991077492186536763e-03,  9.91901429055192807027e+00,  2.49592172215209329167e+00,  2.17949780843674956188e+00 ],
 [   "oce aux",       :w, -1.66963788550371410252e-04,  1.64989216196191281925e-04,  5.43118005323253843601e-07,  1.44011197008218777164e-05 ],
 [   "oce aux",    :pkin, -8.84700789178211977060e+00,  0.00000000000000000000e+00, -3.26178716377690980366e+00,  2.50117471654691314598e+00 ],
 [   "oce aux",     :wz0, -2.12922215345145814233e-05,  3.08186566412866553883e-05, -1.53993475195290930982e-10,  7.82327443240865498639e-06 ],
 [   "oce aux", "u_d[1]", -1.57099271619415420398e-01,  1.22268940698781983234e-01, -7.84250724128123411372e-05,  1.15452721709468873745e-02 ],
 [   "oce aux", "u_d[2]", -2.12964763257990491452e-01,  2.30578860790304179806e-01,  2.76337873040526178152e-05,  1.75119371638010334902e-02 ],
 [   "oce aux", "ΔGu[1]", -2.33686060871737704252e-06,  3.34281015324516805469e-06, -4.07662146058456868485e-08,  3.37377005914367152159e-07 ],
 [   "oce aux", "ΔGu[2]", -2.46095864522509354306e-06,  2.23139964389116549296e-06,  1.29016375934756163675e-06,  7.46140899703298511983e-07 ],
 [   "oce aux",       :y,  0.00000000000000000000e+00,  4.00000000000000046566e+06,  2.00000000000000000000e+06,  1.15573163901915703900e+06 ],
 [ "baro Q_2D",   "U[1]", -1.77642579751613922667e+01,  6.90024701711891168543e+01,  4.64758402961474637038e+00,  1.81125164737756669808e+01 ],
 [ "baro Q_2D",   "U[2]", -3.48668457221064898022e+01,  8.56043954389619301537e+01, -2.60107324488959568143e+00,  1.80557567005594421516e+01 ],
 [ "baro Q_2D",       :η, -8.06729084084832237522e-01,  3.47845108442749906263e-01, -3.03371430044830942430e-04,  2.92173862421732766226e-01 ],
 [  "baro aux",  "Gᵁ[1]", -3.34281015324516816989e-03,  2.33686060871737691716e-03,  4.07662146058456834074e-05,  3.37393707332951826861e-04 ],
 [  "baro aux",  "Gᵁ[2]", -2.23139964389116544213e-03,  2.46095864522509347530e-03, -1.29016375934756159609e-03,  7.46177836457347174598e-04 ],
 [  "baro aux", "U_c[1]", -1.77641349600227265171e+01,  6.90014815427443437557e+01,  4.64756464081321674087e+00,  1.81123414932683921563e+01 ],
 [  "baro aux", "U_c[2]", -3.48658691554727866446e+01,  8.56044062434442594167e+01, -2.60101037971774573521e+00,  1.80557220201615180599e+01 ],
 [  "baro aux",     :η_c, -8.06721661019456748321e-01,  3.47844179114868090608e-01, -3.03380420083561807253e-04,  2.92173242116136766544e-01 ],
 [  "baro aux", "U_s[1]", -1.77642579751613922667e+01,  6.90024701711891168543e+01,  4.64758402961474637038e+00,  1.81125164737756669808e+01 ],
 [  "baro aux", "U_s[2]", -3.48668457221064898022e+01,  8.56043954389619301537e+01, -2.60107324488959568143e+00,  1.80557567005594421516e+01 ],
 [  "baro aux",     :η_s, -8.06729084084832237522e-01,  3.47845108442749906263e-01, -3.03371430044830942430e-04,  2.92173862421732766226e-01 ],
 [  "baro aux",  "Δu[1]", -4.51957131810988945505e-04,  3.99130692959488437947e-04, -1.09363921539353876238e-05,  1.02327771113594037156e-04 ],
 [  "baro aux",  "Δu[2]", -2.42568998424025800446e-04,  3.82991529986384212410e-04,  5.18845385039475763891e-05,  7.47269801857268440920e-05 ],
 [  "baro aux",  :η_diag, -8.06666755113148337131e-01,  3.47953196151883525911e-01, -3.02575448959868274160e-04,  2.92168272060050748795e-01 ],
 [  "baro aux",      :Δη, -5.84450730272745300198e-04,  7.45283859099332701703e-04, -8.04971123704421912570e-07,  1.64194305918131369573e-04 ],
 [  "baro aux",       :y,  0.00000000000000000000e+00,  4.00000000000000046566e+06,  2.00000000000000000000e+06,  1.15578885204060329124e+06 ],
]
parr = [
 [  "oce Q_3D",   "u[1]",    12,    12,    12,    12 ],
 [  "oce Q_3D",   "u[2]",    12,    12,    12,    12 ],
 [  "oce Q_3D",       :η,    12,    12,     8,    12 ],
 [  "oce Q_3D",       :θ,    12,    12,    12,    12 ],
 [   "oce aux",       :w,    12,    12,     8,    12 ],
 [   "oce aux",    :pkin,    12,    12,    12,    12 ],
 [   "oce aux",     :wz0,    12,    12,     8,    12 ],
 [   "oce aux", "u_d[1]",    12,    12,    12,    12 ],
 [   "oce aux", "u_d[2]",    12,    12,    12,    12 ],
 [   "oce aux", "ΔGu[1]",    12,    12,    12,    12 ],
 [   "oce aux", "ΔGu[2]",    12,    12,    12,    12 ],
 [   "oce aux",       :y,    12,    12,    12,    12 ],
 [ "baro Q_2D",   "U[1]",    12,    12,    12,    12 ],
 [ "baro Q_2D",   "U[2]",    12,    12,    12,    12 ],
 [ "baro Q_2D",       :η,    12,    12,     8,    12 ],
 [  "baro aux",  "Gᵁ[1]",    12,    12,    12,    12 ],
 [  "baro aux",  "Gᵁ[2]",    12,    12,    12,    12 ],
 [  "baro aux", "U_c[1]",    12,    12,    12,    12 ],
 [  "baro aux", "U_c[2]",    12,    12,    12,    12 ],
 [  "baro aux",     :η_c,    12,    12,     8,    12 ],
 [  "baro aux", "U_s[1]",    12,    12,    12,    12 ],
 [  "baro aux", "U_s[2]",    12,    12,    12,    12 ],
 [  "baro aux",     :η_s,    12,    12,     8,    12 ],
 [  "baro aux",  "Δu[1]",    12,    12,    12,    12 ],
 [  "baro aux",  "Δu[2]",    12,    12,    12,    12 ],
 [  "baro aux",  :η_diag,    12,    12,     8,    12 ],
 [  "baro aux",      :Δη,    12,    12,     8,    12 ],
 [  "baro aux",       :y,    12,    12,    12,    12 ],
]
# END SCPRINT
# SC ====================================================================================

    append!(refVals ,[ varr ] )
    append!(refPrecs,[ parr ] )

#! format: on
