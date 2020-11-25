##### Get prognostic variable list

import ..BalanceLaws: prognostic_vars, all_prognostic_vars

prognostic_vars(::DryModel) = ()
prognostic_vars(::NoTurbConv) = ()
prognostic_vars(::EquilMoist) = (TotalMoisture(),)

prognostic_vars(::NonEquilMoist) =
    (TotalMoisture(), LiquidMoisture(), IceMoisture())

prognostic_vars(::NoPrecipitation) = ()
prognostic_vars(::RainModel) = (Rain(),)

prognostic_vars(m::AtmosModel) =
    (Mass(), Momentum(), Energy())

all_prognostic_vars(m::AtmosModel) =
    (prognostic_vars(m)...,
     prognostic_vars(m.moisture)...,
     prognostic_vars(m.precipitation)...,
     prognostic_vars(m.turbconv)...,
     )
