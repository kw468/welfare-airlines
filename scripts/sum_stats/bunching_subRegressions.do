

/*
    This script estimates the dynamic substitution regressions that appear in
    "The Welfare Effects of Dynamic Pricing:Evidence from Airline Markets".
--------------------------------------------------------------------------------
change log:
    v0.0.1  Mon 14 Jun 2021
-------------------------------------------------------------------------------
notes:
    `stata` must be an executable.
    You might need to add a symbolic links for `stata` to work.
--------------------------------------------------------------------------------
contributors:
    Kevin:
        name:   Kevin Williams
        email:  kevin.williams@yale.edu
--------------------------------------------------------------------------------
Copyright 2021 Yale University
*/

clear

* SET THE PATHS
global INPUT = "../../data"
global OUTPUT = "../../output"

log using "${OUTPUT}subReg.log", replace

* OPEN THE MASTER FILE
insheet using "${INPUT}/subregs.csv"

egen MY = group(year ddmonth)

replace lf = lf * 100


orthpoly ttdate, deg(6) gen(_t*)


reghdfe lf _t* apd*, absorb(route MY) cluster(route MY)
eststo m1

reghdfe lf _t* apd*, absorb(route MY dowd dows) cluster(route MY)
eststo m2

reghdfe lf _t* apd*, absorb(route MY dowd dows flightnum) cluster(route MY)
eststo m3

esttab m1 m2 m3 using "${OUTPUT}subReg.tex", ///
	drop(_cons) indicate("time = _t*") se r2 label replace

log close

clear
exit
