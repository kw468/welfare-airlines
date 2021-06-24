clear

/* 	This script estimates the dynamic substitution regressions that appear in 
	Williams (2021)
	Edited by: Kevin Williams
	Edit date:6/15/2021
*/

* SET THE PATHS
global pathIn = "/home/kw468/Projects/airlines_jmp/output/"

global regTable = "${pathIn}subReg.tex"

log using "${pathIn}subReg.log", replace

* OPEN THE MASTER FILE
insheet using "$pathIn/subregs.csv"		

egen MY = group(year ddmonth)

replace lf = lf*100


orthpoly ttdate, deg(6) gen(_t*)


reghdfe lf _t* apd*, absorb(route MY) cluster(route MY)
eststo m1

reghdfe lf _t* apd*, absorb(route MY dowd dows) cluster(route MY)
eststo m2

reghdfe lf _t* apd*, absorb(route MY dowd dows flightnum) cluster(route MY)
eststo m3

esttab m1 m2 m3 using $regTable, drop(_cons) indicate("time = _t*")  se r2 label replace

log close

clear
exit
