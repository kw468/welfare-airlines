/*
    This script takes the raw UA.com data
	and extracts columns of the data for analysis
--------------------------------------------------------------------------------
change log:
    v0.0.1  Mon 14 Jun 2021
-------------------------------------------------------------------------------
notes: 

--------------------------------------------------------------------------------
contributors:
    Kevin:
        name:       Kevin Williams
        email:      kevin.williams@yale.edu
--------------------------------------------------------------------------------
Copyright 2021 Yale University
*/

clear all
set more off

* SET THE PATHS
global INPUT = "../../data"
global OUTPUT = "../../data"

* OPEN THE MAIN DATA - REPORTED NUMBERS
insheet using "$INPUT/NUMS_result.txt"

* RENAME VARS
rename v1 ddate_y
rename v2 ddate_m
rename v3 ddate_d
rename v4 origin
rename v5 dest
rename v6 fnum
rename v7 sdate_y
rename v8 sdate_m
rename v9 sdate_d
rename v12 cap
rename v13 book
rename v14 check
* DROP VARS
drop v10 v11

* SAVE FOR NOW (WILL USE LATER)
tempfile nums
save `nums'

* OPEN THE MAIN DATA - SEAT MAP NUMBERS
insheet using "$INPUT/SM_result.txt", clear

* RENAME VARS
rename v1 ddate_y
rename v2 ddate_m
rename v3 ddate_d
rename v4 origin
rename v5 dest
rename v6 fnum
rename v7 sdate_y
rename v8 sdate_m
rename v9 sdate_d
rename v12 avail
rename v13 occ
* DROP VARS
drop v10 v11

* JOIN SEAT MAP AND REPORTED NUMBERS
joinby ddate* origin dest fnum sdate* using `nums', unmatched(both)

* KEEP ONLY MERGED RESULTS
keep if _merge == 3
drop _merge

* PROCESS THE DATA - EXTRACT BLOCKED NUMBERS FROM THE SEAT MAPS
replace cap = "" if cap == "-" | cap == ":"
destring cap, replace

split book, gen(bk) p("(+")
split bk2, gen(blocked) p(" blocked)")
rename blocked1 blocked
drop bk2 book

replace bk1 = "" if bk1 == "-"
rename bk1 booked

replace blocked = "0" if blocked == ""
destring blocked booked, replace

gen result = booked + blocked

gen capsm = avail + occ
gen blocked_from_sm = cap - capsm

gen dif1 = booked - occ //naive seat map numbers
gen dif2 = result - occ - blocked_from_sm //taking into account blocked seat maps

gen pc1 = dif1 / cap // calculate pct dif 1
gen pc2 = dif2 / cap // calculate pct dif 2

gen ddate = mdy(ddate_m, ddate_d, ddate_y) // create dep date var
gen sdate = mdy(sdate_m, sdate_d, sdate_y) // create search date var
gen ttdate = sdate - ddate // gen ttdate

egen itin = group(origin dest fnum ddate) // create itinerary var

* DROP MISSING VALUES
drop if pc1 == .
drop if pc2 == .

* EXPORT DATA
export delimited itin ttdate pc2 pc1 occ cap using "$OUTPUT/united_data.csv", replace
