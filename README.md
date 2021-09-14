# The Welfare Effects of Dynamic Pricing: Evidence from Airline Markets
by Kevin Williams

# Description
This respository contains replication code for "The Welfare Effects of Dynamic Pricing: Evidence from Airline Markets."

Step 1: Unzip the JSON files into data

Step 2: Execute data cleaning using extract_AS_data and extract_EF_data in scripts/data_processing 

Step 3: Estimate dynamic model using estim_model_jax_multiFE_AS and estim_model_jax_multiFE_EF with relaxed bounds using scripts in scripts/estimation 

Step 4: Confirm estimates using estim_model_robust

Step 5: Execute counterfactuals, main_counterfactual, cf_results, price_freq_counterfactual, stochastic_limit_counterfactual, cf_sl_results

Step 6: To recreate all tables and figures, execute all scripts in the plots (scripts/plots/) and sum_stats (scripts/sum_stats) folders.

Computation calls knitro and gpu-enabled jax. A license and the callable python api of knitro can be obtained Artleys. The programs expect the correct cuda drivers are installed so that function calls are sent to local nividia GPUs using the jax callable api.
