# EOS-CG-MSIBI
This repo contains scripts and target states used to create potentials for CER EOS CG beads

1. VERSIONS OF THE MSIBI CODE:

run_msibi_v1: this version of the code runs all the pairs present in all systems (pair must be present in all systems). The initial guesses for these to be optimised pairs is parash's .txt files (his v20.6 txt files)

run_msibi_v1.2: This version of the code is similar to v1 apart from the initial guesses being taken from v18 txt files (the ones we normally use). Initial guesses for ESTER beads are taken from AMIDE, taild and tail2 from tail, and STICKY from OH2.

run_msibi_v2: This version of the code runs only the pairs present in a particular stage(s). There is also an option to exclude (not optimise) a pair even if its present in the currently looked at stage. The initial guesses for these to be optimised pairs is parash's .txt files (his v20.6 txt files).

run_msibi_v2.2: This version of the code is similar to v2 apart from the initial guesses being taken from v18 txt files (the ones we normally use). Initial guesses for EST
ER beads are taken from AMIDE, taild and tail2 from tail, and STICKY from OH2.

run_msibi_v3: This version added stage wise updates + made sure that if stage >=2 was being optimised, the already optimised pair potentials from the previous stages were used. Discontinued since the experimental build this version was running on was changed.

run_msibi_v4: Run on an experimental msibi build (new rdfs). Same as v3 but with the addition of pair.set_state_params(
                    state=s,
                    optimize_against=False,
                    #exclude_all_bonded = True,
                    exclude_bond_depth = 3,
                )

Allows us to set a pair to not be optimised by a particular state + instead of excluding all bonded interactions (which in turn did not take any interaction from the same molecule into consideration), a bond depth was added, where only atoms +-x from the bead type were included in calculating the rdfs.

run_msibi_v5: Runs on the main build. Tries to copy whats been done in v4 with those v4 features added in script instead of msibi directly.

2. SUBMIT SCRIPTS:

2 submit scripts are present, one using the cpu and one the gpu. CPU is only used for TESTING PURPOSES, and not an actual run since it is quite slow.

3. target_states:

Above folder contains the CG mapped atomistic target states (all 7 of them) from v2.2 of the cer EOS model from Parash's dissertation chp 8 (table 8.4).

4. Forcefields

v18 forcefields are stored in the cg folder while Parash's v20.6 forcefields are stored in the eosFF_6 folder.
