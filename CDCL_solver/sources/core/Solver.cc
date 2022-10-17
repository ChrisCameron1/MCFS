/***************************************************************************************[Solver.cc]
MiniSat -- Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
           Copyright (c) 2007-2010, Niklas Sorensson
 
Chanseok Oh's MiniSat Patch Series -- Copyright (c) 2015, Chanseok Oh
 
Maple_LCM, Based on MapleCOMSPS_DRUP -- Copyright (c) 2017, Mao Luo, Chu-Min LI, Fan Xiao: implementing a learnt clause minimisation approach
Reference: M. Luo, C.-M. Li, F. Xiao, F. Manya, and Z. L. , “An effective learnt clause minimization approach for cdcl sat solvers,” in IJCAI-2017, 2017, pp. to–appear.

Maple_LCM_Dist, Based on Maple_LCM -- Copyright (c) 2017, Fan Xiao, Chu-Min LI, Mao Luo: using a new branching heuristic called Distance at the beginning of search

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 **************************************************************************************************/

#include <torch/script.h>
//#include <torch/torch.h>

#include <float.h>
#include <math.h>
//#include <pair.h>
#include <random>
#include <signal.h>
#include <string.h>
#include <regex>
//#include <algorithm.h>
#include <unistd.h>
#include <map>
#include <sys/stat.h>
//#include <utility>

#include <chrono> ///looking for bottlenecks

//#include "Python.h"

#include "core/Solver.h"
#include "mtl/Sort.h"

#include "utils/System.h"
#include "core/Dimacs.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <functional> //for std::hash
#include <time.h>


//#include <string>

using namespace Minisat;
//using namespace std;

#ifdef BIN_DRUP
int Solver::buf_len = 0;
unsigned char Solver::drup_buf[2 * 1024 * 1024];
unsigned char* Solver::buf_ptr = drup_buf;
#endif

//=================================================================================================
// Options:

static const char*_cat = "CORE";

// Base solver pararams
static DoubleOption opt_step_size(_cat, "step-size", "Initial step size", 0.40, DoubleRange(0, false, 1, false));
static DoubleOption opt_step_size_dec(_cat, "step-size-dec", "Step size decrement", 0.000001, DoubleRange(0, false, 1, false));
static DoubleOption opt_min_step_size(_cat, "min-step-size", "Minimal step size", 0.06, DoubleRange(0, false, 1, false));
static DoubleOption opt_var_decay(_cat, "var-decay", "The variable activity decay factor", 0.80, DoubleRange(0, false, 1, false));
static DoubleOption opt_clause_decay(_cat, "cla-decay", "The clause activity decay factor", 0.999, DoubleRange(0, false, 1, false));
static DoubleOption opt_random_var_freq(_cat, "rnd-freq", "The frequency with which the decision heuristic tries to choose a random variable", 0, DoubleRange(0, true, 1, true));
static DoubleOption opt_random_seed(_cat, "rnd-seed", "Used by the random variable selection", 91648253, DoubleRange(0, false, HUGE_VAL, false));
static IntOption opt_ccmin_mode(_cat, "ccmin-mode", "Controls conflict clause minimization (0=none, 1=basic, 2=deep)", 2, IntRange(0, 2));
static IntOption opt_phase_saving(_cat, "phase-saving", "Controls the level of phase saving (0=none, 1=limited, 2=full)", 2, IntRange(0, 2));
static BoolOption opt_rnd_init_act(_cat, "rnd-init", "Randomize the initial activity", false);
static IntOption opt_restart_first(_cat, "rfirst", "The base restart interval", 100, IntRange(1, INT32_MAX));
static DoubleOption opt_restart_inc(_cat, "rinc", "Restart interval increase factor", 2, DoubleRange(1, false, HUGE_VAL, false));
static DoubleOption opt_garbage_frac(_cat, "gc-frac", "The fraction of wasted memory allowed before a garbage collection is triggered", 0.20, DoubleRange(0, false, HUGE_VAL, false));

static IntOption opt_decisions_cutoff(_cat, "decisions_cutoff", "How many decisions before terminating", INT32_MAX, IntRange(2, INT32_MAX));
static BoolOption opt_conflict_driven_search(_cat, "conflict-driven-search", "Do Conflict Driven Search or not?", false);
static IntOption opt_aux_solver_cutoff(_cat, "aux-solver-cutoff", "How many times do we use Auxiliary Solver to update the table", 1e5);
static BoolOption opt_inquire_tree(_cat, "inquire-tree", "Inquire tree search distribution or not?", false);
static BoolOption opt_cleanup_datatable(_cat, "cleanup-datatable", "Clean up data table or not", false); 
static BoolOption opt_use_qd(_cat, "use-qd", "Initialize value function to Qd or not?", false);
static StringOption opt_cnf_hash(_cat, "cnf-hash", "String of Instance CNF Hash", "default");
static BoolOption opt_params_tuning(_cat, "params-tuning", "Use Parameters Tuning debugging or not", false);

// Monte Carlo Tree Search params:
static BoolOption opt_dpll(_cat, "dpll", "Run as a DPLL solver (no learnt clausses, chronological backtracking)", true);
static DoubleOption opt_prop_mcts(_cat, "prop-mcts", "Weight of mcts probabilities vs nn", 1, DoubleRange(0, true, 1, true));
static StringOption opt_external_solver_executable(_cat, "external-solver-executable", "Path to external subsolver executable", "");
static StringOption opt_rollout_policy(_cat, "rollout-policy", "type of rollout policy: options: {FULL_SUBTREE, VALUE_NET}", "VALUE_NET");
static BoolOption opt_use_node_lookup(_cat, "use-node-lookup","Whether nodes with the same variable state share lookahead data", true);
static BoolOption opt_empty_child_table(_cat, "empty-child-table","Whether sub-searches inherit the parent's data table", false);
static BoolOption opt_do_monte_carlo_tree_search(_cat, "do-monte-carlo-tree-search", "Whether or not to do Monte Carlo tree search for variable branching", false);
static IntOption opt_num_monte_carlo_tree_search_samples(_cat, "num-monte-carlo-tree-search-samples", "Number of MCTS samples per node", 2, IntRange(1, 10000000));
static IntOption opt_cutoff_time(_cat, "cutoff-time", "Maximum Monte Carlo tree search depth terminating with a capped run", INT32_MAX, IntRange(2, INT32_MAX));
static BoolOption opt_refresh_lookahead_tree(_cat, "refresh-lookahead-tree", "Whether to refresh lookahead tree after taking action", false);
static BoolOption opt_subproblem_termination(_cat, "subproblem-termination", "Whether to terminate lookaheads when subproblem is proved UNSAT", false);
static IntOption opt_depth_before_rollout(_cat, "depth-before-rollout", "Depth of lookahead before rolling out with current policy defined by model", INT32_MAX, IntRange(1, INT32_MAX));
static IntOption opt_num_variables_for_rollout(_cat, "num-variables-for-rollout", "Number of variables below which roll out with current policy defined by model or subsolver", 2, IntRange(0, INT32_MAX));
static BoolOption opt_dynamic_node_expansion(_cat, "dynamic-node-expansion", "Expand a MCTS node only if you have been there before.", false);
static IntOption opt_solution_cache_size(_cat, "solution-cache-size", "Size of solution cache", 10000000, IntRange(1, INT32_MAX));
static BoolOption opt_terminate_mcts_path_at_conflict(_cat, "terminate-mcts-path-at-conflict", "Terminate sequence of MCTS decisions (outer loop) when reach a conflict. Otherwise wait until problem solver.", true);
static DoubleOption opt_value_error_threshold(_cat, "value-error-threshold", "Threshold for multiplicative value error at leaf nodes. Half prob of subsolver call as error halves", 1.0, DoubleRange(0, true, 100, true));
static BoolOption opt_do_importance_sampling(_cat, "do-importance-sampling", "Whether to do importance sampling for selecting T/F assignment for variable", false);

//Bandit Policy parameters
static StringOption opt_bandit_policy(_cat, "bandit-policy", "Bandit Policy for MCTS", "Knuth");
static BoolOption opt_normalize_rewards(_cat, "normalize-rewards", "Whether to project rewards onto -1,1", true);
static BoolOption opt_inverse_objective(_cat, "inverse-objective", "Whether to optimize for the inverse objectve (1/decision count)", false); 
static BoolOption opt_min_knuth_tree_estimate(_cat, "min-knuth-tree-estimate", "Whether to optimize for knuth's tree size estimate", true);
static IntOption opt_mcts_max_decisions(_cat, "mcts-max-decisions", "How many decisions before stop using MCTS policy",  INT32_MAX, IntRange(1, INT32_MAX));
static DoubleOption opt_failure_prob(_cat, "failure-prob", "Failure probability of Knuth estimate lower bound", 0.05, DoubleRange(0, true, 1, true));
static IntOption opt_mcts_samples_per_lb_update(_cat, "mcts-samples-per-lb-update", "Number of samples between lower bound updates", 1, IntRange(1, INT32_MAX));
static DoubleOption opt_random_action_prob(_cat, "random-action-prob", "Probability of taking random action", 0.05, DoubleRange(0, true, 1, true));
static BoolOption opt_chernoff_level_bounds(_cat, "chernoff-level-bounds", "Whether to use Chernoff levels for finding tree lower bound", false); 
static DoubleOption opt_BETA(_cat, "BETA", "Drift constant to stretch confidence interval", 0.1, DoubleRange(0, true, 1000, true));
static BoolOption opt_fix_mcts_policy_to_prior(_cat, "fix-mcts-policy-to-prior", "Whether to fix MCTS policy to prior", false);

// Experimental setup params
static StringOption opt_log_bandit(_cat, "log-bandit", "Log prior, q, u, or qu during MCTS", "None");
static StringOption opt_data_dir(_cat, "data-dir", "Data directory for saving MCTS data", "./");
static StringOption opt_experiment_name(_cat, "experiment-name", "Name used to fetch the NN model and save datapoints for predictions; no saving by default", "test");
static StringOption opt_model_filename(_cat, "model-path", "Path to torch neural network", "");
static StringOption opt_torch_device(_cat, "torch-device", "Path to torch neural network", "CUDA"); 

// Branching params
static BoolOption opt_use_neural_branching(_cat, "use-neural-branching", "Use neural network predictions for variable branching", true);
static BoolOption opt_use_bsh_branching(_cat, "use-bsh-branching", "Use bsh heuristic for branching (that used in kcnfs)", false);
static BoolOption opt_use_random_branching(_cat, "use-random-branching", "Randomly select variables to branch on", false);


//Neural Net prior params:
static IntOption opt_neural_net_prior_depth(_cat, "neural-net-prior-depth", "How deep to use neural net prior in lookahead tree", INT32_MAX, IntRange(2, INT32_MAX));
static IntOption opt_neural_net_refresh_rate(_cat, "neural-net-refresh-rate", "How often to update neural net prediction", 1, IntRange(1, 10000000));
static IntOption opt_neural_net_depth_threshold(_cat, "neural-net-depth-threshold", "Depth at which to never query network below", INT32_MAX, IntRange(1, INT32_MAX));
static BoolOption opt_use_neural_net_prior(_cat, "use-neural-net-prior", "Whether to use neural net prior for MCTS lookaheads", true);
static BoolOption opt_uniform_random_ucb(_cat, "uniform-random-ucb", "Whether to use random branching in UCB", false);
static DoubleOption opt_prior_temp(_cat, "prior-temp", "Temperature for soft max of prior. Less than 1, harden. More than 1, soften", 1, DoubleRange(0, true, 10000, true));

//Saving out intermediate states
static BoolOption opt_save_solver_state(_cat, "save-solver-state", "Whether to save solver state after each decision", false);
static BoolOption opt_save_subsolver_rollout_calls(_cat, "save-subsolver-rollout-calls", "Whether to save solver state after each subsolver rollout", false);
//Database options:
static BoolOption opt_use_mcts_db(_cat, "use-mcts-db", "Send MCTS data to arrowdb instead of storing locally", false);


//=================================================================================================
// Constructor/Destructor:


Solver::Solver() :

//TODO: How do we know which type all of these are?? All of these attributes are typed in Solver.h

// Parameters (user settable):
//
drup_file(NULL)
, verbosity(0)
, step_size(opt_step_size)
, step_size_dec(opt_step_size_dec)
, min_step_size(opt_min_step_size)
, timer(5000)
, var_decay(opt_var_decay)
, clause_decay(opt_clause_decay)
, random_var_freq(opt_random_var_freq)
, random_seed(opt_random_seed)
, VSIDS(false)
, ccmin_mode(opt_ccmin_mode)
, phase_saving(opt_phase_saving)
, rnd_pol(false)
, rnd_init_act(opt_rnd_init_act)
, garbage_frac(opt_garbage_frac)
, restart_first(opt_restart_first)
, restart_inc(opt_restart_inc)

// Parameters (the rest):
//
, learntsize_factor((double) 1 / (double) 3)
, learntsize_inc(1.1)

// Parameters (experimental):
//
, learntsize_adjust_start_confl(100)
, learntsize_adjust_inc(1.5)

// Conflict Driven Search addition 
, conflict_driven_search(opt_conflict_driven_search)
, aux_solver_cutoff(opt_aux_solver_cutoff)
, count_aux_solves(0)

// Inquire Tree addition
, inquire_tree(opt_inquire_tree)
, cleanup_datatable(opt_cleanup_datatable)
, use_qd(opt_use_qd)
, cnf_hash(opt_cnf_hash)
, params_tuning(opt_params_tuning)
// Monte Carlo additions:
//
, dpll(opt_dpll)
, experiment_name(opt_experiment_name)
, use_neural_branching(opt_use_neural_branching)
, use_bsh_branching(opt_use_bsh_branching)
, use_random_branching(opt_use_random_branching)
, do_monte_carlo_tree_search(opt_do_monte_carlo_tree_search)
, subproblem_termination(opt_subproblem_termination)
, refresh_lookahead_tree(opt_refresh_lookahead_tree)
, use_node_lookup(opt_use_node_lookup)
, empty_child_table(opt_empty_child_table)
, num_monte_carlo_tree_search_samples(opt_num_monte_carlo_tree_search_samples)
, cutoff_time(opt_cutoff_time)
, data_dir(opt_data_dir)
, model_filename(opt_model_filename)
, torch_device(opt_torch_device)
, bandit_policy_class(opt_bandit_policy)
, external_solver_executable(opt_external_solver_executable)
, prop_mcts(opt_prop_mcts)
, log_bandit(opt_log_bandit)
, use_neural_net_prior(opt_use_neural_net_prior)
, neural_net_prior_depth_threshold(opt_neural_net_prior_depth)
, neural_net_refresh_rate(opt_neural_net_refresh_rate)
, neural_net_depth_threshold(opt_neural_net_depth_threshold)
, uniform_random_ucb(opt_uniform_random_ucb)
, depth_before_rollout(opt_depth_before_rollout)
, num_variables_for_rollout(opt_num_variables_for_rollout)
, save_solver_state(opt_save_solver_state)
, save_subsolver_rollout_calls(opt_save_subsolver_rollout_calls)
, inverse_objective(opt_inverse_objective)
, min_knuth_tree_estimate(opt_min_knuth_tree_estimate)
, mcts_max_decisions(opt_mcts_max_decisions)
, normalize_rewards(opt_normalize_rewards)
, decisions_cutoff(opt_decisions_cutoff)
, failure_prob(opt_failure_prob)
, mcts_samples_per_lb_update(opt_mcts_samples_per_lb_update)
, random_action_prob(opt_random_action_prob)
, chernoff_level_bounds(opt_chernoff_level_bounds)
, BETA(opt_BETA)
, use_mcts_db(opt_use_mcts_db)
, rollout_policy(opt_rollout_policy)
, dynamic_node_expansion(opt_dynamic_node_expansion)
, prior_temp(opt_prior_temp)
, solution_cache_size(opt_solution_cache_size)
, terminate_mcts_path_at_conflict(opt_terminate_mcts_path_at_conflict)
, fix_mcts_policy_to_prior(opt_fix_mcts_policy_to_prior)
, value_error_threshold(opt_value_error_threshold)
, do_importance_sampling(opt_do_importance_sampling)

// Statistics: (formerly in 'SolverStats')
//
, solves(0), starts(0), decisions(0), rnd_decisions(0), propagations(0), conflicts(0), conflicts_VSIDS(0)
, dec_vars(0), clauses_literals(0), learnts_literals(0), max_literals(0), tot_literals(0)

, ok(true)
, cla_inc(1)
, var_inc(1)
, watches_bin(WatcherDeleted(ca))
, watches(WatcherDeleted(ca))
, qhead(0)
, simpDB_assigns(-1)
, simpDB_props(0)
, order_heap_CHB(VarOrderLt(activity_CHB))
, order_heap_VSIDS(VarOrderLt(activity_VSIDS))
, order_heap_distance(VarOrderLt(activity_distance))
, progress_estimate(0)
, remove_satisfied(true)

, core_lbd_cut(3)
, global_lbd_sum(0)
, lbd_queue(50)
, next_T2_reduce(10000)
, next_L_reduce(15000)

, counter(0)

// Resource constraints:
//
, conflict_budget(-1)
, propagation_budget(-1)
, asynch_interrupt(false)

// simplfiy
, nbSimplifyAll(0)
, s_propagations(0)

// simplifyAll adjust occasion
, curSimplify(1)
, nbconfbeforesimplify(1000)
, incSimplify(1000)

, var_iLevel_inc(1)
, my_var_decay(0.6)
, DISTANCE(true)

{
    start_time = time(NULL);

    //Make data directory
    printf("Get to creating directory...");
    //exit(0);
    int check = mkdir(data_dir,0777); 
    std::string root_dir;
    root_dir.append(data_dir);
    root_dir.append("/");
    std::string str(model_filename);
    root_dir.append(getFileName(str));
    // std::string new_root_dir = "new string";
    // new_root_dir = root_dir;
    char* c = const_cast<char*>(root_dir.c_str());
    check = mkdir(c,0777);

    // Create directory for subinstances
    root_dir.append("/subproblems");
    subproblem_dir = const_cast<char*>(root_dir.c_str());
    check = mkdir(subproblem_dir,0777);

    // check if directory is created or not 
    if (check == 0) 
        printf("Directory created\n"); 
    else { 
        printf("Unable to create directory\n"); 
    } 

    //TODO: Create UCBPolicy Object
    if (strcmp(bandit_policy_class,"PUCT") == 0)
    {
        printf("%s ", bandit_policy_class);
        std::cerr << "Bandit policy not recognized.";
        throw;
    }
    else if(strcmp(bandit_policy_class,"Knuth") == 0)
    {
        banditPolicy = new KnuthSampleLowerBound(random_action_prob);
    }
    else
    {
        printf("%s ", bandit_policy_class);
        std::cerr << "Bandit policy not recognized.";
        throw;
    } 

    // Each node in the mcts tree (data_table) is a DataCentre object, indexed by the vector of assignments
    data_table = new std::unordered_map<std::vector<bool>, DataCentre *>();
    mcts_data_ids = new std::unordered_map<std::vector<bool>, int>();
    mcts_values = new std::unordered_map<std::vector<bool>, double>();
    nn_solution_cache = new LRUCache(solution_cache_size);
    rollout_cache = new std::unordered_map<std::vector<bool>, double>();
    importance_sample_ratio = new std::unordered_map<std::vector<bool>, double>();

    value_net_leaf_error = 0;
    value_net_leaf_count = 0;

    std::regex regstr("crash");

    if (strcmp(model_filename,"") != 0) {
        // try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            std::cout << "Deserializing model...\n"; 
            module = torch::jit::load(model_filename);
            std::cout << "Loaded model.\n"; 
            if (strcmp(torch_device,"CPU") == 0) {
                module.to(at::kCPU);
            } else if (strcmp(torch_device,"CUDA") == 0) {
                // if (!torch::cuda::cudnn_is_available())
                // {
                module.to(at::kCUDA);
                // } else
                // {
                //     std::cerr << "CUDA not available. Moving model to CPU\n";
                //     module.to(at::kCPU);
                // }
            }
            else {
                printf("%s ", torch_device);
                std::cerr << "Device not recognized.";
                throw;
            }
            module.eval();

            std::cout << "Model successfully loaded" << std::endl;
        // }
        // catch (torch::jit::ErrorReport e) {
        //     printf("%s ", torch_device);
        //     std::cerr << "error loading the model\n";
        //     std::cout << e.msg() << std::endl; 
        //     //throw;
        // }
    }
}

Solver::~Solver()
{ ///monte carlo references really mess this up (classic use case for reference counting)
//    delete banditPolicy;
//    if (use_node_lookup) {
//        delete lookahead_table;
//        delete probability_table;
//    }
}

Solver::Solver(const Solver& anotherSolver) :

// Parameters (user settable):
//
drup_file(anotherSolver.drup_file)
, verbosity(anotherSolver.verbosity)
, step_size(anotherSolver.step_size)
, step_size_dec(anotherSolver.step_size_dec)
, min_step_size(anotherSolver.min_step_size)
, timer(anotherSolver.timer)
, var_decay(anotherSolver.var_decay)
, clause_decay(anotherSolver.clause_decay)
, random_var_freq(anotherSolver.random_var_freq)
, random_seed(anotherSolver.random_seed)
, VSIDS(anotherSolver.VSIDS)
, ccmin_mode(anotherSolver.ccmin_mode)
, phase_saving(anotherSolver.phase_saving)
, rnd_pol(anotherSolver.rnd_pol)
, rnd_init_act(anotherSolver.rnd_init_act)
, garbage_frac(anotherSolver.garbage_frac)
, restart_first(anotherSolver.restart_first)
, restart_inc(anotherSolver.restart_inc)

// Parameters (the rest):
//
, learntsize_factor(anotherSolver.learntsize_factor)
, learntsize_inc(anotherSolver.learntsize_inc)

// Parameters (experimental):
//
, learntsize_adjust_start_confl(anotherSolver.learntsize_adjust_start_confl)
, learntsize_adjust_inc(anotherSolver.learntsize_adjust_inc)

// Conflict driven search additions:
, conflict_driven_search(anotherSolver.conflict_driven_search)
, aux_solver_cutoff(anotherSolver.aux_solver_cutoff)
, count_aux_solves(anotherSolver.count_aux_solves)

// Inquire tree additions:
, inquire_tree(anotherSolver.inquire_tree)
, cleanup_datatable(anotherSolver.cleanup_datatable)
, use_qd(anotherSolver.use_qd)
, cnf_hash(anotherSolver.cnf_hash)
, params_tuning(anotherSolver.params_tuning)
// Monte Carlo additions:
//
, dpll(anotherSolver.dpll)
, experiment_name(anotherSolver.experiment_name)
, use_neural_branching(anotherSolver.use_neural_branching)
, use_bsh_branching(anotherSolver.use_bsh_branching)
, use_random_branching(anotherSolver.use_random_branching)
, do_monte_carlo_tree_search(anotherSolver.do_monte_carlo_tree_search)
, subproblem_termination(anotherSolver.subproblem_termination)
, refresh_lookahead_tree(anotherSolver.refresh_lookahead_tree)
, use_node_lookup(anotherSolver.use_node_lookup)
, empty_child_table(anotherSolver.empty_child_table)
, num_monte_carlo_tree_search_samples(anotherSolver.num_monte_carlo_tree_search_samples)
, cutoff_time(anotherSolver.cutoff_time)
, data_dir(anotherSolver.data_dir)
, subproblem_dir(anotherSolver.subproblem_dir)
, model_filename(anotherSolver.model_filename)
, torch_device(anotherSolver.torch_device)
, bandit_policy_class(anotherSolver.bandit_policy_class)
, external_solver_executable(anotherSolver.external_solver_executable)
, prop_mcts(anotherSolver.prop_mcts)
, log_bandit(anotherSolver.log_bandit)
, use_neural_net_prior(anotherSolver.use_neural_net_prior)
, neural_net_prior_depth_threshold(anotherSolver.neural_net_prior_depth_threshold)
, neural_net_refresh_rate(anotherSolver.neural_net_refresh_rate)
, neural_net_depth_threshold(anotherSolver.neural_net_depth_threshold)
, uniform_random_ucb(anotherSolver.uniform_random_ucb)
, depth_before_rollout(anotherSolver.depth_before_rollout)
, num_variables_for_rollout(anotherSolver.num_variables_for_rollout)
, save_solver_state(anotherSolver.save_solver_state)
, save_subsolver_rollout_calls(anotherSolver.save_subsolver_rollout_calls)
, inverse_objective(anotherSolver.inverse_objective)
, min_knuth_tree_estimate(anotherSolver.min_knuth_tree_estimate)
, mcts_max_decisions(anotherSolver.mcts_max_decisions)
, normalize_rewards(anotherSolver.normalize_rewards)
, decisions_cutoff(anotherSolver.decisions_cutoff)
, failure_prob(anotherSolver.failure_prob)
, mcts_samples_per_lb_update(anotherSolver.mcts_samples_per_lb_update)
, random_action_prob(anotherSolver.random_action_prob)
, chernoff_level_bounds(anotherSolver.chernoff_level_bounds)
, BETA(anotherSolver.BETA)
, use_mcts_db(anotherSolver.use_mcts_db)
, rollout_policy(anotherSolver.rollout_policy)
, dynamic_node_expansion(anotherSolver.dynamic_node_expansion)
, prior_temp(anotherSolver.prior_temp)
, solution_cache_size(anotherSolver.solution_cache_size)
, terminate_mcts_path_at_conflict(anotherSolver.terminate_mcts_path_at_conflict)
, fix_mcts_policy_to_prior(anotherSolver.fix_mcts_policy_to_prior)
, value_error_threshold(anotherSolver.value_error_threshold)
, do_importance_sampling(anotherSolver.do_importance_sampling)
//
, solves(anotherSolver.solves)
, starts(anotherSolver.starts)
, decisions(0)//anotherSolver.decisions)
, rnd_decisions(anotherSolver.rnd_decisions)
, propagations(anotherSolver.propagations)
, conflicts(anotherSolver.conflicts)
, conflicts_VSIDS(anotherSolver.conflicts_VSIDS)
, dec_vars(anotherSolver.dec_vars)
, clauses_literals(anotherSolver.clauses_literals)
, learnts_literals(anotherSolver.learnts_literals)
, max_literals(anotherSolver.max_literals)
, tot_literals(anotherSolver.tot_literals)

// Solver state:
, ok(anotherSolver.ok)
, cla_inc(anotherSolver.cla_inc)
, var_inc(anotherSolver.var_inc)
, watches_bin(WatcherDeleted(ca))
, watches(WatcherDeleted(ca))
, qhead(anotherSolver.qhead)
, simpDB_assigns(anotherSolver.simpDB_assigns)
, simpDB_props(anotherSolver.simpDB_props)
, order_heap_CHB(VarOrderLt(anotherSolver.activity_CHB))
, order_heap_VSIDS(VarOrderLt(anotherSolver.activity_VSIDS))
, order_heap_distance(VarOrderLt(anotherSolver.activity_distance))
, progress_estimate(anotherSolver.progress_estimate)
, remove_satisfied(anotherSolver.remove_satisfied)

, core_lbd_cut(anotherSolver.core_lbd_cut)
, global_lbd_sum(anotherSolver.global_lbd_sum)
, lbd_queue(50)
, next_T2_reduce(anotherSolver.next_T2_reduce)
, next_L_reduce(anotherSolver.next_L_reduce)

, counter(anotherSolver.counter)
, max_learnts(anotherSolver.max_learnts)
, learntsize_adjust_confl(anotherSolver.learntsize_adjust_confl)
, learntsize_adjust_cnt(anotherSolver.learntsize_adjust_cnt)

// Resource constraints:
//
, conflict_budget(anotherSolver.conflict_budget)
, propagation_budget(anotherSolver.propagation_budget)
, asynch_interrupt(anotherSolver.asynch_interrupt)

// simplfiy
, trailRecord(anotherSolver.trailRecord)
, nbSimplifyAll(anotherSolver.nbSimplifyAll)
, simplified_length_record(anotherSolver.simplified_length_record)
, original_length_record(anotherSolver.original_length_record)
, s_propagations(anotherSolver.s_propagations)

// simplifyAll adjust occasion
, curSimplify(anotherSolver.curSimplify)
, nbconfbeforesimplify(anotherSolver.nbconfbeforesimplify)
, incSimplify(anotherSolver.incSimplify)

, nbcollectfirstuip(anotherSolver.nbcollectfirstuip)
, nblearntclause(anotherSolver.nblearntclause)
, nbDoubleConflicts(anotherSolver.nbDoubleConflicts)
, nbTripleConflicts(anotherSolver.nbTripleConflicts)
, uip1(anotherSolver.uip1)
, uip2(anotherSolver.uip2)
, previousStarts(anotherSolver.previousStarts)

, var_iLevel_inc(anotherSolver.var_iLevel_inc)
, my_var_decay(anotherSolver.my_var_decay)
, DISTANCE(anotherSolver.DISTANCE)

, value_net_leaf_error(anotherSolver.value_net_leaf_error)
, value_net_leaf_count(anotherSolver.value_net_leaf_count)

{
    start_time = time(NULL);
    module = anotherSolver.module; // immutable so just set a pointer to module
    // if (strcmp(model_filename,"") != 0)
    // {
    //     try {
    //         // Deserialize the ScriptModule from a file using torch::jit::load().
    //         module = torch::jit::load(model_filename);
    //         if (strcmp(torch_device,"CPU") == 0)
    //         {
    //             module.to(at::kCPU);
    //         } else if (strcmp(torch_device,"CUDA") == 0)
    //         {
    //             module.to(at::kCUDA);
    //         }
    //         else
    //         {
    //             printf("%s ", torch_device);
    //             std::cerr << "Device not recognized.";
    //             throw;
    //         }
    //         module.eval();

    //         //std::cout << "Model successfully loaded" << std::endl;
    //     }
    //     catch (const c10::Error& e) {
    //         printf("%s ", torch_device);
    //         std::cerr << "error loading the model\n";
    //         throw;
    //     }
    // }


    //TODO: Create UCBPolicy Object
    if (strcmp(bandit_policy_class,"PUCT") == 0)
    {
        printf("%s ", bandit_policy_class);
        std::cerr << "Bandit policy not recognized.";
        throw;
    } 
    else if(strcmp(bandit_policy_class,"Knuth") == 0)
    {
        banditPolicy = new KnuthSampleLowerBound(random_action_prob);
    }
    else
    {
        printf("%s ", bandit_policy_class);
        std::cerr << "Bandit policy not recognized.";
        throw;
    }
    
    // note this is a pointer to the origional table, so data will be accumulated across searches (feature or bug?)
    if (anotherSolver.use_node_lookup) { ///note in this version node lookup is on by default
        if (anotherSolver.empty_child_table) {
            data_table = new std::unordered_map<std::vector<bool>, DataCentre *>();
        } else {
            data_table = anotherSolver.data_table;
        }
    }
    mcts_data_ids = anotherSolver.mcts_data_ids;
    mcts_values = anotherSolver.mcts_values;

    nn_solution_cache = anotherSolver.nn_solution_cache;
    rollout_cache = anotherSolver.rollout_cache;
    importance_sample_ratio = anotherSolver.importance_sample_ratio;


    /* Copy Header file */

    // Extra results: (read-only member variable)
    //
    //vec<lbool> model;             // If problem is satisfiable, this vector contains the model (if any).
    //printf("Gets in solver copy constructor\n");
    //model.growTo(anotherSolver.model.size());
    for (int i = 0; i < anotherSolver.model.size(); i++) {
        model.push(anotherSolver.model[i]);
    }
    //printf("Gets past first copy\n");

    // vec<lbool> *model = new vec<lbool>(anotherSolver.model.size());
    // model = new vec<lbool>(anotherSolver.model.size());
    // model;
    // //&(model) = new vec<lbool>(anotherSolver.model.size());
    // anotherSolver.model.copyTo(model);

    //vec<Lit>   conflict;          // If problem is unsatisfiable (possibly under assumptions),
    //conflict.growTo(anotherSolver.conflict.size());
    for (int i = 0; i < anotherSolver.conflict.size(); i++) {
        conflict.push(anotherSolver.conflict[i]);
    }
    // This vector represent the final conflict clause expressed in the assumptions.

    // Statistics: (read-only member variable)
    //
    //vec<uint32_t> picked;
    //picked.growTo(anotherSolver.picked.size());
    for (int i = 0; i < anotherSolver.picked.size(); i++) {
        picked.push(anotherSolver.picked[i]);
    }

    //vec<uint32_t> conflicted;
    //conflicted.growTo(anotherSolver.conflicted.size());
    for (int i = 0; i < anotherSolver.conflicted.size(); i++) {
        conflicted.push(anotherSolver.conflicted[i]);
    }

    //vec<uint32_t> almost_conflicted;
    //almost_conflicted.growTo(anotherSolver.almost_conflicted.size());
    for (int i = 0; i < anotherSolver.almost_conflicted.size(); i++) {
        almost_conflicted.push(anotherSolver.almost_conflicted[i]);
    }

    //TODO: Not sure about these ifdefs 
    // #ifdef ANTI_EXPLORATION
    //     vec<uint32_t> canceled;
    //     canceled= new vec<uint32_t>(anotherSolver.canceled.size())
    //     anotherSolver.canceled.copyTo(canceled)
    // #endif

    // Solver state:
    //vec<CRef>           clauses;          // List of problem clauses.
    //clauses.growTo(anotherSolver.clauses.size());
    for (int i = 0; i < anotherSolver.clauses.size(); i++) {
        clauses.push(anotherSolver.clauses[i]);
    }

    //vec<CRef>           learnts_core,     // List of learnt clauses.
    //learnts_tier2,
    //learnts_local;
    //printf("Gets to first CREF copy\n");
    // learnts_core.growTo(anotherSolver.learnts_core.size());
    // learnts_tier2.growTo(anotherSolver.learnts_tier2.size());
    // learnts_local.growTo(anotherSolver.learnts_local.size());

    // TODO: What is wrong with this?? How is CREF copied
    for (int i = 0; i < anotherSolver.learnts_core.size(); i++) {
        learnts_core.push(anotherSolver.learnts_core[i]);
    }

    for (int i = 0; i < anotherSolver.learnts_tier2.size(); i++) {
        learnts_tier2.push(anotherSolver.learnts_tier2[i]);
    }

    for (int i = 0; i < anotherSolver.learnts_local.size(); i++) {
        learnts_local.push(anotherSolver.learnts_local[i]);
    }

    //vec<double>         activity_CHB,     // A heuristic measurement of the activity of a variable.
    //activity_VSIDS,activity_distance;
    //activity_CHB.growTo(anotherSolver.activity_CHB.size());
    for (int i = 0; i < anotherSolver.activity_CHB.size(); i++) {
        activity_CHB.push(anotherSolver.activity_CHB[i]);
    }

    //activity_VSIDS.growTo(anotherSolver.activity_VSIDS.size());
    for (int i = 0; i < anotherSolver.activity_VSIDS.size(); i++) {
        activity_VSIDS.push(anotherSolver.activity_VSIDS[i]);
    }

    //activity_distance.growTo(anotherSolver.activity_distance.size());
    for (int i = 0; i < anotherSolver.activity_distance.size(); i++) {
        activity_distance.push(anotherSolver.activity_distance[i]);
    }

    //OccLists<Lit, vec<Watcher>, WatcherDeleted>
    //watches_bin,      // Watches for binary clauses only.
    //watches;          // 'watches[lit]' is a list of constraints watching 'lit' (will go there if literal becomes true).
    //TODO: no idea how to copy OCClists by value. Pass by reference for now. Perhaps will not matter
    // This is not going to work. Needs a pointer anotherSolver's attribute

    //watches_bin = anotherSolver.watches_bin;
    //printf("Copying watches_bin...\n");
    watches_bin.update(anotherSolver.watches_bin.getoccs(), anotherSolver.watches_bin.getdirty(), anotherSolver.watches_bin.getdirties(), anotherSolver.watches_bin.getdeleted());
    //printf("Copying watches...\n");
    watches.update(anotherSolver.watches.getoccs(), anotherSolver.watches.getdirty(), anotherSolver.watches.getdirties(), anotherSolver.watches.getdeleted());

    //watches = anotherSolver.watches;
    // for (int i =0; i < anotherSolver.watches.size(), i++)
    // {
    //     this.watches.init(anotherSolver.watches[i]);
    // }

    //vec<lbool>          assigns;          // The current assignments.
    //assigns.growTo(anotherSolver.assigns.size());
    for (int i = 0; i < anotherSolver.assigns.size(); i++) {
        //printf("Assigns before:%d and after:%d\n", toInt(anotherSolver.assigns[i]), toInt(assigns[i]));
        assigns.push(anotherSolver.assigns[i]);
        //printf("Assigns before:%d and after:%d\n", toInt(anotherSolver.assigns[i]), toInt(assigns[i]));
    }
    //printf("Size orig:%d, Size new:%d\n", anotherSolver.assigns.size(), assigns.size());

    //vec<char>           polarity;         // The preferred polarity of each variable.
    //polarity.growTo(anotherSolver.polarity.size());
    for (int i = 0; i < anotherSolver.polarity.size(); i++) {
        polarity.push(anotherSolver.polarity[i]);
    }

    //vec<char>           decision;         // Declares if a variable is eligible for selection in the decision heuristic.
    //decision.growTo(anotherSolver.decision.size());
    for (int i = 0; i < anotherSolver.decision.size(); i++) {
        decision.push(anotherSolver.decision[i]);
    }

    //vec<Lit>            trail;            // Assignment stack; stores all assigments made in the order they were made.
    //trail.growTo(anotherSolver.trail.size());
    for (int i = 0; i < anotherSolver.trail.size(); i++) {
        trail.push(anotherSolver.trail[i]);
    }

    //vec<int>            trail_lim;        // Separator indices for different decision levels in 'trail'.
    //trail_lim.growTo(anotherSolver.trail_lim.size());
    for (int i = 0; i < anotherSolver.trail_lim.size(); i++) {
        trail_lim.push(anotherSolver.trail_lim[i]);
    }

    //vec<VarData>        vardata;          // Stores reason and level for each variable.
    //TODO: Watch out for VarData. This won't guarantee a deep copy
    //printf("Gets Line to copygin VarData\n");
    //vardata.growTo(anotherSolver.vardata.size());
    for (int i = 0; i < anotherSolver.vardata.size(); i++) {
        vardata.push(anotherSolver.vardata[i]);
    }
    //printf("Finishes copying VarData\n");

    //vec<Lit>            assumptions;      // Current set of assumptions provided to solve by the user.
    //assumptions.growTo(anotherSolver.assumptions.size());
    for (int i = 0; i < anotherSolver.assumptions.size(); i++) {
        assumptions.push(anotherSolver.assumptions[i]);
    }

    //Heap<VarOrderLt>    order_heap_CHB,   // A priority queue of variables ordered with respect to the variable activity.
    //order_heap_VSIDS,order_heap_distance;
    //TODO: Not sure how to copy Heap efficiently. Just copy by reference by now
    // This REALLY needs to copied properly if actually using their heursitic because it changs variable ordering
    // Needs to get a pointer

    // order_heap_CHB = anotherSolver.order_heap_CHB;
    //printf("heap distance empty: %d, heap CHB empty: %d, heap VSIDS empty: %d\n", order_heap_distance.size(), order_heap_CHB.size(), order_heap_VSIDS.size());

    for (int i = 0; i < anotherSolver.order_heap_CHB.size(); i++) {
        order_heap_CHB.update(anotherSolver.order_heap_CHB[i]);
        //printf("Index: %d in heap is %d and copy is %d\n", i, anotherSolver.order_heap_CHB[i], order_heap_CHB[i] );
    }

    // order_heap_VSIDS = anotherSolver.order_heap_VSIDS;
    for (int i = 0; i < anotherSolver.order_heap_VSIDS.size(); i++) {
        order_heap_VSIDS.update(anotherSolver.order_heap_VSIDS[i]);
    }

    // order_heap_distance = anotherSolver.order_heap_distance;
    for (int i = 0; i < anotherSolver.order_heap_distance.size(); i++) {
        order_heap_distance.update(anotherSolver.order_heap_distance[i]);
    }

    //printf("heap distance empty: %d, heap CHB empty: %d, heap VSIDS empty: %d\n", order_heap_distance.size(), order_heap_CHB.size(), order_heap_VSIDS.size());

    //MyQueue<int>        lbd_queue;  // For computing moving averages of recent LBD values.
    //TODO: Not sure how to copy ClauseAllocator. Just copy by reference
    //This is not copying by reference!. Need to do that

    //lbd_queue = anotherSolver.lbd_queue

    //ClauseAllocator     ca;
    //TODO: Not sure how to copy ClauseAllocator. Just copy by reference
    //ca = anotherSolver.ca;
    anotherSolver.ca.copyTo(ca);

    // Temporaries (to reduce allocation overhead). Each variable is prefixed by the method in which it is
    // used, exept 'seen' wich is used in several places.
    //
    //vec<char>           seen;

    //seen.growTo(anotherSolver.seen.size());
    for (int i = 0; i < anotherSolver.seen.size(); i++) {
        seen.push(anotherSolver.seen[i]);
    }

    //vec<Lit>            analyze_stack;
    //analyze_stack.growTo(anotherSolver.analyze_stack.size());
    for (int i = 0; i < anotherSolver.analyze_stack.size(); i++) {
        analyze_stack.push(anotherSolver.analyze_stack[i]);
    }

    //vec<Lit>            analyze_toclear;
    //analyze_toclear.growTo(anotherSolver.analyze_toclear.size());
    for (int i = 0; i < anotherSolver.analyze_toclear.size(); i++) {
        analyze_toclear.push(anotherSolver.analyze_toclear[i]);
    }

    //vec<Lit>            add_tmp;
    //add_tmp.growTo(anotherSolver.add_tmp.size());
    for (int i = 0; i < anotherSolver.add_tmp.size(); i++) {
        add_tmp.push(anotherSolver.add_tmp[i]);
    }

    //vec<Lit>            add_oc;
    //add_oc.growTo(anotherSolver.add_oc.size());
    for (int i = 0; i < anotherSolver.add_oc.size(); i++) {
        add_oc.push(anotherSolver.add_oc[i]);
    }

    //vec<uint64_t>       seen2;    // Mostly for efficient LBD computation. 'seen2[i]' will indicate if decision level or variable 'i' has been seen.
    //seen2.growTo(anotherSolver.seen2.size());
    for (int i = 0; i < anotherSolver.seen2.size(); i++) {
        seen2.push(anotherSolver.seen2[i]);
    }

    // Resource contraints:
    //
    //TODO: What to do about static varialbes??
    // #ifdef BIN_DRUP
    //     static int buf_len;
    //     this.buf_len = new int
    //     *(buf_len) = *(anotherSolver.buf_len)

    //     static unsigned char drup_buf[];
    //     this.buf_len = new int
    //     *(buf_len) = *(anotherSolver.buf_len)

    //     static unsigned char*buf_ptr;

    // simplify
    //
    //vec<Lit> simp_learnt_clause;
    //simp_learnt_clause.growTo(anotherSolver.simp_learnt_clause.size());
    for (int i = 0; i < anotherSolver.simp_learnt_clause.size(); i++) {
        simp_learnt_clause.push(anotherSolver.simp_learnt_clause[i]);
    }

    //vec<CRef> simp_reason_clause;
    //simp_reason_clause.growTo(anotherSolver.simp_reason_clause.size());
    for (int i = 0; i < anotherSolver.simp_reason_clause.size(); i++) {
        simp_reason_clause.push(anotherSolver.simp_reason_clause[i]);
    }

    // adjust simplifyAll occasion

    //vec<double> var_iLevel,var_iLevel_tmp;
    //var_iLevel.growTo(anotherSolver.var_iLevel.size());
    for (int i = 0; i < anotherSolver.var_iLevel.size(); i++) {
        var_iLevel.push(anotherSolver.var_iLevel[i]);
    }

    //var_iLevel_tmp.growTo(anotherSolver.var_iLevel_tmp.size());
    for (int i = 0; i < anotherSolver.var_iLevel_tmp.size(); i++) {
        var_iLevel_tmp.push(anotherSolver.var_iLevel_tmp[i]);
    }

    //vec<int> pathCs;
    //pathCs.growTo(anotherSolver.pathCs.size());
    for (int i = 0; i < anotherSolver.pathCs.size(); i++) {
        pathCs.push(anotherSolver.pathCs[i]);
    }

    //vec<Lit> involved_lits;
    //involved_lits.growTo(anotherSolver.involved_lits.size());
    for (int i = 0; i < anotherSolver.involved_lits.size(); i++) {
        involved_lits.push(anotherSolver.involved_lits[i]);
    }
}

// simplify All
//

CRef Solver::simplePropagate()
{
    CRef confl = CRef_Undef;
    int num_props = 0;
    watches.cleanAll();
    watches_bin.cleanAll();
    while (qhead < trail.size()) {
        Lit p = trail[qhead++]; // 'p' is enqueued fact to propagate.
        vec<Watcher>& ws = watches[p];
        Watcher *i, *j, *end;
        num_props++;

        // First, Propagate binary clauses
        vec<Watcher>& wbin = watches_bin[p];

        for (int k = 0; k < wbin.size(); k++) {

            Lit imp = wbin[k].blocker;

            if (value(imp) == l_False) {
                return wbin[k].cref;
            }

            if (value(imp) == l_Undef) {
                simpleUncheckEnqueue(imp, wbin[k].cref);
            }
        }
        for (i = j = (Watcher*) ws, end = i + ws.size(); i != end;) {
            // Try to avoid inspecting the clause:
            Lit blocker = i->blocker;
            if (value(blocker) == l_True) {
                *j++ = *i++;
                continue;
            }

            // Make sure the false literal is data[1]:
            CRef cr = i->cref;
            Clause& c = ca[cr];
            Lit false_lit = ~p;
            if (c[0] == false_lit)
                c[0] = c[1], c[1] = false_lit;
            assert(c[1] == false_lit);
            //  i++;

            // If 0th watch is true, then clause is already satisfied.
            // However, 0th watch is not the blocker, make it blocker using a new watcher w
            // why not simply do i->blocker=first in this case?
            Lit first = c[0];
            //  Watcher w     = Watcher(cr, first);
            if (first != blocker && value(first) == l_True) {
                i->blocker = first;
                *j++ = *i++;
                continue;
            }

                // Look for new watch:
                //if (incremental)
                //{ // ----------------- INCREMENTAL MODE
                //  int choosenPos = -1;
                //  for (int k = 2; k < c.size(); k++)
                //  {
                //    if (value(c[k]) != l_False)
                //    {
                //      if (decisionLevel()>assumptions.size())
                //      {
                //        choosenPos = k;
                //        break;
                //      }
                //      else
                //      {
                //        choosenPos = k;

                //        if (value(c[k]) == l_True || !isSelector(var(c[k]))) {
                //          break;
                //        }
                //      }

                //    }
                //  }
                //  if (choosenPos != -1)
                //  {
                //    // watcher i is abandonned using i++, because cr watches now ~c[k] instead of p
                //    // the blocker is first in the watcher. However,
            //    // the blocker in the corresponding watcher in ~first is not c[1]
                //    Watcher w = Watcher(cr, first); i++;
                //    c[1] = c[choosenPos]; c[choosenPos] = false_lit;
                //    watches[~c[1]].push(w);
                //    goto NextClause;
                //  }
                //}
            else { // ----------------- DEFAULT  MODE (NOT INCREMENTAL)
                for (int k = 2; k < c.size(); k++) {

                    if (value(c[k]) != l_False) {
                        // watcher i is abandonned using i++, because cr watches now ~c[k] instead of p
                        // the blocker is first in the watcher. However,
                        // the blocker in the corresponding watcher in ~first is not c[1]
                        Watcher w = Watcher(cr, first);
                        i++;
                        c[1] = c[k];
                        c[k] = false_lit;
                        watches[~c[1]].push(w);
                        goto NextClause;
                    }
                }
            }

            // Did not find watch -- clause is unit under assignment:
            i->blocker = first;
            *j++ = *i++;
            if (value(first) == l_False) {
                confl = cr;
                qhead = trail.size();
                // Copy the remaining watches:
                while (i < end)
                    *j++ = *i++;
            }
            else {
                simpleUncheckEnqueue(first, cr);
            }
NextClause:
            ;
        }
        ws.shrink(i - j);
    }

    s_propagations += num_props;

    return confl;
}

void Solver::simpleUncheckEnqueue(Lit p, CRef from)
{
    assert(value(p) == l_Undef);
    assigns[var(p)] = lbool(!sign(p)); // this makes a lbool object whose value is sign(p)
    vardata[var(p)].reason = from;
    trail.push(p);
}

void Solver::cancelUntilTrailRecord()
{
    for (int c = trail.size() - 1; c >= trailRecord; c--) {
        Var x = var(trail[c]);
        assigns[x] = l_Undef;

    }
    qhead = trailRecord;
    trail.shrink(trail.size() - trailRecord);

}

vec<int>* Solver::getActiveLiterals()
{
    int num_literals = nVars()*2;
    vec<int> *active_literals = new vec<int>();
    for( int l =0; l < num_literals; l++)
    {
        if(value(toLit(l)) == l_Undef)
        {
            active_literals->push(l);
        }
    }
    return active_literals;
}

vec<Var>* Solver::getActiveVars()
{

    int num_vars = nVars();
    vec<Var> *active_vars = new vec<Var>();
    for( int i =0; i < num_vars; i++)
    {
        Var v = var(toLit(i*2));
        if(value(v) == l_Undef)
        {
            active_vars->push(v);
        }
        // else
        // {
        //     printf("Active:%d\n", i);
        // }
    }
    return active_vars;
}


void Solver::litsEnqueue(int cutP, Clause& c)
{
    for (int i = cutP; i < c.size(); i++) {
        simpleUncheckEnqueue(~c[i]);
    }
}

bool Solver::removed(CRef cr)
{
    return ca[cr].mark() == 1;
}

//=================================================================================================
// Minor methods:

// Creates a new SAT variable in the solver. If 'decision' is cleared, variable will not be
// used as a decision variable (NOTE! This has effects on the meaning of a SATISFIABLE result).
//

Var Solver::newVar(bool sign, bool dvar)
{
    int v = nVars();
    watches_bin.init(mkLit(v, false));
    watches_bin.init(mkLit(v, true));
    watches .init(mkLit(v, false));
    watches .init(mkLit(v, true));
    assigns .push(l_Undef);
    vardata .push(mkVarData(CRef_Undef, 0));
    activity_CHB .push(0);
    activity_VSIDS.push(rnd_init_act ? drand(random_seed) * 0.00001 : 0);

    picked.push(0);
    conflicted.push(0);
    almost_conflicted.push(0);
#ifdef ANTI_EXPLORATION
    canceled.push(0);
#endif

    seen .push(0);
    seen2 .push(0);
    polarity .push(sign);
    decision .push();
    trail .capacity(v + 1);
    setDecisionVar(v, dvar);

    activity_distance.push(0);
    var_iLevel.push(0);
    var_iLevel_tmp.push(0);
    pathCs.push(0);
    return v;
}

bool Solver::addClause_(vec<Lit>& ps)
{
    assert(decisionLevel() == 0);
    if (!ok) return false;

    // Check if clause is satisfied and remove false/duplicate literals:
    sort(ps);
    Lit p;
    int i, j;

    if (drup_file) {
        add_oc.clear();
        for (int i = 0; i < ps.size(); i++) add_oc.push(ps[i]);
    }

    for (i = j = 0, p = lit_Undef; i < ps.size(); i++)
        if (value(ps[i]) == l_True || ps[i] == ~p)
            return true;
        else if (value(ps[i]) != l_False && ps[i] != p)
            ps[j++] = p = ps[i];
    ps.shrink(i - j);

    if (drup_file && i != j) {
#ifdef BIN_DRUP
        binDRUP('a', ps, drup_file);
        binDRUP('d', add_oc, drup_file);
#else
        for (int i = 0; i < ps.size(); i++)
            fprintf(drup_file, "%i ", (var(ps[i]) + 1) * (-2 * sign(ps[i]) + 1));
        fprintf(drup_file, "0\n");

        fprintf(drup_file, "d ");
        for (int i = 0; i < add_oc.size(); i++)
            fprintf(drup_file, "%i ", (var(add_oc[i]) + 1) * (-2 * sign(add_oc[i]) + 1));
        fprintf(drup_file, "0\n");
#endif
    }

    if (ps.size() == 0)
        return ok = false;
    else if (ps.size() == 1) {
        uncheckedEnqueue(ps[0]);
        return ok = (propagate() == CRef_Undef);
    }
    else {
        CRef cr = ca.alloc(ps, false);
        clauses.push(cr);
        attachClause(cr);
    }

    return true;
}

void Solver::attachClause(CRef cr)
{
    const Clause& c = ca[cr];
    assert(c.size() > 1);
    OccLists<Lit, vec<Watcher>, WatcherDeleted>& ws = c.size() == 2 ? watches_bin : watches;
    ws[~c[0]].push(Watcher(cr, c[1]));
    ws[~c[1]].push(Watcher(cr, c[0]));
    if (c.learnt()) learnts_literals += c.size();
    else clauses_literals += c.size();
}

void Solver::detachClause(CRef cr, bool strict)
{
    const Clause& c = ca[cr];
    assert(c.size() > 1);
    OccLists<Lit, vec<Watcher>, WatcherDeleted>& ws = c.size() == 2 ? watches_bin : watches;

    if (strict) {
        remove(ws[~c[0]], Watcher(cr, c[1]));
        remove(ws[~c[1]], Watcher(cr, c[0]));
    }
    else {
        // Lazy detaching: (NOTE! Must clean all watcher lists before garbage collecting this clause)
        ws.smudge(~c[0]);
        ws.smudge(~c[1]);
    }

    if (c.learnt()) learnts_literals -= c.size();
    else clauses_literals -= c.size();
}

void Solver::removeClause(CRef cr)
{
    Clause& c = ca[cr];

    if (drup_file) {
        if (c.mark() != 1) {
#ifdef BIN_DRUP
            binDRUP('d', c, drup_file);
#else
            fprintf(drup_file, "d ");
            for (int i = 0; i < c.size(); i++)
                fprintf(drup_file, "%i ", (var(c[i]) + 1) * (-2 * sign(c[i]) + 1));
            fprintf(drup_file, "0\n");
#endif
        }
        else
            printf("c Bug. I don't expect this to happen.\n");
    }

    detachClause(cr);
    // Don't leave pointers to free'd memory!
    if (locked(c)) {
        Lit implied = c.size() != 2 ? c[0] : (value(c[0]) == l_True ? c[0] : c[1]);
        vardata[var(implied)].reason = CRef_Undef;
    }
    c.mark(1);
    ca.free(cr);
}

bool Solver::satisfied(const Clause& c) const
{
    for (int i = 0; i < c.size(); i++)
        if (value(c[i]) == l_True)
            return true;
    return false;
}

std::string gen_random(const int len) {
    
    std::string tmp_s;
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    
    srand( (unsigned) time(NULL) * getpid());

    tmp_s.reserve(len);

    for (int i = 0; i < len; ++i) 
        tmp_s += alphanum[rand() % (sizeof(alphanum) - 1)];
    
    
    return tmp_s;
    
}

// Revert to the state at given level (keeping all assignment at 'level' but not beyond).
//

void Solver::cancelUntil(int level)
{
    if (decisionLevel() > level) {
        for (int c = trail.size() - 1; c >= trail_lim[level]; c--) {
            Var x = var(trail[c]);

            if (!VSIDS) {
                uint32_t age = conflicts - picked[x];
                if (age > 0) {
                    double adjusted_reward = ((double) (conflicted[x] + almost_conflicted[x])) / ((double) age);
                    double old_activity = activity_CHB[x];
                    activity_CHB[x] = step_size * adjusted_reward + ((1 - step_size) * old_activity);
                    if (order_heap_CHB.inHeap(x)) {
                        if (activity_CHB[x] > old_activity)
                            order_heap_CHB.decrease(x);
                        else
                            order_heap_CHB.increase(x);
                    }
                }
#ifdef ANTI_EXPLORATION
                canceled[x] = conflicts;
#endif
            }

            assigns [x] = l_Undef;
            if (phase_saving > 1 || (phase_saving == 1) && c > trail_lim.last())
                polarity[x] = sign(trail[c]);
            insertVarOrder(x);
        }
        qhead = trail_lim[level];
        trail.shrink(trail.size() - trail_lim[level]);
        trail_lim.shrink(trail_lim.size() - level);
    }
}

//=================================================================================================
// Major methods:

/*TODO: This is where Python bridge should be to integrate. We need objects for all of:
        1. Variables currently set
        2. Full problem (and learnt clauses)
        
        This must be written into Python object and passed through variable_prediction.py

        Then it must read matrix of variable probabilities and variable True/False settings

        When conflict happens needs to keep write out training data to file for Python program (variables, problem, NN prediction, Ground truth (from MCTS))
 */
Lit Solver::pickBranchLit(vec<double> *counts, vec<double> *values, bool refresh_literal_probabilities, int current_depth)
{
    Var next = var_Undef;

    if (use_random_branching) {
        double counts_sum = 0;

        for (int i = 0; i < counts->size(); i++) {
            counts_sum += (*counts)[i];
        }

        //No MCTS lookaheads. Purely random choice...
        if (counts_sum == 0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(0, nVars() - 1);

            while (value(next) != l_Undef) {
                next = dis(gen);

                bool flag = false;
                for (int i = 0; i < nVars(); i++) {
                    if (value(i) == l_Undef) {
                        flag = true;
                    }
                }

                if (!flag) {
                    return lit_Undef;
                }
            }

            return mkLit(next, pickPolarityFromDistribution(0.5));
        }

        // Project Q values onto [1,-1]
        vec<double> *q_vec = new vec<double>(counts->size(), 0);
        double max_q = imin;
        double min_q = imax;

        for (int i = 0; i < counts->size(); i++) {
            if ((*counts)[i] == 0) {
                (*q_vec)[i] = 0;
                continue;
            }

            else {
                (*q_vec)[i] = (*values)[i] / (*counts)[i] * -1;
            }

            if ((*q_vec)[i] > max_q) {
                max_q = (*q_vec)[i];
            }

            if ((*q_vec)[i] < min_q) {
                min_q = (*q_vec)[i];
            }
        }

        for (int i = 0; i < counts->size(); i++) {
            if ((*counts)[i] != 0 && max_q != min_q) {
                double mean = (*values)[i] / (*counts)[i] * -1;
                (*q_vec)[i] = 2 * ((mean - min_q) / (max_q - min_q)) - 1;
            }

            else {
                (*q_vec)[i] = 0;
            }
        }

        double c_puct = nVars() * 2;
        double max_qu = imin;
        int max_action = toInt(lit_Undef);

        for (int i = 0; i < counts->size(); i++) {
            double q = (*q_vec)[i];
            double u = c_puct * (1.0 / (nVars() * 2)) * sqrt(counts_sum) / (1 + (*counts)[i]);

            double qu = q + u;

            if (qu > max_qu && value(toLit(i)) == l_Undef) {
                max_qu = qu;
                max_action = i;
            }
        }
        delete q_vec; ///previously a memory leak
        return toLit(max_action);
    }
    else if (use_bsh_branching)
    {
        if (getActiveLiterals()->size() == 0)
        {
            return lit_Undef;
        }
        literal_bsh_scores = at::zeros(1);
        literal_bsh_scores = get_bsh_lit_vals(literal_bsh_scores);
        fflush(stdout);
        Lit selected_literal = selectLiteral(literal_bsh_scores, "max");//'sample'
        assert(value(selected_literal) == l_Undef);
        return selected_literal;
    }
    else if (use_neural_branching) {
        
        if ((refresh_literal_probabilities or previous_depth > current_depth) and current_depth <= neural_net_depth_threshold)
        {
            //printf("pickBranchLit via neural net\n");
            std::vector<torch::Tensor> outputs = getNeuralPrediction(at::zeros(1)); //TODO: sometimes literals are getting -inf predictions!!
            global_literal_probabilities = outputs[0];
            global_literal_probabilities = global_literal_probabilities.max(at::zeros(nVars()*2));
            previous_depth = current_depth;
            nn_decisions++;
        }
        if (global_literal_probabilities.size(0) == 1)
        {
            return lit_Undef;
        }
        torch::Tensor literal_probabilities = at::ones(nVars()*2)*-1;
        vec<int> *active_literals = getActiveLiterals();
        if (active_literals->size() == 0)
        {
            delete active_literals; ///would be a leak
            return lit_Undef;
        }
        // Set to literal probabilities to zero if unactive
        for (int i =0; i < active_literals->size(); i++)
        {
            int lit = (*active_literals)[i];
            if (global_literal_probabilities[lit].item<double>() > 0.0)
            {
                literal_probabilities[lit] = global_literal_probabilities[lit];
            }
            else
            {
                literal_probabilities[lit] = 0.0;
            }
        }
        
        Lit selected_literal = selectLiteral(literal_probabilities, "max");//'sample'

        // x
        // std::cout << global_literal_probabilities << std::endl;

        // Bunch of floats being mapping to -1 or 0
        delete active_literals; ///would be a leak
        assert(value(selected_literal) == l_Undef);
        return selected_literal;
    }


    else {
        Heap<VarOrderLt>& order_heap = DISTANCE ? order_heap_distance : ((!VSIDS) ? order_heap_CHB : order_heap_VSIDS);

        // Activity based decision:
        while (next == var_Undef || value(next) != l_Undef || !decision[next])
            if (order_heap.empty())
                return lit_Undef;
            else {
#ifdef ANTI_EXPLORATION
                if (!VSIDS) {
                    Var v = order_heap_CHB[0];
                    uint32_t age = conflicts - canceled[v];
                    while (age > 0) {
                        double decay = pow(0.95, age);
                        activity_CHB[v] *= decay;
                        if (order_heap_CHB.inHeap(v))
                            order_heap_CHB.increase(v);
                        canceled[v] = conflicts;
                        v = order_heap_CHB[0];
                        age = conflicts - canceled[v];
                    }
                }
#endif
                next = order_heap.removeMin();
            }
    }

    return mkLit(next, polarity[next]);
}

std::string Solver::getFileName(std::string filePath)
{
    //printf("Gets to get file name");
    //exit(0);
//    bool withExtension = false;
    char seperator = '/';
    char dot = '.';
    // Get last dot position
    std::size_t dotPos = filePath.rfind(dot);
    std::size_t sepPos = filePath.rfind(seperator);
    
    std::string basename = filePath.substr(sepPos + 1, dotPos - (sepPos+1));

    return basename;//filePath.size() - (withExtension || dotPos != std::string::npos ? 1 : dotPos) );

}


std::string Solver::getStateName()
{
    std::string str;
    str.append(getFileName(instance_name));
    str.append("_");
    // str.append(getFileName(model_filename));
    // str.append("_");
    str.append(std::to_string(num_monte_carlo_tree_search_samples));
    str.append("_");
    str.append(std::to_string(decisions));

    return str;
}

Lit Solver::selectLiteral(torch::Tensor probabilities, std::string mode)
{
    int literal_index = -1;
    //printf("Selecting literal...\n");
    //std::cout << probabilities << std::endl;

    if (mode == "max")
    {
        literal_index = probabilities.argmax().item<int64_t>();
    }
    else if (mode == "sample")
    {
        probabilities = probabilities.softmax(0);
        try{
            literal_index = probabilities.multinomial(1)[0].item<int64_t>();
        }catch (std::exception){
            std::cerr << probabilities;
        }
    }
    else
    {
        std::cerr << "select literal mode not recognized.";
        throw;
    }
    //printf("Selected literal: %d\n", literal_index);
    return toLit(literal_index);
}

static std::string base64_encode(const std::string &in) {

    std::string out;

    int val = 0, valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(val>>valb)&0x3F]);
            valb -= 6;
        }
    }
    if (valb>-6) out.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[((val<<8)>>(valb+8))&0x3F]);
    while (out.size()%4) out.push_back('=');
    return out;
}

//TODO: Change type to return a tuple of (value double, counts tensor)
std::vector<torch::Tensor> Solver::getNeuralPrediction(torch::Tensor probabilities, lbool result, torch::Tensor unnorm_reward, torch::Tensor lit_counts)
{

    //std::cout << probabilities << std::endl;

    //printf("Getting neural prediction...");

    // Get values and index of current state of search tree
    Pair<vec<int64_t>>* problem = getProblem(); ///this has its own dynamic mem plus its data mem

    vec<int64_t> * problem_values = problem->getFirst();
    vec<int64_t> * problem_indices = problem->getSecond();

    if (problem_values->size() == 0)
    {
        delete problem;
        std::vector<torch::Tensor> network_outputs;
        network_outputs.push_back(at::zeros(1));
        network_outputs.push_back(at::zeros(1));
        return network_outputs;
    }

    // Change ordering of indices to be all clauses, then all variables
    vec<int64_t> * clauses = new vec<int64_t>(0,0); ///!!!!
    vec<int64_t> * vars = new vec<int64_t>(0,0);    ///!!!!

    std::map<int64_t,int64_t> var_reindexing;
    std::map<int64_t,int64_t> clause_reindexing;

    for (int i=0; i < problem_indices->size();)
    {

        clauses->push((*problem_indices)[i]);
        vars->push((*problem_indices)[i+1]);
        i=i+2;
    }

    problem_indices = new vec<int64_t>(0,0);        ///!!!!
    int64_t clause_counter = 0;
    for (int i=0; i < clauses->size(); i++)
    {
        int64_t clause = (*clauses)[i];
        if (clause_reindexing.find(clause) == clause_reindexing.end())
        {
            clause_reindexing.insert(std::pair<int64_t,int64_t>(clause,clause_counter));
            clause_counter++;
        }

        problem_indices->push(clause_reindexing[clause]);
    }
    // Variables are reindexed in the order they appear in the cnf. e.g., if the first literal of the first clause, it will get relabeled to variable 0.
    int64_t var_counter = 0;
    for (int i=0; i < vars->size(); i++)
    {
        int64_t var_i = (*vars)[i];
        if (var_reindexing.find(var_i) == var_reindexing.end())
        {
            var_reindexing.insert(std::pair<int64_t,int64_t>(var_i,var_counter));
            var_counter++;
        }
        problem_indices->push(var_reindexing[var_i]);
    }
    // Get backwards mapping
    std::map<int64_t,int64_t> var_backwards_reindexing;
    for (std::map<int64_t,int64_t>::iterator it=var_reindexing.begin(); it!=var_reindexing.end(); ++it)
    {
        var_backwards_reindexing.insert(std::pair<int64_t,int64_t>(it->second,it->first));
        // if (value(toLit(it->first)) != l_Undef)
        // {
        //     std::cerr << "error: literal is already assigned!";
        // }
    }


    //torch::Device device;
    torch::Device device(at::kCPU);
    if (strcmp(torch_device, "CPU") == 0)
    {
        torch::Device device(at::kCPU);
    } else if (strcmp(torch_device, "CUDA") == 0)
    {
        torch::Device device(at::kCUDA);
    }
    else
    {
        printf("Error: Device %s not recognized.", torch_device);
        throw;
    }

    // Convert to values,index into torch
    auto index_options = torch::TensorOptions().dtype(torch::kInt64);//.device(device); // kGPU, 1
    auto values_options = torch::TensorOptions().dtype(torch::kInt64);//.device(device); // kGPU, 1
    torch::Tensor values = torch::from_blob(*problem_values,{problem_values->size()}, values_options).to(device);//.toBackend(c10::Backend::SparseCUDA);//.toBackend(SparseGPU);
    torch::Tensor index = torch::from_blob(*problem_indices,{problem_indices->size()}, index_options).to(device);//.toBackend(c10::Backend::SparseCUDA);;
    
    // Convert nnz*2 1D tensor to nnz,2 2D tensor
    index = index.reshape({2, index.size(0)/2});
    int num_clauses = index[0].max().item<int64_t>();
    int num_vars = index[1].max().item<int64_t>();
    //printf("NN input dims: #clauses:%d, #vars:%d\n", num_clauses, num_vars);

    // Add dim to values
    values = values.reshape({values.size(0), 1});
    

    auto options = torch::TensorOptions(torch::Layout::Sparse).dtype(torch::kInt64).device(device);
    // TODO: Set size dynamically based on num vars and clauses
    torch::Tensor sparse_matrix = torch::_sparse_coo_tensor_with_dims_and_tensors(2, 1, {num_clauses*2,num_vars*2,1},index, values, options);//_sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::Tensor &indices, const at::Tensor &values, const at::TensorOptions &options)
    
    if (strcmp(torch_device, "CUDA") == 0) {
        sparse_matrix = sparse_matrix.to(at::kCUDA);
    }

    if (probabilities.size(0) > 1) {
        printf("Writing out MCTS data point...\n");
        int num_variables = var_backwards_reindexing.size();
        torch::Tensor mcts_lit_probabilities = at::zeros(num_variables*2);
        torch::Tensor unnormalized_reward = at::zeros(num_variables*2);
        torch::Tensor mcts_lit_counts = at::zeros(num_variables*2);
        for (int i=0; i < num_variables*2; i++)
        {
            bool polarity = sign(toLit(i));
            int64_t variable = toInt(var(toLit(i)));

            variable = var_backwards_reindexing[variable];
            int index = toInt(mkLit(variable, polarity));
            
            unnormalized_reward[i] = unnorm_reward[index];
            mcts_lit_probabilities[i] = probabilities[index];
            mcts_lit_counts[i] = lit_counts[index];
        }

        std::vector<torch::Tensor> pickle_inputs;
        pickle_inputs.push_back(values);
        pickle_inputs.push_back(index);
        pickle_inputs.push_back(mcts_lit_probabilities);
        pickle_inputs.push_back(unnormalized_reward);
        pickle_inputs.push_back(mcts_lit_counts);

        std::vector<char> pickled_data = torch::jit::pickle_save(pickle_inputs);
        std::string data_point_string(pickled_data.begin(), pickled_data.end());
        data_point_string = base64_encode(data_point_string);

        if (use_mcts_db) {

            std::string base = "mysql -N -B -h address -P port -u username -p\"password\" database -e ";
            std::string instance_id = "'" + std::to_string(this->db_instance_key) + "'";
            std::string experiment_id = "'" + std::to_string(this->db_experiment_key) + "'";
            std::string experiment_name = "'" + (std::string) this->experiment_name + "'";
            std::string cnf_hash = "'" + (std::string) this->cnf_hash + "'";
            std::string status = "'" + (std::string) (result == l_True ? "SAT" : (result == l_False ? "UNSAT" : "CAPPED" )) + "'";
            std::string subdepth = "'" + std::to_string(decisionLevel()) + "'";
            std::string dummy = "";
            std::string sql_command = dummy + "INSERT IGNORE INTO Data (data_id, experiment_name, experiment_id, cnf_hash, status, time, subsolver_depth, bytes) VALUES ("
                                       + "default" + "," + experiment_name + "," + experiment_id + "," + cnf_hash + "," + status + ",NOW()," + subdepth + "," + "'" + data_point_string + "'" + ");  SELECT LAST_INSERT_ID();";
            std::string sql_file_name = "/tmp/sql_command.sql";
            std::string random_suffix = gen_random(10);
            sql_file_name += random_suffix;
            std::ofstream sql_file;
            sql_file.open(sql_file_name);
            sql_file << sql_command;
            sql_file.close();

            std::string command = base + "< " + sql_file_name;
            // Make file executable
            std::string chmod_command = "chmod +x " + sql_file_name;
            try {
                system(chmod_command.c_str());
                int data_id = std::strtoul(exec(command.c_str()).c_str(), NULL, 0); // Returns experiment_id of last insert\"";
                (*mcts_data_ids)[assignToVec(assigns)] = data_id;
            
            } catch (std::exception& e) {
                std::cout << sql_file_name;
                printf("Error with executing mysql command: %s\n", e.what());
                throw;
            }
            std::remove(sql_file_name.c_str());
        } else {
            auto bytes = torch::jit::pickle_save(pickle_inputs);
            std::string filename;
            filename.append(data_dir);
            filename.append("/"); 
            filename.append(getFileName(model_filename));
            filename.append("/");
            filename.append(getStateName());
            filename.append("_");
            filename.append(result == l_True ? "SAT" : (result == l_False ? "UNSAT" : "CAPPED" ));
            filename.append(".zip");
            std::ofstream fout(filename, std::ios::out | std::ios::binary);
            fout.write(bytes.data(), bytes.size());
            fout.close();
        }
        printf("Done writing out MCTS data point.\n");
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(sparse_matrix);

    torch::Tensor literal_probabilities;
    torch::Tensor value = -1 * at::ones(1);
    try{
        torch::NoGradGuard no_grad_guard;
        literal_probabilities = module.forward(inputs).toTensor();// at::ones(vars->size()*2);//;
        if (true)//strcmp(rollout_policy, "VALUE_NET") == 0)
        {
            // Value is last element of tensor
            torch::Tensor log2_value = literal_probabilities.index({torch::indexing::Slice(literal_probabilities.size(0)-1, torch::indexing::None)});
            value = torch::pow(2, log2_value);
            if (value[0].item<double>() < 1)
            {
                //printf("Warning: Neural Net prediction was less than 1. Setting to 1.\n");
                value = at::ones(1);
            }
            literal_probabilities = literal_probabilities.index({torch::indexing::Slice(torch::indexing::None, literal_probabilities.size(0)-1)});

            // Get all but last element of tensor

        }
        //printf("Literal probabilities size: %d, Num lits: %d\n", literal_probabilities.size(0), nVars()*2);
    } catch (std::runtime_error err) {
        std::cerr << err.what();
        std::cerr << "error call forward";
        std::cout << sparse_matrix << std::endl;
        throw;
    }
    // If network output is larger (because of value net), just take first nVars*2 elements
    literal_probabilities = literal_probabilities.index({torch::indexing::Slice(torch::indexing::None, getActiveLiterals()->size())});
    literal_probabilities = literal_probabilities.flatten();
    literal_probabilities = literal_probabilities.softmax(0);

    torch::Tensor reindexed_literal_probabilties = at::zeros(nVars()*2); // zeroes vector of length nVars()*2
    reindexed_literal_probabilties.fill_(-INFINITY);

    for (int i=0; i < literal_probabilities.size(0); i++)
    {
        bool polarity = sign(toLit(i));
        int64_t variable = toInt(var(toLit(i)));

        variable = var_backwards_reindexing[variable];
        int index = toInt(mkLit(variable, polarity));
        reindexed_literal_probabilties[index] = literal_probabilities[i]; 
    }

    std::vector<torch::Tensor> network_outputs;
    network_outputs.push_back(reindexed_literal_probabilties);
    network_outputs.push_back(value);

    delete clauses; ///these three previously leaked
    delete vars;
    delete problem_indices;
    delete problem; ///this should delete the pair as well as its contents per the pair destructor
    return network_outputs;

}

Pair<vec<int64_t>>* Solver::getProblem(){


    vec<int64_t> * values = new vec<int64_t>(0, 0);
    vec<int64_t> * indices = new vec<int64_t>(0, 0);

    // Go through problem clauses
    for (int i = 0; i < nClauses(); i++) {
        Clause& c = ca[clauses[i]];

        // Check for true; if so, skip clause        
        for (int j = 0; j < c.size(); j++) {
            if (value(c[j]) == l_True) {
                goto problemCont;
            }
        }

        // Add remaining unassigned pairs
        for (int j = 0; j < c.size(); j++) {
            if (value(c[j]) == l_False) {
                continue;
            }

            indices->push(i);
            indices->push(var(c[j]));

             // Get sign
            if (sign(c[j]) == true) {
                values->push( (int64_t) 1);
            }
            else {
                values->push( (int64_t) 0);
            }
        }

problemCont:
        ;
    }

    // TODO: Investigate if we want to include all types of learnt clauses
    // Go through learnt clauses
    for (int i = 0; i < learnts_core.size(); i++) {
        Clause& c = ca[learnts_core[i]];

        // Check for true; if so, skip clause        
        for (int j = 0; j < c.size(); j++) {
            if (value(c[j]) == l_True) {
                goto learntCont;
            }
        }

        // Add remaining unassigned pairs
        for (int j = 0; j < c.size(); j++) {
            if (value(c[j]) == l_False) {
                continue;
            }

            indices->push(i+nClauses());
            indices->push(var(c[j]));

            // Get sign
            if (sign(c[j]) == true) {
                values->push((int64_t) 1);
            }
            else {
                values->push((int64_t) 0);
            }
        }

learntCont:
        ;
    }

    return new Pair<vec<int64_t>>(values, indices);
}

std::map<int64_t,int64_t> Solver::stateToDimacs(FILE*f)
{

    std::map<int64_t,int64_t> var_reindexing;
    int64_t var_counter = 1;
    int clause_counter = 0;
    std::string dimacs_cnf = "";

    for (int i = 0; i < nClauses(); i++) {
        Clause& c = ca[clauses[i]];

        // Check for true; if so, skip clause        
        for (int j = 0; j < c.size(); j++) {
            if (value(c[j]) == l_True) {
                goto problemCont;
            }
        }
        clause_counter++;
        // Add remaining unassigned pairs
        int curr_var;
        for (int j = 0; j < c.size(); j++) {
            int64_t var_j = var(c[j]);
            if (value(c[j]) == l_False) {
                continue;
            }

            if (var_reindexing.find(var(c[j])) == var_reindexing.end())
            {
                curr_var = var_counter;
                var_reindexing.insert(std::pair<int64_t,int64_t>(var_j,var_counter));
                var_counter++;
            } else
            {
                curr_var = var_reindexing[var_j];
            }

            dimacs_cnf.append( (sign(c[j]) ? "-" : "") + std::to_string(curr_var) + " ");

        }
        dimacs_cnf.append("0\n"); //End clause

        problemCont:
        ;
    }


    fprintf(f, "p cnf %ld %d\n", var_counter-1, clause_counter);
    const char *dimacs_cnf_c = dimacs_cnf.c_str();
    fprintf(f, dimacs_cnf_c);

    return var_reindexing;

}


Var Solver::pickRandomVarFromDistribution(double* distribution)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Get unassigned variables and renormalize
    double sum = 0;
    vec<Var> unassigned;

    for (int i = 0; i < nVars(); i++) {
        if (value(i) == l_Undef) {
            unassigned.push(i);
            sum += distribution[i];
        }
        else {
            distribution[i] = 0;
        }
    }

    if (unassigned.size() == 0) {
        return var_Undef;
    }

    for (int i = 0; i < nVars(); i++) {
        if (distribution[i] != 0) {
            distribution[i] /= sum;
        }
    }

restart:
    double val = dis(gen), probabilitySum = 0;
    int index;

    for (index = 0; index < unassigned.size(); index++) {
        probabilitySum += distribution[unassigned[index]];
        if (val < probabilitySum) {
            break;
        }
    }

    int varIndex = unassigned[index];

    if (value(varIndex) != l_Undef) {
        goto restart;
    }

    unassigned.clear(true);
    return varIndex;
}

// Probability of true
bool Solver::pickPolarityFromDistribution(double probability)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double val = dis(gen);

    return val < probability;
}


// Check if 'p' can be removed. 'abstract_levels' is used to abort early if the algorithm is
// visiting literals at levels that cannot be removed later.

bool Solver::litRedundant(Lit p, uint32_t abstract_levels)
{
    analyze_stack.clear();
    analyze_stack.push(p);
    int top = analyze_toclear.size();
    while (analyze_stack.size() > 0) {
        assert(reason(var(analyze_stack.last())) != CRef_Undef);
        Clause& c = ca[reason(var(analyze_stack.last()))];
        analyze_stack.pop();

        // Special handling for binary clauses like in 'analyze()'.
        if (c.size() == 2 && value(c[0]) == l_False) {
            assert(value(c[1]) == l_True);
            Lit tmp = c[0];
            c[0] = c[1], c[1] = tmp;
        }

        for (int i = 1; i < c.size(); i++) {
            Lit p = c[i];
            if (!seen[var(p)] && level(var(p)) > 0) {
                if (reason(var(p)) != CRef_Undef && (abstractLevel(var(p)) & abstract_levels) != 0) {
                    seen[var(p)] = 1;
                    analyze_stack.push(p);
                    analyze_toclear.push(p);
                }
                else {
                    for (int j = top; j < analyze_toclear.size(); j++)
                        seen[var(analyze_toclear[j])] = 0;
                    analyze_toclear.shrink(analyze_toclear.size() - top);
                    return false;
                }
            }
        }
    }

    return true;
}

void Solver::uncheckedEnqueue(Lit p, CRef from) {
    assert(value(p) == l_Undef);
    Var x = var(p);
    if (!VSIDS) {
        picked[x] = conflicts;
        conflicted[x] = 0;
        almost_conflicted[x] = 0;
#ifdef ANTI_EXPLORATION
        uint32_t age = conflicts - canceled[var(p)];
        if (age > 0) {
            double decay = pow(0.95, age);
            activity_CHB[var(p)] *= decay;
            if (order_heap_CHB.inHeap(var(p)))
                order_heap_CHB.increase(var(p));
        }
#endif
    }
    //printf("Assigned %d", var(p));

    assigns[x] = lbool(!sign(p));
    vardata[x] = mkVarData(from, decisionLevel());
    trail.push(p);
}

/*_________________________________________________________________________________________________
|
|  propagate : [void]  ->  [Clause*]
|  
|  Description:
|    Propagates all enqueued facts. If a conflict arises, the conflicting clause is returned,
|    otherwise CRef_Undef.
|  
|    Post-conditions:
|      * the propagation queue is empty, even if there was a conflict.
|________________________________________________________________________________________________@*/
CRef Solver::propagate() {
    CRef confl = CRef_Undef;
    int num_props = 0;
    watches.cleanAll();
    watches_bin.cleanAll();
    //printf("Starting propagate...\n");
    while (qhead < trail.size()) {
        Lit p = trail[qhead++]; // 'p' is enqueued fact to propagate.
        //printf("------------------------------------- Enqueue %d to propagate  -------------------------------------  \n", toInt(p));

        vec<Watcher>& ws = watches[p];
        Watcher *i, *j, *end;
        num_props++;

        vec<Watcher>& ws_bin = watches_bin[p]; // Propagate binary clauses first.
        //printf("propagate binary clauses of size: %d...\n" , ws_bin.size());
        for (int k = 0; k < ws_bin.size(); k++) {
            Lit the_other = ws_bin[k].blocker;
            //printf("Get binary blocker...\n");
            if (value(the_other) == l_False) {
                confl = ws_bin[k].cref; // Why conflict? What is literal p satisfies clause?
#ifdef LOOSE_PROP_STAT
                return confl;
#else
                goto ExitProp;
#endif
            }
            else if (value(the_other) == l_Undef) {
                printf("Trying to enqueue 2030 \n");
                //printf("Unqueue var in binary clause for propagation...\n");
                uncheckedEnqueue(the_other, ws_bin[k].cref); // Found lit to add to propagate queue
                //printf("------------------------------------- Enqueue %d to propagate  -------------------------------------  \n", toInt(the_other));
            }

        }
        //printf("Loop through watchers...\n");
        for (i = j = (Watcher*) ws, end = i + ws.size(); i != end;) {
            // Try to avoid inspecting the clause:

            //CRef     cr        = i->cref;
            //printf("CRef is %d\n", cr);
            //i++;
            //continue;

            Lit blocker = i->blocker;
            if (value(blocker) == l_True) { // If any of the variables are true, clause is already satisfied
                *j++ = *i++;
                continue;
            }

            // Make sure the false literal is data[1]:
            //cr        = i->cref;
            CRef cr = i->cref;
            //printf("CRef is %d\n", cr);
            Clause& c = ca[cr];
            Lit false_lit = ~p;
            //printf("c[0] is %d, c[1] is %d, false_lit is %d\n", toInt(c[0]), toInt(c[1]), toInt(false_lit));
            if (c[0] == false_lit) {
                c[0] = c[1];
                c[1] = false_lit;
            }
            assert(c[1] == false_lit); // How are we guaranteed false lit in one of two first indices
            i++;

            // If 0th watch is true, then clause is already satisfied.
            Lit first = c[0];
            Watcher w = Watcher(cr, first);
            if (first != blocker && value(first) == l_True) {
                *j++ = w;
                continue;
            }

            // Look for new watch:
            for (int k = 2; k < c.size(); k++)
                if (value(c[k]) != l_False) { // Could be unassigned
                    c[1] = c[k];
                    c[k] = false_lit;
                    watches[~c[1]].push(w); // There is unit prop in the event of litera ~c[1]. This means than c[1] would cause clause to become true
                    goto NextClause;
                }

            // Did not find watch -- clause is unit under assignment:
            *j++ = w;
            // If that variable that is left over is opposite the literal it needs to be set to, then we have a conflict.
            // Otherwise, the literal and value is aligned and we must propagate out that variable
            if (value(first) == l_False) {
                confl = cr;
                qhead = trail.size();
                // Copy the remaining watches:
                while (i < end)
                    *j++ = *i++;
            }
            else
                uncheckedEnqueue(first, cr);

NextClause:
            ;
        }
        ws.shrink(i - j);
    }

#ifndef LOOSE_PROP_STAT
ExitProp:
    ;
    propagations += num_props;
    simpDB_props -= num_props;
#endif

    return confl;
}

/*_________________________________________________________________________________________________
|
|  reduceDB : ()  ->  [void]
|  
|  Description:
|    Remove half of the learnt clauses, minus the clauses locked by the current assignment. Locked
|    clauses are clauses that are reason to some assignment. Binary clauses are never removed.
|________________________________________________________________________________________________@*/
struct reduceDB_lt {
    ClauseAllocator& ca;

    reduceDB_lt(ClauseAllocator& ca_) : ca(ca_)
    {
    }

    bool operator()(CRef x, CRef y) const {
        return ca[x].activity() < ca[y].activity();
    }
};

void Solver::reduceDB() {
    int i, j;
    //if (local_learnts_dirty) cleanLearnts(learnts_local, LOCAL);
    //local_learnts_dirty = false;

    sort(learnts_local, reduceDB_lt(ca));

    int limit = learnts_local.size() / 2;
    for (i = j = 0; i < learnts_local.size(); i++) {
        Clause& c = ca[learnts_local[i]];
        if (c.mark() == LOCAL)
            if (c.removable() && !locked(c) && i < limit)
                removeClause(learnts_local[i]);
            else {
                if (!c.removable()) limit++;
                c.removable(true);
                learnts_local[j++] = learnts_local[i];
            }
    }
    learnts_local.shrink(i - j);
    checkGarbage();
}

void Solver::reduceDB_Tier2()
{
    int i, j;
    for (i = j = 0; i < learnts_tier2.size(); i++) {
        Clause& c = ca[learnts_tier2[i]];
        if (c.mark() == TIER2)
            if (!locked(c) && c.touched() + 30000 < conflicts) {
                learnts_local.push(learnts_tier2[i]);
                c.mark(LOCAL);
                //c.removable(true);
                c.activity() = 0;
                claBumpActivity(c);
            }
            else
                learnts_tier2[j++] = learnts_tier2[i];
    }
    learnts_tier2.shrink(i - j);
}

void Solver::removeSatisfied(vec<CRef>& cs)
{
    int i, j;
    for (i = j = 0; i < cs.size(); i++) {
        Clause& c = ca[cs[i]];
        if (satisfied(c))
            removeClause(cs[i]);
        else
            cs[j++] = cs[i];
    }
    cs.shrink(i - j);
}

void Solver::safeRemoveSatisfied(vec<CRef>& cs, unsigned valid_mark)
{
    int i, j;
    for (i = j = 0; i < cs.size(); i++) {
        Clause& c = ca[cs[i]];
        if (c.mark() == valid_mark)
            if (satisfied(c))
                removeClause(cs[i]);
            else
                cs[j++] = cs[i];
    }
    cs.shrink(i - j);
}



CRef Solver::propagateLits(vec<Lit>& lits)
{
    Lit lit;
    int i;

    for (i = lits.size() - 1; i >= 0; i--) {
        lit = lits[i];
        if (value(lit) == l_Undef) {
            newDecisionLevel();
            uncheckedEnqueue(lit);
            CRef confl = propagate();
            if (confl != CRef_Undef) {
                return confl;
            }
        }
    }
    return CRef_Undef;
}



vec<Lit>* Solver::get_unassigned_clause(Clause& c )
{
    vec<Lit>* unassigned = new vec<Lit>();
    for (int j = 0; j < c.size(); j++)
    {
        if (value(c[j]) == l_Undef)
        {
            unassigned->push(c[j]);
        }

    }
    return unassigned;         
}

Lit Solver::lit_contained_in(vec<Lit>* literals, vec<Lit>* unassigned_clause)
{
    for (int i=0; i < literals->size(); i++)
    {
        Lit lit_i = (*literals)[i];
        for (int j=0; j < unassigned_clause->size(); j++)
        {
            Lit lit_j = (*unassigned_clause)[j];
            if (lit_i == lit_j)
            {
                return lit_i;
            }
        }
    }
    return toLit(-1);
}

vec<vec<Lit>*>* Solver::get_implied_binary_clauses(Lit literal)
{
    //std::cerr << "Gets to implied binary...\n";
    vec<vec<Lit>*>* binary_clauses = new vec<vec<Lit>*>();
    vec<Lit>* literals = new vec<Lit>();
    literals->push(literal);
    // Get all literals that force literal in binary clauses
    for (int i = 0; i < clauses.size(); i++)
    {
        Clause& c = ca[clauses[i]];
        if (!satisfied(c)) {
            vec<Lit>* unassigned_clause = get_unassigned_clause(c);
            if (unassigned_clause->size() == 2 && (literal == c[0] or literal == c[1]))
            {
                if (literal == c[0])
                {
                    literals->push(c[1]);
                }
                else
                {
                    literals->push(c[0]);
                }
            }
            delete unassigned_clause;
        }
    }
    // Search from ternary clauses that contain any of the literals
    for (int i = 0; i < clauses.size(); i++)
    {
        Clause& c = ca[clauses[i]];
        if (!satisfied(c)) 
        {
            vec<Lit>* unassigned_clause = get_unassigned_clause(c);
            Lit lit_contained = lit_contained_in(literals, unassigned_clause);
            if (toInt(lit_contained) >=0 && unassigned_clause->size() == 3)
            {
                // Create binary clause from rest of clause
                vec<Lit>* binary_clause = new vec<Lit>();
                for (int j=0; j < unassigned_clause->size(); j++)
                {
                    Lit lit_to_add = (*unassigned_clause)[j];
                    if (lit_contained != lit_to_add)
                    {
                        binary_clause->push(lit_to_add);
                    }
                }
                assert(binary_clause->size() == 2);
                binary_clauses->push(binary_clause);
            }
            delete unassigned_clause;
        }

    }
    delete literals;
    return binary_clauses;
}

vec<int>* Solver::get_num_contained_clauses(int length)
{
    //std::cout << "Gets to num contained...\n";
    vec<int>* counter = new vec<int>(nVars()*2,1);
    //TODO: Make a counter map: literal -> counter
    for (int i = 0; i < clauses.size(); i++)
    {
        Clause& c = ca[clauses[i]];
        if (!satisfied(c)) 
        {
            vec<Lit>* unassigned_clause = get_unassigned_clause(c);
            if (unassigned_clause->size() == length)
            {
                for(int j = 0; j < unassigned_clause->size(); j++)
                {
                    Lit lit = (*unassigned_clause)[j];
                    (*counter)[toInt(lit)]++;
                }
            }
        }
    }
    return counter;
}

vec<vec<double>*>* Solver::update_propagate_matrix(torch::Tensor bsh_vals)
{
    /*
     propagate_matrix[i][j] = product over literal in clause i s.t. neq j: bsh_vals of negation of literal
    */
   //std::cout << "Updating propagate matrix...\n";
   vec<vec<double>*>* propagate_matrix = new vec<vec<double>*>();
   for (int i = 0; i < clauses.size(); i++)
    {
        Clause& c = ca[clauses[i]];
        if (!satisfied(c)) 
        {
            vec<Lit>* unassigned_clause = get_unassigned_clause(c);
            vec<double>* clause_propagate = new vec<double>();
            for(int j_1 = 0; j_1 < unassigned_clause->size(); j_1++)
            {
                Lit lit_1 = (*unassigned_clause)[j_1];
                double prod = 1;
                for(int j_2 = 0; j_2 < unassigned_clause->size(); j_2++)
                {
                    Lit lit_2 = (*unassigned_clause)[j_2];
                    if (lit_1 == lit_2) continue;
                    prod *= bsh_vals[toInt(~j_2)].item<double>();
                }
                clause_propagate->push(prod);
            }
            propagate_matrix->push(clause_propagate);
        }
    }
    return propagate_matrix;

}

torch::Tensor Solver::get_bsh_lit_vals(torch::Tensor bsh_vals)
{

    /* Get difference between formula last time. And update values for new added/deleted clauses */
    //std::cout << "Getting bsh vals...\n";
    vec<int>* literals = getActiveLiterals();
    assert(literals->size() > 0);
    
    torch::Tensor bsh = at::ones(nVars()*2);

    if (bsh_vals.size(0) <= 1)
    {
        bsh_vals = at::ones(nVars()*2);
        // Get number of binary / ternary clauses each literal particpates in
        vec<int>* num_binary = get_num_contained_clauses(2);
        vec<int>* num_ternary = get_num_contained_clauses(3);
        for (int i=0; i<literals->size(); i++)
        {
            int index = (*literals)[i];
            bsh_vals[index] = 1 + 2*(*num_binary)[index] + (*num_ternary)[index]; 
        }
    }
    //std::cout << bsh_vals << std::endl;
    vec<vec<double>*>* propagate_matrix = update_propagate_matrix(bsh_vals); //Each index contains the product of bsh from negation of other two literals
    //std::cout << "Finished updating propagate matrix...\n";
    int sat_clause_counter = 0;
    for (int i = 0; i < clauses.size(); i++)
    {
        Clause& c = ca[clauses[i]];
        if (!satisfied(c)) 
        {
            vec<Lit>* unassigned_clause = get_unassigned_clause(c);
            for(int j= 0; j< unassigned_clause->size(); j++)
            {
                Lit lit = (*unassigned_clause)[j];
                bsh[toInt(lit)] += (*(*propagate_matrix)[sat_clause_counter])[j];
            }
            sat_clause_counter++;
        }
    }
    //std::cout << "Set bsh...\n";
    
    return bsh;

//TODO: What about unary clauses? Maybe should be num_unary, num_binary instead??

    // for (int i=0; i<literals->size(); i++)
    // {
    //     double sum = 0.01;
    //     Lit literal = toLit((*literals)[i]);
    //     vec<vec<Lit>*>* binary_clauses = get_implied_binary_clauses(literal);
    //     //std::cerr << "Computing score...\n";
    //     //std::cerr << toInt(literal);
        
    //     //Compute this matrix for all literal all at once. 
    //     for (int j=0; j<binary_clauses->size(); j++)
    //     {
    //         vec<Lit>* binary_clause = (*binary_clauses)[j];
    //         Lit not_u = ~((*binary_clause)[0]);
    //         Lit not_v = ~((*binary_clause)[1]);

    //         sum += (2*(*num_binary)[toInt(not_u)] + (*num_ternary)[toInt(not_u)]) * (2*(*num_binary)[toInt(not_v)] + (*num_ternary)[toInt(not_v)]);
    //         /* above should just be for initialization. After it's just positive bsh * negative bsh */
    //         // if (no_init)
    //         // {
    //         //     sum += (prop_2[not_u]*WEIGHT + prop_3[not_u]) * (prop_2[u]*WEIGHT + prop_3[u]);
    //         // }
    //         delete binary_clause;
    //     }
    //     //std::cerr << "assigning bsh...\n";
    //     bsh[toInt(literal)] = sum;
    //     delete binary_clauses;
    // }

    // /* Looks like bsh should in fact be product of negative and positive literal values. 
    // We'd pick variable and I guess randomly decide which literal after */

    // //std::cerr << bsh;
    // delete num_binary;
    // delete num_ternary;
    // delete literals;
    // return bsh;
}

bool Solver::binResMinimize(vec<Lit>& out_learnt)
{
    // Preparation: remember which false variables we have in 'out_learnt'.
    counter++;
    for (int i = 1; i < out_learnt.size(); i++)
        seen2[var(out_learnt[i])] = counter;

    // Get the list of binary clauses containing 'out_learnt[0]'.
    const vec<Watcher>& ws = watches_bin[~out_learnt[0]];

    int to_remove = 0;
    for (int i = 0; i < ws.size(); i++) {
        Lit the_other = ws[i].blocker;
        // Does 'the_other' appear negatively in 'out_learnt'?
        if (seen2[var(the_other)] == counter && value(the_other) == l_True) {
            to_remove++;
            seen2[var(the_other)] = counter - 1; // Remember to remove this variable.
        }
    }

    // Shrink.
    if (to_remove > 0) {
        int last = out_learnt.size() - 1;
        for (int i = 1; i < out_learnt.size() - to_remove; i++)
            if (seen2[var(out_learnt[i])] != counter)
                out_learnt[i--] = out_learnt[last--];
        out_learnt.shrink(to_remove);
    }
    return to_remove != 0;
}

void Solver::analyze(CRef confl, vec<Lit>& out_learnt, int& out_btlevel, int& out_lbd)
{
    int pathC = 0;
    Lit p = lit_Undef;

    // Generate conflict clause:
    //
    out_learnt.push(); // (leave room for the asserting literal)
    int index = trail.size() - 1;

    do {
        assert(confl != CRef_Undef); // (otherwise should be UIP)
        Clause& c = ca[confl];

        // For binary clauses, we don't rearrange literals in propagate(), so check and make sure the first is an implied lit.
        if (p != lit_Undef && c.size() == 2 && value(c[0]) == l_False) {
            assert(value(c[1]) == l_True);
            Lit tmp = c[0];
            c[0] = c[1], c[1] = tmp;
        }

        // Update LBD if improved.
        if (c.learnt() && c.mark() != CORE) {
            int lbd = computeLBD(c);
            if (lbd < c.lbd()) {
                if (c.lbd() <= 30) c.removable(false); // Protect once from reduction.
                c.set_lbd(lbd);
                if (lbd <= core_lbd_cut) {
                    learnts_core.push(confl);
                    c.mark(CORE);
                }
                else if (lbd <= 6 && c.mark() == LOCAL) {
                    // Bug: 'cr' may already be in 'learnts_tier2', e.g., if 'cr' was demoted from TIER2
                    // to LOCAL previously and if that 'cr' is not cleaned from 'learnts_tier2' yet.
                    learnts_tier2.push(confl);
                    c.mark(TIER2);
                }
            }

            if (c.mark() == TIER2)
                c.touched() = conflicts;
            else if (c.mark() == LOCAL)
                claBumpActivity(c);
        }

        for (int j = (p == lit_Undef) ? 0 : 1; j < c.size(); j++) {
            Lit q = c[j];

            if (!seen[var(q)] && level(var(q)) > 0) {
                if (VSIDS) {
                    varBumpActivity(var(q), .5);
                    add_tmp.push(q);
                }
                else
                    conflicted[var(q)]++;
                seen[var(q)] = 1;
                if (level(var(q)) >= decisionLevel()) {
                    pathC++;
                }
                else
                    out_learnt.push(q);
            }
        }

        // Select next clause to look at:
        while (!seen[var(trail[index--])]);
        p = trail[index + 1];
        confl = reason(var(p));
        seen[var(p)] = 0;
        pathC--;

    }
    while (pathC > 0);
    out_learnt[0] = ~p;

    // Simplify conflict clause:
    //
    int i, j;
    out_learnt.copyTo(analyze_toclear);
    if (ccmin_mode == 2) {
        uint32_t abstract_level = 0;
        for (i = 1; i < out_learnt.size(); i++)
            abstract_level |= abstractLevel(var(out_learnt[i])); // (maintain an abstraction of levels involved in conflict)

        for (i = j = 1; i < out_learnt.size(); i++)
            if (reason(var(out_learnt[i])) == CRef_Undef || !litRedundant(out_learnt[i], abstract_level))
                out_learnt[j++] = out_learnt[i];

    }
    else if (ccmin_mode == 1) {
        for (i = j = 1; i < out_learnt.size(); i++) {
            Var x = var(out_learnt[i]);

            if (reason(x) == CRef_Undef)
                out_learnt[j++] = out_learnt[i];
            else {
                Clause& c = ca[reason(var(out_learnt[i]))];
                for (int k = c.size() == 2 ? 0 : 1; k < c.size(); k++)
                    if (!seen[var(c[k])] && level(var(c[k])) > 0) {
                        out_learnt[j++] = out_learnt[i];
                        break;
                    }
            }
        }
    }
    else
        i = j = out_learnt.size();

    max_literals += out_learnt.size();
    out_learnt.shrink(i - j);
    tot_literals += out_learnt.size();

    out_lbd = computeLBD(out_learnt);
    if (out_lbd <= 6 && out_learnt.size() <= 30) // Try further minimization?
        if (binResMinimize(out_learnt))
            out_lbd = computeLBD(out_learnt); // Recompute LBD if minimized.

    // Find correct backtrack level:
    //
    if (out_learnt.size() == 1)
        out_btlevel = 0;
    else {
        int max_i = 1;
        // Find the first literal assigned at the next-highest level:
        for (int i = 2; i < out_learnt.size(); i++)
            if (level(var(out_learnt[i])) > level(var(out_learnt[max_i])))
                max_i = i;
        // Swap-in this literal at index 1:
        Lit p = out_learnt[max_i];
        out_learnt[max_i] = out_learnt[1];
        out_learnt[1] = p;
        out_btlevel = level(var(p));
    }

    if (VSIDS) {
        for (int i = 0; i < add_tmp.size(); i++) {
            Var v = var(add_tmp[i]);
            if (level(v) >= out_btlevel - 1)
                varBumpActivity(v, 1);
        }
        add_tmp.clear();
    }
    else {
        seen[var(p)] = true;
        for (int i = out_learnt.size() - 1; i >= 0; i--) {
            Var v = var(out_learnt[i]);
            CRef rea = reason(v);
            if (rea != CRef_Undef) {
                const Clause& reaC = ca[rea];
                for (int i = 0; i < reaC.size(); i++) {
                    Lit l = reaC[i];
                    if (!seen[var(l)]) {
                        seen[var(l)] = true;
                        almost_conflicted[var(l)]++;
                        analyze_toclear.push(l);
                    }
                }
            }
        }
    }

    for (int j = 0; j < analyze_toclear.size(); j++) seen[var(analyze_toclear[j])] = 0; // ('seen[]' is now cleared)
}

bool Solver:: checkLitExist(Lit p, vec<Lit> *v) {
    if (v->size() == 0) return false;
    bool retval = false;
    for (int i = 0; i < v->size(); i++) {
        if ((*v)[i] == p) {
            retval = true;
        }
    }
    return retval;
}

void Solver:: removeRedundantLits(vec<Lit>& v) {
    if (v.size() <= 1) return;
    Lit last = v.last();
    v.pop();
    removeRedundantLits(v);
    if (!checkLitExist(last, &v)) {
        v.push(last);
    }
}


double Solver:: get_value_net_sampling_prob(double value_net_leaf_error){
    //Every halving of error, half as close to probability 1.0
    if (value_net_leaf_error > value_error_threshold){
        return 0.0;
    }
    else{
        return 1- (value_net_leaf_error / value_error_threshold);
    }
}

/*_________________________________________________________________________________________________
|
|  search : (nof_conflicts : int) (params : const SearchParams&)  ->  [lbool]
 *   @param  do_monte_carlo_lookahead is boolean identifying whether or not to monte_carlo_lookahead for branch prediction
 *   @param  within_monte_carlo_lookahead is boolean identifying whether the search procedure is within a Monte Carlo lookahead state
 *   @param  branching_literals are a list of literals that were branched on before last conflict
|  
|  Description:
|    Search for a model the specified number of conflicts. 
|  
|  Output:
|    'l_True' if a partial assigment that is consistent with respect to the clauseset is found. If
|    all variables are decision variables, this means that the clause set is satisfiable. 'l_False'
|    if the clause set is unsatisfiable. 'l_Undef' if the bound on number of conflicts is reached.
|________________________________________________________________________________________________@*/

/*
TODO: Figure out state when conflicts arise and how to propogate for this method. We may need to build a separate search over this solver. We don't MCTS updating global state.

 */


lbool Solver::search(int& nof_conflicts, bool search_in_order, bool do_monte_carlo_lookahead, bool within_monte_carlo_lookahead, vec<Lit>*branching_literals, int starting_decision_level, uint64_t decisions_cap, std::vector<std::vector<bool>> * decision_states)
{
    assert(ok);

    // Think this all the information about conflict:
    int backtrack_level;
    int lbd = 0;
    int base_decision_level = decisionLevel();
    vec<Lit> learnt_clause;
    vec<Lit> conflict_clause;
    vec<Lit> flipped_learnt_clause;
    vec<Lit> dpll_stack;
    vec<SearchTrailData> search_trail;
    //int network_calls = 0;
    torch::Tensor last_dist;
    bool rollout;
    literal_bsh_scores = at::zeros(1);
    assert(propagate() == CRef_Undef); ///testing if there will ever be a conflict on the first propegate (i think even with assumptions simplify in main should take care of this)
    bool pure_literal_elimination = false;
    bool rollout_from_unvisited_node = false;
    torch::Tensor last_parent_probs = at::zeros(0);
    bool set_probs_to_parent = false;
    int last_parent_decision_level = decisionLevel();
    // What if we call search from a partial state. We need to reset local state of search
    for (;;) {
        /* if (search_in_order) {
            printf("Doing conflict probing\n");
        } */

        if (search_in_order && conflict_clause.size() == 0) {
            return l_Undef;
        }

        CRef confl = propagate();
        // Print trail
        if (verbosity > 2) {
            printf("Trail:");
            for (int i=0; i < trail.size(); i++)
            {
                printf("%d->", toInt(trail[i]));
            }
            printf("\n");
        }

        /* If conflict, either (1) backtrack DPLL style or (2) return for Knuth probe */
        if (confl != CRef_Undef) {
            printf("Conflict found at decisions level:%d, num active literals:%d\n", decisionLevel(), getActiveLiterals()->size());
            if (pure_literal_elimination) {
                throw std::invalid_argument("At conflict, yet pure literal elimination turned on. Must be bug!");
            }
            conflicts++; 
            nof_conflicts--;

            // If decision level is 0, instance must be UNSAT
            if (decisionLevel() == 0) return l_False;
            backtrack_level = decisionLevel();

            cancelUntil(backtrack_level);

            learnt_clause.clear();
            analyze(confl, learnt_clause, backtrack_level, lbd); 
            conflict_clause.clear();

            removeRedundantLits(conflict_clause);

            for (int i = 0; i < learnt_clause.size(); i++) {
                conflict_clause.push(~learnt_clause[i]);
            }

            //Return immediately if hit a conflict within knuth probe
            if (min_knuth_tree_estimate && within_monte_carlo_lookahead) {
                int current_level = decisionLevel();
                onlineProbeUpdate(current_level-1, base_decision_level, search_trail);
                if (conflict_driven_search && count_aux_solves <= aux_solver_cutoff) {
                    if (search_in_order) {
                        return l_Undef;
                    } else {
                        int new_nofs_conflicts = 2;
                        std::vector<std::vector<bool>> *new_decision_states = new std::vector<std::vector<bool>>();
                        
                        search(new_nofs_conflicts, true, true, true, branching_literals, decisionLevel(), std::numeric_limits<uint64_t>::max(), new_decision_states);

                        delete new_decision_states;
                        count_aux_solves++;
                        if (count_aux_solves % 200 == 0) {
                            printf("Number of auxiliary solves: %d     \r", count_aux_solves);
                            fflush(stdout);
                        }

                        return l_Undef;
                    }
                }
                return l_Undef;
            } else {

                if (do_monte_carlo_lookahead) {
                    int bt_level = (dpll_stack.size() == 0 ? base_decision_level - 1 : vardata[var(dpll_stack.last())].level - 1);
                    int current_level = decisionLevel() - 1;
                    // If subproblem proved UNSAT, update mcts value of state
                    if (terminate_mcts_path_at_conflict){
                        //Update value of probe and terminate
                        printf("Terminating MCTS path at conflict decisions level:%d\n", decisionLevel());
                        updateMCTSValueProbe(current_level, base_decision_level, search_trail, 0);
                        return l_Undef;
                     }else{
                        updateMCTSValues(bt_level, current_level, base_decision_level, decisions, search_trail, 0);
                    }
                }
                
                if (dpll_stack.size() == 0) return l_False;
                cancelUntil(vardata[var(dpll_stack.last())].level - 1);
                newDecisionLevel();
                printf("Decisions: %lu\t Decision Level: %d\t Forced Literal: %d\n", decisions, decisionLevel(), toInt(dpll_stack.last()));
                uncheckedEnqueue(dpll_stack.last());
                dpll_stack.pop();
            }
               
        } else {
            // NO CONFLICT
            //printf("Choose branching variable\n");
            Lit next = lit_Undef;
            int lit_index;
            
            if (!within_monte_carlo_lookahead and save_solver_state) {
                save_state();
            }
            rollout = rollout_from_unvisited_node or decisionLevel() >= depth_before_rollout or getActiveLiterals()->size() < num_variables_for_rollout*2;
            decisions++;
            if (!rollout and !(*data_table)[assignToVec(assigns)] and (do_monte_carlo_lookahead || within_monte_carlo_lookahead)) {
                //printf("Create new node\n");
                if (sum_values_by_depth.count(decisionLevel()) == 0) {
                    sum_values_by_depth.insert(std::pair<int, double>(decisionLevel(), 0));
                    occurances_by_depth.insert(std::pair<int, int>(decisionLevel(), 1));
                } else if (occurances_by_depth.count(decisionLevel()) == 0) {
                    std::cerr << "In search() - no conflict: Should have reached this depth, there's a bug.";
                    throw;
                } else {
                    occurances_by_depth[decisionLevel()]++;
                }
                (*data_table)[assignToVec(assigns)] = new DataCentre((*data_table).size(), assignToVec(assigns).size(),
                                                    min_knuth_tree_estimate, failure_prob, mcts_samples_per_lb_update, 
                                                    BETA, decisionLevel(), prior_temp, use_qd, fix_mcts_policy_to_prior);
                double new_Qd = sum_values_by_depth[decisionLevel()]/occurances_by_depth[decisionLevel()];
                if (new_Qd > 0) {
                    (*data_table)[assignToVec(assigns)]->setQd(new_Qd);
                }
                // if (within_monte_carlo_lookahead){
                //     rollout = true;
                // }   
                if (dynamic_node_expansion and within_monte_carlo_lookahead){
                    //printf("Rollout from unvisited node next branch\n");
                    rollout_from_unvisited_node = true;
                    rollout = true;
                }
            }

            int branching_procedure;
            if (search_in_order) {
                branching_procedure = constants::CONFLICT_PROBING;
            } else if (rollout) {
                // No decision made if rolling out
                decisions--;

                // When (1) solving online or (2) when VALUE_NET turned off, call subsolver
                if (!within_monte_carlo_lookahead or strcmp(rollout_policy, "VALUE_NET") != 0) {
                    branching_procedure = constants::SUBSOLVER_ROLLOUT;
                }
                // When using value net and not at leaf node, use value net
                else if (decisionLevel() < depth_before_rollout){
                    branching_procedure = constants::VALUE_NET;
                // Rollout at leaf node. Select between solver and value net based on value net error
                } else{

                    if(!(*data_table)[assignToVec(assigns)])
                    {
                        (*data_table)[assignToVec(assigns)] = new DataCentre((*data_table).size(), assignToVec(assigns).size(),
                                                    min_knuth_tree_estimate, failure_prob, mcts_samples_per_lb_update, 
                                                    BETA, decisionLevel(), prior_temp, use_qd, fix_mcts_policy_to_prior);
                    }

                    // Rollout, so always call network
                    set_neural_net_prior(assignToVec(assigns), at::zeros(0), false);

                    //printf("\nDecision level: %d, Value net multiplicative error: %f\n", decisionLevel(), value_net_leaf_error);
                    double prob_sampling_value_net = get_value_net_sampling_prob(value_net_leaf_error);
                    bool call_value_net = value_net_leaf_count > 0 and (drand(random_seed) < prob_sampling_value_net);
                    if (verbosity >= 2)
                    {
                        if (call_value_net){
                            printf("Rollout from leaf node. Calling value net\n");
                        } else{
                            printf("Rollout from leaf node. Calling subsolver\n");
                        }
                    }
                    call_value_net ? branching_procedure = constants::VALUE_NET : branching_procedure = constants::SUBSOLVER_ROLLOUT;
                    //printf("\nDecision level: %d, Value net multiplicative error: %f, Calling value net: %d\n", decisionLevel(), value_net_leaf_error, call_value_net);
                }
            } 
            else if (do_monte_carlo_lookahead) {
                branching_procedure = constants::MONTE_CARLO_LOOKAHEAD;
            } else if (within_monte_carlo_lookahead) {
                branching_procedure = constants::BANDIT_BRANCHING;
            } else {
                branching_procedure = constants::STANDARD_BRANCHING;
            }

            switch (branching_procedure) {
                case constants::CONFLICT_PROBING: {
                    if (conflict_clause.size() == 0) {
                        throw std::invalid_argument("Conflict clause size is 0, should have a conflict and returned before this point \n");
                    }
                    next = conflict_clause.last();
                    conflict_clause.pop();
                    while (value(next) != l_Undef && conflict_clause.size() > 0) { 
                        next = conflict_clause.last();
                        conflict_clause.pop();
                    }
                    if (conflict_clause.size() == 0) {
                        int current_level = decisionLevel();
                        onlineProbeUpdate(current_level-1, base_decision_level, search_trail);
                        return l_Undef;
                    } 
                    break;
                }
                case constants::MONTE_CARLO_LOOKAHEAD : {
                    set_neural_net_prior(assignToVec(assigns), at::zeros(0), false);
                    next = monteCarloLookahead();
                    lit_index = toInt(next);
                    printf("Finished monteCarloLookahead\n");
                    break;
                }
                case constants::BANDIT_BRANCHING : {

                    //printf("Bandit branching within lookahead\n");
                    int decision_level_difference = decisionLevel() - last_parent_decision_level;
                    // Set the node to last queried parent if parent prob non null and within depth threshold
                    if (decision_level_difference < neural_net_refresh_rate and last_parent_probs.size(0) > 1 and decisionLevel() > 1){
                        set_probs_to_parent = true;
                    } else{
                        set_probs_to_parent = false;
                    }
                    set_neural_net_prior(assignToVec(assigns), last_parent_probs, set_probs_to_parent);
                    torch::Tensor lit_probs = (*data_table)[assignToVec(assigns)]->getNeuralProbabilities();
                    if (!set_probs_to_parent){
                        last_parent_probs = lit_probs;
                        last_parent_decision_level = decisionLevel();
                    }
                    vec<int> * active_lits = getActiveLiterals();
                    torch::Tensor upper_bounds = (*data_table)[assignToVec(assigns)]->getRewardUpperBounds();
                    //std::cout << "upper bounds: " << upper_bounds << std::endl;
                    lit_index = banditPolicy->selectAction(upper_bounds,
                                                         (*data_table)[assignToVec(assigns)]->getCounts(),
                                                         active_lits,
                                                         lit_probs);
                    
                    break;
                }
                case constants::STANDARD_BRANCHING : {
                    //printf("Standard branching\n");
                    // Either (1) BSH, (2) Random, (3) Neural network
                    vec<double> *counts = new vec<double>(nVars()*2, 0);
                    vec<double> *values = new vec<double>(nVars()*2, 0);
                    bool refresh_literal_probabilities= false;
                    if ( (decisions-1) % neural_net_refresh_rate == 0){
                        refresh_literal_probabilities = true;
                    }
                    next = pickBranchLit(counts, values, refresh_literal_probabilities, decisionLevel());
                    lit_index = toInt(next);
                    break;
                }
                case constants::SUBSOLVER_ROLLOUT : {
                    // Solve complete subproblem with external solver
                    int external_solver_decisions;
                    lbool solved_status = l_Undef;
                    // Check if problem is in rollout_cache
                    if (!(*rollout_cache)[assignToVec(assigns)])
                    {   
                        solved_status = this->rollout_with_external_solver(&external_solver_decisions);
                        if (verbosity > 1){
                            printf("Rollout full subtree with external solver: %d\n", external_solver_decisions);
                        }
                        (*rollout_cache)[assignToVec(assigns)] = external_solver_decisions;
                    } else {
                        external_solver_decisions = (*rollout_cache)[assignToVec(assigns)];
                        if (verbosity > 1){
                            printf("Rollout full subtree with cached value: %d\n", external_solver_decisions);
                        }
                    }
                    assert(external_solver_decisions >= 1);
                    decisions += external_solver_decisions; // external_solver_decisions in different units than network calls. But network calls roughly held constant across samples.
                    external_subsolver_calls++;
                    if (within_monte_carlo_lookahead) {
                        //Evaluate error at leaf node
                        if(decisionLevel() == depth_before_rollout and strcmp(rollout_policy, "VALUE_NET") == 0 )
                        {
                            double value = (*data_table)[assignToVec(assigns)]->getValue();
                            double error = abs(log2(value) - log2(external_solver_decisions));
                            // Convert to multiplicative error so consistent across value net scaling
                            value_net_leaf_error += (error - value_net_leaf_error) / (value_net_leaf_count+1);
                            value_net_leaf_count += 1;
                        }
                        assignToVec(assigns);
                        if (solved_status != l_True) {
                            int current_level = decisionLevel() - 1;
                            if (min_knuth_tree_estimate) {
                                onlineProbeUpdate(current_level, base_decision_level, search_trail, log2(external_solver_decisions));
                                decisions -= external_solver_decisions;
                                decisions = pow(2,decisions) * external_solver_decisions + pow(2,decisions)-1;
                                return l_Undef;
                            }
                        }
                    }
                    if (solved_status == l_True){
                        return l_True;
                    } else{

                        if (do_monte_carlo_lookahead){
                            printf("Rollout from leaf: %d decisions\n", external_solver_decisions);
                            int bt_level = (dpll_stack.size() == 0 ? base_decision_level - 1 : vardata[var(dpll_stack.last())].level - 1);
                            int current_level = decisionLevel() - 1;
                            // If subproblem proved UNSAT, update mcts value of state 
                            if (terminate_mcts_path_at_conflict){
                                //Update value of probe and terminate
                                double expected_tree_size = pow(2, decisionLevel()) * external_solver_decisions;
                                printf("Terminating MCTS path at depth %d for external solver rollout with %d decisions. Expected tree size:%f\n", decisionLevel(), external_solver_decisions, expected_tree_size);
                                updateMCTSValueProbe(current_level, base_decision_level, search_trail, log2(external_solver_decisions));
                                return l_Undef;
                            }else{
                                updateMCTSValues(bt_level, current_level, base_decision_level, decisions, search_trail, 0);
                                
                            }
                        }
                        if (dpll_stack.size() == 0) return l_False;
                        cancelUntil(vardata[var(dpll_stack.last())].level - 1);
                        newDecisionLevel();
                        uncheckedEnqueue(dpll_stack.last());
                        dpll_stack.pop();
                        continue;
                    }
                    break;
                }
                
                case constants::VALUE_NET : {
                    if (!within_monte_carlo_lookahead){
                        throw std::invalid_argument("Value net can only be used outside of monte carlo lookahead\n");
                    }
                    if(!(*data_table)[assignToVec(assigns)])
                    {
                        (*data_table)[assignToVec(assigns)] = new DataCentre((*data_table).size(), assignToVec(assigns).size(),
                                                    min_knuth_tree_estimate, failure_prob, mcts_samples_per_lb_update, 
                                                    BETA, decisionLevel(), prior_temp, use_qd, fix_mcts_policy_to_prior);
                    }

                    set_neural_net_prior(assignToVec(assigns), at::zeros(0), false);
                    double value_prediction = (*data_table)[assignToVec(assigns)]->getValue();
                    int current_level = decisionLevel() - 1;
                    onlineProbeUpdate(current_level, base_decision_level, search_trail, log2(value_prediction));
                    decisions = pow(2,decisions) * value_prediction + pow(2,decisions)-1;
                    return l_Undef;

                    break;
                }

            }


            if (!search_in_order) {
                next = selectPolarity(var(toLit(lit_index)));
            }

            if (next != lit_Undef) {
                branching_literals->push(next);
                search_trail.push(mkSearchData(next,new std::vector<bool>(assignToVec(assigns)), decisions));        
            }

            if (next == lit_Undef) {
                // Search is finished - update datapoints with true
                return l_True;
            }
            
            dpll_stack.push(~next); /// Next to bracktrack to ~next when we find conflict
            newDecisionLevel();

            if (!within_monte_carlo_lookahead) {
                printf("Decisions: %lu\t Decision Level: %d\t Decided Literal: %d\n", decisions, decisionLevel(), toInt(next));
                if (inquire_tree && !search_in_order) {
                    printf("Made a decision, now return from search\n");
                    return l_Undef;
                }
                if (cleanup_datatable) {
                    int old_size = data_table->size();
                    printf("------------------- Start cleaning up data table -------------------\n");
                    printf("Size before:%d\n", old_size);
                    doCleanUpDataTable();
                    int new_size = data_table->size();
                    printf("Size after:%d\n", new_size);
                    double percentage_deleted = 100*(old_size-new_size)/old_size;
                    printf("Deleted %.3f percent of data table\n", percentage_deleted);
                    printf("------------------- Finish cleaning up data table -------------------\n");
                   
                }
            }
            
            uncheckedEnqueue(next);
            if (verbosity >=3){
                printf("Enqueued literal");
            }
        }
    }
}

Lit Solver::selectPolarity(Var v){

    if (verbosity >=3){
        printf("Selecting polarity for variable");
    }

    if (do_importance_sampling){
        double false_value = 0;
        double true_value = 0;
        for (int i = 0; i < 2; i++){
            // Create false literal
            Lit l;
            if(i == 0){
                l = mkLit(v, false);
            }else{
                l = mkLit(v, true);
            }
            // Assign literal
            newDecisionLevel();
            uncheckedEnqueue(l);
            std::vector<bool> assignment = assignToVec(assigns);
            if(!(*data_table)[assignment]) // If doesn't exist, use uniform sampling
            {
                cancelUntil(decisionLevel()-1);
                (*importance_sample_ratio)[assignToVec(assigns)] = 0.5;
                return mkLit(v, pickPolarityFromDistribution(0.5));  
            }
            // Get value estimate
            set_neural_net_prior(assignment, at::zeros(0), false);
            if (i == 0){
                false_value = (*data_table)[assignment]->getValue();
            } else {
                true_value = (*data_table)[assignment]->getValue();
            }
            // Undo assignment
            cancelUntil(decisionLevel()-1);
        }
        // Save ratio of value estimates to dict
        if (verbosity >=3){
            printf("Doing importance sampling\n");
        }
        
        double prob_true = true_value / (true_value + false_value);
        (*importance_sample_ratio)[assignToVec(assigns)] = prob_true;
        return mkLit(v, pickPolarityFromDistribution(prob_true));
    }
    
    //Uniform random
    if (verbosity >=3){
        printf("Selecting polarity from uniform distribution");
    }
    return mkLit(v, pickPolarityFromDistribution(0.5));   
    

}

void Solver::save_state(){

    printf("Saving solver state\n");
    std::string file;
    file.append(data_dir);
    file.append("/");
    file.append(getFileName(model_filename));
    file.append("/subproblems/");
    file.append(getFileName(instance_name));
    file.append("_numactive");
    file.append(std::to_string(getActiveLiterals()->size()));
    file.append("_");
    file.append(std::to_string(decisions));
    file.append(".cnf");

    //TODO: convert string to const char*
    const char *file_c = file.c_str();
    FILE*f = fopen(file_c, "wr");
    if (f == NULL)
        fprintf(stderr, "could not open file %s\n", file.c_str()), exit(1);
    stateToDimacs(f);
    fclose(f);
                
}
/* Set the neural network prior if not already set. */
void Solver::set_neural_net_prior(std::vector<bool> state, torch::Tensor last_parent_probs, bool set_probs_to_parent){ 
    // Set neural net prior if not already set
    
    DataCentre* node = (*data_table)[state];
    if (node->getNeuralProbabilities().size(0) <= 1) {
        if ( (strcmp(rollout_policy, "VALUE_NET") == 0) or (use_neural_net_prior and decisionLevel() < neural_net_prior_depth_threshold) ) {
            
            
            torch::Tensor probabilities;
            double value;
            if (set_probs_to_parent) {
                //printf("seting probs to parent\n");
                probabilities = last_parent_probs;
                value = 1;
            } else {
                if (verbosity >= 3) {
                    printf("Getting value and probabilities from neural net\n");
                }
                std::vector<torch::Tensor> network_outputs = getNeuralPrediction(at::zeros(1));
                probabilities = network_outputs[0];
                value = network_outputs[1][0].item<double>();
            }

            if (use_neural_net_prior){
                //printf("Setting neural net prior\n");
                assert(probabilities.size(0) > 1);
                node->setNeuralProbabilities(probabilities);
            }
            node->setValue(value);
            // Update linked list 
            //printf("Update linked list\n");
            if (nn_solution_cache->exists(state)) {
                nn_solution_cache->moveFromMiddle(state);
            } else {
                std::vector<bool> deleted_key = nn_solution_cache->insert(state);
                if (deleted_key.size() > 0) {
                    (*data_table)[deleted_key]->setNeuralProbabilities(at::zeros(1));
                }
            }
        }
        else {
            (*data_table)[state]->setNeuralProbabilities(at::zeros(1));//TODO: Set to sum to 1 over active variables??
        }
    }
}

lbool Solver::rollout_with_external_solver(int *external_solver_decisions){
    //Write current state to .cnf
    std::string file;
    std::string random_suffix = gen_random(10);
    file.append("/tmp/");
    // Number of active variables
    file.append(std::to_string(getActiveLiterals()->size()));
    // Number of variables
    file.append("_");
    file.append(std::to_string(nVars()));
    file.append("_");
    file.append(std::to_string(decisionLevel()));
    file.append(random_suffix);
    file.append(".cnf");
    const char *file_c = file.c_str();
    FILE*f = fopen(file_c, "wr");
    if (f == NULL)
        fprintf(stderr, "could not open file %s\n", file.c_str()), exit(1);
    std::map<int64_t,int64_t> var_reindexing = stateToDimacs(f);
    fclose(f);

    std::map<int64_t,int64_t> var_backwards_reindexing;
    for (std::map<int64_t,int64_t>::iterator it=var_reindexing.begin(); it!=var_reindexing.end(); ++it)
    {
        var_backwards_reindexing.insert(std::pair<int64_t,int64_t>(it->second,it->first));

    }

    //Execute solver command string with .cnf (enforce that it is a wrapper that return a generic format)
    std::string external_solver;
    // Check in external_solver_executable contains the string "kcnfs"

    if (std::string(external_solver_executable).find("kcnfs") != std::string::npos)
    {
        external_solver = "kcnfs";
    }
    else if (std::string(external_solver_executable).find("minisat") != std::string::npos)
    {
        external_solver = "minisat";
    }
    else 
    {
        throw std::runtime_error("Unknown external solver");
    }
   
    std::string call_string = external_solver_executable;
    
    // Check whether solver is "minisat"
    if (external_solver == "kcnfs")
    {
        call_string.append(" -nop "); // kcnfs does not count decisions properly if this option not set
    }
    else{
        call_string.append(" ");
    }
    call_string.append(file);
    call_string.append(" > ");
    std::string solver_output = "/tmp/extern_solver.out";
    solver_output.append(random_suffix);
    call_string.append(solver_output);
    if (verbosity > 1)
    {
        printf("Calling %s\n", call_string.c_str());
    }
    //int call_string_code = 
    system(call_string.c_str());
    // if (call_string_code == -1)
    // {
    //     printf("Call string Failed: %s\n",call_string.c_str());//System call status:%d\n", system_call_status);
    //     std::cout << "Error: " << strerror(errno) << '\n';
    // }

    //Read solver output
    std::ifstream infile(solver_output);
    std::string line;
    lbool solved = l_Undef;
    while (std::getline(infile, line))
    {
        std::size_t decisions_found;
        if (external_solver == "kcnfs"){
            decisions_found= line.find("Size of search tree");
        }
        else if (external_solver == "minisat"){
            decisions_found= line.find("decisions");
        }
        else{
            throw std::runtime_error("Unknown external solver");
        }
        std::size_t unsat_status_found = line.find("UNSATISFIABLE");
        std::size_t sat_status_found = line.find("SATISFIABLE");
        if (decisions_found!=std::string::npos)
        {
            size_t first_position = line.find_first_of(":") + 1;
            size_t last_position;
            if (external_solver == "kcnfs"){
                last_position = line.find_first_of("nodes") - 1;
            }
            else if (external_solver == "minisat"){
                last_position = line.find_first_of("(") - 1;
            }
            else{
                throw std::runtime_error("Unknown external solver");
            }
            std::string sub_string = line.substr(first_position, last_position - first_position);
            *external_solver_decisions = std::stoi(sub_string);//
        }
        else if(unsat_status_found != std::string::npos)
        {
            solved = l_False;
        }
        else if(sat_status_found != std::string::npos)
        {
            solved = l_True;
        }       
    }
    infile.close();

    // If SATisfiable, assign variables accordingly
    if (solved == l_True) {
        gzFile input_stream = gzopen(solver_output.c_str(), "rb");
        StreamBuffer in(input_stream);
        vec<Lit>* subsolver_assignments = parse_DIMACS_Solution(in);
        

        for (int i =0; i < subsolver_assignments->size(); i++) {
            Lit p = (*subsolver_assignments)[i];
            int var_index = int(var(p));
            Var true_var = var_backwards_reindexing[var_index+1];

            assert(value(true_var) == l_Undef);
            assigns[true_var] = lbool(!sign(p));
        }
        gzclose(input_stream); 
    }

    if (solved==l_Undef) {
        printf("Call string failed: %s\n", call_string.c_str());
        std::ifstream infile(solver_output);
        std::string line;
        while (std::getline(infile, line))
        {
            std::cerr << line;
        }
         std::cout << "Error: " << strerror(errno) << '\n';
        throw;
    }
    if (verbosity > 2) {
        printf("Solved subproblem in %d decisions\n", *external_solver_decisions);
    }
    //Delete these /tmp files
    std::remove(solver_output.c_str());

    if (save_subsolver_rollout_calls){  
        // TODO: need to get instance name
        std::string rollout_file;
        rollout_file.append(data_dir);
        rollout_file.append("/");
        rollout_file.append(getFileName(model_filename));
        rollout_file.append("/subproblems/");
        rollout_file.append(getFileName(instance_name));
        rollout_file.append("_");
        rollout_file.append(std::to_string(getActiveLiterals()->size()));
        rollout_file.append("_");
        rollout_file.append(std::to_string(nVars()));
        rollout_file.append("_");
        rollout_file.append(std::to_string(decisionLevel()));
        rollout_file.append("_");
        rollout_file.append(std::to_string(*external_solver_decisions));
        rollout_file.append("_");
        rollout_file.append(random_suffix);
        rollout_file.append(".cnf");
        const char *file_c = rollout_file.c_str();
        FILE*f = fopen(file_c, "wr");
        if (f == NULL)
            fprintf(stderr, "could not open file %s\n", rollout_file.c_str()), exit(1);
        stateToDimacs(f);
        fclose(f);
    }
    std::remove(file.c_str());
    return solved;
}

/* Update (backpropagate) lookahead tree with Knuth probe result */
void Solver::onlineProbeUpdate(int cur_level, int base_level, vec<SearchTrailData> & search_trail, double starting_depth) {
    
    assert(starting_depth >= 0);
    //printf("starting depth %f\n", starting_depth);
    double itr = starting_depth;
    double depth_above_leaf = 0;
    double tree_size_below_leaf = pow(2, itr);
    double tree_size_above_leaf;
    double tree_size;
    SearchTrailData * sd = nullptr;
    while (cur_level >= base_level) {
        
        sd = &search_trail[cur_level - base_level];
        if (!(*data_table)[*(sd->s_state)]){
            throw std::runtime_error(std::string("Data table not initialized for node at depth "));//Can't update node that is not yet created
        } else{
            if (sum_values_by_depth.count((*data_table)[*(sd->s_state)]->getDepth()) == 0) {
                sum_values_by_depth.insert(std::pair<int, double>((*data_table)[*(sd->s_state)]->getDepth(), 0));
                occurances_by_depth.insert(std::pair<int, int>((*data_table)[*(sd->s_state)]->getDepth(), 1));
            }
            // Tree size = below rollouts + above rollouts
            if (do_importance_sampling){
                tree_size_below_leaf /= (*importance_sample_ratio)[*(sd->s_state)];
            }
            else{
                tree_size_below_leaf *= 2;
            }
            tree_size_above_leaf =  pow(2, depth_above_leaf+1) -1;
            tree_size = tree_size_below_leaf + tree_size_above_leaf;
            double log_tree_size = log2(tree_size);
            (*data_table)[*(sd->s_state)]->addNodeData(new LookaheadData(toInt(sd->lit), log_tree_size, false, log_tree_size));
            (*data_table)[*(sd->s_state)]->addNodeData(new LookaheadData(toInt(~sd->lit), log_tree_size, false, log_tree_size));
        
            occurances_by_depth[(*data_table)[*(sd->s_state)]->getDepth()]++;
            double new_reward = (*data_table)[*(sd->s_state)]->getStateMeanTreeSize(); //printf("Inverted new reward line 3243: %lf\n", 1/new_reward);
            sum_values_by_depth[(*data_table)[*(sd->s_state)]->getDepth()] += new_reward; //- old_reward;
            if (occurances_by_depth.count((*data_table)[*(sd->s_state)]->getDepth()) == 0 || sum_values_by_depth.count((*data_table)[*(sd->s_state)]->getDepth()) == 0) {
                std::cerr << "In onlineProbeUpdate(): Should have created a node here, there's a bug.\n";
                throw;
            }
            double new_Qd = sum_values_by_depth[(*data_table)[*(sd->s_state)]->getDepth()]/occurances_by_depth[(*data_table)[*(sd->s_state)]->getDepth()];
            (*data_table)[*(sd->s_state)]->setQd(new_Qd);
        } 
        //printf("OnlineProbeUpdate: itr %f, Lit %d\n", itr, toInt(sd->lit));
        search_trail.pop();
        cur_level--;
        itr++;
        depth_above_leaf++;

        delete sd->s_state;
    }
}

/* Update knuth probe value for MCTS decisions */
void Solver::updateMCTSValueProbe(int cur_level, int base_level, vec<SearchTrailData> & search_trail, double starting_depth) {
    
    double itr = starting_depth;
    SearchTrailData * sd = nullptr;
    while (cur_level >= base_level) {
        sd = &search_trail[cur_level - base_level];
        if (!(*data_table)[*(sd->s_state)]){
            throw std::runtime_error(std::string("Data table not initialized for node. Can't update Table."));
        } else {
            double tree_size = pow(2,itr+1);//2^depth of probe
            if (use_mcts_db) {
                int data_id = (*mcts_data_ids)[*(sd->s_state)];
                std::string base = "mysql -N -B -h address -P port -u username -p\"password\" database -e ";
                std::string command = base + "\"UPDATE Data SET value = " + std::to_string(tree_size) + " WHERE data_id = " + std::to_string(data_id) + ";\"";
                //std::cout << command;
                exec(command.c_str());
            }
        }
        search_trail.pop();
        cur_level--;
        itr++;

        delete sd->s_state;
    }
}


/* Update MCTS values for completed subproblems */
void Solver::updateMCTSValues(int bt_level, int cur_level, int base_level, int decisions, vec<SearchTrailData> & search_trail, int starting_decision_count) {
    
    int itr = 0;
    SearchTrailData * last_sd = nullptr;
    SearchTrailData * sd = nullptr;
    while (cur_level > bt_level) {
        sd = &search_trail[cur_level - base_level];
        sd->reward_f = sd->decisions_f - decisions;
        double full_reward = sd->reward_f + sd->reward_d - 1;
        /* Update database with MCTS values */
        if (use_mcts_db) {
            int data_id = (*mcts_data_ids)[*(sd->s_state)];
            std::string base = "mysql -N -B -h address -P port -u username -p\"password\" database -e ";
            std::string command = base + "\"UPDATE Data SET value = " + std::to_string(-1*full_reward) + " WHERE data_id = " + std::to_string(data_id) + ";\"";
            std::cout << command;
            exec(command.c_str());
        }
        (*mcts_values)[*(sd->s_state)]= full_reward;

        if (itr > 0) {
            delete last_sd->s_state;
        }
        last_sd = sd;
        search_trail.pop();
        cur_level--;
        itr++;
    }

    if (search_trail.size()) {
        sd = &search_trail[cur_level - base_level];
        sd->reward_d = sd->decisions_d - decisions;
        sd->decisions_f = decisions;
        sd->cur_forced = true;
        sd->lit = ~sd->lit;

        if (itr > 0)
        {
            delete last_sd->s_state;
        }
    }
    else
    {
        if (itr > 0)
        {
            delete sd->s_state;
        }
    }
}

Lit Solver::pickUnconstrainedLiteral() {

    bool flag = false;
    for (int i = 0; i < nVars(); i++) {
        if (value(i) == l_Undef) {
            flag = true;
        }
    }

    if (!flag) {
        return lit_Undef;
    }

    vec<int>* unconstrained_literals = getUnconstrainedLiterals();
    Lit return_lit = lit_Error;

    if (unconstrained_literals->size() > 0) {
        for (int i = 0; i < unconstrained_literals->size(); i++) {
            return_lit = toLit((*unconstrained_literals)[i]);
            if (value(return_lit) != l_Undef) {
                return_lit = lit_Error;
                continue;
            }
            else
            {
                break;
            }
        }

    }

    unconstrained_literals->~vec();
    //delete unconstrained_literals;
    return return_lit;
}

vec<int>* Solver::getUnconstrainedLiterals() {

    vec<int> *candidate_unconstrained_lits = new vec<int>(0, 0);
    vec<int> *constrained_lits = new vec<int>(0, 0);
    vec<int> *unconstrained_lits = new vec<int>(0, 0);
    vec<int> *lits_to_be_returned = new vec<int>(0, 0);
    vec<int> *candidate_pure_lits = new vec<int>(0, 0);


    for (int i = 0; i < nVars()*2; i++) {
        candidate_pure_lits->push(0);
    }
    

    for (int i = 0; i < nClauses(); i++) {
        Clause& c = ca[clauses[i]];

        // Check for true literal (clause is satisfied)
        bool flag = false;
        for (int j = 0; j < c.size(); j++) {
            if (value(c[j]) == l_True) {
                flag = true;
            }
        }

        // Literals in this clause may be unconstrained because it is satisfied
        if (flag) {
            for (int j = 0; j < c.size(); j++) {
                int literal_as_int = toInt(c[j]);
                candidate_unconstrained_lits->push(literal_as_int);
            }
        }
        // Variables are definitely constrained
        else {
            for (int j = 0; j < c.size(); j++) {
                int literal_as_int = toInt(c[j]);
                constrained_lits->push(literal_as_int);
                if (value(c[j]) == l_Undef) {
                    (*candidate_pure_lits)[literal_as_int] += 1;
                }
            }
        }
    }

    for (int i = 0; i < candidate_unconstrained_lits->size(); i++) {
        for (int j = 0; j < constrained_lits->size(); j++) {
            // Not actually unconstrained - moving on to next literal
            if ((*constrained_lits)[j] == (*candidate_unconstrained_lits)[i]) {
                goto nextLiteral;
            }
        }

        unconstrained_lits->push((*candidate_unconstrained_lits)[i]);
nextLiteral:
        continue;
    }

    for (int i = 0; i < unconstrained_lits->size(); i++) {
        int literal_to_check = (*unconstrained_lits)[i];
        int converse_literal_to_check = toInt(~toLit(literal_to_check));

        bool flag = false;
        for (int j = 0; j < unconstrained_lits->size(); j++) {
            if ((*unconstrained_lits)[j] == converse_literal_to_check) {
                flag = true;
            }
        }

        if (flag) {
            lits_to_be_returned->push(literal_to_check);
            lits_to_be_returned->push(converse_literal_to_check);
        }
    }
    

    for (int i = 0; i < candidate_pure_lits->size(); i+=2) 
    {
        // Pure literal if only one literal of variable ever set
        if (  (*candidate_pure_lits)[i] == 0 and (*candidate_pure_lits)[i+1] > 0)
        {
            lits_to_be_returned->push(i+1);
        }
        else if((*candidate_pure_lits)[i] > 0 and (*candidate_pure_lits)[i+1] == 0)
        {
            lits_to_be_returned->push(i);
        }
    }

    delete candidate_unconstrained_lits;
    delete constrained_lits;
    //delete unconstrained_lits;
    delete candidate_pure_lits;

    return lits_to_be_returned;
}

/**
 *   @brief  Perform MCTS lookaheads
 *   
 *   @return Best Lit based on MCTS lookahead
 */
Lit Solver::monteCarloLookahead()//int& backtrack_level_arg, int& lbd_arg, vec<Lit> learnt_clause_arg, bool cached_arg)
{
    printf("Doing MonteCarlo Lookahead for decision %lu (depth %d, %d active literals). Peak memory used: %.2f MB. CPU time %.2fs\n", decisions, decisionLevel(), getActiveLiterals()->size(), memUsedPeak(), cpuTime());

    int num_samples_in_tree = ((*data_table)[assignToVec(assigns)] ? (*data_table)[assignToVec(assigns)]->getDataCount() : 0);
    num_samples_in_tree /= 2;
    int num_samples = num_monte_carlo_tree_search_samples - num_samples_in_tree;

    printf("Lookahead tree has %d samples of %d needed\n", num_samples_in_tree, num_monte_carlo_tree_search_samples);
    bool monte_carlo_search = false;
    bool within_monte_carlo_lookahead = true;
    lbool result;/// this will be a bug when we dont need to sample
    bool required_run = true;
    //Dictionary of counts by depth
    std::unordered_map<int, int> * samples_by_depth = new std::unordered_map<int, int>();

    vec<vec<lbool> * > *sat_solutions = new vec<vec<lbool> * >();
    int decisions_cap =  cutoff_time;
    //Create copy of solver state once before taking MCTS sample, so the global state in the outer loop remains untouched.
    Solver lookahead_solver = Solver(*this); // Create new solver copy for every lookahead search. It's a pointer so Solver should take pointer as input

    time_t last_cpu_time;
    time(&last_cpu_time);
    int logging_frequency = 100;
    for (int mcts_iter = 0; mcts_iter < num_samples or required_run; mcts_iter++) {

        if (mcts_iter % logging_frequency == 1)
        {
            logProgress(last_cpu_time, mcts_iter, num_samples, logging_frequency);
        }        
        time(&last_cpu_time);
        // Create new branching literals for every monte-carlo sample
        vec<Lit> *branching_literals = new vec<Lit>();
        std::vector<std::vector<bool>> *decision_states = new std::vector<std::vector<bool>>();
      
        int nof_conflicts = 1000000;
        vec<int> * active_lits = getActiveLiterals();
        result = lookahead_solver.search(nof_conflicts, false, monte_carlo_search, within_monte_carlo_lookahead, branching_literals, decisionLevel(), decisions_cap, decision_states);
        int depth_of_lookahead = branching_literals->size();
        if (samples_by_depth->count(depth_of_lookahead) == 0) {
            samples_by_depth->insert(std::pair<int, int>(depth_of_lookahead, 0));
        }
        (*samples_by_depth)[depth_of_lookahead] += 1;
        if (verbosity >= 2)
        {
            printf("Completed lookahead\n");
            double lower_bound = 1/(*data_table)[assignToVec(assigns)]->getRewards()[toInt((*branching_literals)[0])].item<double>();
            int count = (*data_table)[assignToVec(assigns)]->getCounts()[toInt((*branching_literals)[0])].item<int>();
            printf("Decision:%lu, Sample:%d, Lit:%d, # branch:%lu, Mean tree size:%.1f, Counts:%d, Peak mem: %.2f MB, CPU time: %.2fs, Wall time: %fs\n", decisions, mcts_iter, toInt((*branching_literals)[0]), lookahead_solver.decisions, lower_bound, count, memUsedPeak(), cpuTime(), difftime( time(NULL), start_time)); 
            printf("Chain:");        
            for (int j=0; j < branching_literals->size(); j++)
            {
                printf("%d->", toInt((*branching_literals)[j]));
            }
            printf("\n");  
        }
        
        delete active_lits;
        delete branching_literals;
        required_run = false;

        lookahead_solver.decisions = 0;
        lookahead_solver.cancelUntil(decisionLevel()); // Cancel decisions until outer loop decision level

        //If converged, terminate lookaheads
        if (mcts_iter % 100 == 0)
        {
            if (banditPolicy->isConverged((*data_table)[assignToVec(assigns)]->getRewardLowerBounds(), (*data_table)[assignToVec(assigns)]->getRewardUpperBounds(), getActiveLiterals()))
            {
                printf("Would have converged after %d lookaheads. Continuing...\n", mcts_iter);
               // break;
            }
        }
        
    }

    int current_best = (*data_table)[assignToVec(assigns)]->getCounts().argmax().item<int>();
    double selected_literal_tree_size_mcts = 1/(*data_table)[assignToVec(assigns)]->getRewards()[toInt(current_best)].item<double>();
    double selected_literal_lower_bound_mcts = 1/(*data_table)[assignToVec(assigns)]->getRewardUpperBounds()[toInt(current_best)].item<double>();
    int count = (*data_table)[assignToVec(assigns)]->getCounts()[toInt(current_best)].item<int>();
    printf("\nAfter MCTS samples: Completed %d samples, highest counts literal:%d, Mean tree size:%f, Lower bound:%f, Count:%d,  Qd:%f\n", 
                                num_samples, current_best ,selected_literal_tree_size_mcts, 
                                selected_literal_lower_bound_mcts, count, (*data_table)[assignToVec(assigns)]->getQd());
    fflush(stdout);

    printf("Samples by depth: \n");
    for (auto const& x : *samples_by_depth)
    {
        printf("%d:%d, ", x.first, x.second);
    }
    printf("\n");
    
    torch::Tensor action_probabilities;
    torch::Tensor unnorm_reward = at::zeros(0);
    torch::Tensor lit_counts = at::zeros(0);
    vec<int> * active_lits = getActiveLiterals();
    action_probabilities = banditPolicy->getActionProbabilities((*data_table)[assignToVec(assigns)]->getRewards(), (*data_table)[assignToVec(assigns)]->getCounts(), 
                                                                active_lits, nVars(), (*data_table)[assignToVec(assigns)]->getRewardUpperBounds());
    unnorm_reward = (*data_table)[assignToVec(assigns)]->getRewards();//* 1/(*data_table)[assignToVec(assigns)]->getCounts();
    lit_counts = (*data_table)[assignToVec(assigns)]->getCounts();

    std::vector<torch::Tensor> outputs = getNeuralPrediction(action_probabilities, result, unnorm_reward, lit_counts);
    torch::Tensor neural_net_lit_probabilities = outputs[0];

    if (neural_net_lit_probabilities.size(0) == 1)
    {
        for (int i = 0; i < sat_solutions->size(); i++) {
            delete (*sat_solutions)[i];
        }
        delete sat_solutions;
        return lit_Undef;
    }
    action_probabilities = action_probabilities.softmax(0);
    neural_net_lit_probabilities = neural_net_lit_probabilities.softmax(0);
    torch::Tensor summed_probs = (action_probabilities * prop_mcts) + (neural_net_lit_probabilities * (1.0-prop_mcts));

    Lit selected_literal = selectLiteral(summed_probs, "max");//"sample");
    int selected_literal_count = (*data_table)[assignToVec(assigns)]->getCounts()[toInt(selected_literal)].item<int>();
    double selected_literal_tree_size = 1/(*data_table)[assignToVec(assigns)]->getRewards()[toInt(selected_literal)].item<double>();
    double selected_literal_lower_bound = 1/(*data_table)[assignToVec(assigns)]->getRewardUpperBounds()[toInt(selected_literal)].item<double>();

    printf("Selected literal %d was sampled %d times. Mean tree size:%f, Lower bound:%f\n", toInt(selected_literal), selected_literal_count, selected_literal_tree_size, selected_literal_lower_bound);

    if (value(selected_literal) != l_Undef) {
        printf("Chosen literal has already been assigned!. Must have picked literal with 0 probability. This is a bug!\n");
        std::cout << "MCTS literal problabilities: " << std::endl;
        std::cout << summed_probs;
    }
    return selected_literal;
}


void Solver::logProgress(time_t last_cpu_time, int mcts_iter, int num_samples, int logging_frequency){

    time_t current_time;
    time(&current_time);
    double elapsed_time = difftime(current_time, last_cpu_time);
    
    //TODO: Get literal amongst ones currently in the problem
    vec<int>* active_literals = getActiveLiterals();
    torch::Tensor boolean_map = torch::zeros(active_literals->size(), torch::TensorOptions().dtype(torch::kLong));
    for (int i=0; i < active_literals->size(); i++) {
        boolean_map[i] = (*active_literals)[i];
    }
    torch::Tensor counts = (*data_table)[assignToVec(assigns)]->getCounts();
    counts = counts.index({boolean_map});
    int lit_index = counts.argmax().item<int>();
    int current_best = (*active_literals)[lit_index];
    double selected_literal_tree_size = 1/(*data_table)[assignToVec(assigns)]->getRewards()[toInt(current_best)].item<double>();
    double selected_literal_lower_bound = 1/(*data_table)[assignToVec(assigns)]->getRewardUpperBounds()[toInt(current_best)].item<double>();
    double selected_Qd = (*data_table)[assignToVec(assigns)]->getQd();
    int count = (*data_table)[assignToVec(assigns)]->getCounts()[toInt(current_best)].item<int>();
    //TODO: Adjust for all the variable not yet selected
    double mean_counts = (*data_table)[assignToVec(assigns)]->getCounts().index({boolean_map}).mean().item<double>();
    double min_count = (*data_table)[assignToVec(assigns)]->getCounts().index({boolean_map}).min().item<int>();
    int mcts_tree_size = data_table->size();
    printf("%d/%d (%.1f /s), #nodes:%d, #cached:%d, Lit:%d, Tree size:%.2f, LB:%.2f, Count:%d, Mean count:%.1f, Min count:%.1f, Mem: %.1f, Qd:%.1f   \r", 
            mcts_iter, num_samples, logging_frequency/elapsed_time,mcts_tree_size, nn_solution_cache->getSize(), 
            current_best,selected_literal_tree_size, selected_literal_lower_bound, count, 
            mean_counts, min_count, memUsedPeak(), selected_Qd);
    fflush(stdout);
}


double Solver::progressEstimate() const {
    double progress = 0;
    double F = 1.0 / nVars();

    for (int i = 0; i <= decisionLevel(); i++) {
        int beg = i == 0 ? 0 : trail_lim[i - 1];
        int end = i == decisionLevel() ? trail.size() : trail_lim[i];
        progress += pow(F, i) * (end - beg);
    }

    return progress / nVars();
}


static bool switch_mode = false;

static void SIGALRM_switch(int signum) {
    switch_mode = true;
}

// NOTE: assumptions passed in member-variable 'assumptions'.

// This is the big solve function

lbool Solver::solve_() {
    Solver aux_solver = Solver(*this);
    signal(SIGALRM, SIGALRM_switch);
    alarm(2500);

    model.clear();
    conflict.clear();
    if (!ok) return l_False;

    solves++;


    lbool status = l_Undef;

    if (verbosity >= 1) {
        printf("c ============================[ Search Statistics ]==============================\n");
        printf("c | Conflicts |          ORIGINAL         |          LEARNT          | Progress |\n");
        printf("c |           |    Vars  Clauses Literals |    Limit  Clauses Lit/Cl |          |\n");
        printf("c ===============================================================================\n");
    }

    add_tmp.clear();

    VSIDS = true;
    int init = 10000;
    if (do_monte_carlo_tree_search) {
        printf("c Running Solver with Monte Carlo tree search\n");
    }

    if (use_mcts_db) {
        try {
            std::string base = "mysql -N -B -h address -P port -u username -p\"password\" database -e ";
            std::string variables = std::to_string(nVars());
            std::string clauses = std::to_string(nClauses());
            std::string cnf = "'" + readFileIntoString(std::string(this->instance_name)) + "'";
            cnf = std::regex_replace(cnf, std::regex("`"), ""); 
            //std::hash<std::string> hasher;
            //this->db_instance_key = hasher(cnf); //returns std::size_t
            /* std::string command = base + "\"INSERT IGNORE INTO Instances (instance_id,n_vars,n_clauses,cnf) VALUES ("
                                       + std::to_string(db_instance_key) + "," + variables + "," + clauses + ","
                                       + cnf + ");\"";
            //std::cout << command;
            exec(command.c_str());
 */
            std::string model_id = "'" + (std::string) this->model_filename + "'";
            std::string callstring  = "'" + (std::string) this->callstring + "'";
            std::string options = "'" + create_options_dict() + "'";
            std::string experiment_string = "'" + (std::string) this->experiment_name + "'" ;
            std::string cnf_hash = "'" + (std::string) this->cnf_hash + "'";
            std::string instance_id = "'" + std::to_string(this->db_instance_key) + "'";
            std::string command = base + "\"INSERT IGNORE INTO Experiments (experiment_id, experiment_name, cnf_hash, time, model_id, call_string, options) VALUES ("
                                       + "default" + "," + experiment_string + "," + cnf_hash + "," + "NOW()" + "," + model_id + "," +  callstring + "," + options + "); SELECT LAST_INSERT_ID()\"";
            //this->db_experiment_key = std::strtoul(exec(command.c_str()).c_str(), NULL, 0); // Returns experiment_id of last insert
            exec(command.c_str());
            printf("Inserted into Experiments table\n");
            //printf("Command: %s\n", command.c_str());
        } catch (std::exception &e) {
            printf("Error setting up database: TODO");
        }
    }
    vec<Lit>*branching_literals = new vec<Lit>();
    status = search(init, false, do_monte_carlo_tree_search, false, branching_literals, 0, std::numeric_limits<uint64_t>::max()); // Somehow search sets to True without setting variables
    
    if (use_mcts_db) {
        try {
            std::string base = "mysql -N -B -h address -P port -u username -p\"password\" database -e ";
            std::string decisions = "'" + std::to_string(this->decisions) + "'";
            //std::string instance_id = "'" + std::to_string(this->db_instance_key) + "'";
            std::string cnf_hash = "'" + (std::string) this->cnf_hash + "'";
            //std::string command = base + "\"UPDATE Experiments SET decisions = " + decisions + " WHERE instance_id = " + instance_id +  ";\"";
            std::string command = base + "\"UPDATE Experiments SET decisions = " + decisions + " WHERE cnf_hash = " + cnf_hash +  ";\"";
            exec(command.c_str());
            printf("Updated Experiments table\n");
            printf("Command: %s\n", command.c_str());
        } catch (std::exception &e) {
            printf("Error setting up database: TODO");
        }
    }

    if (verbosity >= 1)
        printf("c ===============================================================================\n");

#ifdef BIN_DRUP
    if (drup_file && status == l_False) binDRUP_flush(drup_file);
#endif

    if (status == l_True) {
        // Extend & copy model:
        model.growTo(nVars());
        for (int i = 0; i < nVars(); i++) model[i] = value(i);
    }
    else if (status == l_False && conflict.size() == 0)
        ok = false;
    
    delete branching_literals;
    cancelUntil(0);

    if (inquire_tree) {
        std::vector<int> state_lengths;
        std::vector<int> num_samples;
        std::vector<int> depths;
        saveSearchTreeStatistics(state_lengths, num_samples, depths);

        std::ofstream myFile("stat.csv");
        std::vector<std::pair<std::string, std::vector<int>>> vals = {{"State Lengths", state_lengths}, {"Num Samples", num_samples}, {"Depths", depths}};

        write_csv("tree_stat.csv", vals);

    }

    printf("Before ifstream:\n");
    if (params_tuning) {
        std::vector<std::pair<std::string, std::vector<int>>> current_stats = read_csv("mcts_stats.csv");
        printf("Read mcts_stats.csv done\n");
        std::vector<int> current_decisions = current_stats[0].second;
        std::vector<int> current_betas = current_stats[1].second;
        std::vector<int> current_lookaheads = current_stats[2].second;
        current_decisions.push_back(this->decisions);
        current_betas.push_back(this->BETA*1000000);
        current_lookaheads.push_back(this->num_monte_carlo_tree_search_samples);
        std::vector<std::pair<std::string, std::vector<int>>> new_stats = {{"Decisions", current_decisions}, {"Betas", current_betas}, {"Lookaheads", current_lookaheads}};
        printf("Start writing new stats\n");
        write_csv("mcts_stats.csv", new_stats);
    }
    
    return status;
}

//TODO: Create dictionary for solver call options
std::string Solver::create_options_dict() {

    std::string options_dict = "{{rollout_depth:" + std::to_string(this->depth_before_rollout) + "}," + 
                         "{num_rollouts:" + std::to_string(this->num_monte_carlo_tree_search_samples) + "}," +
                         "{prop_mcts:" + std::to_string(this->prop_mcts) + "}," +
                         "{beta:" + std::to_string(this->BETA) + "}," +
                         "{use_prior:" + std::to_string(this->use_neural_net_prior) + "}" +
                         "}" ;

    return options_dict;
}

std::string Solver::readFileIntoString(const std::string& path) {
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        // cerr << "Could not open the file - '"
        //      << path << "'" << endl;
        exit(EXIT_FAILURE);
    }
    return std::string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
}

//=================================================================================================
// Writing CNF to DIMACS:
// 
// FIXME: this needs to be rewritten completely.

static Var mapVar(Var x, vec<Var>& map, Var& max)
{
    if (map.size() <= x || map[x] == -1) {
        map.growTo(x + 1, -1);
        map[x] = max++;
    }
    return map[x];
}

void Solver::toDimacs(FILE*f, Clause& c, vec<Var>& map, Var& max)
{
    if (satisfied(c)) return;

    for (int i = 0; i < c.size(); i++)
        if (value(c[i]) != l_False)
            fprintf(f, "%s%d ", sign(c[i]) ? "-" : "", mapVar(var(c[i]), map, max) + 1);
    fprintf(f, "0\n");
}

void Solver::toDimacs(const char *file, const vec<Lit>& assumps)
{
    FILE*f = fopen(file, "wr");
    if (f == NULL)
        fprintf(stderr, "could not open file %s\n", file), exit(1);
    toDimacs(f, assumps);
    fclose(f);
}

void Solver::toDimacs(FILE*f, const vec<Lit>& assumps)
{
    // Handle case when solver is in contradictory state:
    if (!ok) {
        fprintf(f, "p cnf 1 2\n1 0\n-1 0\n");
        return;
    }

    vec<Var> map;
    Var max = 0;

    // Cannot use removeClauses here because it is not safe
    // to deallocate them at this point. Could be improved.
    int cnt = 0;
    for (int i = 0; i < clauses.size(); i++)
        if (!satisfied(ca[clauses[i]]))
            cnt++;

    for (int i = 0; i < clauses.size(); i++)
        if (!satisfied(ca[clauses[i]])) {
            Clause& c = ca[clauses[i]];
            for (int j = 0; j < c.size(); j++)
                if (value(c[j]) != l_False)
                    mapVar(var(c[j]), map, max);
        }

    // Assumptions are added as unit clauses:
    cnt += assumptions.size();

    fprintf(f, "p cnf %d %d\n", max, cnt);

    for (int i = 0; i < assumptions.size(); i++) {
        assert(value(assumptions[i]) != l_False);
        fprintf(f, "%s%d 0\n", sign(assumptions[i]) ? "-" : "", mapVar(var(assumptions[i]), map, max) + 1);
    }

    for (int i = 0; i < clauses.size(); i++)
        toDimacs(f, ca[clauses[i]], map, max);

    if (verbosity > 0)
        printf("c Wrote %d clauses with %d variables.\n", cnt, max);
}

//=================================================================================================
// Garbage Collection methods:

void Solver::relocAll(ClauseAllocator& to)
{
    // All watchers:
    //
    // for (int i = 0; i < watches.size(); i++)
    watches.cleanAll();
    watches_bin.cleanAll();
    for (int v = 0; v < nVars(); v++)
        for (int s = 0; s < 2; s++) {
            Lit p = mkLit(v, s);
            // printf(" >>> RELOCING: %s%d\n", sign(p)?"-":"", var(p)+1);
            vec<Watcher>& ws = watches[p];
            for (int j = 0; j < ws.size(); j++)
                ca.reloc(ws[j].cref, to);
            vec<Watcher>& ws_bin = watches_bin[p];
            for (int j = 0; j < ws_bin.size(); j++)
                ca.reloc(ws_bin[j].cref, to);
        }

    // All reasons:
    //
    for (int i = 0; i < trail.size(); i++) {
        Var v = var(trail[i]);

        if (reason(v) != CRef_Undef && (ca[reason(v)].reloced() || locked(ca[reason(v)])))
            ca.reloc(vardata[v].reason, to);
    }

    // All learnt:
    //
    for (int i = 0; i < learnts_core.size(); i++)
        ca.reloc(learnts_core[i], to);
    for (int i = 0; i < learnts_tier2.size(); i++)
        ca.reloc(learnts_tier2[i], to);
    for (int i = 0; i < learnts_local.size(); i++)
        ca.reloc(learnts_local[i], to);

    // All original:
    //
    int i, j;
    for (i = j = 0; i < clauses.size(); i++)
        if (ca[clauses[i]].mark() != 1) {
            ca.reloc(clauses[i], to);
            clauses[j++] = clauses[i];
        }
    clauses.shrink(i - j);
}

void Solver::garbageCollect()
{
    // Initialize the next region to a size corresponding to the estimated utilization degree. This
    // is not precise but should avoid some unnecessary reallocations for the new region:
    ClauseAllocator to(ca.size() - ca.wasted());

    relocAll(to);
    if (verbosity >= 2)
        printf("c |  Garbage collection:   %12d bytes => %12d bytes             |\n",
               ca.size() * ClauseAllocator::Unit_Size, to.size() * ClauseAllocator::Unit_Size);
    to.moveTo(ca);
}

void Solver::testPrinter() {
    for (int i = 0; i < assigns.size(); i++) {
        if (i % 10 == 0) {std::cout << std::endl;}
        std::cout << toInt(assigns[i]) << " ";
    }
}

std::vector<bool> Solver::assignToVec(vec<lbool> & assignment_state) {
    std::vector<bool> lit_state(2*assignment_state.size(),false);
    for (int i = 0; i < assignment_state.size(); i++) {
        if (assignment_state[i] == l_True) {
            lit_state[2*i] = true;
        } else if (assignment_state[i] == l_False){
            lit_state[2*i + 1] = true;
        }
    }
    return lit_state;
}

void Solver::doCleanUpDataTable() {
    for (auto it = data_table->begin(); it != data_table->end(); ) {
        DataCentre *curr_node = it->second; 
        if (curr_node->getDataCount() <= 1) {
            it = data_table->erase(it);
        } else {
            ++it;
        }
    }
}

void Solver::saveSearchTreeStatistics(std::vector<int> &state_lengths, std::vector<int> &num_samples, std::vector<int> &depths) {
    for (std::pair<std::vector<bool>, DataCentre*> element : *data_table) {
        DataCentre *curr_node = element.second; 
        int num_decisions = 0;
        for (int i = 0; i < (int) assignToVec(assigns).size()/2; i++) {
            if (assignToVec(assigns)[2*i] || assignToVec(assigns)[2*i+1]) {
                num_decisions++;
            }
        }
        state_lengths.push_back(num_decisions);
        num_samples.push_back(curr_node->getDataCount());
        depths.push_back(curr_node->getDepth());
    }
}

#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair

void Solver::write_csv(std::string filename, std::vector<std::pair<std::string, std::vector<int>>> dataset) {
    std::ofstream myFile(filename);
    for (int j = 0; j < (int) dataset.size(); ++j) {
        myFile << dataset.at(j).first;
        if(j != (int) dataset.size() - 1) myFile << ","; // No comma at end of line
    }
    myFile << "\n";
    for (int i = 0; i < (int) dataset.at(0).second.size(); ++i) {
        for (int j = 0; j < (int) dataset.size(); ++j) {
            myFile << dataset.at(j).second.at(i);
            if (j != (int) dataset.size() - 1) myFile << ","; // No comma at end of line
        }
        myFile << "\n";
    }
    myFile.close();
}

std::vector<std::pair<std::string, std::vector<int>>> Solver::read_csv(std::string filename) {
    std::vector<std::pair<std::string, std::vector<int>>> result;
    std::ifstream myFile(filename);
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");
    std::string line, colname;
    int val;
    if(myFile.good()) {
        std::getline(myFile, line);
        std::stringstream ss(line);
        while(std::getline(ss, colname, ',')){
            result.push_back({colname, std::vector<int> {}});
        }
    }
    while(std::getline(myFile, line)) {
        std::stringstream ss(line);
        int colIdx = 0;
        while(ss >> val){
            result.at(colIdx).second.push_back(val);
            if(ss.peek() == ',') ss.ignore();
            colIdx++;
        }
    }
    myFile.close();
    return result;
}