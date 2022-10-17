import os
import tqdm
import sys
import subprocess
import glob
import numpy as np
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy import stats
import pandas as pd

import time

def plotECDF(data,color='k',linewidth=1.0,marker=None,hollow=False,linestyle='-',label=None,alpha=1.0,cutoff=100000000,solved_instances=1000000):
    """Plots the empirical cumulative distribution function of the given data.

    Arguments:
        data -- a list of real data points.

    Keyword arguments:
        color -- the color of the plot line.
    """

    #filter below cutoff
    temp = [i for i in data if i < cutoff]

    x = sorted(data)
    x = x[:solved_instances]
    y = np.array(range(1,len(x)+1))/float(len(x))

    markeredgecolor=color
    markerfacecolor=color
    if hollow:
        markeredgecolor=color
        markerfacecolor='None'


    if label == None:
        plt.plot(x,y,color=color,linestyle=linestyle,linewidth=linewidth,alpha=alpha)
    else:
        plt.plot(x,y,color=color,label=label,marker=marker,linestyle=linestyle,linewidth=linewidth,alpha=alpha)


def get_num_branches_from_logs(lines):

	for line in str(lines).split('\\n'):

		if 'c decisions' in line:
			num_decisions = int(float(line.split(':')[1].strip().split(' ')[0]))
			return num_decisions

	print('Warning: Decision not found in log')
	return -1

def get_satisfiability(lines):

	for line in str(lines).split('\\n'):
		if line:
			if line[0] == 's':
				if len(line.split(' ')) < 2:
					print('Warning: No satisfiability status found in log')
					return 'UNSATISFIABLE'
				return line.split(' ')[1]


def cdcl_evaluation(instances, 
					model=None,
					checkpt='val',
					data_dir='./',
					exec_dir='./',
					read_logs_only=False,
					neural_branching=True,
					random_branching=False,
					mcts=False,
					num_lookaheads=1000,
					verb=1,
					depth_before_rollout=5,
					rollout_policy='FULL_SUBTREE',
					dynamic_node_expansion=True,
					bandit_policy="Knuth",
					rerun=True,
					prop_mcts=1.0,
					nn_prior=True,
					seed=1,
					uniform_random_ucb=False,
					neural_net_prior_depth=30,
					min_knuth_tree_estimate=True,
					use_max_rewards=False,
                    verbose=False,
                    decisions_cutoff=1000,
					external_solver_executable='',
					neural_net_refresh_rate=1,
					torch_device='CUDA'
					):
	"""
	Arguments:
	    instances -- a list of instance filenames.

	Keyword arguments:
	    color -- the color of the plot line.

	Returns:
		decisions -- a list of the decisions made by CDCL solver for each instance
	"""

	if not os.path.isfile(model):
		raise Exception('Model not found at ' + model)

	decisions=[]
	mean_decisions = 0.
	sat_decisions =[]
	unsat_decisions=[]
	failed_instances= []
	sat_statuses = []
	runtimes = []
	# print(checkpt)
	# print(data_dir)

	checkpt_data_directory = os.path.join(data_dir, 'cdcl_logs_' + checkpt)
	if not os.path.isdir(checkpt_data_directory):
		os.mkdir(checkpt_data_directory)
	t = tqdm.tqdm(instances, desc=f'Mean performance: ', leave=True)
	for instance in t:

		t.set_description(f"Mean performance: {mean_decisions}")
		t.refresh()

		instance = instance.replace('*','')
		instance = instance.replace('\n','')
		if not os.path.isfile(instance):
			raise Exception('Instance not found at ' + instance)

		solver_output_file = os.path.join(checkpt_data_directory,os.path.basename(instance)+'.out')

		if not os.path.isfile(solver_output_file) or rerun:
			if read_logs_only:
				raise Exception('Output file missing for instance: ' + instance)

			args = []
			args.append(f'-rnd-seed={seed}')
			args.append(f'-verb={verb}')
			args.append(f'-torch-device={torch_device}')
			if min_knuth_tree_estimate:
				args.append('-min-knuth-tree-estimate')
			if neural_branching:
				args.append('-use-neural-branching')
				#args.append(f'-decisions_cutoff={decisions_cutoff}')
				args.append(f'-model-path={model}')
			elif random_branching:
				args.append('-use-random-branching')

			if mcts:
				#print("Using MCTS")
				args.append(f'-bandit-policy={bandit_policy}')
				args.append(f'-model-path={model}')
				args.append('-do-monte-carlo-tree-search')
				args.append(f'-num-monte-carlo-tree-search-samples={num_lookaheads}')
				args.append('-data-dir=mcts_data')
				args.append(f'-prop-mcts={prop_mcts}')
				args.append(f'-rollout-policy={rollout_policy}')
				if (dynamic_node_expansion):
					args.append(f'-dynamic-node-expansion')
				#args.append(f'-neural-net-prior-depth={neural_net_prior_depth}')
				if nn_prior:
					args.append('-use-neural-net-prior')
				if uniform_random_ucb:
					args.append('-uniform-random-ucb')
				if use_max_rewards:
					args.append('-use-max-rewards')


			#if num_variables_for_rollout:
			args.append(f'-depth-before-rollout={depth_before_rollout}')
			args.append(f'-external-solver-executable={external_solver_executable}')
			args.append(f'-neural-net-refresh-rate={neural_net_refresh_rate}')

			#perl_cutoff_command = f'perl -e \'alarm shift @ARGV; exec @ARGV\' {cutoff_time}'

			call_string = f'export LD_LIBRARY_PATH=/libtorch/lib;cd {exec_dir};./glucose {instance} ' + ' '.join(args)
			#call_string = perl_cutoff_command + ' ' + call_string
			#print('Calling: ' + call_string)
			if verbose:
				print('Calling: ' + call_string)
			start = time.time()
			process = subprocess.run(call_string,shell=True,stdout=subprocess.PIPE)
			end = time.time()
			runtimes.append(end-start)
			#print('complete')
			# with open(solver_output_file, 'wb') as f:
			# 	f.write(process.stdout)

		#with open(solver_output_file, 'rb') as f:
			#output = f.read()
			num_decisions = get_num_branches_from_logs(process.stdout)
			status = get_satisfiability(process.stdout)

		if num_decisions != -1:
			decisions.append(num_decisions)
			if verbose:
				print(np.mean(decisions))
			if status == 'SATISFIABLE':
				sat_statuses.append('SAT')
				sat_decisions.append(num_decisions)
			elif status == 'UNSATISFIABLE':
				sat_statuses.append('UNSAT')
				unsat_decisions.append(num_decisions)
		else:
			print('Warning: Decision missing. Solving failed for instance!')
			print(f'Failed call string: {call_string}')
			failed_instances.append(instance)
			sat_decisions.append(10000000)
			unsat_decisions.append(1000000)

		mean_decisions = np.mean(sat_decisions + unsat_decisions)

	if len(failed_instances) > 0:
		print(f'{len(failed_instances)}/{len(decisions)} failed instances')

	if verbose:
		print("Decisions stats:")
		print(pd.Series(decisions).describe())


		if sat_decisions:
			print("SATISFIABLE:")
			print(pd.Series(sat_decisions).describe())
		if unsat_decisions:
			print("UNSATISFIABLE:")
			print(pd.Series(unsat_decisions).describe())


	# Save out to file
	# output_file = os.path.join(data_dir, checkpt+'.csv')
	# with open(output_file, 'w') as f:
	# 	header = 'instance,num_lookaheads,nn_prior,c_puct,random_ucb,num_decisions,seed,max_rewards'
	# 	f.write(header + '\n')
	# 	for i, instance in enumerate(instances):
	# 		f.write(f'{instance},{num_lookaheads},{nn_prior},{c_puct},{uniform_random_ucb},{decisions[i]},{seed},{use_max_rewards}\n')

	return decisions, runtimes, sat_statuses


def ecdf_comparison(experiments, figure_name=None):

	print('Plotting...')
	palette = itertools.cycle(sns.color_palette())

	for experiment in experiments:
		decisions = experiments[experiment]
		plotECDF(decisions,color=next(palette),label=experiment,linewidth=2.0)

	plt.legend()
	plt.ylabel('Frequency')
	plt.xlabel('Num Decisions')

	plt.xscale('log')

	plt.title('CDCL evaluation')

	if not figure_name:
		figure_name = 'ecdf_cdcl_decisions'.png

	plt.savefig(figure_name + '.png', dpi=500)
	plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', nargs='+', default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--checkpt', type=str, default="")
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--exec_dir', type=str)
    parser.add_argument('--read_logs_only', action='store_true', default=False)
    parser.add_argument('--no_neural_branching', action='store_true', default=False)
    parser.add_argument('--random_branching', action='store_true', default=False)

    parser.add_argument('--mcts', action='store_true', default=True)
    parser.add_argument('--num_lookaheads', type=int, default=10000)
    parser.add_argument('--nn_prior', action='store_true', default=False)
    parser.add_argument('--uniform_random_ucb', action='store_true', default=False)
    parser.add_argument('--seed', type=float, default=12.0)
    parser.add_argument('--neural_net_prior_depth', type=int, default=30)
    parser.add_argument('--use_max_rewards', action='store_true', default=False)
    
    parser.add_argument('--dpll', action='store_true', default=False)
    

    args = parser.parse_args()

    cdcl_evaluation(args.instances, 
    				model=args.model, 
    				checkpt=args.checkpt, 
    				data_dir=args.data_dir, 
    				exec_dir=args.exec_dir, 
    				read_logs_only=args.read_logs_only,
    				neural_branching=not(args.no_neural_branching),
    				random_branching=args.random_branching,
    				mcts=args.mcts,
    				num_lookaheads=args.num_lookaheads,
    				rerun=True,
    				nn_prior=args.nn_prior,
    				seed=args.seed,
    				uniform_random_ucb=args.uniform_random_ucb,
    				neural_net_prior_depth=args.neural_net_prior_depth,
    				use_max_rewards=args.use_max_rewards,
                    dpll=args.dpll)




