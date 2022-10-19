#include "mtl/Vec.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <torch/script.h>
#include "mtl/Alg.h"

#include <math.h>
#include <map>


using namespace Minisat;

class LookaheadData {
    public:
        int action;
        double reward;
        bool capped;
        int shortest_path;
        
        virtual ~LookaheadData() {}

        LookaheadData(int action, double reward, bool capped) {
            this->action = action;
            this->reward = reward;
            this->capped = capped;
            this->shortest_path = std::numeric_limits<int>::max();
        }

        LookaheadData(int action, double reward, bool capped, int shortest_path) {
            this->action = action;
            this->reward = reward;
            this->capped = capped;
            this->shortest_path = shortest_path;
        }
};

class IBanditPolicy {
    public:
        virtual ~IBanditPolicy() {}
        virtual int selectAction(torch::Tensor rewards, torch::Tensor counts, vec<int> *active_literals, torch::Tensor neural_net_probabilities) = 0;
        virtual torch::Tensor getActionProbabilities(torch::Tensor rewards, torch::Tensor counts, vec<int> *active_literals, int n_vars, torch::Tensor reward_lower_bounds) = 0;
        //virtual bool isConverged(vec<LookaheadData*> *node_data, vec<int> *active_literals) {}
        virtual bool isConverged(torch::Tensor reward_lower_bounds, torch::Tensor reward_upper_bounds, vec<int> *active_literals) = 0;
};

class KnuthSampleLowerBound : public IBanditPolicy {
    public:
        double random_action_prob; // probability to select random action

        KnuthSampleLowerBound(double random_action_prob) {
            this->random_action_prob = random_action_prob;
        }

        /** Return normalized distribution of counts **/
        torch::Tensor getActionProbabilities(torch::Tensor rewards, torch::Tensor all_counts, vec<int> *active_literals, int n_vars, torch::Tensor reward_upper_bounds) {
            
            //std::cout << "counts: " << all_counts << std::endl;
            double sum_counts = all_counts.sum().item<double>();
            double sum_rewards = rewards.sum().item<double>();
            auto options = torch::TensorOptions().dtype(torch::kFloat64);
            torch::Tensor action_probabilities = at::ones(n_vars*2, options) * -INFINITY; // initialize to -inf
            for (int i=0; i < active_literals->size(); i++) {
                int index = (*active_literals)[i];
                //printf("Literal:%d, counts:%d, tree size:%f, lower bounds: %f\n", index, all_counts[index].item<int>(), 1/rewards[index].item<double>(), 1/reward_lower_bounds[index].item<double>());
                double normalized_count = all_counts[index].item<double>()/sum_counts;
                double normalized_reward = rewards[index].item<double>()/sum_rewards;
                action_probabilities[index] = normalized_count + (normalized_reward * 0.001); // add small amount of reward to make sure we tiebreak counts on rewards
            }
            return action_probabilities;

        }

        int selectAction(torch::Tensor reward_upper_bounds, torch::Tensor all_counts, vec<int> *active_literals, torch::Tensor neural_net_probabilities) {
            torch::Tensor boolean_map = torch::zeros(active_literals->size(), torch::TensorOptions().dtype(torch::kLong));
            for (int i=0; i < active_literals->size(); i++) {
                boolean_map[i] = (*active_literals)[i];
            }
            reward_upper_bounds = reward_upper_bounds.index({boolean_map});
            all_counts = all_counts.index({boolean_map});            
            int argmax_index = reward_upper_bounds.argmax().item<int>();

            //printf("Picked literal:%d, with tree size lower bound:%f, with count:%d\n", (*active_literals)[argmax_index], 1/reward_upper_bounds[argmax_index].item<double>(), all_counts[argmax_index].item<int>());
            return (*active_literals)[argmax_index];
        }

        bool isConverged(torch::Tensor reward_lower_bounds, torch::Tensor reward_upper_bounds, vec<int> *active_literals) {
            torch::Tensor boolean_map = torch::zeros(active_literals->size(), torch::TensorOptions().dtype(torch::kLong));
            for (int i=0; i < active_literals->size(); i++) {
                boolean_map[i] = (*active_literals)[i];
            }
            reward_lower_bounds = reward_lower_bounds.index({boolean_map});
            double max_lower_bound = reward_lower_bounds.max().item<double>();
            int argmax_lower_bound = reward_lower_bounds.argmax().item<int>();
            int negative_lit = 0;
            if (argmax_lower_bound % 2 == 0) {
                negative_lit = argmax_lower_bound+1;
            } else {
                negative_lit = argmax_lower_bound-1;
            }
            // zero out literal
            boolean_map[argmax_lower_bound] = 0;
            boolean_map[negative_lit] = 0;
            reward_upper_bounds = reward_upper_bounds.index({boolean_map});
            double second_best_upper_bound = reward_upper_bounds.max().item<double>();
            //int second_best_argmax_upper_bound = reward_upper_bounds.argmax().item<int>();

            //printf("Best Lower:%.5f (lit %d), Second best upper:%.5f (lit %d)\n", max_lower_bound, argmax_lower_bound, second_best_upper_bound, second_best_argmax_upper_bound);

            if (max_lower_bound > second_best_upper_bound) {
                return true;
            }
            else {
                return false;
            }
        }
};