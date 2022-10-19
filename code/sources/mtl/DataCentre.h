#ifndef Minisat_DataCenter_h
#define Minisat_DataCenter_h

#include "mtl/Vec.h"
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <limits>
#include <stdexcept>

#include "core/SolverTypes.h"

namespace Minisat {

    class DataCentre {
        
    private:
        int id; 
        int state_size;
        
        bool tree_size;
        double Delta;
        int samples_per_update;
        double BETA;

        double depth;
        double prior_temperature;// = 2; // 1 is softmax, > 1 softer, < 1 harder

        bool use_Qd;
        bool fix_policy_to_prior;

        double Qd;
        int n_data;
        vec<LookaheadData*> * node_data_list;
        torch::Tensor neural_net_probabilities;
        double value;
        
        std::unordered_map<int, int> * counts;
        std::unordered_map<int, double> * rewards; 
        std::unordered_map<int, double> * reward_lower_bounds;
        std::unordered_map<int, double> * reward_upper_bounds;
        std::unordered_map<int, double> * mean_tree_size;

    public:
        
        DataCentre(int id, int state_size, bool tree_size, double Delta, 
                    int samples_per_update, double BETA, int depth, 
                    double prior_temperature, bool use_Qd, bool fix_policy_to_prior) {
            this->id = id;
            this->state_size = state_size;
            this->tree_size = tree_size;
            this->Delta = Delta;
            this->samples_per_update = samples_per_update;
            this->BETA = BETA;
            this->depth = depth;
            this->prior_temperature = prior_temperature;
            this->use_Qd = use_Qd; 
            this->fix_policy_to_prior = fix_policy_to_prior;

            this->Qd = 1;
            this->n_data = 0;
            this->node_data_list = new vec<LookaheadData*>();
            this->neural_net_probabilities = at::zeros(1);
            this->value = -1;

            this->counts = new std::unordered_map<int, int>();
            this->rewards = new std::unordered_map<int, double>();
            (*this->rewards)[-1] = 1; // Initialize to 1 for first sample before intialized properly
            this->reward_lower_bounds = new std::unordered_map<int, double>();
            this->reward_upper_bounds = new std::unordered_map<int, double>();
            this->mean_tree_size = new std::unordered_map<int, double>();
        }
        
        ~DataCentre() {
            for (int i = 0; i < this->node_data_list->size(); i++) {
                delete this->node_data_list[i];
            }
            delete this->node_data_list;
            delete this->counts;
            delete this->rewards;
            delete this->reward_lower_bounds;
            delete this->reward_upper_bounds;
            delete this->mean_tree_size;
        }
        
        torch::Tensor getNeuralProbabilities() {
            return this->neural_net_probabilities;
        }
        
        void setNeuralProbabilities(torch::Tensor new_prob) {
            //std::cout << "neural net probs: " << new_prob << std::endl;
            this->neural_net_probabilities = new_prob;
        }

        void setValue(double value) {
            this->value = value;
        }
        
        vec<LookaheadData*> * getNodeData() {
            return this->node_data_list;
        }
        
        void setNodeData(vec<LookaheadData*> * new_data) {
            for (int i = 0; i < this->node_data_list->size(); i++) {
                delete this->node_data_list[i];
            }
            delete this->node_data_list;
            this->node_data_list = new_data;
            for (int i = 0; i < new_data->size(); i++) {
                updateAggregate((*new_data)[i]);
            }
            this->n_data = new_data->size();
        }
        
        void addNodeData(LookaheadData* lookahead) {
            if(lookahead->reward < 1)
            {
                printf("Warning: Tree size for lookkahead is < 1: %f.\n", lookahead->reward);
            }
            if (this->n_data == 0) {
                if (this->value > 0) {
                    // if value has been set, initialize with value estimate
                    initialize_q(log2(this->value));
                } else {
                    initialize_q(lookahead->reward);
                }
            }
            updateAggregate(lookahead);
            delete lookahead;
            this->n_data++;
        }

        void initialize_q(double reward) {
            //assert(reward >= 1);
            int num_actions = this->state_size;
            for (int i=0; (i < num_actions) && (this->mean_tree_size->count(i) > 0); i++) { 
                if (tree_size) {
                    (*this->mean_tree_size)[i] = pow(2,reward);
                } else {
                    (*this->mean_tree_size)[i] = reward;
                }
                (*this->rewards)[i] = 1/(*this->mean_tree_size)[i];
            }
            if (tree_size) {
                (*this->mean_tree_size)[-1] = pow(2,reward);
            } else {
                (*this->mean_tree_size)[-1] = reward;
            }
            (*this->rewards)[-1] = 1/(*this->mean_tree_size)[-1];
            this->Qd = (*this->mean_tree_size)[-1];
        }

        void updateAggregate(LookaheadData * node) {
            if (this->counts->count(node->action) == 0) {
                this->counts->insert(std::pair<int, int>(node->action, 1));
            } else {
                (*this->counts)[node->action] += 1;
            }
            if (this->mean_tree_size->count(node->action) > 0) {
                if (tree_size) {
                    (*this->mean_tree_size)[node->action] += ((pow(2, node->reward)) - (*this->mean_tree_size)[node->action]) / (1 + (*this->counts)[node->action]);
                } else {
                    (*this->mean_tree_size)[node->action] += (node->reward - (*this->mean_tree_size)[node->action]) / (1 + (*this->counts)[node->action]);
                }
            } else {
                if (tree_size) {
                    (*this->mean_tree_size)[node->action] = (*this->mean_tree_size)[-1] + ((pow(2, node->reward)) - (*this->mean_tree_size)[-1]) / (1 + (*this->counts)[node->action]);
                } else {
                    (*this->mean_tree_size)[node->action] = (*this->mean_tree_size)[-1] + (node->reward - (*this->mean_tree_size)[-1]) / (1 + (*this->counts)[node->action]);
                }
            }
            (*this->rewards)[node->action] = 1/(*this->mean_tree_size)[node->action];
        } 

        torch::Tensor getRewards() {
            torch::Tensor ret_rewards = at::zeros(this->state_size);
            for (int i = 0; i < (int) this->state_size; i++) {
                if (this->rewards->count(i) > 0) {
                    ret_rewards[i] = (*this->rewards)[i];
                } else {
                    ret_rewards[i] = (*this->rewards)[-1];
                }
            }
            return ret_rewards;
        }

        torch::Tensor getRewardLowerBounds() {
            torch::Tensor confidence_interval = getConfidenceInterval();    
            torch::Tensor ret_lower_bounds = at::zeros(this->state_size);
            for (int i = 0; i < (int) this->state_size; i++) {
                if (this->rewards->count(i) > 0) {
                    ret_lower_bounds[i] = (*this->rewards)[i] - confidence_interval[i];
                } else {
                    ret_lower_bounds[i] = (*this->rewards)[-1] - confidence_interval[i];
                }
            }
            return ret_lower_bounds;
        }

        torch::Tensor getRewardUpperBounds() {

            torch::Tensor confidence_interval = getConfidenceInterval();
            torch::Tensor ret_upper_bounds = at::zeros(this->state_size);
            for (int i = 0; i < (int) this->state_size; i++) {
                if (this->fix_policy_to_prior)
                {
                    ret_upper_bounds[i] = this->neural_net_probabilities[i];
                }
                else{
                    if (this->rewards->count(i) > 0) {
                        ret_upper_bounds[i] = (*this->rewards)[i] + confidence_interval[i];
                    } else {
                        ret_upper_bounds[i] = (*this->rewards)[-1] + confidence_interval[i];
                    }
                }
            }
            return ret_upper_bounds;
        }

        torch::Tensor getConfidenceInterval() {
            double drift_multiplier;
            if (!this->use_Qd) {
                drift_multiplier = BETA * 1/ this->value; //pow(2, this->depth);
            } else {
                //assert(this->Qd > 0);
                drift_multiplier = BETA * 1/this->Qd;
            }
            torch::Tensor confidence_interval = at::zeros(this->state_size);
            double uniform_probability = 1.0 / this->state_size;
            bool uniform_prior;
            if (this->neural_net_probabilities.size(0) <= 1){
                uniform_prior = true;
                printf("Using uniform prior.\n");
            } else {
                uniform_prior = false;
            }
            for (int i = 0; i < (int) this->state_size; i++) {
                double counts_prior;
                if (uniform_prior) {
                    counts_prior = uniform_probability;
                } else {
                    counts_prior = this->neural_net_probabilities[i].item<double>(); // Maybe divide by sum to assure sum to zero
                }
                //std::cout << "counts_prior: " << counts_prior << std::endl;
                //assert(counts_prior >= 0);
                if (this->counts->count(i) > 0) {
                    confidence_interval[i] =  (drift_multiplier * counts_prior * this->state_size)  * pow(this->n_data + 1, 0.5) / (1 + (*this->counts)[i]);
                } else {
                    confidence_interval[i] =  (drift_multiplier * counts_prior * this->state_size)  * pow(this->n_data + 1, 0.5);
                }
                //assert(confidence_interval[i].item<double>()>=0);
            }
            //std::cout << "confidence_interval: " << confidence_interval << std::endl;
            return confidence_interval;
        }
        
        torch::Tensor getCounts() {
            torch::Tensor ret_counts = at::zeros(this->state_size);
            for (std::pair<int, int> element: (*this->counts)) {
                ret_counts[element.first] = element.second;
            }
            return ret_counts;
        }

        double getStateMeanTreeSize() {
            double sum = 0;
            int count = 0;
            for (std::pair<int, double> element : (*this->mean_tree_size)) {
                sum += element.second;
                count ++;
            }
            if (count > 0) {
                return sum/count;
            } else {
                return 0;
            }
        }

        int getDataCount() {
            return this->n_data;
        }

        int getNodeId() {
            return this->id;
        }
        
        double getQd() {
            return this->Qd;
        }

        int getDepth() {
            return this->depth;
        }

        double getValue() {
            return this->value;
        }

        void setQd(double new_Qd) {
            this->Qd = new_Qd;
        }
    };
}
#endif