#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <ctime>
#include <vector>
#include <string.h>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include <functional> // For std::hash
#include <set>
#include <utility> // for std::pair
#include <thread>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <bits/stdc++.h>
#include <map>
#include <string>
#include <iomanip> // for std::fixed and std::setprecision
#include <thread>
#include <vector>
#include <chrono>
#include<filesystem>
#include <regex>
#define MINVALUE  -99999999
#define MAXVALUE 99999999
#define MAXNUM 100
#define DEBUG
#define Pop_Num 10
#include <functional>
#include <mutex>

using namespace std;
namespace fs = std::filesystem;
std::mutex global_data_mutex;

std::mutex log_mutex;  // for thread-safe console and file output
// Inside Hyper_heuristic.h or a relevant header
typedef std::function<int(int)> AlgorithmFunction;
//instance data
string file_name; // Global variable used in initializing()
// Global instanceName variable
std::string instanceName;
std::map<int, int> heuristic_usage_count;         // Number of times each heuristic was used
int num_node;			//number of practitioners
int num_each_t;			//number of practitioners in each crew
int num_team;			//number of crews
double min_div;			//the diversity threshold of each crew
double **div_in;		//the diversity between each pair of practitioners
int *eff;				//efficiency of practitioners
int f_cur_div;
double d_min = 0.8;		//d_min = {0.8,1.0,1.05}
double beta= 0.4;			//beta = 0.4
int fbest_eff ;
int fbest_div ;
//for tabu search
int **tabu_list;		//tabu_list[num_node][num_team+1]
int tl;					//disabled
int tabu_tenure;		//tabu tenure
int fls_depth;			//search depth of feasible local search
int ils_depth;			//search depth of infeasible local search
int *num_t_cur;			//record the number of practitioners of each crew in current solution
int **team_check;		//for the use of verifying solution
double start_time, end_time, time_limit;
int f_cur_eff;
//parameters for feasible and infeasible local search
double p_factor;			//penalty factor
int p_count;
int lamba;
int u1, u2;
int tow;
double deg_cur;			//infeasible degree of the current solution
double *degree_inf;		//infeasible degree of each crew
int best_cost_eff;
int best_cost_div;
double initial_temp = 100.0;
double cooling_rate = 0.995;
double x0 = 10.0; // Initial guess
int    max_iterations = 1500;
double Result;
int  best_eff = 0, best_div = 0;
double time_taken = 0.0;
double average_objective = 0.0;//average objective function
double average_diversity = 0.0;
int worst_objective = 0; //Worst Objective Function Value
double average_cpu_time = 0.0;//Average CPU Time per Iteration
//variables that need to be updated
int *address;			//record the address of each practitioner
int  **team;				//record the practitioners for each crew

int **team1;
int **delta_div;
int *state;				//record which crew the practitioners are in
int *w_div;				//record the diversity of each crew
int *w_eff;				//record the efficiency of each crew
int f_cur;				//the objective value of current solution
int *best_inn;			//the best solution obtained during fits search
int f_best_inn;			//the objective value of the f_best_inn
int S[100]; // Population size
int nm[100];
int m[100];
int fprod;
int l;
int *aa = new int[num_team + 1];

//for the memetic algorithm
typedef struct population{
	int *p;
	int cost;
}population;
population *pop, *A;		//population pop
int generations;		//the generations of evolutionary process

int HMS = 5;         // Harmony Memory Size
// Define the Solution struct
//preserve the best solution over 30 independent runs
int iter;
int fbest;
int ffbest;
int *best_solution;
int *fbest_solution;
int *div_best;
int *eff_best;
int *div_fbest;
int *eff_fbest;
int K;
double lower_bound1=109;
double upper_bound1=1000;
double avg_time =0.0;
enum EfficiencyLevel { LOW_EFF, MEDIUM_EFF, HIGH_EFF };
enum DiversityLevel { LOW_DIV, MEDIUM_DIV, HIGH_DIV };


struct State {
    EfficiencyLevel eff_level;
    DiversityLevel div_level;
    int iteration_bin;
    // Add more discrete features as needed
};

// Enum to specify the MAB strategy
enum MABStrategy {
    UCB,
    EPSILON_GREEDY,
    THOMPSON_SAMPLING
};

// Define a structure to hold heuristic performance metrics
struct Heuristic {
    int id;                 // Heuristic identifier
    std::string name;      // Heuristic name
    int usage_count;       // Number of times the heuristic was used
    double total_reward;   // Cumulative reward
    double average_reward; // Average reward

    Heuristic(int heuristic_id, const std::string& heuristic_name)
        : id(heuristic_id), name(heuristic_name), usage_count(0),
          total_reward(0.0), average_reward(0.0) {}
};

// Function to encode state into a unique string
std::string encode_state(const State& state) {
    return std::to_string(state.eff_level) + "_" +
           std::to_string(state.div_level) + "_" +
           std::to_string(state.iteration_bin);
}

State determine_current_state(double cost_eff, double cost_div, int iteration) {
    State state;

    // Discretize Efficiency
    if (cost_eff < 50)
        state.eff_level = LOW_EFF;
    else if (cost_eff < 100)
        state.eff_level = MEDIUM_EFF;
    else
        state.eff_level = HIGH_EFF;

    // Discretize Diversity
    if (cost_div < 30)
        state.div_level = LOW_DIV;
    else if (cost_div < 60)
        state.div_level = MEDIUM_DIV;
    else
        state.div_level = HIGH_DIV;

    // Discretize Iteration Number into Bins of 100
    state.iteration_bin = (iteration / 100) + 1; // e.g., 0-99: 1, 100-199: 2, etc.

    return state;
}

// Structure to hold all 39 features
struct Features {
    // Basic Features (F1-F9)
    int F1; // |Θ| = n
    int F2_nb_ge_1; // |Θ|nb≥1
    int F3_iterations; // # iterations
    int F4_total_nb; // ∑ni=1 |hi| nb
    int F5_total_imp; // ∑ni=1 |hi| imp
    int F6_total_wrs; // ∑ni=1 |hi| wrs
    int F7_total_eq; // ∑ni=1 |hi| eq
    int F8_total_ac; // ∑ni=1 |hi| ac
    int F9_total_uq; // ∑ni=1 |hi| uq

    // Quality-based Features (F10-F24)
    double F10;
    double F11;
    double F12;
    double F13;
    double F14;
    double F15;
    double F16;
    double F17;
    double F18;
    double F19;
    double F20;
    double F21;
    double F22;
    double F23;
    double F24;

    // Time-based Features (F25-F28)
    double F25_max_th;
    double F26_min_th;
    double F27_avg_th;
    double F28_variance_th;

    // Mixed (Ratio) Features (F29-F39)
    double F29;
    double F30;
    double F31;
    double F32;
    double F33;
    double F34;
    double F35;
    double F36;
    double F37;
    double F38;
    double F39;
};









class Hyper_heuristic {
// Method to implement Algorithm 7: Selection Hyper-Heuristic Framework

private:

    // -------------------------------------------------------------------------
    // Variable Declarations
    // -------------------------------------------------------------------------
    typedef int** (Hyper_heuristic::*LLH_Function)(int**);
    vector<LLH_Function> LLH_set;

    int**  Sbest;
    double fbest_efficiency;
    double fbest_diversity;
    std::map<int, double> heuristicPerformance;
    std::vector<std::vector<int>> Sinput;
    std::vector<std::vector<int>> Soutput;
    std::vector<int> sbest;
    double bestObjectiveValue = -std::numeric_limits<double>::infinity();
    int    maxIterations = 100;
    double terminationThreshold = 0.01;

    // -------------------------------------------------------------------------
    // Internal Helper Functions
    // -------------------------------------------------------------------------
    LLH_Function SelectLowLevelHeuristic();
    int** ApplyHeuristic(LLH_Function heuristic, int** Scurrent);
    bool  Accept(int** Scurrent, int** Snew, double f_current, double f_new);
    void  updateBestSolution(int** Scurrent, double f_current);
    bool  TerminationCriterionSatisfied(int iter, int max_iter);

public:

    // -------------------------------------------------------------------------
    // Struct Definitions
    // -------------------------------------------------------------------------
    struct Inner {
        std::vector<int> ind1;
    };

    struct Outer {
        std::string outerName;
        std::vector<std::vector<Inner>> HM1;
        std::vector<Inner> HM6;
    };

    Outer HM, HM3, HM5, HM2, ImHM;

    // Harmony Memory fitness arrays
    std::vector<std::vector<int>> HMEF;
    std::vector<std::vector<int>> HMDF;
    std::vector<int> SF;
    Inner Route;
    std::vector<int> best_heuristics;
    int* team_size;
    // -------------------------------------------------------------------------
    // Entry Point
    // -------------------------------------------------------------------------
    int selection_hyper_heuristic(int max_iterations);

    // -------------------------------------------------------------------------
    void  print_harmony_memory(const Outer &, int, int ,const std::vector<std::vector<int>>& ,const std::vector<std::vector<int>>& );
    // State-vector structure (10D feature vector)
    struct StateFeatures {
    double f_eff_norm;        // Normalized efficiency
    double f_div_norm;        // Normalized diversity
    double delta_eff_norm;    // Efficiency improvement
    double delta_div_norm;    // Diversity improvement
    double iter_ratio;        // Current iteration progress
    double accept_ratio;      // Acceptance ratio
    double reward_avg;        // Average recent reward
    double div_std_norm;      // Normalized diversity std. deviation
    double temp_norm;         // Normalized temperature (SA)
    double flex_norm;         // Normalized flexibility threshold
     };



    // -------------------------------------------------------------------------
    // Objective & Evaluation Functions
    // -------------------------------------------------------------------------
    void   compute_mindiv();
    void   objective_Function(int **);
    void   objective_Function1(int **);
    int    min_func(int*, int);
    int    sec_func(int*, int, int);
    int    th_func(int *,int, int, int);
    int    randomInt(int);
    int    max_func(int*, int); // <-- ✅ Add this
    int    rand_func(int *, int);
    int    calculate_efficiency(const Inner&);
    double calculate_diversity(const Inner&);
    int    select_max_multiple(int *, int *, int &);
    int**  Apply_LS_OP(int, int , int**);
    // -------------------------------------------------------------------------
    // Meta-Heuristic Algorithms Declaration
    // -------------------------------------------------------------------------
    int**   great_deluge_algorithm();
    int     guided_local_search();
    int**   late_acceptance_hill_climbing();
    int     variable_neighborhood_descent();
    int     variable_neighborhood_search(int);
    int     Hill_Late_Acceptance();
    int**   iterated_local_search();
    int     iterated_local_search12();
    int**   simulated_annealing();
    int**   memetic();
    void    HH_RL_GD();

    // -------------------------------------------------------------------------
    // Local Search Operators
    // -------------------------------------------------------------------------
    int** local_search();
    int** infeasible_local_search();
    int** local_search1();
    int   feasible_local_search();

    // -------------------------------------------------------------------------
    // Initialization Functions
    // -------------------------------------------------------------------------
    void Parameters();
    void initialization(const string &);
    void generate_initial123();
    void initialsolRand();
    void initialSolRefined1();
    void initial_population();
    int  generate_initialrandom();
    int  generate_initial();


    // -------------------------------------------------------------------------
    // Repair Functions
    // -------------------------------------------------------------------------
    void repairSolutions();
    void repair_solution();


    // -------------------------------------------------------------------------
    // Swap and Mutation Operators
    // -------------------------------------------------------------------------
    int   swap1(int , int &, int &);
    int   swap_ils(int , int &, int &);
    int   swap_min(int , int &, int &);
    void  inverse_operator();
    void  randomSwap();

    // -------------------------------------------------------------------------
    // Crossover Operators
    // -------------------------------------------------------------------------
    void  cross_over2();

    // -------------------------------------------------------------------------
    // Tabu & Utility Functions
    // -------------------------------------------------------------------------
    bool isTabu(const std::vector<std::vector<int>>&, int, int, int);
    void applySwap(int, int);
    void updateTabuList(std::vector<std::vector<int>>&, int, int, int, int);
    void update_delta(int, int , int);
    void update_delta12(int , int , int );
    void update_populaion(int *, int);
    int ** fits();

    // -------------------------------------------------------------------------
    // Function Selection (Hyper-Heuristic) and Application
    // -------------------------------------------------------------------------
    int** ApplyHeuristic(int, int**);
    int** ApplyMeta_Heuristic(int, int**);
    void  Apply(char *);
    void  ApplySequence(const std::vector<int>& , int** );
    int** AdaptiveHeuristicSelection(int, double );
    void  Multi_Armed_bandit();
    void  opposite_Based_L();

    // -------------------------------------------------------------------------
    // Function Selection (TRI-LEVEL Hyper-Heuristic) and Application
    // -------------------------------------------------------------------------
    int** SimulatedAnnealing(int** Sstart, int OPj, double T0, double alpha);
    int** IteratedLocalSearch(int** Sstart, int OPj);
    int** LateAcceptance(int** Sstart, int OPj, int Lwindow);
    int** GreatDeluge(int** Sstart, int OPj, double level0, double delta);
    int** TabuSearch(int** Sstart, int OPj);
    // ---- Operator selection (Level 2) ----
    int** apply_LLHop(int op_id, int** sol);

    // -------------------------------------------------------------------------
    // Low-Level Heuristics (LLH)
    // -------------------------------------------------------------------------
    int** LLH1(int**); int** LLH2(int**); int** LLH3(int**);
    int** LLH4(int**); int** LLH5(int**); int** LLH6(int**);
    int** LLH7(int**); int** LLH8(int**); int** LLH9(int**);
    int** LLH10(int**); int** LLH11(int**); int** LLH12(int**);
    int** LLH13(int**); int** LLH14(int**); int** LLH15(int**);
    int** LLH16(int**); int** LLH17(int**);int** LLH18(int**);
    int** LLH19(int**);int** LLH20(int**);int** LLH21(int**);
    int** LLH22(int**);int** LLH23(int**);int** LLH24(int**);
    int** LLH25(int**);

    // -------------------------------------------------------------------------
    // Selection Hyper-Heuristic Strategies
    // -------------------------------------------------------------------------
    void  hyper_heuristic1();
    void  Random_Selection_Hyperheuristic_CMCEE(int );
    void  HH_Choice_Function_Selection_CMCEE(int);
    void  MAB_Selection_Hyperheuristic_CMCEE(int);
    void  Greedy_Selection_Hyperheuristic_CMCEE(int);
    // Q-Learning Based Selection Function
    void Q_Learning_Selection_Hyperheuristic_CMCEE(int);
    void TriLevel_HH_Qlearning_CMCEE(int);
     

    // Choice Function Refined
    void execute_algorithm(int , const std::string&);
    void runSingleOptimizationAlgorithm(int , int );

    // -------------------------------------------------------------------------
    // Utility and Display
    // -------------------------------------------------------------------------
    void display(int**);
    void check_best_solution();
    bool dominates(int , int , int , int );


    // -------------------------------------------------------------------------
    // Deep Copy / Free Utilities
    // -------------------------------------------------------------------------
    // ======================================================================
// Deep Copy of a 2D Team Structure with Variable Team Sizes
// ======================================================================
int** deep_copy_solution(int** source_team,
                                          int num_node,
                                          int num_team,
                                          int num_each_t)
{
    // 1. Determine the size of each team
    std::vector<int> team_size(num_team + 1);
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    // 2. Allocate memory for the copied structure
    int** new_team = new int*[num_team + 1];
    for (int i = 0; i <= num_team; ++i)
        new_team[i] = new int[team_size[i]];

    // 3. Copy values from the source team
    for (int i = 0; i <= num_team; ++i)
        for (int j = 0; j < team_size[i]; ++j)
            new_team[i][j] = source_team[i][j];

    return new_team;
}

// ======================================================================
// Free Allocated Memory of a 2D Team Structure
// ======================================================================
void free_solution(int** team_copy,
                                    int num_node,
                                    int num_team,
                                    int num_each_t)
{
    if (!team_copy) return;

    // Determine team sizes again for proper deallocation
    std::vector<int> team_size(num_team + 1);
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    // Free each row
    for (int i = 0; i <= num_team; ++i)
        delete[] team_copy[i];

    // Free the top-level pointer
    delete[] team_copy;
}

    // ---- new added for tri level Feature extraction ----

    StateFeatures compute_state_vector(
    int f_eff, int f_div,
    int prev_eff, int prev_div,
    int f_eff_max, int f_div_max,
    int iter, int max_iter,
    int accepted_moves, int total_moves,
    const std::vector<double>& reward_hist,
    const std::vector<int>& div_values,
    double temp, double temp0,
    double flex, double flex_max
    );
    StateFeatures compute_state_vector1(
    int, int, int, int, int, int, int, int, int, int,
    const std::vector<double>&,
    const std::vector<int>&,
    double, double);

    // ---- Utility functions ----
    double compute_delta(int f_curr, int f_prev);
    double reward_from_delta(double delta);
    std::string discretize_state(const StateFeatures& s);
    double rolling_average(const std::vector<double>& values, int window);

    // ---- Local search dispatch (Level 1) ----
    enum LSAlgo { LS_SA = 1, LS_ILS, LS_TS, LS_GD, LS_LAHC, LS_BasicLS };
    int** run_LS(LSAlgo algo, int** team);


    // ---- Feature and State Computation ----
    double compute_div_std_norm(const std::vector<int>& div_values, double div_max);

    // ---- Move acceptance (Level 3) ----
    enum MA_Strategy { MA_ONLY_IMPROVE = 1, MA_ACCEPT_ALL, MA_SA, MA_R2R, MA_THRESHOLD };
    bool accept_move(MA_Strategy rule, int cur_eff, int cur_div, int new_eff, int new_div, double min_div);
      // -------------------------------------------------------------------------
    // Destructor
    // -------------------------------------------------------------------------
    ~Hyper_heuristic(){
       // Free allocated memory
        delete[] eff;
        delete[] state;
        delete[] best_solution;
        delete[] fbest_solution;
        delete[] best_inn;
        delete[] address;
        delete[] w_div;
        delete[] w_eff;
        delete[] num_t_cur;
        delete[] div_best;
        delete[] eff_best;
        delete[] div_fbest;
        delete[] eff_fbest;
        for (int i = 0; i < num_node; ++i) {
            delete[] div_in[i];
            delete[] delta_div[i];
            delete[] tabu_list[i];
        }
        for (int i = 0; i <= num_team; ++i) {
            delete[] team[i];
            delete[] team_check[i];
        }
        delete[] div_in;
        delete[] degree_inf;
        delete[] delta_div;
        delete[] tabu_list;
        delete[] team;
        delete[] team_check;
        delete[] aa;
    }
};

// Function to calculate features (implement actual calculations as per your problem

// External function to calculate features
// Corrected function signature
// Complete version of calculate_features with F1–F39 implemented
Features calculate_features(
    const std::set<int>& S_imp,
    const std::set<int>& S_wrs,
    const std::set<int>& S_eq,
    const std::set<int>& S_ac,
    const std::set<int>& S_uq,
    const std::set<int>& S_nb,
    double Sfirst,
    double Sbest,
    double Sworst,
    const std::vector<double>& fitness_history,    // history of f(S)
    const std::map<int, double>& heuristic_total_time, // per-heuristic time
    int total_heuristics,
    int iteration_count
) {
    Features features;

    // ----------------------------
    // Basic Features (F1–F9)
    // ----------------------------
    features.F1  = total_heuristics;                     // |Θ| = n
    features.F2_nb_ge_1 = S_nb.size();                   // |Θ|nb≥1
    features.F3_iterations = iteration_count;            // # iterations
    features.F4_total_nb   = S_nb.size();                // total best counts
    features.F5_total_imp  = S_imp.size();               // total improvements
    features.F6_total_wrs  = S_wrs.size();               // total worsenings
    features.F7_total_eq   = S_eq.size();                // total equals
    features.F8_total_ac   = S_ac.size();                // total accepted
    features.F9_total_uq   = S_uq.size();                // total unique

    // ----------------------------
    // Quality-based Features (F10–F24)
    // ----------------------------
    features.F10 = Sfirst != 0.0 ? std::abs(Sbest - Sfirst) / Sfirst : 0.0;

    // Improvement ratio relative to total moves
    features.F11 = (iteration_count > 0) ?
                   (static_cast<double>(S_imp.size()) / iteration_count) : 0.0;

    // Worsening ratio relative to total moves
    features.F12 = (iteration_count > 0) ?
                   (static_cast<double>(S_wrs.size()) / iteration_count) : 0.0;

    // Equality ratio
    features.F13 = (iteration_count > 0) ?
                   (static_cast<double>(S_eq.size()) / iteration_count) : 0.0;

    // Ratio of accepted to total moves
    features.F14 = (iteration_count > 0) ?
                   (static_cast<double>(S_ac.size()) / iteration_count) : 0.0;

    // Relative improvement of best over worst
    features.F15 = (Sworst != 0.0) ? std::abs(Sbest - Sworst) / std::abs(Sworst) : 0.0;

    // Stability: variance of fitness history
    if (!fitness_history.empty()) {
        double mean = std::accumulate(fitness_history.begin(), fitness_history.end(), 0.0) / fitness_history.size();
        double var  = 0.0;
        for (double f : fitness_history) var += (f - mean) * (f - mean);
        features.F16 = var / fitness_history.size(); // variance
    } else {
        features.F16 = 0.0;
    }

    // Relative variance normalized by mean
    features.F17 = (features.F16 > 0.0 && !fitness_history.empty()) ?
                   std::sqrt(features.F16) /
                   (std::accumulate(fitness_history.begin(), fitness_history.end(), 0.0) / fitness_history.size())
                   : 0.0;

    // Convergence speed (best - first)/iterations
    features.F18 = (iteration_count > 0) ? (Sbest - Sfirst) / iteration_count : 0.0;

    // Deviation of last from best
    if (!fitness_history.empty())
        features.F19 = std::abs(fitness_history.back() - Sbest);
    else
        features.F19 = 0.0;

    // Improvement frequency (improvements vs. worsens)
    features.F20 = (S_wrs.size() > 0) ?
                   static_cast<double>(S_imp.size()) / S_wrs.size() : 0.0;

    // Progress ratio: (best - worst)/first
    features.F21 = (Sfirst != 0.0) ? (Sbest - Sworst) / Sfirst : 0.0;

    // Improvement stability: improvements / total heuristics
    features.F22 = (total_heuristics > 0) ?
                   static_cast<double>(S_imp.size()) / total_heuristics : 0.0;

    // Acceptance stability: accepted / total heuristics
    features.F23 = (total_heuristics > 0) ?
                   static_cast<double>(S_ac.size()) / total_heuristics : 0.0;

    // Worst stagnation = (first - worst)/first
    features.F24 = (Sfirst != 0.0) ? (Sfirst - Sworst) / Sfirst : 0.0;

    // ----------------------------
    // Time-based Features (F25–F28)
    // ----------------------------
    if (!heuristic_total_time.empty()) {
        // F25: max(th)
        features.F25_max_th = std::max_element(heuristic_total_time.begin(), heuristic_total_time.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->second;

        // F26: min(th)
        features.F26_min_th = std::min_element(heuristic_total_time.begin(), heuristic_total_time.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->second;

        // F27: average time per iteration
        double total_time = 0.0;
        for (const auto& [heuristic, time] : heuristic_total_time) {
            total_time += time;
        }
        features.F27_avg_th = (iteration_count > 0) ? total_time / iteration_count : 0.0;

        // F28: variance of heuristic times
        double mean_time = features.F27_avg_th;
        double variance = 0.0;
        for (const auto& [heuristic, time] : heuristic_total_time) {
            variance += (time - mean_time) * (time - mean_time);
        }
        features.F28_variance_th = heuristic_total_time.size() > 1 ?
                                   variance / (heuristic_total_time.size() - 1) : 0.0;
    } else {
        features.F25_max_th = 0.0;
        features.F26_min_th = 0.0;
        features.F27_avg_th = 0.0;
        features.F28_variance_th = 0.0;
    }

    // ----------------------------
    // Mixed Features (F29–F39)
    // ----------------------------
    features.F29 = (features.F25_max_th - features.F26_min_th) != 0.0 ?
                   features.F27_avg_th / (features.F25_max_th - features.F26_min_th) : 0.0;
    features.F30 = (iteration_count != 0) ?
                   static_cast<double>(S_eq.size()) / iteration_count : 0.0;
    features.F31 = (S_wrs.empty()) ? 0.0 :
                   static_cast<double>(S_eq.size()) / S_wrs.size();
    features.F32 = (total_heuristics > 0) ?
                   static_cast<double>(features.F4_total_nb) / total_heuristics : 0.0;
    features.F33 = (S_wrs.empty()) ? 0.0 :
                   static_cast<double>(S_imp.size()) / S_wrs.size();
    features.F34 = (iteration_count != 0) ?
                   static_cast<double>(S_imp.size()) / iteration_count : 0.0;
    features.F35 = (S_wrs.empty()) ? 0.0 :
                   static_cast<double>(S_imp.size()) / S_wrs.size();
    features.F36 = (iteration_count != 0) ?
                   static_cast<double>(S_nb.size()) / iteration_count : 0.0;
    features.F37 = (iteration_count != 0) ?
                   static_cast<double>(S_uq.size()) / iteration_count : 0.0;
    features.F38 = (iteration_count != 0) ?
                   static_cast<double>(S_wrs.size()) / iteration_count : 0.0;
    features.F39 = (S_eq.empty()) ? 0.0 :
                   static_cast<double>(S_wrs.size()) / S_eq.size();

    return features;
}


int Hyper_heuristic::select_max_multiple(int *best_node, int *best_team, int &num_best)
{
	double max = -1000000;
	double product = 0;
	num_best = -1;
	double beta= 0.4;
	for (int i = 0; i < num_node; i++)
	{
		if (state[i] == 0)
		{
			for (int j = 1; j <= num_team; j++)
			{
				if (num_t_cur[j] < num_each_t) //if team have less  than Mt number of practitionar
				{
					product = sqrt(delta_div[i][j] * beta * eff[i]);
					//	cout<< " i= " << i << " j= " << j << " Delta= " << Delta_Div[i][j] << " Eff= " << Eff[i] << " product= "<< product << " max= " << max << " num_best = " << num_best << endl;
					//	getchar();
					if (product > max)
					{
						max = product;
						best_node[0] = i;
						best_team[0] = j;
						num_best = 1;
					}
					else if ((fabs(product - max)<0.01) && num_best<50)
					{
						best_node[num_best] = i;
						best_team[num_best] = j;
						num_best++;
					}
				}
			}
		}
	}
	if (num_best > 0)
		return 1;
	else
		return -1;
}

// Solution is feasible// Implementation of the ruin_and_recreate function

int Hyper_heuristic::swap1(int iter, int &d1, int &d2)
{
	double cha1, cha2, cha;
	int num1 = -1, num2 = -1, dt1, dt2;
	double eff_best = MINVALUE, eff_tabu_best = MINVALUE;
	double delta_eff;
	int idx_min_eff, idx_sec_eff;
	for (int i = 0; i < num_node; i++)
		delta_div[i][0] = 0;
	idx_min_eff = min_func(w_eff, num_team);
	idx_sec_eff = sec_func(w_eff, idx_min_eff, num_team);
	int can_i[MAXNUM], can_j[MAXNUM];
	int tabu_can_i[MAXNUM], tabu_can_j[MAXNUM];
	int ij_len = 0, tabu_ij_len = 0;
	/* swap a member in the team with minimum eff and a member that is not assigned */
	for (int i = 0; i < num_node; i++)
	{
		if (state[i] == 0)
		{
			for (int j = 0; j < num_each_t; j++)
			{
				int k = team[idx_min_eff][j];
				cha = delta_div[i][state[k]] - delta_div[k][state[k]] - div_in[i][k];
				double inf_deg;
				if (w_div[state[k]] + cha >= min_div)
				inf_deg = 0;
				else
				inf_deg = min_div - (w_div[state[k]] + cha);
				if (w_div[state[k]] + cha >= min_div)
				{
					double delta1 = eff[i] - eff[k];
					if (delta1 + w_eff[idx_min_eff] > w_eff[idx_sec_eff])
						delta_eff = w_eff[idx_sec_eff] - w_eff[idx_min_eff];
					else
						delta_eff = delta1;
					if ((tabu_list[i][state[k]] <= iter) && (tabu_list[k][0] <= iter))
					{
						if (delta_eff > eff_best)
						{
							eff_best = delta_eff;
							//d1 = i;
							//d2 = k;
							//num1 = 1;
							ij_len = 0;
							can_i[ij_len] = i;
							can_j[ij_len] = k;
							ij_len++;
						}
						else if (fabs(delta_eff - eff_best) <= 1.0e-5 && ij_len < MAXNUM)
						{
							can_i[ij_len] = i;
							can_j[ij_len] = k;
							ij_len++;
						}
					}
					else
					{
						if (delta_eff > eff_tabu_best)
						{
							eff_tabu_best = delta_eff;
							tabu_ij_len = 0;
							tabu_can_i[tabu_ij_len] = i;
							tabu_can_j[tabu_ij_len] = k;
							tabu_ij_len++;
						}
						else if (fabs(delta_eff - eff_tabu_best) <= 1.0e-5 && tabu_ij_len < MAXNUM)
						{
							tabu_can_i[tabu_ij_len] = i;
							tabu_can_j[tabu_ij_len] = k;
							tabu_ij_len++;
						}
					}
				}
			}
		}
	}
	/* swap a member in the team with minimum eff and a member in another team */
	for (int i = 0; i < num_node; i++)
	{
		if (state[i] != 0 && state[i] != idx_min_eff)
		{
			for (int j = 0; j < num_each_t; j++)
			{
				int k = team[idx_min_eff][j];
				cha1 = delta_div[k][state[i]] - delta_div[i][state[i]] - div_in[i][k];
				cha2 = delta_div[i][state[k]] - delta_div[k][state[k]] - div_in[i][k];
				double delta1 = eff[i] - eff[k];
				double delta2 = eff[k] - eff[i];
				delta_eff = delta1;
				if (delta1 + w_eff[idx_min_eff] > delta2 + w_eff[state[i]])
					delta_eff = delta2 + w_eff[state[i]] - w_eff[idx_min_eff];
				if (delta_eff + w_eff[idx_min_eff] > w_eff[idx_sec_eff])
					delta_eff = w_eff[idx_sec_eff] - w_eff[idx_min_eff];
				if ((state[i] != state[k]) && (w_div[state[i]] + cha1 >= min_div) && (w_div[state[k]] + cha2 >= min_div))
				{
					if ((tabu_list[i][state[k]] <= iter) && (tabu_list[k][state[i]] <= iter))
					{
						//delta_eff = (eff[i] - eff[k]) + alpha*(cha2);
						if (delta_eff > eff_best)
						{
							eff_best = delta_eff;
							ij_len = 0;
							can_i[ij_len] = i;
							can_j[ij_len] = k;
							ij_len++;
						}
						else if (fabs(eff_best - delta_eff) < 1.0e-5 && ij_len < MAXNUM)
						{
							can_i[ij_len] = i;
							can_j[ij_len] = k;
							ij_len++;
						}
					}
					else
					{
						if (delta_eff > eff_tabu_best)
						{
							eff_tabu_best = delta_eff;
							tabu_ij_len = 0;
							tabu_can_i[tabu_ij_len] = i;
							tabu_can_j[tabu_ij_len] = k;
							tabu_ij_len++;
						}
						else if (fabs(delta_eff - eff_tabu_best) <= 1.0e-5 && tabu_ij_len < MAXNUM)
						{
							tabu_can_i[tabu_ij_len] = i;
							tabu_can_j[tabu_ij_len] = k;
							tabu_ij_len++;
						}
					}
				}
			}
		}
	}
	//aspiration creterion
	if ((tabu_ij_len > 0 && eff_tabu_best > eff_best && f_cur + eff_tabu_best > fbest) || (tabu_ij_len > 0 && ij_len == 0))
	{
		int rx = rand() % tabu_ij_len;
		d1 = tabu_can_i[rx];
		d2 = tabu_can_j[rx];
	}
	else if (ij_len > 0)
	{
		int rx = rand() % ij_len;
		d1 = can_i[rx];
		d2 = can_j[rx];
	}
	return eff_best;
}

int Hyper_heuristic::swap_ils(int iter, int &d1, int &d2)
{
	double cha1, cha2, cha;
	int num1 = -1, num2 = -1, dt1, dt2;
	double eff_best = MINVALUE, eff_tabu_best = MINVALUE;
	double delta_eff, delta_inf_deg;
	int idx_min_eff, idx_sec_eff;
	for (int i = 0; i < num_node; i++)
		delta_div[i][0] = 0;
	idx_min_eff = min_func(w_eff, num_team);
	idx_sec_eff = sec_func(w_eff, idx_min_eff, num_team);
	int can_i[MAXNUM], can_j[MAXNUM];
	int tabu_can_i[MAXNUM], tabu_can_j[MAXNUM];
	int ij_len = 0, tabu_ij_len = 0;
	/* swap a member in the team with minimum eff and a member that is not assigned */
	for (int i = 0; i < num_node; i++)
	{
		if (state[i] == 0)
		{
			for (int j = 0; j < num_each_t; j++)
			{
				int k = team[idx_min_eff][j];
				cha = delta_div[i][state[k]] - delta_div[k][state[k]] - div_in[i][k];
				double delta1 = eff[i] - eff[k];
				if (delta1 + w_eff[idx_min_eff] > w_eff[idx_sec_eff])
					delta_eff = w_eff[idx_sec_eff] - w_eff[idx_min_eff];
				else
					delta_eff = delta1;
				double inf_deg;
				if (w_div[state[k]] + cha >= min_div)
					inf_deg = 0;
				else
					inf_deg = min_div - (w_div[state[k]] + cha);
				delta_inf_deg = inf_deg - degree_inf[state[k]];
				delta_eff = delta_eff - p_factor*delta_inf_deg;
				if ((tabu_list[i][state[k]] <= iter) && (tabu_list[k][0] <= iter))
				{
					if (delta_eff > eff_best)
					{
						eff_best = delta_eff;
						ij_len = 0;
						can_i[ij_len] = i;
						can_j[ij_len] = k;
						ij_len++;
					}
					else if (fabs(delta_eff - eff_best) <= 1.0e-5 && ij_len < MAXNUM)
					{
						can_i[ij_len] = i;
						can_j[ij_len] = k;
						ij_len++;
					}
				}
				else
				{
					if (delta_eff > eff_tabu_best)
					{
						eff_tabu_best = delta_eff;
						tabu_ij_len = 0;
						tabu_can_i[tabu_ij_len] = i;
						tabu_can_j[tabu_ij_len] = k;
						tabu_ij_len++;
					}
					else if (fabs(delta_eff - eff_tabu_best) <= 1.0e-5 && tabu_ij_len < MAXNUM)
					{
						tabu_can_i[tabu_ij_len] = i;
						tabu_can_j[tabu_ij_len] = k;
						tabu_ij_len++;
					}
				}
			}
		}
	}
	/* swap a member in the team with minimum eff and a member in another team */
	for (int i = 0; i < num_node; i++)
	{
		if (state[i] != 0 && state[i] != idx_min_eff)
		{
			for (int j = 0; j < num_each_t; j++)
			{
				int k = team[idx_min_eff][j];
				cha1 = delta_div[k][state[i]] - delta_div[i][state[i]] - div_in[i][k];
				cha2 = delta_div[i][state[k]] - delta_div[k][state[k]] - div_in[i][k];
				double delta1 = eff[i] - eff[k];
				double delta2 = eff[k] - eff[i];
				delta_eff = delta1;
				if (delta1 + w_eff[idx_min_eff] > delta2 + w_eff[state[i]])
					delta_eff = delta2 + w_eff[state[i]] - w_eff[idx_min_eff];
				if (delta_eff + w_eff[idx_min_eff] > w_eff[idx_sec_eff])
					delta_eff = w_eff[idx_sec_eff] - w_eff[idx_min_eff];

				double inf_deg1;
				if (w_div[state[k]] + cha2 >= min_div)
					inf_deg1 = 0;
				else
					inf_deg1 = min_div - (w_div[state[k]] + cha2);
				double inf_deg2;
				if (w_div[state[i]] + cha1 >= min_div)
					inf_deg2 = 0;
				else
					inf_deg2 = min_div - (w_div[state[i]] + cha1);

				delta_inf_deg = inf_deg1 - degree_inf[state[k]] + inf_deg2 - degree_inf[state[i]];
				delta_eff = delta_eff - p_factor*delta_inf_deg;
				if ((tabu_list[i][state[k]] <= iter) && (tabu_list[k][state[i]] <= iter))
				{
					if (delta_eff > eff_best)
					{
						eff_best = delta_eff;
						ij_len = 0;
						can_i[ij_len] = i;
						can_j[ij_len] = k;
						ij_len++;
					}
					else if (fabs(eff_best - delta_eff) < 1.0e-5 && ij_len < MAXNUM)
					{
						can_i[ij_len] = i;
						can_j[ij_len] = k;
						ij_len++;
					}
				}
				else
				{
					if (delta_eff > eff_tabu_best)
					{
						eff_tabu_best = delta_eff;
						tabu_ij_len = 0;
						tabu_can_i[tabu_ij_len] = i;
						tabu_can_j[tabu_ij_len] = k;
						tabu_ij_len++;
					}
					else if (fabs(delta_eff - eff_tabu_best) <= 1.0e-5 && tabu_ij_len < MAXNUM)
					{
						tabu_can_i[tabu_ij_len] = i;
						tabu_can_j[tabu_ij_len] = k;
						tabu_ij_len++;
					}
				}
			}
		}
	}
	//aspiration criterion
	if ((tabu_ij_len > 0 && eff_tabu_best > eff_best && f_cur + eff_tabu_best > fbest) || (tabu_ij_len > 0 && ij_len == 0))
	{
		int rx = rand() % tabu_ij_len;
		d1 = tabu_can_i[rx];
		d2 = tabu_can_j[rx];
	}
	else if (ij_len > 0)
	{
		int rx = rand() % ij_len;
		d1 = can_i[rx];
		d2 = can_j[rx];
	}
	return eff_best;
}

int Hyper_heuristic::swap_min(int iter, int &c1, int &c2)
{
	double cha, cha1, cha2;
	double delta_best = MINVALUE;
	for (int i = 0; i < num_node; i++)
		delta_div[i][0] = 0;
	int idx_min_div = min_func(w_div, num_team);
	/* swap a member in the team with mininum div and a member that is not assigned */
	for (int i = 0; i < num_node; i++)
	{
		if (state[i] == 0)
		{
			for (int j = 0; j < num_each_t; j++)
			{
				int k = team[idx_min_div][j];
				if ((tabu_list[i][state[k]] <= iter) && (tabu_list[k][0] <= iter))
				{
					cha1 = delta_div[i][state[k]];
					cha2 = -delta_div[k][state[k]];
					cha = cha1 + cha2 - div_in[i][k];
					if (cha > delta_best)
					{
						delta_best = cha;
						c1 = i;
						c2 = k;
					}
				}
			}
		}
	}
	/* swap a member in the team with minmium  div and a member that assigned in another team */
	for (int i = 0; i < num_node; i++)
	{
		if (state[i] != 0)
		{
			for (int j = 0; j < num_each_t; j++)
			{
				int k = team[idx_min_div][j];
				cha1 = delta_div[i][state[k]] - delta_div[k][state[k]] - div_in[i][k];
				cha2 = delta_div[k][state[i]] - delta_div[i][state[i]] - div_in[i][k];
				if (((w_div[state[i]] + cha2 >= min_div) || cha2 > 0) && state[i] != state[k])
				{
					if ((tabu_list[i][state[k]] <= iter) && (tabu_list[k][state[i]] <= iter))
					{
						if (cha1 > delta_best)
						{
							delta_best = cha1;
							c1 = i;
							c2 = k;
						}
					}
				}
			}
		}
	}
	return delta_best;
}


void Hyper_heuristic::compute_mindiv()
{
	double dd = 0;
	min_div = 0;
	for (int i = 0; i < num_node; i++)
		for (int j = 0; j < num_node; j++)
			dd = dd + div_in[i][j];
	dd = dd / ((num_node - 1)*num_node);
	min_div = d_min*num_each_t*(num_each_t - 1) / 2 * dd;
	cout << "dd=" << dd << " MinDiv= " << min_div << endl;
}

int Hyper_heuristic::randomInt(int n)
{
	return rand() % n;
}

//identify the index of the minimum number in the array aa
int Hyper_heuristic::min_func(int *aa, int len)
{
	int min_value = MAXVALUE;
	int idx;
	for (int i = 1; i <= len; i++)
	{
		if (aa[i] < min_value)
		{
			min_value = aa[i];
			idx = i;
		}
	}
	return idx;
}

//identify the index of the second minimum number in the array aa
int Hyper_heuristic::sec_func(int *aa, int idx_min, int len)
{
	int min_value = MAXVALUE;
	int idx;
	for (int i = 1; i <= len; i++)
	{  //cout<<aa[i];
		if (aa[i] < min_value && i != idx_min)
		{
			min_value = aa[i];
			idx = i;
		}
	}
	return idx;
}

//identify the index of the third minimum number in the array aa
int Hyper_heuristic::th_func(int *aa,int idx_min1, int idx_min2, int len)
{
	int min_value = MAXVALUE;
	int idx;
	for (int i = 1; i <= len; i++)
	{  //cout<<aa[i];
		if (aa[i] < min_value && i != idx_min1 && i != idx_min2)
		{
			min_value = aa[i];
			idx = i;
		}
	}
	return idx;
}

//identify the index of the maximum number in the array aa
int Hyper_heuristic::max_func(int *aa, int len)
{
	int pos;
	int max_value = MINVALUE;
	for (int i = 0; i<len; i++)
	{
		if (aa[i] > max_value)
		{
			max_value = aa[i];
			pos = i;
		}
	}
	return pos;
}
int Hyper_heuristic::rand_func(int *aa, int len)
{
	int pos;
	int max_value = MINVALUE;
	for (int i = 0; i<len; i++)
	{
		if (aa[i])
		{
			max_value = aa[i];
			pos = i;
		}
	}
	return pos;
}
void Hyper_heuristic::update_delta(int node, int newn, int old)
{
	if ((newn != 0) && (old == 0))
		for (int i = 0; i < num_node; i++)
			delta_div[i][newn] = delta_div[i][newn] + div_in[i][node];
	else if ((newn != 0) && (old != 0))
	{
		for (int i = 0; i < num_node; i++)
			delta_div[i][newn] += div_in[i][node];
		for (int i = 0; i < num_node; i++)
			delta_div[i][old] -= div_in[i][node];
	}
	else if ((newn == 0) && (old != 0))
		for (int i = 0; i < num_node; i++)
			delta_div[i][old] -= div_in[i][node];
}


void Hyper_heuristic::repair_solution()
{
	int c1 = -1, c2 = -1;
	double min, best;
	int node1, node2, team_min, team_old;
	int iter = 0;
	//cout<<"begin repair solution "<<endl;
	//for (int t=1; t<=num_each_t; t++){
	int idx = min_func(w_div, num_team);
	min = w_div[idx];
	//cout<<idx;
	for (int i = 0; i < num_node; i++)
		for (int j = 0; j <= num_team; j++)
			tabu_list[i][j] = 0;//
	while (min < min_div)
	{
		best = swap_min(iter, c1, c2);
		node1 = c1;									 /* move in */
		node2 = c2;									 /* move out */
		team_min = state[c2];						/*team with min div*/
		team_old = state[c1];
		int a1 = address[node1];
		int a2 = address[node2];
		if (team_old == 0)
		{
			tabu_list[node1][team_old] = iter + tl + randomInt(tabu_tenure);
			w_div[team_min] = w_div[team_min] + delta_div[node1][team_min] - delta_div[node2][team_min] - div_in[node1][node2];
			w_eff[team_min] = w_eff[team_min] + eff[node1];
			update_delta(node1, team_min, team_old);
			state[node1] = team_min;
			team[team_min][a2] = node1;
			address[node1] = a2;

			tabu_list[node2][team_min] = iter + tl + randomInt(tabu_tenure);
			w_eff[team_min] = w_eff[team_min] - eff[node2];
			update_delta(node2, team_old, team_min);
			state[node2] = team_old;
			team[team_old][a1] = node2;
			address[node2] = a1;
		}
		else
		{
			tabu_list[node1][team_old] = iter + tl + randomInt(tabu_tenure);
			w_div[team_min] = w_div[team_min] + delta_div[node1][team_min] - delta_div[node2][team_min] - div_in[node1][node2];
			w_div[team_old] = w_div[team_old] + delta_div[node2][team_old] - delta_div[node1][team_old] - div_in[node1][node2];
			w_eff[team_min] = w_eff[team_min] + eff[node1];
			w_eff[team_old] = w_eff[team_old] - eff[node1];
			update_delta(node1, team_min, team_old);
			state[node1] = team_min;
			team[team_min][a2] = node1;
			address[node1] = a2;

			tabu_list[node2][team_min] = iter + tl + randomInt(tabu_tenure);
			w_eff[team_old] = w_eff[team_old] + eff[node2];
			w_eff[team_min] = w_eff[team_min] - eff[node2];
			update_delta(node2, team_old, team_min);
			state[node2] = team_old;
			team[team_old][a1] = node2;
			address[node2] = a1;
		}
		iter++;
		int idx = min_func(w_div, num_team);
		min = w_div[idx];
		if (min >= min_div || iter > 100000)
			break;
	}

	idx = min_func(w_eff, num_team);
	f_cur = w_eff[idx];

	//cout<<f_cur;
}

//generate an initial solution Greedy Construction
int Hyper_heuristic::generate_initial()
{
    //srand((unsigned)time(NULL));
	int *sort_eff = new int[num_node];
	int *best_node = new int[num_node];
	int *best_team = new int[num_node];
	int *arr_t = new int[num_team + 1];
	int num_best = -1;
	memset(state, 0, sizeof(int)*num_node);  //initial all members in state array to team zero
	memset(best_solution, 0, sizeof(int)*num_node);
	for (int i = 0; i < num_node; i++)
	{
		for (int j = 0; j <= num_team; j++)
		{
			tabu_list[i][j] = 0;
			delta_div[i][j] = 0;
		}
	}
	for (int i = 0; i <= num_team; i++)
	{
		w_div[i] = 0;
		w_eff[i] = 0;
		div_best[i] = 0;
		eff_best[i] = 0;
	}
	for (int i = 1; i <= num_team; i++)
		num_t_cur[i] = 0;
	for (int i = 0; i < num_node; i++){
		sort_eff[i] = eff[i];
    }
        for (int i = 0; i < num_node; i++){
		int v = max_func(sort_eff, num_node);
		// cout<<"sort_eff="<< v <<"  ";
		 }
//       //cout<<endl<<endl;
//    cout<<"----------------------------------------------------"<<endl;
//	cout<<"(3)Starting Greedy Construction initial solution:"<<endl;
	int m = 1;
	while (1)
	{
		//allocate a practitioner with the with any efficiency to each crew
		if (m == 1)
		{
			m = -1;
			for (int i = 1; i <= num_team; i++)
			{
				int v = max_func(sort_eff, num_node);
				sort_eff[v] = 0;
				int k = i;
				w_eff[k] = w_eff[k] + eff[v];
				w_div[k] = w_div[k] + delta_div[v][k];
				num_t_cur[k] = num_t_cur[k] + 1;
				update_delta(v, k, state[v]);
				state[v] = k;
			}
		}
		else
		{
			int l = select_max_multiple(best_node, best_team, num_best);
			if (l == 1)
			{
				int n = randomInt(num_best);
				int v = best_node[n];
				int k = best_team[n];
				w_eff[k] = w_eff[k] + eff[v];
				w_div[k] = w_div[k] + delta_div[v][k];
				num_t_cur[k] = num_t_cur[k] + 1;
				update_delta(v, k, state[v]);
				state[v] = k;

			}
			else
				break;
		}
	}
  int j;
	for (int i = 0; i <= num_team; i++)
		for (int j = 0; j < num_node; j++)
			team[i][j] = -1;
	for (int i = 0; i <= num_team; i++)
	{
		arr_t[i] = 0;
		for (int j = 0; j < num_node; j++)
		{
			if (state[j] == i)
			{
				team[i][arr_t[i]] = j;
				address[j] = arr_t[i];
				arr_t[i]++;
			}
		}

	}
	for (int i = 0; i < num_node; i++)
		delta_div[i][0] = 0;
	int idx = min_func(w_eff, num_team);
	f_cur = w_eff[idx];
	//cout<<f_cur;
    int idx11 = min_func(w_div, num_team);
	int f_cur_div = w_div[idx11];
    repair_solution();
//
    /*    cout<<endl;
        cout<<"team 0:";
    for (int i = 0; i < num_node-(num_team*num_each_t); ++i) {
         cout<<team[0][i]<<" ";
         }
         cout<<endl;
         cout<<endl;
	for (int t=1; t <= num_team ; t++){
         cout<<"team "<<t<<": ";
        for (int j=0; j < num_each_t; j++){
         cout<<team[t][j]<<"\t";
        }
         cout<<"eff="<<w_eff[t]<<" "<<"div="<<w_div[t]<<"\t";
        cout<<endl;
	}
	cout<<endl;*/
	//cout<<endl;
	//cout<<"---------------------------------------------------"<<endl;
	delete [] sort_eff;
 	delete [] best_node;
	delete [] best_team;
	delete [] arr_t;
	return 0;
}

int Hyper_heuristic::generate_initialrandom()
{
	int *sort_eff = new int[num_node];
	int *best_node = new int[num_node];
	int *best_team = new int[num_node];
	int *arr_t = new int[num_team + 1];
	int num_best = -1;
	int **team1;
	memset(state, 0, sizeof(int)*num_node);
	memset(best_solution, 0, sizeof(int)*num_node);
	for (int i = 0; i < num_node; i++)
	{
		for (int j = 0; j <= num_team; j++)
		{
			tabu_list[i][j] = 0;
			delta_div[i][j] = 0;
		}
	}
	for (int i = 0; i <= num_team; i++)
	{
		w_div[i] = 0;
		w_eff[i] = 0;
		div_best[i] = 0;
		eff_best[i] = 0;
	}
	for (int i = 1; i <= num_team; i++)
		num_t_cur[i] = 0;
	for (int i = 0; i < num_node; i++)
		sort_eff[i] = eff[i];

    //cout<<"----------------------------------------------------"<<endl;
    //cout<<"(4)Starting Random Construction initial solution:"<<endl;
	int m = 1;
	while (1)
	{
		if (m == 1)
		{
			m = -1;
			for (int i = 1; i <= num_team; i++)
			{
				int v = rand_func(sort_eff, num_node);
				sort_eff[v] = 0;
				int k = i;
				w_eff[k] = w_eff[k] + eff[v];
				w_div[k] = w_div[k] + delta_div[v][k];
				num_t_cur[k] = num_t_cur[k] + 1;
				update_delta(v, k, state[v]);
				state[v] = k;
			}
		}
		else
		{
			int l = select_max_multiple(best_node, best_team, num_best);
			if (l == 1)
			{
				int n = randomInt(num_best);
				int v = best_node[n];
				int k = best_team[n];
				w_eff[k] = w_eff[k] + eff[v];
				w_div[k] = w_div[k] + delta_div[v][k];
				num_t_cur[k] = num_t_cur[k] + 1;
				update_delta(v, k, state[v]);
				state[v] = k;
			}
			else
				break;
		}
	}
  int j;
	for (int i = 0; i <= num_team; i++)
		for (int j = 0; j < num_node; j++)
			team[i][j] = -1;

	for (int i = 0; i <= num_team; i++)
	{
		arr_t[i] = 0;
		for (int j = 0; j < num_node; j++)
		{
			if (state[j] == i)
			{
				team[i][arr_t[i]] = j;
				address[j] = arr_t[i];
				arr_t[i]++;
			}
		}

	}
	for (int i = 0; i < num_node; i++)
		delta_div[i][0] = 0;
	int idx = min_func(w_eff, num_team);
	f_cur = w_eff[idx];
	int idx1 = min_func(w_div, num_team);
	f_cur_div = w_div[idx1];
	repair_solution();
	team1 = team;
	/* cout<<"team 0:";
        for (int i = 0; i < num_node-(num_team*num_each_t); ++i) {
         cout<<team[0][i]<<" ";
         }
         cout<<endl;
         cout<<endl;
	for (int t=1; t <= num_team ; t++){
         cout<<"team "<<t<<": ";
        for (int j=0; j < num_each_t; j++){
         cout<<team[t][j]<<"\t";
        }
        cout<<"eff="<<w_eff[t]<<"\t"<<"div="<<w_div[t]<<" ";
        cout<<endl;
	}
	cout<<endl;
	cout<<endl;
	cout<<"---------------------------------------------------"<<endl;*/

	delete [] sort_eff;
 	delete [] best_node;
	delete [] best_team;
	delete [] arr_t;
	return 0;
}

void Hyper_heuristic::initialization(const string &current_file_name)
{
    ifstream fic;
    file_name = current_file_name; // Set the global file_name
    fic.open(file_name.c_str());
    if (fic.fail()) {
        cout << "### Error opening file: " << file_name << endl;
        exit(0);
    }
    if (fic.eof()) {
        cout << "### Error reading file: " << file_name << endl;
        exit(0);
    }
	char str_reading[100];
	double nn[4];
	for (int i = 0; i<4; i++)
	{
		double x;
		fic >> str_reading;
		fic >> str_reading;
		fic >> str_reading;
		fic >> x;
		nn[i] = floor(x);
		fic >> str_reading;
	}
	num_node = nn[0];
	num_team = nn[1];
	num_each_t = nn[2];
	min_div = nn[3];
    //cout << " num_node= " << num_node << "  " << " num_team= " << num_team << ",num_each_t=" << num_each_t << "  " << " min_div= " << min_div << endl;

	eff = new int[num_node];
	state = new int[num_node];
	best_solution = new int[num_node];
	fbest_solution = new int[num_node];
	best_inn = new int[num_node];
	address = new int[num_node];

	w_div = new int[num_team + 1];
	w_eff = new int[num_team + 1];
	num_t_cur = new int[num_team + 1];
	div_best = new int[num_team + 1];
	eff_best = new int[num_team + 1];
	div_fbest = new int[num_team + 1];
	eff_fbest = new int[num_team + 1];
	div_in = new double*[num_node];
	degree_inf = new double[num_team + 1];
	for (int i = 0; i < num_node; i++)
		div_in[i] = new double[num_node];
	delta_div = new int*[num_node];
	for (int i = 0; i < num_node; i++)
		delta_div[i] = new int[num_team + 1];
	tabu_list = new int*[num_node];
	for (int i = 0; i < num_node; i++)
		tabu_list[i] = new int[num_team + 1];
	team = new int*[num_team + 1];
	for (int i = 0; i <= num_team; i++)
		team[i] = new int[num_node];
	team_check = new int*[num_team + 1];
	for (int i = 0; i <= num_team; i++)
		team_check[i] = new int[num_node];

	fic >> str_reading;
	fic >> str_reading;
	fic >> str_reading;
	for (int i = 0; i < num_node; i++)
	{
		int xx;
		fic >> str_reading;
		fic >> xx;
		eff[i] = xx;
	}
	cout << endl;
	fic >> str_reading;
	fic >> str_reading;
	fic >> str_reading;
	fic >> str_reading;
	for (int i = 0; i < num_node; i++)
	{
		for (int j = 0; j < num_node; j++)
		{
			double yy;
			fic >> str_reading;
			fic >> yy;
			div_in[i][j] = yy;
		}
	}
	fic.close();
	cout << "End of reading file" << endl;
	cout << "-----------------------------------------" << endl;
}

int** Hyper_heuristic::iterated_local_search(){
        //double avg_time =0.0;
        int ls_depth = 100;//or 200
        int iter = 0;
        int f_Sintial, f_Sbest, F_perturb, f_S, f_Spar,  f_Spar2;
        double best;
	    int node1, node2, team_min, team_old;
	    int a1, a2;
	    int d1 = 0, d2 = 0;
	    int **Sinitial,**Sbest, **S, **Spar, **Spar2;
	    // begin of iterated local search
        //generate_initial();//generate initial solution s0
        Sinitial = team;
        Sbest = Sinitial;
        f_Sintial = f_cur;
        f_Sbest = f_Sintial;
        f_S = feasible_local_search(); // local search for initial solution Sinitial
        S = team;
        // ---------- NEW: Convergence Setup ----------
        std::vector<int> convergence;
        convergence.push_back(f_Sbest);

        std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
        if (!std::filesystem::exists(folder_path))
            std::filesystem::create_directories(folder_path);

        std::filesystem::path conv_path = folder_path / "iterated_local_search_Convergence.csv";
        std::ofstream conv_file(conv_path);
        if (conv_file.is_open()) {
            conv_file << "Iteration,Fbest\n";
        }
        // --------------------------------------------
        //start_time = clock();
        //while (1.0*(clock()-start_time)/CLOCKS_PER_SEC<time_limit)
        while (iter < ls_depth)
        {        //start_time = clock();
                 Spar = simulated_annealing();
                 f_Spar = f_cur;
                 f_Spar2 = feasible_local_search();
                 Spar2 = team;
                //accept (S,S'')---->S=S''
                if (f_Spar2 > f_S){
                    f_S = f_Spar2;
                    S = Spar2;
                    fbest = f_Spar2;
                    team = S;
                    end_time = clock();

                }

            if (f_Spar2 > f_best_inn)
            {
                f_best_inn = f_Spar2 ;
                for (int m = 0; m < num_node; m++)
                    best_inn[m] = state[m];
                for (int m = 0; m < num_node; m++)
                       best_solution[m] = state[m];
                for (int m = 1; m <= num_team; m++)
                       {
                       eff_best[m] = w_eff[m];
                       div_best[m] = w_div[m];
                   }
            // ---- Save Convergence ----
            convergence.push_back(f_best_inn);
            }

            if (conv_file.is_open()) {
                conv_file << iter + 1 << "," << f_best_inn << "\n";
            }
           iter++;
        }

        if (conv_file.is_open())
            conv_file.close();

        //std::cout << "Convergence data saved to: " << conv_path << std::endl;
        repair_solution();

        return team;
}


int** Hyper_heuristic::simulated_annealing() {
    double best;
    int node1, node2, team_min, team_old;
    int a1, a2;
    double current_temp = initial_temp;
    int iter = 0;
    double f_s0, f_s1;
    int fbest1;
    int d1 = 0, d2 = 0;
    float Tmin = 0.001;

    int** S0_current = team;   // initial solution (greedy constructed)
    int** S1_solution = nullptr;
    int** S1 = S0_current;

    f_s0 = f_cur;
    fbest1 = f_s0;

    // ---------- NEW: Convergence Setup ----------
    std::vector<int> convergence;
    convergence.push_back(f_s0);

    std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "simulated_annealing_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest,Temperature\n";
    }
    // --------------------------------------------

    while (current_temp > Tmin) {
        while (iter < 100) {
            best = swap1(iter, d1, d2);
            node1 = d1;							    /* move in */
            node2 = d2;								/* move out */
            team_min = state[d2];					/* team with min eff */
            team_old = state[d1];
            a1 = address[node1];
            a2 = address[node2];

            if (team_old == 0) {
                tabu_list[node1][team_old] = iter + tl + randomInt(tabu_tenure);
                w_eff[team_min] = w_eff[team_min] + eff[node1] - eff[node2];
                w_div[team_min] = w_div[team_min] + delta_div[node1][team_min] -
                                  delta_div[node2][team_min] - div_in[node1][node2];
                update_delta(node1, team_min, team_old);
                state[node1] = team_min;
                team[team_min][a2] = node1;
                address[node1] = a2;

                tabu_list[node2][team_min] = iter + tl + randomInt(tabu_tenure);
                update_delta(node2, team_old, team_min);
                state[node2] = team_old;
                team[team_old][a1] = node2;
                address[node2] = a1;
            } else {
                tabu_list[node1][team_old] = iter + tl + randomInt(tabu_tenure);
                w_eff[team_min] = w_eff[team_min] + eff[node1] - eff[node2];
                w_div[team_min] = w_div[team_min] + delta_div[node1][team_min] -
                                  delta_div[node2][team_min] - div_in[node1][node2];
                w_div[team_old] = w_div[team_old] + delta_div[node2][team_old] -
                                  delta_div[node1][team_old] - div_in[node1][node2];
                update_delta(node1, team_min, team_old);
                state[node1] = team_min;
                team[team_min][a2] = node1;
                address[node1] = a2;

                tabu_list[node2][team_min] = iter + tl + randomInt(tabu_tenure);
                w_eff[team_old] = w_eff[team_old] - eff[node1] + eff[node2];
                update_delta(node2, team_old, team_min);
                state[node2] = team_old;
                team[team_old][a1] = node2;
                address[node2] = a1;
            }

            iter++;
            int idx = min_func(w_eff, num_team);
            f_s1 = w_eff[idx];
            S1_solution = team;
            double delta_fitness = f_s1 - f_s0;

            // ---- Acceptance Criteria ----
            if (delta_fitness >= 0) {
                end_time = clock();
                S0_current = S1_solution;
                f_s0 = f_s1;
            } else if (exp(-delta_fitness / current_temp) > (rand() / (double)RAND_MAX)) {
                end_time = clock();
                S0_current = S1_solution;
                f_s0 = f_s1;
            }

            // ---- Update Best Solution ----
            if (f_s1 > fbest1) {
                fbest1 = f_s1;
                fbest = fbest1;
                end_time = clock();

                for (int m = 0; m < num_node; m++) {
                    best_solution[m] = state[m];
                    best_inn[m] = state[m];
                }
                for (int m = 1; m <= num_team; m++) {
                    eff_best[m] = w_eff[m];
                    div_best[m] = w_div[m];
                }
                // ---- Save Convergence ----
                convergence.push_back(fbest1);

            }


        current_temp *= cooling_rate;
        if (conv_file.is_open()) {
                conv_file << iter << "," << fbest1 << "," << current_temp << "\n";
            }

    }
    }
    if (conv_file.is_open())
        conv_file.close();

    repair_solution();
    //std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return team;
}

int** Hyper_heuristic::IteratedLocalSearch(int** Sstart, int OPj)
{
    //-----------------------------------------------------------------
    // Initial setup
    //-----------------------------------------------------------------
    int** Sinitial = Sstart;
    objective_Function1(Sinitial);
    int f_Sinitial = f_cur;

    int** Sbest = Sinitial;
    int f_Sbest = f_Sinitial;

    // Apply initial local search
    team = Sinitial;
    int f_S = feasible_local_search();
    int** S = team;
    f_S = f_cur;

    //-----------------------------------------------------------------
    // Convergence setup
    //-----------------------------------------------------------------
    int max_iter = 100;
    int iter = 0;

    std::vector<int> convergence;
    convergence.push_back(f_Sbest);

    std::filesystem::path folder_path =
        "D:/Datasets/TRI_LEVEL_HH_MODELS/TRI_LEVEL_MHS_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "IteratedLocalSearch_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest\n";
    }

    //-----------------------------------------------------------------
    // ILS Main Loop (lines 6–13)
    //-----------------------------------------------------------------
    while (iter < max_iter)
    {
        //--------------------------------------------------------------
        // 7. Perturbation: S′ ← Perturbation(S)
        //--------------------------------------------------------------
        team = S;
        int** Spert = apply_LLHop(OPj, team);
        objective_Function1(Spert);
        int f_Spert = f_cur;

        //--------------------------------------------------------------
        // 8. Local Search on S′: S′′ ← LocalSearch(S′)
        //--------------------------------------------------------------
        team = Spert;
        int f_Spp = feasible_local_search();   // after local search
        int** Spp = team;                      // S′′
        f_Spp = f_cur;

        //--------------------------------------------------------------
        // 9. Maintain Best Solution
        //--------------------------------------------------------------
        int best_of_three = f_S;
        int** bestSol_of_three = S;

        if (f_Spert > best_of_three && f_cur_div >= min_div) {
            best_of_three = f_Spert;
            bestSol_of_three = Spert;
        }

        if (f_Spp > best_of_three && f_cur_div >= min_div) {
            best_of_three = f_Spp;
            bestSol_of_three = Spp;
        }

        // Update global best
        if (best_of_three > f_Sbest && f_cur_div >= min_div)
        {
            Sbest = bestSol_of_three;
            f_Sbest = best_of_three;

            for (int i = 0; i < num_node; i++)
                best_solution[i] = state[i];
            for (int t = 1; t <= num_team; t++) {
                eff_best[t] = w_eff[t];
                div_best[t] = w_div[t];
            }
        convergence.push_back(f_Sbest);
        }

        //--------------------------------------------------------------
        // 10–11. Acceptance Rule: Accept if S′′ is better
        //--------------------------------------------------------------
        if (f_Spp >= f_S && f_cur_div >= min_div)
        {
            S = Spp;
            f_S = f_Spp;
        }

        //--------------------------------------------------------------
        // 12. Log convergence
        //--------------------------------------------------------------

        if (conv_file.is_open()) {
            conv_file << iter + 1 << "," << f_Sbest << "\n";
        }

        iter++;
    }

    if (conv_file.is_open())
        conv_file.close();

    //-----------------------------------------------------------------
    // 14. Return best solution
    //-----------------------------------------------------------------
    team = Sbest;
    //std::cout << "Convergence saved to: " << conv_path << std::endl;
    return team;
}



int** Hyper_heuristic::SimulatedAnnealing(int** Sstart, int OPj, double T0, double alpha)
{
    //------------------------------------------------------------
    // Initialization (Algorithm lines 1–5)
    //------------------------------------------------------------
    double T = T0;
    double Tmin = 0.001;

    int** S = Sstart;
    objective_Function1(S);
    int f_S = f_cur;

    int** Sbest = S;
    int f_best = f_S;

    //------------------------------------------------------------
    // Convergence setup
    //------------------------------------------------------------
    std::vector<int> convergence;
    convergence.push_back(f_best);

    std::filesystem::path folder_path =
        "D:/Datasets/TRI_LEVEL_HH_MODELS/TRI_LEVEL_MHS_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "SimulatedAnnealing_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest,Temperature\n";
    }

    //------------------------------------------------------------
    // Simulated Annealing Main Loop (Algorithm lines 6–19)
    //------------------------------------------------------------
    int iter = 0;
    while (T > Tmin)
    {
        //--------------------------------------------------------
        // (7) Generate(S′) using operator OPj
        //--------------------------------------------------------
        int** Spert = apply_LLHop(OPj, S);

        //--------------------------------------------------------
        // Evaluate(S′)
        //--------------------------------------------------------
        objective_Function1(Spert);
        int f_pert = f_cur;

        //--------------------------------------------------------
        // (8) Maintain best-so-far
        //--------------------------------------------------------
        if (f_pert > f_best && f_cur_div >= min_div)
        {
            f_best = f_pert;
            Sbest = Spert;

            // store global best
            for (int i = 0; i < num_node; i++)
                best_solution[i] = state[i];
            for (int t = 1; t <= num_team; t++) {
                eff_best[t] = w_eff[t];
                div_best[t] = w_div[t];
            }
            convergence.push_back(f_best);
        }

        //--------------------------------------------------------
        // Acceptance decision (Algorithm lines 9–17)
        //--------------------------------------------------------
        if (f_pert > f_S && f_cur_div >= min_div)
        {
            // (10) Accept improving solution
            S = Spert;
            f_S = f_pert;
        }
        else
        {
            // (13–16) SA probability for worse moves
            double delta = f_S - f_pert;          // positive if S′ is worse
            double p = exp(-delta / T);           // SA probability
            double r = (double)rand() / RAND_MAX; // uniform [0,1]

            if (r < p)
            {
                // Accept worse solution
                S = Spert;
                f_S = f_pert;
            }
        }

        //--------------------------------------------------------
        // (18) Temperature update: T = α·T
        //--------------------------------------------------------
        T *= alpha;

        //--------------------------------------------------------
        // Log convergence (iteration, best fitness, temperature)
        //--------------------------------------------------------

        if (conv_file.is_open()) {
            conv_file << iter + 1 << "," << f_best << "," << T << "\n";
        }

        iter++;
    }

    if (conv_file.is_open())
        conv_file.close();

    //------------------------------------------------------------
    // (20) Return Sbest
    //------------------------------------------------------------
    team = Sbest;
    objective_Function1(team);

    //std::cout << "Convergence saved to: " << conv_path << std::endl;
    return team;
}

int** Hyper_heuristic::GreatDeluge(int** Sstart, int OPj, double level0, double rainSpeed)
{
    //-----------------------------------------------------
    // Initialization (Algorithm lines 1–6)
    //-----------------------------------------------------
    int** S = Sstart;
    objective_Function1(S);
    int f_S = f_cur;

    int** Sbest = S;
    int f_best = f_S;

    double level = level0;   // τ ← f0 (initial water level)

    int max_iter = 100;
    int iter = 0;

    //-----------------------------------------------------
    // Convergence Setup
    //-----------------------------------------------------
    std::vector<int> convergence;
    convergence.push_back(f_best);

    std::filesystem::path folder_path =
        "D:/Datasets/TRI_LEVEL_HH_MODELS/TRI_LEVEL_MHS_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "GreatDeluge_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest,WaterLevel\n";
    }

    //-----------------------------------------------------
    // Main Great Deluge Loop (Algorithm lines 7–19)
    //-----------------------------------------------------
    while (iter < max_iter)
    {
        //-------------------------------------------------
        // (8) Generate(S′) using operator OPj
        //-------------------------------------------------
        int** Spert = apply_LLHop(OPj, S);

        //-------------------------------------------------
        // Evaluate(S′)
        //-------------------------------------------------
        objective_Function1(Spert);
        int f_pert = f_cur;

        //-------------------------------------------------
        // (9) Maintain best-so-far
        //-------------------------------------------------
        if (f_pert > f_best && f_cur_div >= min_div) {
            f_best = f_pert;
            Sbest = Spert;

            // Store global best
            for (int i = 0; i < num_node; i++)
                best_solution[i] = state[i];
            for (int t = 1; t <= num_team; t++) {
                eff_best[t] = w_eff[t];
                div_best[t] = w_div[t];
            }
        convergence.push_back(f_best);
        }

        //-------------------------------------------------
        // (10–16) Acceptance rules
        //-------------------------------------------------
        if (f_pert > f_S && f_cur_div >= min_div)
        {
            // Accept better solution
            S = Spert;
            f_S = f_pert;
        }
        else
        {
            // Accept if solution value >= current water level
            if (f_pert >= level && f_cur_div >= min_div)
            {
                S = Spert;
                f_S = f_pert;
            }
        }

        //-------------------------------------------------
        // (18) Update water level τ = τ - B
        //-------------------------------------------------
        level -= rainSpeed;

        //-------------------------------------------------
        // Record convergence
        //-------------------------------------------------

        if (conv_file.is_open()) {
            conv_file << iter + 1 << "," << f_best << "," << level << "\n";
        }

        iter++;
    }

    if (conv_file.is_open())
        conv_file.close();

    //-----------------------------------------------------
    // (20) Return Sbest
    //-----------------------------------------------------
    team = Sbest;
    objective_Function1(team);

    //std::cout << "Convergence saved to: " << conv_path << std::endl;
    return team;
}


int** Hyper_heuristic::LateAcceptance(int** Sstart, int OPj, int Lwindow)
{
    //---------------------------------------------------------
    // Initialization (Algorithm lines 1–8)
    //---------------------------------------------------------
    std::vector<int> history(Lwindow);

    int** S = Sstart;
    objective_Function1(S);
    int f_S = f_cur;

    int** Sbest = S;
    int f_best = f_S;

    // Initialize history array with f(S)
    for (int i = 0; i < Lwindow; i++)
        history[i] = f_S;

    //---------------------------------------------------------
    // Iteration control
    //---------------------------------------------------------
    int max_iter = 100;
    int iter = 0;

    //---------------------------------------------------------
    // Convergence Setup
    //---------------------------------------------------------
    std::vector<int> convergence;
    convergence.push_back(f_best);

    std::filesystem::path folder_path =
        "D:/Datasets/TRI_LEVEL_HH_MODELS/TRI_LEVEL_MHS_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "LateAcceptance_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest\n";
    }

    //---------------------------------------------------------
    // Main LAHC Loop (Algorithm lines 9–20)
    //---------------------------------------------------------
    while (iter < max_iter)
    {
        //-----------------------------------------------------
        // (11) Generate(S′) using selected operator OPj
        //-----------------------------------------------------
        int** Spert = apply_LLHop(OPj, S);

        //-----------------------------------------------------
        // (12) Evaluate(S′)
        //-----------------------------------------------------
        objective_Function1(Spert);
        int f_pert = f_cur;

        //-----------------------------------------------------
        // (13) Maintain best-so-far
        //-----------------------------------------------------
        if (f_pert > f_best)
        {
            f_best = f_pert;
            Sbest = Spert;

            for (int i = 0; i < num_node; i++)
                best_solution[i] = state[i];
            for (int t = 1; t <= num_team; t++) {
                eff_best[t] = w_eff[t];
                div_best[t] = w_div[t];
            }
        convergence.push_back(f_best);
        }

        //-----------------------------------------------------
        // Late Acceptance rule (14–17)
        //-----------------------------------------------------
        int c = iter % Lwindow;

        if (f_pert >= history[c])
        {
            S = Spert;
            f_S = f_pert;
        }

        //-----------------------------------------------------
        // (18) f(c) = Evaluate(S)
        //-----------------------------------------------------
        history[c] = f_S;

        //-----------------------------------------------------
        // Save convergence info
        //-----------------------------------------------------

        if (conv_file.is_open()) {
            conv_file << iter + 1 << "," << f_best << "\n";
        }

        iter++;
    }

    if (conv_file.is_open())
        conv_file.close();

    //---------------------------------------------------------
    // Return best solution (Algorithm line 21)
    //---------------------------------------------------------
    team = Sbest;
    objective_Function1(team);

    //std::cout << "Convergence saved to: " << conv_path << std::endl;
    return team;
}

// ======================================================================
//  FITS: Lightweight Tabu Search on CMCEE solution (NO deep copies)
//  - Uses ONLY team[][]
//  - Neighbors generated by apply_LLHop(OPj, team)
//  - Tabu list stores only forbidden operators
// ======================================================================
int** Hyper_heuristic::TabuSearch(int** Sstart, int OPj)
{
    //----------------------------------------------------------
    // INITIALIZATION
    //----------------------------------------------------------
    int** S = Sstart;
    objective_Function1(S);
    int f_S = f_cur;
    int tabu_tenure = 10;

    int** Sbest = S;
    int f_best = f_S;

    std::vector<int> tabu_list;
    int backup[num_team + 1][num_each_t];
    int max_iter = 100;

    //----------------------------------------------------------
    // Convergence setup
    //----------------------------------------------------------
    std::vector<int> convergence;
    convergence.push_back(f_best);

    std::filesystem::path folder_path =
        "D:/Datasets/TRI_LEVEL_HH_MODELS/TRI_LEVEL_MHS_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "TabuSearch_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest\n";
    }

    //----------------------------------------------------------
    // MAIN LOOP (Algorithm lines 5–12)
    //----------------------------------------------------------
    for (int iter = 0; iter < max_iter; iter++)
    {
        //------------------------------------------------------
        // BACKUP CURRENT SOLUTION
        //------------------------------------------------------
        for (int t = 0; t <= num_team; t++)
            for (int j = 0; j < num_each_t; j++)
                backup[t][j] = team[t][j];

        //------------------------------------------------------
        // (6) GENERATE S′ USING OPj THAT IS NOT TABU
        //------------------------------------------------------
        int move = OPj;
        bool tabu_move = false;

        for (int x : tabu_list)
            if (x == move) tabu_move = true;

        if (tabu_move)
            continue; // skip this iteration

        int** Spert = apply_LLHop(move, S);

        objective_Function1(Spert);
        int f_pert = f_cur;

        //------------------------------------------------------
        // (8) UPDATE BEST SOLUTION
        //------------------------------------------------------
        if (f_pert > f_best && f_cur_div >= min_div)
        {
            f_best = f_pert;
            Sbest = Spert;

            for (int i = 0; i < num_node; i++)
                best_solution[i] = state[i];
            for (int t = 1; t <= num_team; t++) {
                eff_best[t] = w_eff[t];
                div_best[t] = w_div[t];
            }
        convergence.push_back(f_best);
        }

        //------------------------------------------------------
        // (9–10) ACCEPTANCE RULE
        //------------------------------------------------------
        if (f_pert > f_S && f_cur_div >= min_div)
        {
            S = Spert;
            f_S = f_pert;
        }
        else
        {
            // Tabu search allows non-improving moves
            S = Spert;
            f_S = f_pert;
        }

        //------------------------------------------------------
        // (7) UPDATE TABU LIST
        //------------------------------------------------------
        tabu_list.push_back(move);
        if ((int)tabu_list.size() > tabu_tenure)
            tabu_list.erase(tabu_list.begin());

        //------------------------------------------------------
        // Save convergence (iteration, f_best)
        //------------------------------------------------------

        if (conv_file.is_open()) {
            conv_file << iter + 1 << "," << f_best << "\n";
        }
    }

    if (conv_file.is_open())
        conv_file.close();

    //----------------------------------------------------------
    // (13) RETURN BEST SOLUTION
    //----------------------------------------------------------
    team = Sbest;
    objective_Function1(team);

    //std::cout << "Convergence saved to: " << conv_path << std::endl;
    return team;
}


void Hyper_heuristic::display(int** team){
    cout<<endl;
    cout<<endl;
    cout<<"team 0:";
    for (int i = 0; i < num_node-(num_team*num_each_t); ++i) {
         cout<<team[0][i]<<" ";
         }
         cout<<endl;
         cout<<endl;
	for (int t=1; t <= num_team ; t++){
         cout<<"team "<<t<<": ";
        for (int j=0; j < num_each_t; j++){
         cout<<team[t][j]<<"\t";
        }
         cout<<"eff="<<w_eff[t]<<"\t\t"<<"div="<<w_div[t];
        cout<<endl;

       }
       //check_best_solution();
       cout<<endl;

}

void Hyper_heuristic::check_best_solution()
{
    int i;
	int *aa = new int[num_team + 1];
	int *sum_eff = new int[num_team + 1];
	int *sum_div = new int[num_team + 1];
	for (int i = 0; i <= num_team; i++)
		for (int j = 0; j < num_node; j++)
			team_check[i][i] = -1;
    //cout<<"team_check[i][i]"<<team_check[i][i]<<" ";
	for (int i = 0; i <= num_team; i++)
	{
		aa[i] = 0;
		for (int j = 0; j < num_node; j++)
		{
			if (fbest_solution[j] == i)
			{  // cout<<fbest_solution[j] ;
				team_check[i][aa[i]] = j;
				//cout<<aa[i]<<" ";
				aa[i]++;
			}
		}
	}

	for (int i = 1; i <= num_team; i++)
		for (int j = 0; j < aa[i] - 1; j++)
			for (int k = j + 1; k < aa[i]; k++)
				if (team_check[i][k] == team_check[i][j])
				{
		cout << " you are wrong the team has two same members" << endl;
		getchar();
				}
	//int sum_eff[num_team + 1], sum_div[num_team + 1];

	cout << "---------------------------------" << endl;
	cout << "begin check best solution" << endl;
	for (int i = 1; i <= num_team; i++)
	{
		sum_eff[i] = 0;
		sum_div[i] = 0;
		for (int j = 0; j < aa[i]; j++)
		{
			sum_eff[i] = sum_eff[i] + eff[team_check[i][j]];
			for (int k = j + 1; k < aa[i]; k++)
				sum_div[i] = sum_div[i] + div_in[team_check[i][j]][team_check[i][k]];
		}
	}
	for (int i = 1; i <= num_team; i++)
	{
		if (sum_div[i] < min_div)
		{
			cout << " you are wrong, the solution is infeasible " << ",i=" << i << ",sum_div=" << sum_div[i] << endl;
			getchar();
		}
	}
    for (int i = 1; i <= num_team; i++)
	{

		if (sum_eff[i] != eff_fbest[i])
		{
			cout << " you are wrong with w_eff of team: " <<i<< endl;
			getchar();
		}
	}
	cout << endl;
	cout << endl;
	for (int i = 1; i <= num_team; i++)
	{
		if (sum_div[i] != div_fbest[i])
		{
			cout << " you are wrong with w_div of team: " <<i<< endl;
			getchar();
		}
	}
	int t = 0;
		cout << "team_check " << t << ":  ";
		for (int j = 0; j < aa[t]; j++){
			cout << team_check[t][j] <<"\t";

			}
      cout << endl;
      cout << endl;

    for (int i = 1; i <= num_team; i++)
	{
		cout << "team_check " << i << ":  ";
		for (int j = 0; j < aa[i]; j++){

			cout << team_check[i][j] <<"\t";

		    }
	cout << "sum_eff=" << sum_eff[i]<< " ";
    cout << "sum_div=" << sum_div[i] ;
    cout<<endl<<endl;
	}
	cout << endl;
	cout << " finish check best solution " << endl;
	cout << "---------------------------------" << endl;
	delete [] aa;
	delete [] sum_eff;
	delete [] sum_div;
}

int Hyper_heuristic::feasible_local_search(){
	double best;
	int node1, node2, team_min, team_old;
	int a1, a2;
	int d1 = 0, d2 = 0;
	int iter = 0;
	// ---------- NEW: Convergence Setup ----------
    std::vector<int> convergence;
    convergence.push_back(f_cur);

    std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "LocalSearch_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest\n";
    }
	for (int i = 0; i < num_node; i++)
		for (int j = 0; j <= num_team; j++)
			tabu_list[i][j] = 0;
	while (iter < fls_depth)
	{
		best = swap1(iter, d1, d2);
		node1 = d1;							    /* move in */
		node2 = d2;  							/* move out */
		//cout<<"d1="<<d1<<"  "<<"d2="<<d2<<"\n";
		team_min = state[d2];					/*team with min eff*/
		team_old = state[d1];
		a1 = address[node1];
		a2 = address[node2];
		//cout << "in tabu method, iter=" << iter << ",node1=" << node1 << ",node2=" << node2 << endl;
		if (team_old == 0)
		{
			tabu_list[node1][team_old] = iter + tl + randomInt(tabu_tenure);
			w_eff[team_min] = w_eff[team_min] + eff[node1] - eff[node2];
			w_div[team_min] = w_div[team_min] + delta_div[node1][team_min] - delta_div[node2][team_min] - div_in[node1][node2];
			update_delta(node1, team_min, team_old);
			state[node1] = team_min;
			team[team_min][a2] = node1;
			address[node1] = a2;

			tabu_list[node2][team_min] = iter + tl + randomInt(tabu_tenure);
			update_delta(node2, team_old, team_min);
			state[node2] = team_old;
			team[team_old][a1] = node2;
			address[node2] = a1;
		}
		else
		{
			tabu_list[node1][team_old] = iter + tl + randomInt(tabu_tenure);
			w_eff[team_min] = w_eff[team_min] + eff[node1] - eff[node2];
			w_div[team_min] = w_div[team_min] + delta_div[node1][team_min] - delta_div[node2][team_min] - div_in[node1][node2];
			w_div[team_old] = w_div[team_old] + delta_div[node2][team_old] - delta_div[node1][team_old] - div_in[node1][node2];
			update_delta(node1, team_min, team_old);
			state[node1] = team_min;
			team[team_min][a2] = node1;
			address[node1] = a2;

			tabu_list[node2][team_min] = iter + tl + randomInt(tabu_tenure);
			w_eff[team_old] = w_eff[team_old] - eff[node1] + eff[node2];
			update_delta(node2, team_old, team_min);
			state[node2] = team_old;
			team[team_old][a1] = node2;
			address[node2] = a1;
		}
		iter++;
		int idx = min_func(w_eff, num_team);
		f_cur = w_eff[idx];
		if (f_cur > fbest)
		{
			fbest = f_cur;
			end_time = clock();
			for (int m = 0; m < num_node; m++)
				best_solution[m] = state[m];
			for (int m = 1; m <= num_team; m++)
			{
				eff_best[m] = w_eff[m];
				div_best[m] = w_div[m];
			}
		}
		if (f_cur > f_best_inn) {
            f_best_inn = f_cur;
            convergence.push_back(f_best_inn);
            for (int m = 0; m < num_node; m++)
                best_inn[m] = state[m];
        }

        // ---------- NEW: Save Convergence ----------
        if (conv_file.is_open()) {
            conv_file << iter << "," << fbest << "\n";
        }
        // --------------------------------------------
    }

    int idx11 = min_func(w_div, num_team);
    f_cur_div = w_div[idx11];
    repair_solution();

    if (conv_file.is_open())
        conv_file.close();

    //std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return f_best_inn;
}

int** Hyper_heuristic::local_search(){
	double best;
	int node1, node2, team_min, team_old;
	int a1, a2;
	int d1 = 0, d2 = 0;
	int iter = 0;
	// ---------- NEW: Convergence Setup ----------
    std::vector<int> convergence;
    convergence.push_back(f_cur);

    std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "LocalSearch_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest\n";
    }
	for (int i = 0; i < num_node; i++)
		for (int j = 0; j <= num_team; j++)
			tabu_list[i][j] = 0;
	while (iter < fls_depth)
	{
		best = swap1(iter, d1, d2);
		node1 = d1;							    /* move in */
		node2 = d2;  							/* move out */
		//cout<<"d1="<<d1<<"  "<<"d2="<<d2<<"\n";
		team_min = state[d2];					/*team with min eff*/
		team_old = state[d1];
		a1 = address[node1];
		a2 = address[node2];
		//cout << "in tabu method, iter=" << iter << ",node1=" << node1 << ",node2=" << node2 << endl;
		if (team_old == 0)
		{
			tabu_list[node1][team_old] = iter + tl + randomInt(tabu_tenure);
			w_eff[team_min] = w_eff[team_min] + eff[node1] - eff[node2];
			w_div[team_min] = w_div[team_min] + delta_div[node1][team_min] - delta_div[node2][team_min] - div_in[node1][node2];
			update_delta(node1, team_min, team_old);
			state[node1] = team_min;
			team[team_min][a2] = node1;
			address[node1] = a2;

			tabu_list[node2][team_min] = iter + tl + randomInt(tabu_tenure);
			update_delta(node2, team_old, team_min);
			state[node2] = team_old;
			team[team_old][a1] = node2;
			address[node2] = a1;
		}
		else
		{
			tabu_list[node1][team_old] = iter + tl + randomInt(tabu_tenure);
			w_eff[team_min] = w_eff[team_min] + eff[node1] - eff[node2];
			w_div[team_min] = w_div[team_min] + delta_div[node1][team_min] - delta_div[node2][team_min] - div_in[node1][node2];
			w_div[team_old] = w_div[team_old] + delta_div[node2][team_old] - delta_div[node1][team_old] - div_in[node1][node2];
			update_delta(node1, team_min, team_old);
			state[node1] = team_min;
			team[team_min][a2] = node1;
			address[node1] = a2;

			tabu_list[node2][team_min] = iter + tl + randomInt(tabu_tenure);
			w_eff[team_old] = w_eff[team_old] - eff[node1] + eff[node2];
			update_delta(node2, team_old, team_min);
			state[node2] = team_old;
			team[team_old][a1] = node2;
			address[node2] = a1;
		}
		iter++;
		int idx = min_func(w_eff, num_team);
		f_cur = w_eff[idx];
		if (f_cur > fbest)
		{
			fbest = f_cur;
			end_time = clock();
			for (int m = 0; m < num_node; m++)
				best_solution[m] = state[m];
			for (int m = 1; m <= num_team; m++)
			{
				eff_best[m] = w_eff[m];
				div_best[m] = w_div[m];
			}
		}
		if (f_cur > f_best_inn) {
            f_best_inn = f_cur;
            convergence.push_back(f_best_inn);
            for (int m = 0; m < num_node; m++)
                best_inn[m] = state[m];
        }

        // ---------- NEW: Save Convergence ----------
        if (conv_file.is_open()) {
            conv_file << iter << "," << fbest << "\n";
        }
        // --------------------------------------------
    }

    int idx11 = min_func(w_div, num_team);
    f_cur_div = w_div[idx11];
    repair_solution();

    if (conv_file.is_open())
        conv_file.close();

    //std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return team;
}


int** Hyper_heuristic::infeasible_local_search()
{
	double best;
	int node1, node2, team_min, team_old;
	int a1, a2;
	int d1 = 0, d2 = 0, d3=0;
	int iter = 0;
	// ---------- NEW: Convergence Setup ----------
    std::vector<int> convergence;
    convergence.push_back(f_cur);

    std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "infeasible_local_search_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest\n";
    }

	for (int i = 0; i < num_node; i++)
		for (int j = 0; j <= num_team; j++)
			tabu_list[i][j] = 0;
	for (int i = 1; i <= num_team; i++)
    degree_inf[i] = 0;
	deg_cur = 0;
	p_count = 0;
	p_factor = 1;
	lamba = 5;
	u1 = 4;
	u2 = 1;
	tow = 2;
	while (iter < ils_depth)
	{
		best = swap_ils(iter, d1, d2);
		node1 = d1;							    /* move in */
		node2 = d2;								/* move out */
		team_min = state[d2];					/*team with min eff*/
		team_old = state[d1];
		a1 = address[node1];
		a2 = address[node2];
		//cout << "in tabu method, iter=" << iter << ",node1=" << node1 << ",node2=" << node2 << endl;
		if (team_old == 0)
		{
			tabu_list[node1][team_old] = iter + tl + randomInt(tabu_tenure);
			w_eff[team_min] = w_eff[team_min] + eff[node1] - eff[node2];
			w_div[team_min] = w_div[team_min] + delta_div[node1][team_min] - delta_div[node2][team_min] - div_in[node1][node2];
			double deg_old = degree_inf[team_min];
			if (w_div[team_min] >= min_div)
				degree_inf[team_min] = 0;
			else
            degree_inf[team_min] = min_div - w_div[team_min];
			deg_cur += degree_inf[team_min] - deg_old;
			update_delta(node1, team_min, team_old);
			state[node1] = team_min;
			team[team_min][a2] = node1;
			address[node1] = a2;

			tabu_list[node2][team_min] = iter + tl + randomInt(tabu_tenure);
			update_delta(node2, team_old, team_min);
			state[node2] = team_old;
			team[team_old][a1] = node2;
			address[node2] = a1;
		}
		else
		{
			tabu_list[node1][team_old] = iter + tl + randomInt(tabu_tenure);
			w_eff[team_min] = w_eff[team_min] + eff[node1] - eff[node2];
			w_div[team_min] = w_div[team_min] + delta_div[node1][team_min] - delta_div[node2][team_min] - div_in[node1][node2];
			w_div[team_old] = w_div[team_old] + delta_div[node2][team_old] - delta_div[node1][team_old] - div_in[node1][node2];
			double deg_old1 = degree_inf[team_min];
			double deg_old2 = degree_inf[team_old];
			if (w_div[team_min] >= min_div)
				degree_inf[team_min] = 0;
			else
				degree_inf[team_min] = min_div - w_div[team_min];
			deg_cur += degree_inf[team_min] - deg_old1;

			if (w_div[team_old] >= min_div)
				degree_inf[team_old] = 0;
			else
				degree_inf[team_old] = min_div - w_div[team_old];
			deg_cur += degree_inf[team_old] - deg_old2;

			update_delta(node1, team_min, team_old);
			state[node1] = team_min;
			team[team_min][a2] = node1;
			address[node1] = a2;

			tabu_list[node2][team_min] = iter + tl + randomInt(tabu_tenure);
			w_eff[team_old] = w_eff[team_old] - eff[node1] + eff[node2];
			update_delta(node2, team_old, team_min);
			state[node2] = team_old;
			team[team_old][a1] = node2;
			address[node2] = a1;
		}
		iter++;
		int idx = min_func(w_eff, num_team);
		f_cur = w_eff[idx];
		if (f_cur > fbest && deg_cur <= 1.0e-5)
		{
			fbest = f_cur;
			end_time = clock();
			for (int m = 0; m < num_node; m++)
				best_solution[m] = state[m];
			for (int m = 1; m <= num_team; m++)
			{
				eff_best[m] = w_eff[m];
				div_best[m] = w_div[m];
			}
		}
		if (f_cur > f_best_inn && deg_cur <= 1.0e-5)
		{
			f_best_inn = f_cur;
			for (int m = 0; m < num_node; m++)
				best_inn[m] = state[m];
		}

		if (deg_cur > 0)		//infeasible solution
			p_count++;
		if (iter%lamba == 0)
		{
			if (p_count > u1)
				//	p_factor *= tow;
				p_factor += tow;
			else if (p_count < u2)
				//p_factor /= tow;
			{
				p_factor -= tow;
				if (p_factor < 0)
					p_factor = 1;
			}
			p_count = 0;
		}
		// ---------- NEW: Save Convergence ----------
        convergence.push_back(fbest);
        if (conv_file.is_open()) {
            conv_file << iter << "," << fbest << "\n";
        }
        // --------------------------------------------
    }

    repair_solution();

    if (conv_file.is_open())
        conv_file.close();

    //std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return team;

}

// Tabu Search based feasible
int** Hyper_heuristic::fits()
{
	f_best_inn = 0;
	feasible_local_search();
	infeasible_local_search();
	repair_solution();
	end_time=clock();
	return team;
}

//initialize the population
void Hyper_heuristic::initial_population()
{
   for (int i = 0; i < Pop_Num; i++)
	{
		for (int m = 0; m < num_node; m++)
			pop[i].p[m] = state[m];
		    pop[i].cost = f_cur;
	}
}

//backbone crossover: based on maximum match grouping
void Hyper_heuristic::cross_over2()
{
	int **arr1 = new int*[num_team + 1];
	int **arr2 = new int*[num_team + 1];
	for (int i = 0; i <= num_team; i++)
	{
		arr1[i] = new int[num_node];
		arr2[i] = new int[num_node];
	}
	int *len1 = new int[num_team + 1];
	int *len2 = new int[num_team + 1];
	int *match = new int[num_team + 1];
	int *flagC1 = new int[num_team + 1];
	int *flagC2 = new int[num_team + 1];
	int *flagV = new int[num_node];
	int *unassV = new int[num_node];
	int *addressUnaV = new int[num_node];
	int parent1, parent2;
	int ver1, ver2, sharedV, sharedMax;
	int can1, can2, index, unassLen;
	for (int i = 0; i < num_node; i++)
	{
		state[i] = 0;			//crew 0 preserves the practitioners that are not allocated
		flagV[i] = 0;
	}
	parent1 = rand() % Pop_Num;
	parent2 = rand() % Pop_Num;
	while (parent1 == parent2)
		parent2 = rand() % Pop_Num;
	for (int i = 0; i <= num_team; i++)
	{
		len1[i] = 0;
		len2[i] = 0;
		flagC1[i] = 0;
		flagC2[i] = 0;
	}
	for (int i = 0; i < num_node; i++)
	  {
		int clu1 = pop[parent1].p[i];
		//cout<<"clu1="<<clu1<<endl;
		arr1[clu1][len1[clu1]++] = i;
		//cout<<"arr1="<<arr1[clu1][len1[clu1]++];
		int clu2 = pop[parent2].p[i];
		arr2[clu2][len2[clu2]++] = i;
	  }

	//maximum match
	sharedMax = 0;
	for (int it = 1; it <= num_team; it++)
	{
		sharedMax = 0;
		for (int i = 1; i <= num_team; i++)
		{
			if (flagC1[i] == 0)
			{
				for (int m = 1; m <= num_team; m++)
				{
					if (flagC2[m] == 0)
					{
						sharedV = 0;
						//index = 0;
						for (int j = 0; j < len1[i]; j++)
						{
							ver1 = arr1[i][j];
							if (flagV[ver1] == 0)
							{
								for (int n = 0; n < len2[m]; n++)
								{
									ver2 = arr2[m][n];
									if (flagV[ver2] == 0)
									{
										if (ver1 == ver2)
											sharedV++;
										if (ver2 > ver1)
										{
											//index = n;
											break;
										}
									}
								}

							}
						}
						if (sharedV > sharedMax)
						{
							sharedMax = sharedV;
							can1 = i;
							can2 = m;
						}
					}
				}
			}
		}
		match[can1] = can2;
		flagC1[can1] = 1;
		flagC2[can2] = 1;
		index = 0;
		int len000 = 0;
		for (int x1 = 0; x1 < len1[can1]; x1++)
		{
			int ver = arr1[can1][x1];
			for (int x2 = index; x2 < len2[can2]; x2++)
			{
				int ver2 = arr2[can2][x2];
				if (ver == ver2)					//marked the practitioner is being allocated to the offspring solution
				{
					flagV[ver] = 1;
					state[ver] = it;
					len000++;
					index = x2;
					break;
				}
				if (ver2 > ver)
				{
					break;
					index = x2;
				}
			}
		}
		//---------------------------------------------------------------------//
		int coin = 0;
		int len111 = 0;
		int len222 = 0;
		for (int x = 0; x < len1[can1]; x++)
		{
			int ver = arr1[can1][x];
			if (flagV[ver] == 0)
				len111++;
		}
		for (int x = 0; x < len2[can2]; x++)
		{
			int ver = arr2[can2][x];
			if (flagV[ver] == 0)
				len222++;
		}
		double cont_best = 0;
		double cont;
		int select_v;
		while (len000 < num_each_t && (len111>0 || len222 > 0))
		{
			cont_best = 0;
			if (coin % 2 == 0)		//selected from parent 1
			{
				if (len111 > 0)
				{
					for (int x = 0; x < len1[can1]; x++)
					{
						int ver = arr1[can1][x];
						cont = 0;
						if (flagV[ver] == 0)
						{
							for (int j = 0; j < num_node; j++)
							{
								if (state[j] == it)
									cont += div_in[ver][j];
							}
							if (cont*eff[ver] > cont_best)
							{
								cont_best = cont*eff[ver];
								select_v = ver;
							}
						}
					}
					state[select_v] = it;
					flagV[select_v] = 1;
					len111--;
					len000++;
				}
			}
			else				//selected from parent 2
			{
				if (len222 > 0)
				{
					for (int x = 0; x < len2[can2]; x++)
					{
						int ver = arr2[can2][x];
						cont = 0;
						if (flagV[ver] == 0)
						{
							for (int j = 0; j < num_node; j++)
							{
								if (state[j] == it)
									cont += div_in[ver][j];
							}
							if (cont*eff[ver] > cont_best)
							{
								cont_best = cont*eff[ver];
								select_v = ver;
							}
						}
					}
					state[select_v] = it;
					flagV[select_v] = 1;
					len222--;
					len000++;
				}
			}
			coin++;
		}
	}
	for (int i = 1; i <= num_team; i++)
		len1[i] = 0;
	unassLen = 0;
	for (int i = 0; i < num_node; i++)
	{
		if (state[i] != 0)
		{
			int clu = state[i];
			arr1[clu][len1[clu]++] = i;
		}
		else
		{
			unassV[unassLen] = i;
			addressUnaV[i] = unassLen;
			unassLen++;
		}
	}

	//greedy allocate the unallocted practitioners£¬and repair infeasible solution
	int unass_len2 = unassLen;
	while (unass_len2 > (num_node - num_team * num_each_t))
	{
		//int ver = unassV[rand() % unassLen];
		//int k = rand() % num_team + 1;
		double cont = 0;
		double cont_best = 0;
		int ver, crew;
		for (int i = 0; i < num_node; i++)
		{
			cont = 0;
			if (state[i] == 0)
			{
				for (int k = 1; k <= num_team; k++)
				{
					for (int j = 0; j < num_node; j++)
					{
						if (state[j] == k)
							cont += div_in[i][j];
					}
					if (eff[i] * cont > cont_best && len1[k] < num_each_t)
					{
						cont_best = eff[i] * cont;
						ver = i;
						crew = k;
					}
				}
			}
		}
		if(state[ver] == 0 && len1[crew] < num_each_t)
		{
			state[ver] = crew;
			len1[crew]++;
			unass_len2--;
		}
		//cout << "unass_len2=" << unass_len2 << endl;
	}
	/*repair infeasible solution*/
	for (int j = 0; j <= num_team; j++)
	{
		w_eff[j] = 0;
		w_div[j] = 0;
	}
	for (int j = 0; j < num_node; j++)
		for (int k = 0; k <= num_team; k++)
			if (state[j] == k)
				w_eff[k] += eff[j];
	for (int j = 0; j < num_node; j++)
		for (int k = 0; k < num_node; k++)
			if (state[j] == state[k])
				w_div[state[j]] += div_in[j][k];
	for (int j = 0; j <= num_team; j++)
		w_div[j] = w_div[j] / 2;

	for (int i = 0; i < num_node; i++)
		for (int j = 0; j <= num_team; j++)
			delta_div[i][j] = 0;
	for (int i = 0; i < num_node; i++)
		for (int j = 0; j <= num_team; j++)
			for (int k = 0; k < num_node; k++)
				if (state[k] == j)
					delta_div[i][j] += div_in[i][k];
	for (int i = 0; i < num_node; i++)
		delta_div[i][0] = 0;

	for (int i = 0; i <= num_team; i++)
		for (int j = 0; j < num_node; j++)
			team[i][i] = -1;
	for (int i = 0; i <= num_team; i++)
	{
		len1[i] = 0;
		for (int j = 0; j < num_node; j++)
		{
			if (state[j] == i)
			{
				team[i][len1[i]] = j;
				address[j] = len1[i];
				len1[i]++;
			}
		}
	}
	repair_solution();
	for(int i = 0; i < num_team + 1; i++)
	{
		delete[] arr1[i];
	}
	delete [] arr1;
	for(int i = 0; i < num_team + 1; i++)
	{
		delete[] arr2[i];
	}
	delete [] arr2;
	delete [] len1;
	delete [] len2;
	delete [] match;
	delete [] flagC1;
	delete [] flagC2;
	delete [] flagV;
	delete [] unassV;
	delete [] addressUnaV;
}

void Hyper_heuristic::update_populaion(int *child, int cost)
{
	int min_cost = MAXVALUE;
	int select = -1;
	for (int i = 0; i < Pop_Num; i++)
	{
		if (pop[i].cost < min_cost)
		{
			min_cost = pop[i].cost;
			//cout<<"min_cost="<<min_cost<<"  " <<pop[i].cost<<endl;
			select = i;
		}
	}
	if (cost > min_cost)
	{
		for (int i = 0; i < num_node; i++)
			pop[select].p[i] = child[i];
		pop[select].cost = cost;
	}
}

int** Hyper_heuristic::memetic()
{
	int gen = 0;
	fbest = 0;

	pop = new population[Pop_Num];  //population number 10
	for (int i = 0; i < Pop_Num; i++){
		pop[i].p = new int[num_node];

        }
	    initial_population();
	//while (1.0*(clock()-start_time)/CLOCKS_PER_SEC<time_limit)
	while (gen < generations)       //generation 50
	{
		cross_over2();
		feasible_local_search();
		fits();
		update_populaion(best_inn, f_best_inn);
		gen++;
	}
	return team;
}

void Hyper_heuristic::Parameters()
{
    ffbest = 0;
	fbest = 0;
	tl = 0;
	//beta = 0.4
	tabu_tenure = 10;
	generations = 50;
	fls_depth = 3000;
	ils_depth = 1000;
	int    f1, f2, f3, f4, f5, f6, f7, f8, f9;
	double f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23;
	double f24, f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39;
}

void free_memory()
{
	delete [] eff;
	delete [] state;
	delete [] best_solution;
	delete [] fbest_solution;
	delete [] best_inn;
	delete [] address;
	delete [] w_div;
	delete [] w_eff;
	delete [] num_t_cur;
	delete [] div_best;
	delete [] eff_best;
	delete [] div_fbest;
	delete [] eff_fbest;
	delete [] degree_inf;
	for(int i = 0; i < num_node; i++)
	{
		delete [] div_in[i];
	}
	delete [] div_in;
	for(int i = 0; i < num_node; i++)
	{
		delete [] delta_div[i];
	}
	delete [] delta_div;
	for(int i = 0; i < num_node; i++)
	{
		delete [] tabu_list[i];
	}
	delete [] tabu_list;
	for(int i = 0; i < num_team + 1; i++)
	{
		
        delete [] team[i];

		delete [] team_check[i];
	}
	delete [] team;
	delete [] team_check;
	for(int i = 0; i < Pop_Num; i++)
	{
		delete [] pop[i].p;
	}
	delete [] pop;

}

// Implementation of the Selection Hyper-Heuristic Strategies Framework
void Hyper_heuristic::objective_Function(int **team) {

    // Always recompute team 0
    w_eff[0] = 0;
    w_div[0] = 0;

    // Compute efficiency and diversity for teams 1 .. num_team
    for (int t = 1; t <= num_team; t++) {
        w_eff[t] = 0.0;
        w_div[t] = 0.0;

        for (int j = 0; j < num_each_t; j++) {
            int node_j = team[t][j];
            w_eff[t] += eff[node_j];

            for (int k = j + 1; k < num_each_t; k++) {
                int node_k = team[t][k];
                w_div[t] += div_in[node_j][node_k];
            }
        }
    }

    // Determine current objective values
    int idx_eff = min_func(w_eff, num_team);
    int idx_div = min_func(w_div, num_team);

    f_cur     = w_eff[idx_eff];
    f_cur_div = w_div[idx_div];

   /* // Fix membership mapping (ensures no corruption)
    for (int t = 0; t <= num_team; t++) {
        for (int j = 0; j < num_each_t; j++) {
            int node = team[t][j];
            state[node] = t;
            address[node] = j;
        }
    }*/

    /*// Save best solution
    for (int m = 0; m < num_node; m++) {
        best_solution[m] = state[m];
        best_inn[m] = state[m];
    }

    for (int m = 1; m <= num_team; m++) {
        eff_best[m] = w_eff[m];
        div_best[m] = w_div[m];
    }*/
}

void Hyper_heuristic::objective_Function1(int **team){

    // Determine the best solution and update f_cur
    int idx = min_func(w_eff, num_team);
    f_cur = w_eff[idx];
    int idx1 = sec_func(w_eff, idx, num_team);
    //to get maximum effeciency in solution
    int fidx = max_func(w_eff, num_team);
    int maxeff = w_eff[fidx];
    // to compute fairness in solution
    int fair;
    fair= maxeff - f_cur;

    //Determine the best solution and update f_cur_div
    int idx3 = min_func(w_div, num_team);
    f_cur_div = w_div[idx3];
    int idx4 = sec_func(w_div, idx3, num_team);
    //Update best_solution and best_inn
    for (int m = 0; m < num_node; m++){
    best_solution[m] = state[m];
    best_inn[m] = state[m];
    }
    for (int m=1;  m<= num_team  ; m++ ){
        eff_best[m] = w_eff[m];
        div_best[m] = w_div[m];
    }
    //Update f_cur after repair or infeasible local search
    }



int** Hyper_heuristic::great_deluge_algorithm() {
    int max_iter = 100;
    double decay_rate = 0.98;               // water level multiplier (< 1.0)
    int iter = 0;

    int f_s = f_cur;                         // current solution value
    fbest = f_cur;                           // global best
    double level = static_cast<double>(f_s); // water level (starts at current)

    // (optional) convergence log
    std::vector<int> convergence;
    convergence.push_back(f_s);

    // backups for rejection
    std::vector<int>    state_bak(num_node);
    std::vector<int>    address_bak(num_node);
    std::vector<double> w_eff_bak(num_team + 1), w_div_bak(num_team + 1);
    std::vector<std::vector<int>> team_bak(num_team + 1, std::vector<int>(num_node));

    // ---------- NEW: Convergence File ----------
    std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "GreatDeluge_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest,WaterLevel\n";
    }
    // --------------------------------------------

    while (iter < max_iter) {
        // ---- backup current state ----
        for (int i = 0; i < num_node; i++) {
            state_bak[i]   = state[i];
            address_bak[i] = address[i];
        }
        for (int t = 0; t <= num_team; t++) {
            for (int p = 0; p < num_node; p++)
                team_bak[t][p] = team[t][p];
        }
        for (int k = 1; k <= num_team; k++) {
            w_eff_bak[k] = w_eff[k];
            w_div_bak[k] = w_div[k];
        }

        // ---- propose a move (produces a new state/fitness) ----
        int f_new = feasible_local_search();

        // ---- acceptance test (maximize): accept if f_new >= level ----
        if (static_cast<double>(f_new) >= level) {
            f_s = f_new;
            if (f_s > fbest) {
                fbest = f_s;
                for (int m = 0; m < num_node; m++) {
                    best_solution[m] = state[m];
                    best_inn[m]      = state[m];
                }
                for (int i = 1; i <= num_team; i++) {
                    eff_best[i] = w_eff[i];
                    div_best[i] = w_div[i];
                }
            }
        } else {
            // ---- reject: restore previous state ----
            for (int i = 0; i < num_node; i++) {
                state[i]   = state_bak[i];
                address[i] = address_bak[i];
            }
            for (int t = 0; t <= num_team; t++) {
                for (int p = 0; p < num_node; p++)
                    team[t][p] = team_bak[t][p];
            }
            for (int k = 1; k <= num_team; k++) {
                w_eff[k] = w_eff_bak[k];
                w_div[k] = w_div_bak[k];
            }
            int idx = min_func(w_eff, num_team);        // recompute current value
            f_cur = static_cast<int>(w_eff[idx]);
        }

        // ---- lower the water level ----
        level *= decay_rate;                             // decreases over time
        convergence.push_back(fbest);

        // ---- log convergence ----
        if (conv_file.is_open()) {
            conv_file << iter << "," << fbest << "," << level << "\n";
        }

        iter++;
    }

    if (conv_file.is_open())
        conv_file.close();

    //std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return team;
}

int Hyper_heuristic::guided_local_search() {
    std::vector<std::vector<double>> penalties(num_node, std::vector<double>(num_node, 0));
    double lambda_penalty = 0.1;
    int max_iter = 100;
    std::vector<int> convergence = {f_cur};
    auto start_time = std::chrono::high_resolution_clock::now();

    // ---------- NEW: Convergence File ----------
    std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "GuidedLocalSearch_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest\n";
    }
    // --------------------------------------------

    auto augmented_cost = [&](int i, int j) {
        return -eff[i] + lambda_penalty * penalties[i][j];
    };

    for (int iter = 0; iter < max_iter; iter++) {
        double f_base = feasible_local_search();

        // ---- Update penalties ----
        for (int i = 0; i < num_node; i++) {
            for (int j = i + 1; j < num_node; j++) {
                if (augmented_cost(i, j) > 0) {
                    penalties[i][j] += 1;
                }
            }
        }

        // ---- Update best solution ----
        if (f_base > fbest) {
            fbest = f_base;
            for (int i = 1; i <= num_team; i++) {
                eff_best[i] = w_eff[i];
                div_best[i] = w_div[i];
            }
            for (int m = 0; m < num_node; m++) {
                best_solution[m] = state[m];
                best_inn[m] = state[m];
            }
        }

        // ---- Record convergence ----
        convergence.push_back(fbest);
        if (conv_file.is_open()) {
            conv_file << iter << "," << fbest << "\n";
        }
    }

    if (conv_file.is_open())
        conv_file.close();

    //std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return fbest;
}

int** Hyper_heuristic::late_acceptance_hill_climbing() {
    int L = 50;
    int max_iter = 100;
    std::vector<int> history(L, f_cur);
    int f_s = f_cur;
    std::vector<int> convergence = { f_s };
    int iter = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // ---------- NEW: Convergence File ----------
    std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "LateAcceptance_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest\n";
    }
    // --------------------------------------------

    while (iter < max_iter) {
        int f_new = feasible_local_search();

        // ---- Acceptance condition ----
        if (f_new >= history[iter % L]) {
            f_s = f_new;

            // ---- Update best solution ----
            if (f_s > fbest) {
                fbest = f_s;
                for (int i = 1; i <= num_team; i++) {
                    eff_best[i] = w_eff[i];
                    div_best[i] = w_div[i];
                }
                for (int m = 0; m < num_node; m++) {
                    best_solution[m] = state[m];
                    best_inn[m] = state[m];
                }
            }
        }

        history[iter % L] = f_s;
        convergence.push_back(fbest);

        // ---- Log convergence ----
        if (conv_file.is_open()) {
            conv_file << iter << "," << fbest << "\n";
        }

        iter++;
    }

    if (conv_file.is_open())
        conv_file.close();

    //std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return team;
}

// ---------- RNG & index helpers (file-scope) ----------
inline std::mt19937& hh_rng() {
    // ✅ OK: thread_local applies to the static variable, not the function
    static thread_local std::mt19937 gen(
        static_cast<unsigned>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        )
    );
    return gen;
}

// size of the unallocated pool (team 0)
inline int pool_size_unallocated() {
    // team0 size = total nodes - (allocated teams * members per team)
    return std::max(0, num_node - (num_team * num_each_t));
}

// a random allocated team in [1 .. num_team]
inline int rand_alloc_team() {
    std::uniform_int_distribution<int> dist(1, num_team);
    return dist(hh_rng());
}

// two DISTINCT allocated teams in [1 .. num_team]
inline std::pair<int,int> rand_two_distinct_alloc_teams() {
    std::uniform_int_distribution<int> dist(1, num_team);
    int a = dist(hh_rng());
    int b = dist(hh_rng());
    while (b == a) b = dist(hh_rng());
    return {a, b};
}

// a valid member index in [0 .. num_each_t-1]
inline int rand_member_idx() {
    std::uniform_int_distribution<int> dist(0, std::max(1, num_each_t) - 1);
    return dist(hh_rng());
}

// two DISTINCT member indices in [0 .. num_each_t-1]
inline std::pair<int,int> rand_two_distinct_member_idx() {
    if (num_each_t <= 1) return {0, 0};
    std::uniform_int_distribution<int> dist(0, num_each_t - 1);
    int i = dist(hh_rng());
    int j = dist(hh_rng());
    while (j == i) j = dist(hh_rng());
    return {i, j};
}

// a pool index in [0 .. poolSize-1]
inline int rand_pool_idx(int poolSize) {
    std::uniform_int_distribution<int> dist(0, std::max(1, poolSize) - 1);
    return dist(hh_rng());
}

// two DISTINCT pool indices in [0 .. poolSize-1]
inline std::pair<int,int> rand_two_distinct_pool_idx(int poolSize) {
    if (poolSize <= 1) return {0, 0};
    std::uniform_int_distribution<int> dist(0, poolSize - 1);
    int a = dist(hh_rng());
    int b = dist(hh_rng());
    while (b == a) b = dist(hh_rng());
    return {a, b};
}

std::pair<int,int> Hyper_heuristic::objective_after_LLH(int **team) {

    // recompute w_eff and w_div for all teams
    for (int t = 1; t <= num_team; t++) {
        w_eff[t] = 0;
        w_div[t] = 0;

        for (int j = 0; j < num_each_t; j++) {
            int node = team[t][j];
            w_eff[t] += eff[node];
            w_div[t] += delta_div[node][t];
        }
    }

    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);

    f_cur = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];

    return {f_cur, f_cur_div};
}

int** Hyper_heuristic::LLH1(int** team) {
    // LLH1: Swap one unallocated member (team 0) with one from a random allocated team.
    // Only performs permutation and incremental update of eff/div (no objective or best update).

    const int poolSize = pool_size_unallocated();
    //if (num_team < 1 || num_each_t <= 0 || poolSize <= 0) return team;

    const int team0 = 0;
    const int team2 = rand_alloc_team();
    const int memIdx = rand_member_idx();
    const int poolIdx = rand_pool_idx(poolSize);

    // Identify members
    int node_in  = team[team0][poolIdx];  // from pool
    int node_out = team[team2][memIdx];   // from allocated team

    // --- Swap operation ---
    std::swap(team[team0][poolIdx], team[team2][memIdx]);

    // --- Incremental Efficiency Update ---
    w_eff[team2] = w_eff[team2] + eff[node_in] - eff[node_out];

    // --- Incremental Diversity Update ---
    w_div[team2] = w_div[team2]+ delta_div[node_in][team2]- delta_div[node_out][team2]- div_in[node_in][node_out];

    // Update individual delta-diversity values for both nodes
    update_delta(node_in, team2, team0);
    update_delta(node_out, team0, team2);

    // Update membership bookkeeping
    int a_in  = address[node_in];
    int a_out = address[node_out];
    state[node_in]   = team2;
    state[node_out]  = team0;
    address[node_in] = memIdx;
    address[node_out] = poolIdx;

    // Update current efficiency and diversity for feasibility tracking
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);

    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH2(int** team) {
    // LLH2: Pick two DISTINCT allocated teams and swap one member from each with two unallocated members.
    // Performs incremental efficiency/diversity updates, no full objective recomputation.

    const int poolSize = pool_size_unallocated();
    //if (num_team < 2 || num_each_t <= 0 || poolSize <= 1) return team;

    const int team0 = 0;
    auto [t2, t3] = rand_two_distinct_alloc_teams();
    const int idx2 = rand_member_idx();
    const int idx3 = rand_member_idx();
    auto [p1, p2]  = rand_two_distinct_pool_idx(poolSize);

    // Identify members
    int node_in1  = team[team0][p1];   // unallocated → will join team t2
    int node_in2  = team[team0][p2];   // unallocated → will join team t3
    int node_out1 = team[t2][idx2];    // leaving team t2 → goes to pool
    int node_out2 = team[t3][idx3];    // leaving team t3 → goes to pool

    // --- Swap operations ---
    std::swap(team[team0][p1], team[t2][idx2]);
    std::swap(team[team0][p2], team[t3][idx3]);

    // --- Incremental Efficiency Updates ---
    w_eff[t2] = w_eff[t2] + eff[node_in1] - eff[node_out1];
    w_eff[t3] = w_eff[t3] + eff[node_in2] - eff[node_out2];

    // --- Incremental Diversity Updates ---
    w_div[t2] = w_div[t2]
        + delta_div[node_in1][t2]
        - delta_div[node_out1][t2]
        - div_in[node_in1][node_out1];

    w_div[t3] = w_div[t3]
        + delta_div[node_in2][t3]
        - delta_div[node_out2][t3]
        - div_in[node_in2][node_out2];

    // --- Update delta diversity contributions ---
    update_delta(node_in1, t2, team0);
    update_delta(node_out1, team0, t2);
    update_delta(node_in2, t3, team0);
    update_delta(node_out2, team0, t3);

    // --- Update membership bookkeeping ---
    int a_in1  = address[node_in1];
    int a_in2  = address[node_in2];
    int a_out1 = address[node_out1];
    int a_out2 = address[node_out2];

    state[node_in1] = t2;
    state[node_in2] = t3;
    state[node_out1] = team0;
    state[node_out2] = team0;

    address[node_in1] = idx2;
    address[node_in2] = idx3;
    address[node_out1] = p1;
    address[node_out2] = p2;

    // --- Update current eff/div for feasibility check ---
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);

    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH3(int** team) {
    // LLH3: Pick one allocated team; swap TWO of its members with TWO unallocated members.
    // Performs incremental efficiency/diversity updates only (no full objective recomputation).

    const int poolSize = pool_size_unallocated();
    //if (num_team < 1 || num_each_t < 2 || poolSize < 2) return team;

    const int team0 = 0;
    const int t2 = rand_alloc_team();
    auto [iA, iB] = rand_two_distinct_member_idx();   // two members from team t2
    auto [pA, pB] = rand_two_distinct_pool_idx(poolSize); // two unallocated members

    // Identify involved members
    int node_in1  = team[team0][pA];  // from pool → will join t2
    int node_in2  = team[team0][pB];  // from pool → will join t2
    int node_out1 = team[t2][iA];     // leaving t2 → goes to pool
    int node_out2 = team[t2][iB];     // leaving t2 → goes to pool

    // --- Swap operations ---
    std::swap(team[team0][pA], team[t2][iA]);
    std::swap(team[team0][pB], team[t2][iB]);

    // --- Incremental Efficiency Update ---
    w_eff[t2] = w_eff[t2]
        + eff[node_in1] + eff[node_in2]
        - eff[node_out1] - eff[node_out2];

    // --- Incremental Diversity Update ---
    w_div[t2] = w_div[t2]
        + delta_div[node_in1][t2] + delta_div[node_in2][t2]
        - delta_div[node_out1][t2] - delta_div[node_out2][t2]
        - (div_in[node_in1][node_out1] + div_in[node_in2][node_out2]);

    // --- Update delta diversity contributions ---
    update_delta(node_in1, t2, team0);
    update_delta(node_in2, t2, team0);
    update_delta(node_out1, team0, t2);
    update_delta(node_out2, team0, t2);

    // --- Update membership bookkeeping ---
    int a_in1  = address[node_in1];
    int a_in2  = address[node_in2];
    int a_out1 = address[node_out1];
    int a_out2 = address[node_out2];

    state[node_in1] = t2;
    state[node_in2] = t2;
    state[node_out1] = team0;
    state[node_out2] = team0;

    address[node_in1] = iA;
    address[node_in2] = iB;
    address[node_out1] = pA;
    address[node_out2] = pB;

    // --- Update current efficiency/diversity for feasibility checking ---
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);

    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH4(int** team) {
    // LLH4: Pick two DISTINCT allocated teams; swap two consecutive members pairwise.
    // Performs incremental efficiency/diversity updates without recomputing objective function.

    //if (num_team < 2 || num_each_t < 2) return team;

    auto [t1, t2] = rand_two_distinct_alloc_teams();
    std::uniform_int_distribution<int> distIdx(0, num_each_t - 2);
    int i = distIdx(hh_rng());  // ensures i and i+1 are valid indices

    // Identify involved nodes
    int n1a = team[t1][i];
    int n1b = team[t1][i + 1];
    int n2a = team[t2][i];
    int n2b = team[t2][i + 1];

    // --- Perform swaps ---
    int temp1 = team[t1][i];
    team[t1][i] = team[t2][i];
    team[t2][i] = temp1;
    int temp2 = team[t1][i + 1];
    team[t1][i + 1] = team[t2][i + 1];
    team[t2][i + 1] = temp2;
    //std::swap(team[t1][i],     team[t2][i]);
    //std::swap(team[t1][i + 1], team[t2][i + 1]);

    // --- Incremental Efficiency Updates ---
    double delta_eff_t1 = (eff[n2a] + eff[n2b]) - (eff[n1a] + eff[n1b]);
    double delta_eff_t2 = (eff[n1a] + eff[n1b]) - (eff[n2a] + eff[n2b]);
    w_eff[t1] += delta_eff_t1;
    w_eff[t2] += delta_eff_t2;

    // --- Incremental Diversity Updates ---
    double delta_div_t1 =
        delta_div[n2a][t1] + delta_div[n2b][t1]
        - delta_div[n1a][t1] - delta_div[n1b][t1]
        - (div_in[n2a][n1a] + div_in[n2b][n1b]);

    double delta_div_t2 =
        delta_div[n1a][t2] + delta_div[n1b][t2]
        - delta_div[n2a][t2] - delta_div[n2b][t2]
        - (div_in[n1a][n2a] + div_in[n1b][n2b]);

    w_div[t1] += delta_div_t1;
    w_div[t2] += delta_div_t2;

    // --- Update delta-diversity relationships ---
    update_delta(n1a, t2, t1);
    update_delta(n1b, t2, t1);
    update_delta(n2a, t1, t2);
    update_delta(n2b, t1, t2);

    // --- Update membership bookkeeping ---
    int a1a = address[n1a];
    int a1b = address[n1b];
    int a2a = address[n2a];
    int a2b = address[n2b];

    state[n1a] = t2;
    state[n1b] = t2;
    state[n2a] = t1;
    state[n2b] = t1;

    address[n1a] = a2a;
    address[n1b] = a2b;
    address[n2a] = a1a;
    address[n2b] = a1b;

    // --- Update current evaluation values (for feasibility) ---
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);

    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH5(int** team) {
    // LLH5: Pick two DISTINCT allocated teams; swap three consecutive members pairwise.
    // Performs incremental efficiency/diversity updates without full objective recomputation.

    //if (num_team < 2 || num_each_t < 3) return team;

    auto [t1, t2] = rand_two_distinct_alloc_teams();
    std::uniform_int_distribution<int> distIdx(0, num_each_t - 3);
    int i = distIdx(hh_rng());  // ensures i, i+1, i+2 are valid indices

    // Identify involved nodes
    int n1a = team[t1][i];
    int n1b = team[t1][i + 1];
    int n1c = team[t1][i + 2];

    int n2a = team[t2][i];
    int n2b = team[t2][i + 1];
    int n2c = team[t2][i + 2];

    // --- Perform swaps ---
    std::swap(team[t1][i],     team[t2][i]);
    std::swap(team[t1][i + 1], team[t2][i + 1]);
    std::swap(team[t1][i + 2], team[t2][i + 2]);

    // --- Incremental Efficiency Updates ---
    double delta_eff_t1 = (eff[n2a] + eff[n2b] + eff[n2c]) - (eff[n1a] + eff[n1b] + eff[n1c]);
    double delta_eff_t2 = -delta_eff_t1;
    w_eff[t1] += delta_eff_t1;
    w_eff[t2] += delta_eff_t2;

    // --- Incremental Diversity Updates ---
    double delta_div_t1 =
        (delta_div[n2a][t1] + delta_div[n2b][t1] + delta_div[n2c][t1])
        - (delta_div[n1a][t1] + delta_div[n1b][t1] + delta_div[n1c][t1])
        - (div_in[n2a][n1a] + div_in[n2b][n1b] + div_in[n2c][n1c]);

    double delta_div_t2 =
        (delta_div[n1a][t2] + delta_div[n1b][t2] + delta_div[n1c][t2])
        - (delta_div[n2a][t2] + delta_div[n2b][t2] + delta_div[n2c][t2])
        - (div_in[n1a][n2a] + div_in[n1b][n2b] + div_in[n1c][n2c]);

    w_div[t1] += delta_div_t1;
    w_div[t2] += delta_div_t2;

    // --- Update delta-diversity relationships ---
    update_delta(n1a, t2, t1);
    update_delta(n1b, t2, t1);
    update_delta(n1c, t2, t1);
    update_delta(n2a, t1, t2);
    update_delta(n2b, t1, t2);
    update_delta(n2c, t1, t2);

    // --- Update membership bookkeeping ---
    int a1a = address[n1a];
    int a1b = address[n1b];
    int a1c = address[n1c];
    int a2a = address[n2a];
    int a2b = address[n2b];
    int a2c = address[n2c];

    state[n1a] = t2;
    state[n1b] = t2;
    state[n1c] = t2;
    state[n2a] = t1;
    state[n2b] = t1;
    state[n2c] = t1;

    address[n1a] = a2a;
    address[n1b] = a2b;
    address[n1c] = a2c;
    address[n2a] = a1a;
    address[n2b] = a1b;
    address[n2c] = a1c;

    // --- Update current evaluation values (for feasibility check) ---
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);

    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic:: LLH6(int **team){

    // LLH6: Randomly select one individual from a team with minimum efficiency and swap it with an unallocated individual from team 0.

    //std::cout << "\n LLH6 Start (Randomly select one individual from a team with minimum efficiency and swap it with an unallocated individual from team 0.):\n";

	double cha1, cha2, cha;
	int node1, node2, team_min, team_old;
	int a1, a2;

	int d1 = 0, d2 = 0;
	int num1 = -1, num2 = -1, dt1, dt2;
	double eff_best = MINVALUE, eff_tabu_best = MINVALUE;
	double delta_eff;
	int idx_min_eff, idx_sec_eff;
	for (int i = 0; i < num_node; i++)
		delta_div[i][0] = 0;
	idx_min_eff = min_func(w_eff, num_team);
	idx_sec_eff = sec_func(w_eff, idx_min_eff, num_team);
	int can_i[MAXNUM], can_j[MAXNUM];
	int tabu_can_i[MAXNUM], tabu_can_j[MAXNUM];
	int ij_len = 0, tabu_ij_len = 0;
		// swap a member in the team with minimum eff and a member in another team
	for (int i = 0; i < num_node; i++)
	{
		if (state[i] != 0 && state[i] != idx_min_eff)
		{
			for (int j = 0; j < num_each_t; j++)
			{
				int k = team[idx_min_eff][j];
				cha1 = delta_div[k][state[i]] - delta_div[i][state[i]] - div_in[i][k];
				cha2 = delta_div[i][state[k]] - delta_div[k][state[k]] - div_in[i][k];
				double delta1 = eff[i] - eff[k];
				double delta2 = eff[k] - eff[i];
				delta_eff = delta1;
				if (delta1 + w_eff[idx_min_eff] > delta2 + w_eff[state[i]])
					delta_eff = delta2 + w_eff[state[i]] - w_eff[idx_min_eff];
				if (delta_eff + w_eff[idx_min_eff] > w_eff[idx_sec_eff])
					delta_eff = w_eff[idx_sec_eff] - w_eff[idx_min_eff];
				if ((state[i] != state[k]) && (w_div[state[i]] + cha1 >= min_div) && (w_div[state[k]] + cha2 >= min_div))
				{
					if ((tabu_list[i][state[k]] <= iter) && (tabu_list[k][state[i]] <= iter))
					{
						//delta_eff = (eff[i] - eff[k]) + alpha*(cha2);
						if (delta_eff > eff_best)
						{
							eff_best = delta_eff;
							ij_len = 0;
							can_i[ij_len] = i;
							can_j[ij_len] = k;
							ij_len++;
						}
						else if (fabs(eff_best - delta_eff) < 1.0e-5 && ij_len < MAXNUM)
						{
							can_i[ij_len] = i;
							can_j[ij_len] = k;
							ij_len++;
						}
					}
					else
					{
						if (delta_eff > eff_tabu_best)
						{
							eff_tabu_best = delta_eff;
							tabu_ij_len = 0;
							tabu_can_i[tabu_ij_len] = i;
							tabu_can_j[tabu_ij_len] = k;
							tabu_ij_len++;
						}
						else if (fabs(delta_eff - eff_tabu_best) <= 1.0e-5 && tabu_ij_len < MAXNUM)
						{
							tabu_can_i[tabu_ij_len] = i;
							tabu_can_j[tabu_ij_len] = k;
							tabu_ij_len++;
						}
					}
				}
			}
		}
	}
	//aspiration creterion
	if ((tabu_ij_len > 0 && eff_tabu_best > eff_best && f_cur + eff_tabu_best > fbest) || (tabu_ij_len > 0 && ij_len == 0))
	{
		int rx = rand() % tabu_ij_len;
		d1 = tabu_can_i[rx];
		d2 = tabu_can_j[rx];
	}
	else if (ij_len > 0)
	{
		int rx = rand() % ij_len;
		d1 = can_i[rx];
		d2 = can_j[rx];
	}

        node1 = d1;							    // move in //
		node2 = d2;  							// move out //
		//cout<<"d1="<<d1<<"  "<<"d2="<<d2<<"\n";
		team_min = state[d2];					//team with min eff//
		team_old = state[d1];
		a1 = address[node1];
		a2 = address[node2];
		team[team_min][a2] = node1;
        team[team_old][a1] = node2;

	return team ;
}

int** Hyper_heuristic::LLH7(int** team) {
    // LLH7: Swap one random member between min-efficiency and max-efficiency teams.
    // Incremental update without full objective recomputation.

    //if (num_team < 2 || num_each_t <= 0) return team;

    // --- Identify min and max efficiency teams ---
    int tmin = 1, tmax = 1;
    double min_eff = w_eff[1], max_eff = w_eff[1];
    for (int t = 2; t <= num_team; ++t) {
        if (w_eff[t] < min_eff) { min_eff = w_eff[t]; tmin = t; }
        if (w_eff[t] > max_eff) { max_eff = w_eff[t]; tmax = t; }
    }

    // In case all efficiencies are equal, choose two random distinct teams
    if (tmin == tmax) {
        tmin = rand_alloc_team();
        do { tmax = rand_alloc_team(); } while (tmax == tmin);
    }

    // --- Pick one random member from each ---
    int idx_min = rand_member_idx();
    int idx_max = rand_member_idx();
    int node_min = team[tmin][idx_min];
    int node_max = team[tmax][idx_max];

    // --- Swap members between teams ---
    std::swap(team[tmin][idx_min], team[tmax][idx_max]);

    // --- Incremental Efficiency Updates ---
    double delta_eff_tmin = eff[node_max] - eff[node_min];
    double delta_eff_tmax = -delta_eff_tmin;
    w_eff[tmin] += delta_eff_tmin;
    w_eff[tmax] += delta_eff_tmax;

    // --- Incremental Diversity Updates ---
    double delta_div_tmin =
        delta_div[node_max][tmin] - delta_div[node_min][tmin]
        - div_in[node_max][node_min];
    double delta_div_tmax =
        delta_div[node_min][tmax] - delta_div[node_max][tmax]
        - div_in[node_min][node_max];

    w_div[tmin] += delta_div_tmin;
    w_div[tmax] += delta_div_tmax;

    // --- Update delta contributions ---
    update_delta(node_min, tmax, tmin);
    update_delta(node_max, tmin, tmax);

    // --- Update membership bookkeeping ---
    int a_min = address[node_min];
    int a_max = address[node_max];
    state[node_min] = tmax;
    state[node_max] = tmin;
    address[node_min] = a_max;
    address[node_max] = a_min;

    // --- Update current efficiency/diversity for feasibility ---
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);
    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH8(int** team) {
    // LLH8: Swap the first individual from two randomly selected teams (incremental update).

    //if (num_team < 2 || num_each_t <= 0) return team;

    // Randomly select two distinct allocated teams
    auto [t1, t2] = rand_two_distinct_alloc_teams();
    int ind1 = 0;  // swap first member from each team

    int node1 = team[t1][ind1];
    int node2 = team[t2][ind1];

    // --- Perform the swap ---
    std::swap(team[t1][ind1], team[t2][ind1]);

    // --- Incremental Efficiency Update ---
    double delta_eff_t1 = eff[node2] - eff[node1];
    double delta_eff_t2 = -delta_eff_t1;
    w_eff[t1] += delta_eff_t1;
    w_eff[t2] += delta_eff_t2;

    // --- Incremental Diversity Update ---
    double delta_div_t1 =
        delta_div[node2][t1] - delta_div[node1][t1] - div_in[node2][node1];
    double delta_div_t2 =
        delta_div[node1][t2] - delta_div[node2][t2] - div_in[node1][node2];

    w_div[t1] += delta_div_t1;
    w_div[t2] += delta_div_t2;

    // --- Update delta relationships ---
    update_delta(node1, t2, t1);
    update_delta(node2, t1, t2);

    // --- Update membership bookkeeping ---
    int a1 = address[node1];
    int a2 = address[node2];
    state[node1] = t2;
    state[node2] = t1;
    address[node1] = a2;
    address[node2] = a1;

    // --- Update feasibility values ---
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);

    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH9(int** team) {
    // LLH9: Swap the LAST individual from two randomly selected allocated teams.
    // Performs incremental efficiency and diversity updates (no full objective recomputation).

    //if (num_team < 2 || num_each_t <= 0) return team;

    // --- Randomly select two distinct allocated teams ---
    auto [t1, t2] = rand_two_distinct_alloc_teams();
    int ind1 = num_each_t - 1;  // swap the last member in each team

    int node1 = team[t1][ind1];
    int node2 = team[t2][ind1];

    // --- Perform the swap ---
    std::swap(team[t1][ind1], team[t2][ind1]);

    // --- Incremental Efficiency Updates ---
    double delta_eff_t1 = eff[node2] - eff[node1];
    double delta_eff_t2 = -delta_eff_t1;
    w_eff[t1] += delta_eff_t1;
    w_eff[t2] += delta_eff_t2;

    // --- Incremental Diversity Updates ---
    double delta_div_t1 =
        delta_div[node2][t1] - delta_div[node1][t1] - div_in[node2][node1];
    double delta_div_t2 =
        delta_div[node1][t2] - delta_div[node2][t2] - div_in[node1][node2];

    w_div[t1] += delta_div_t1;
    w_div[t2] += delta_div_t2;

    // --- Update delta-diversity relationships ---
    update_delta(node1, t2, t1);
    update_delta(node2, t1, t2);

    // --- Update membership bookkeeping ---
    int a1 = address[node1];
    int a2 = address[node2];
    state[node1] = t2;
    state[node2] = t1;
    address[node1] = a2;
    address[node2] = a1;

    // --- Update current efficiency/diversity for feasibility ---
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);

    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic:: LLH10(int **team){
   //LLH10: Swap individual with minimum efficiency with individual assigned.
	double cha1, cha2, cha;
	int node1, node2, team_min, team_old;
	int a1, a2,d1,d2;
    //std::cout << "\n LLH10 Start(Swap individual with minimum efficiency with individual assigned.):\n";

	//int d1 = 0, d2 = 0;
	int num1 = -1, num2 = -1, dt1, dt2;
	double eff_best = MINVALUE, eff_tabu_best = MINVALUE;
	double delta_eff;
	int idx_min_eff, idx_sec_eff;
	for (int i = 0; i < num_node; i++)
		delta_div[i][0] = 0;
	idx_min_eff = min_func(w_eff, num_team);
	idx_sec_eff = sec_func(w_eff, idx_min_eff, num_team);
	int can_i[MAXNUM], can_j[MAXNUM];
	int tabu_can_i[MAXNUM], tabu_can_j[MAXNUM];
	int ij_len = 0, tabu_ij_len = 0;
		// swap a member in the team with minimum eff and a member in another team
	for (int i = 0; i < num_node; i++)
	{
		if (state[i] != 0 && state[i] != idx_min_eff)
		{
			for (int j = 0; j < num_each_t; j++)
			{
				int k = team[idx_min_eff][j];
				cha1 = delta_div[k][state[i]] - delta_div[i][state[i]] - div_in[i][k];
				cha2 = delta_div[i][state[k]] - delta_div[k][state[k]] - div_in[i][k];
				double delta1 = eff[i] - eff[k];
				double delta2 = eff[k] - eff[i];
				delta_eff = delta1;
				if (delta1 + w_eff[idx_min_eff] > delta2 + w_eff[state[i]])
					delta_eff = delta2 + w_eff[state[i]] - w_eff[idx_min_eff];
				if (delta_eff + w_eff[idx_min_eff] > w_eff[idx_sec_eff])
					delta_eff = w_eff[idx_sec_eff] - w_eff[idx_min_eff];
				if ((state[i] != state[k]) && (w_div[state[i]] + cha1 >= min_div) && (w_div[state[k]] + cha2 >= min_div))
				{
					if ((tabu_list[i][state[k]] <= iter) && (tabu_list[k][state[i]] <= iter))
					{
						//delta_eff = (eff[i] - eff[k]) + alpha*(cha2);
						if (delta_eff > eff_best)
						{
							eff_best = delta_eff;
							ij_len = 0;
							can_i[ij_len] = i;
							can_j[ij_len] = k;
							ij_len++;
						}
						else if (fabs(eff_best - delta_eff) < 1.0e-5 && ij_len < MAXNUM)
						{
							can_i[ij_len] = i;
							can_j[ij_len] = k;
							ij_len++;
						}
					}
					else
					{
						if (delta_eff > eff_tabu_best)
						{
							eff_tabu_best = delta_eff;
							tabu_ij_len = 0;
							tabu_can_i[tabu_ij_len] = i;
							tabu_can_j[tabu_ij_len] = k;
							tabu_ij_len++;
						}
						else if (fabs(delta_eff - eff_tabu_best) <= 1.0e-5 && tabu_ij_len < MAXNUM)
						{
							tabu_can_i[tabu_ij_len] = i;
							tabu_can_j[tabu_ij_len] = k;
							tabu_ij_len++;
						}
					}
				}
			}
		}
	}
	//aspiration creterion
	if ((tabu_ij_len > 0 && eff_tabu_best > eff_best && f_cur + eff_tabu_best > fbest) || (tabu_ij_len > 0 && ij_len == 0))
	{
		int rx = rand() % tabu_ij_len;
		d1 = tabu_can_i[rx];
		d2 = tabu_can_j[rx];
	}
	else if (ij_len > 0)
	{
		int rx = rand() % ij_len;
		d1 = can_i[rx];
		d2 = can_j[rx];
	}

        node1 = d1;							    // move in
		node2 = d2;  							//move out
		//cout<<"d1="<<d1<<"  "<<"d2="<<d2<<"\n";
		team_min = state[d2];					//team with min eff
		team_old = state[d1];
		a1 = address[node1];
		a2 = address[node2];
		team[team_min][a2] = node1;
        team[team_old][a1] = node2;

	return team;
}

int** Hyper_heuristic::LLH11(int** team) {
    // LLH11: Perform backward circular swaps between teams for d members.
    // Each column c is swapped such that member(i, c) ← member(i+1, c+1) circularly.
    // Incremental efficiency/diversity update without objective_Function().

    int d = 4;  // number of members to circularly shift
    //if (num_team < 2 || num_each_t < d) return team;

    // For each column c, perform backward circular swap across all allocated teams
    for (int c = 0; c < d; ++c) {
        // --- Save current members in this column across all teams ---
        std::vector<int> members(num_team + 1);
        for (int t = 1; t <= num_team; ++t)
            members[t] = team[t][c];

        // --- Perform circular shift ---
        int temp = members[1];  // first team member to move last
        for (int t = 1; t < num_team; ++t)
            team[t][c] = members[t + 1];
        team[num_team][c] = temp;
    }

    // --- Incremental Efficiency & Diversity Updates ---
    for (int t = 1; t <= num_team; ++t) {
        double new_eff = 0.0, new_div = 0.0;
        for (int j = 0; j < num_each_t; ++j) {
            int node = team[t][j];
            new_eff += eff[node];
            new_div += delta_div[node][t];
        }
        w_eff[t] = new_eff;
        w_div[t] = new_div;
    }

    // --- Update current efficiency and diversity metrics ---
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);

    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH12(int** team) {
    // LLH12: Swap 50% of the members between the least efficient team and the unallocated team (team 0).
    // Incremental update without recomputing full objective.

    //if (num_team < 1 || num_each_t <= 1) return team;

    // --- Identify the least efficient team ---
    int min_team = 1;
    double min_eff = w_eff[1];
    for (int t = 2; t <= num_team; ++t)
        if (w_eff[t] < min_eff) { min_eff = w_eff[t]; min_team = t; }

    int team0 = 0;
    int swap_count = num_each_t / 2;  // Swap 50% of the members
    int pool_size = pool_size_unallocated();
    if (pool_size < swap_count) swap_count = pool_size;

    // --- Perform incremental swaps ---
    for (int i = 0; i < swap_count; ++i) {
        int node_min = team[min_team][i];
        int node_0   = team[team0][i];

        std::swap(team[min_team][i], team[team0][i]);

        // Efficiency updates
        double delta_eff = eff[node_0] - eff[node_min];
        w_eff[min_team] += delta_eff;

        // Diversity updates
        double delta_div_min =
            delta_div[node_0][min_team] - delta_div[node_min][min_team]
            - div_in[node_0][node_min];
        w_div[min_team] += delta_div_min;

        // Update delta relationships
        update_delta(node_min, team0, min_team);
        update_delta(node_0, min_team, team0);

        // Update state and address
        int a_min = address[node_min];
        int a_0   = address[node_0];
        state[node_min] = team0;
        state[node_0]   = min_team;
        address[node_min] = a_0;
        address[node_0]   = a_min;
    }

    // --- Recalculate current efficiency/diversity status ---
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);
    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH13(int** team) {

    //LLH13: Apply a local hill-climbing search on the current solution.

    //std::cout << "\n LLH13 Start(Apply a local hill-climbing search on the current solution.):\n";

    f_cur = feasible_local_search();
    //repair_solution();
    return team;
}

int** Hyper_heuristic::LLH14(int** team) {

    //LLH14: Ruin and Recreate
    //- Randomly remove a percentage of members from allocated teams.
    //- Reassign them to random teams with available slots (incrementally updated).


    //if (num_team < 1 || num_each_t <= 1) return team;

    int ruin_percentage = 50;
    int total_members = num_each_t * num_team;
    int num_ruin = (total_members * ruin_percentage) / 100;

    std::vector<int> ruined_members;
    ruined_members.reserve(num_ruin);
    std::vector<int> team_sizes(num_team + 1, 0);

    // Count team sizes
    for (int t = 1; t <= num_team; ++t) {
        for (int j = 0; j < num_each_t; ++j) {
            if (team[t][j] != -1) team_sizes[t]++;
        }
    }

    // Randomly remove members from allocated teams
    for (int r = 0; r < num_ruin; ++r) {
        int t = 1 + rand() % num_team;
        int m = rand() % num_each_t;
        int member = team[t][m];
        if (member == -1) continue;

        ruined_members.push_back(member);
        team[t][m] = -1;  // mark removed
        team_sizes[t]--;
        state[member] = 0;  // move to unallocated

        // update efficiency and diversity of team t
        w_eff[t] -= eff[member];
        w_div[t] -= delta_div[member][t];
    }

    // Reassign ruined members to teams with available capacity
    for (int member : ruined_members) {
        std::vector<int> available;
        for (int t = 1; t <= num_team; ++t)
            if (team_sizes[t] < num_each_t)
                available.push_back(t);

        if (available.empty()) break;

        int target_team = available[rand() % available.size()];

        // assign to the first empty slot
        for (int j = 0; j < num_each_t; ++j) {
            if (team[target_team][j] == -1) {
                team[target_team][j] = member;
                team_sizes[target_team]++;
                state[member] = target_team;

                // update efficiency & diversity incrementally
                w_eff[target_team] += eff[member];
                w_div[target_team] += delta_div[member][target_team];
                break;
            }
        }
    }

    // recompute current objective indicators
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);
    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH15(int** team) {

    //LLH15: Randomly perturb the solution by performing a few random swaps
    //between allocated team members and unallocated members (team 0).
    //Incremental efficiency and diversity updates are applied.


    int team0 = 0;  // unallocated pool
    int k = 2;      // number of random swaps
    int pool_size = pool_size_unallocated();
    //if (num_team < 1 || num_each_t <= 0 || pool_size <= 0) return team;

    for (int i = 0; i < k; ++i) {
        int t_alloc = rand_alloc_team();
        int member_idx = rand_member_idx();
        int pool_idx = rand_pool_idx(pool_size);

        int node_alloc = team[t_alloc][member_idx];
        int node_pool  = team[team0][pool_idx];

        // --- Perform swap ---
        std::swap(team[t_alloc][member_idx], team[team0][pool_idx]);

        // --- Incremental efficiency update ---
        double delta_eff = eff[node_pool] - eff[node_alloc];
        w_eff[t_alloc] += delta_eff;

        // --- Incremental diversity update ---
        double delta_div_t = delta_div[node_pool][t_alloc] - delta_div[node_alloc][t_alloc]
                             - div_in[node_pool][node_alloc];
        w_div[t_alloc] += delta_div_t;

        // --- Update states ---
        state[node_alloc] = team0;
        state[node_pool]  = t_alloc;

        // --- Update delta info for local interactions ---
        update_delta(node_alloc, team0, t_alloc);
        update_delta(node_pool, t_alloc, team0);
    }

    // --- Recalculate min efficiency/diversity for the current solution ---
    int min_eff_idx = min_func(w_eff, num_team);
    int min_div_idx = min_func(w_div, num_team);
    f_cur     = w_eff[min_eff_idx];
    f_cur_div = w_div[min_div_idx];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH16(int** team) {
    /*
    LLH17: 3-Chain Move (LLH1-style)
    - Pick 3 distinct allocated teams
    - Pick 1 random member from each
    - Circular exchange:
          team1[i] ← team2[i]
          team2[i] ← team3[i]
          team3[i] ← team1[i]
    - Full recomputation of eff/div for correctness
    */

    //if (num_team < 3 || num_each_t < 1)  return team;

    // ---------------------------------------------------
    // 1. Select three distinct allocated teams
    // ---------------------------------------------------
    int t1 = rand_alloc_team();
    int t2 = rand_alloc_team();
    int t3 = rand_alloc_team();

    while (t2 == t1) t2 = rand_alloc_team();
    while (t3 == t1 || t3 == t2) t3 = rand_alloc_team();

    // ---------------------------------------------------
    // 2. Random member index
    // ---------------------------------------------------
    int idx1 = rand_member_idx();
    int idx2 = rand_member_idx();
    int idx3 = rand_member_idx();

    int n1 = team[t1][idx1];
    int n2 = team[t2][idx2];
    int n3 = team[t3][idx3];

    // ---------------------------------------------------
    // 3. Perform the circular swap
    // ---------------------------------------------------
    team[t1][idx1] = n2;   // team1 gets from team2
    team[t2][idx2] = n3;   // team2 gets from team3
    team[t3][idx3] = n1;   // team3 gets from team1

    // ---------------------------------------------------
    // 4. Repair membership mapping
    // ---------------------------------------------------
    int poolSize = pool_size_unallocated();

    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? poolSize : num_each_t);

        for (int j = 0; j < limit; j++) {
            int node = team[t][j];
            state[node]   = t;
            address[node] = j;
        }
    }

    // ---------------------------------------------------
    // 5. Recompute efficiencies
    // ---------------------------------------------------
    w_eff[0] = 0;

    for (int t = 1; t <= num_team; t++) {
        double sum = 0;
        for (int j = 0; j < num_each_t; j++)
            sum += eff[team[t][j]];
        w_eff[t] = sum;
    }

    // ---------------------------------------------------
    // 6. Recompute pairwise diversity
    // ---------------------------------------------------
    w_div[0] = 0;

    for (int t = 1; t <= num_team; t++) {
        double sum = 0.0;
        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++)
                sum += div_in[a][team[t][j]];
        }
        w_div[t] = sum;
    }

    // ---------------------------------------------------
    // 7. Update objective values
    // ---------------------------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}

// ============================================================
// ApplyHeuristic(h, solution)
// LLHs:    1 → 16
// Metaheuristics: 17 → 24
// Comments taken from LLH function headers
// ============================================================
int** Hyper_heuristic::LLH17(int** team) {
    /*
    LLH13: Apply a local hill-climbing search on the current solution.
    */
    //std::cout << "\n LLH13 Start(Apply a local hill-climbing search on the current solution.):\n";

   feasible_local_search();
    //repair_solution();
    return team;
}
int** Hyper_heuristic::LLH18(int** team) {
    /*
    LLH13: Apply a local hill-climbing search on the current solution.
    */
    //std::cout << "\n LLH13 Start(Apply a local hill-climbing search on the current solution.):\n";

    team = infeasible_local_search();
    //repair_solution();
    return team;
}
int** Hyper_heuristic::LLH19(int** team) {

    team = fits();
        // Metaheuristic: Tabu Search (FITS)
    return team;
}
int** Hyper_heuristic::LLH20(int** team) {
    /*
    LLH13: Apply a local hill-climbing search on the current solution.
    */
    //std::cout << "\n LLH13 Start(Apply a local hill-climbing search on the current solution.):\n";

    team = iterated_local_search();
        // Metaheuristic: Iterated Local Search (ILS)

    //repair_solution();
    return team;
}
int** Hyper_heuristic::LLH21(int** team) {
    /*
    LLH13: Apply a local hill-climbing search on the current solution.
    */
    //std::cout << "\n LLH13 Start(Apply a local hill-climbing search on the current solution.):\n";

    team = simulated_annealing();
        // Metaheuristic: Simulated Annealing (SA)
    //repair_solution();
    return team;
}
int** Hyper_heuristic::LLH22(int** team) {
    /*
    LLH13: Apply a local hill-climbing search on the current solution.
    */
    //std::cout << "\n LLH13 Start(Apply a local hill-climbing search on the current solution.):\n";

    team = memetic();
        // Metaheuristic: Memetic Algorithm
    return team;
}
int** Hyper_heuristic::LLH23(int** team) {

    team = great_deluge_algorithm();
        // Metaheuristic: Great Deluge
    return team;
}
int** Hyper_heuristic::LLH24(int** team) {

    int fbest12 = guided_local_search();
        // Metaheuristic: Guided Local Search
    return team;
}
int** Hyper_heuristic::LLH25(int** team) {
    /*
    LLH13: Apply a local hill-climbing search on the current solution.
    */
    //std::cout << "\n LLH13 Start(Apply a local hill-climbing search on the current solution.):\n";

    team = late_acceptance_hill_climbing();
        // Metaheuristic: LAHC
    return team;
}
int** Hyper_heuristic::ApplyHeuristic(int h, int** solution)
{
    int **Scurrent = solution;
    int fbest12 = 0;

    switch(h)
    {
    // ---------------------------------------------------------
    //                    LOW-LEVEL HEURISTICS
    // ---------------------------------------------------------

    case 1:
        Scurrent = LLH1(solution);
        // LLH1: Swap one unallocated member (team 0) with one from a random allocated team.
        break;

    case 2:
        Scurrent = LLH2(solution);
        // LLH2: Swap TWO unallocated members (team 0) with TWO members from two allocated teams.
        break;

    case 3:
        Scurrent = LLH3(solution);
        // LLH3: Swap one member between two allocated teams.
        break;

    case 4:
        Scurrent = LLH4(solution);
        // LLH4: Move one member from Team A to Team B by replacing B’s chosen member.
        break;

    case 5:
        Scurrent = LLH5(solution);
        // LLH5: Swap three consecutive members between two allocated teams.
        break;

    case 6:
        Scurrent = LLH6(solution);
        // LLH6: Swap one member from the minimum-efficiency team with one from the pool (team 0).
        break;

    case 7:
        Scurrent = LLH7(solution);
        // LLH7: Swap one random member between min-efficiency and max-efficiency teams.
        break;

    case 8:
        Scurrent = LLH8(solution);
        // LLH8: Swap the FIRST individual from two allocated teams.
        break;

    case 9:
        Scurrent = LLH9(solution);
        // LLH9: Swap the LAST individual from two allocated teams.
        break;

    case 10:
        Scurrent = LLH10(solution);
        // LLH10: Swap one member from the min-efficiency team with a random member from another team.
        break;

    case 11:
        Scurrent = LLH11(solution);
        // LLH11: Circularly shift d columns across all allocated teams.
        break;

    case 12:
        Scurrent = LLH12(solution);
        // LLH12: Swap 50% of the members between min-efficiency team and team 0.
        break;

    case 13:
        Scurrent = LLH13(solution);
        // LLH13: Ruin and Recreate (remove a percentage of members then refill).
        break;

    case 14:
        Scurrent = LLH14(solution);
        // LLH14: Random perturbation—perform k random swaps between teams and pool.
        break;

    case 15:
        Scurrent = LLH15(solution);
        // LLH15: Swap a small subset of members between two allocated teams.
        break;

    case 16:
        Scurrent = LLH16(solution);
        // LLH16: 3-chain move — circular exchange among three allocated teams.
        break;

    // ---------------------------------------------------------
    //                    METAHEURISTICS (High-level)
    // ---------------------------------------------------------

    case 17:
        f_cur = feasible_local_search();
        // Metaheuristic: infeasible local search: Apply local hill-climbing / feasible_local_search() on current solution.
        break;

    case 18:
        Scurrent = infeasible_local_search();
        // Metaheuristic: infeasible local search
        break;

    case 19:
        Scurrent = fits();
        // Metaheuristic: Tabu Search (FITS)
        break;

    case 20:
        Scurrent = iterated_local_search();
        // Metaheuristic: Iterated Local Search (ILS)
        break;

    case 21:
        Scurrent = simulated_annealing();
        // Metaheuristic: Simulated Annealing (SA)
        break;

    case 22:
        Scurrent = memetic();
        // Metaheuristic: Memetic Algorithm
        break;

    case 23:
        Scurrent = great_deluge_algorithm();
        // Metaheuristic: Great Deluge
        break;

    case 24:
        fbest12 = guided_local_search();
        // Metaheuristic: Guided Local Search
        break;

    case 25:
        Scurrent = late_acceptance_hill_climbing();
        // Metaheuristic: LAHC
        break;

    default:
        std::cout << "Invalid heuristic index\n";
    }

    return Scurrent;
}

int** Hyper_heuristic::ApplyMeta_Heuristic(int h, int** solution)
{
    int **Scurrent = solution;
    int fbest12 = 0;

    switch(h)
    {
     // ---------------------------------------------------------
    //                    METAHEURISTICS (High-level)
    // ---------------------------------------------------------

    case 0:
		Scurrent = simulated_annealing();
        // Metaheuristic: Simulated Annealing (SA)
        break;
    case 1:
		Scurrent = iterated_local_search();
        // Metaheuristic: Iterated Local Search (ILS)
        break;
    case 2:
        Scurrent = fits();
        // Metaheuristic: Tabu Search (FITS)
        break;

    case 3:
        Scurrent = great_deluge_algorithm();
        // Metaheuristic: Great Deluge
        break;

    case 4:
        Scurrent = late_acceptance_hill_climbing();
        // Metaheuristic: LAHC
        break;

    case 5:
		Scurrent = infeasible_local_search();
        // Metaheuristic: infeasible local search
        break;

    case 6:
        f_cur = feasible_local_search();
        // Metaheuristic: infeasible local search: Apply local hill-climbing / feasible_local_search() on current solution.
        break;

    case 7:
        fbest12 = guided_local_search();
        // Metaheuristic: Guided Local Search
        break;

    case 8:
        Scurrent = memetic();
        // Metaheuristic: Memetic Algorithm
        break;

    default:
        std::cout << "Invalid heuristic index\n";
    }

    return Scurrent;
}

void Hyper_heuristic::Greedy_Selection_Hyperheuristic_CMCEE(int max_time) {
    std::cout <<
        "=============================================================================\n"
        "Greedy Selection Hyper-Heuristic Framework Start its Processes.\n"
        "=============================================================================\n";

    // ------------------------------------------------------------
    // INITIALIZATION
    // ------------------------------------------------------------
    generate_initialrandom();
    objective_Function1(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;

    std::cout << "\tInitial objectives: Efficiency = " << cost_eff
              << ", Diversity = " << cost_div << "\n";

    int best_cost_eff = cost_eff;
    int best_cost_div = cost_div;
    int best_found_at = 0;

    int** current_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** best_solution1   = deep_copy_solution(team, num_node, num_team, num_each_t);

    int* team_size = new int[num_team + 1];
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    // ------------------------------------------------------------
    // HEURISTICS & TRACKING
    // ------------------------------------------------------------
    std::vector<int> heuristics = {0,1,2,3,4,5,6};
    //std::vector<int> heuristics = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 17, 18, 19, 20, 22,23};
    std::map<int,double> heuristic_scores;
    for (auto h : heuristics) heuristic_scores[h] = 0.0;

    std::vector<int> objective_values;
    std::vector<int> diversity_values;
    std::vector<double> iteration_times;
    std::vector<double> rewards_record;

    // ------------------------------------------------------------
    // FILES
    // ------------------------------------------------------------
    std::filesystem::path folder =
        "D:/Datasets/RESULTS_OF_HH_MODELS/Greedy_Convergence/";

    if (!std::filesystem::exists(folder))
        std::filesystem::create_directories(folder);

    std::filesystem::path result_file =
        folder / ("Greedy_HH_CMCEE_" + instanceName + "_results.txt");
    std::filesystem::path trace_file =
        folder / ("Greedy_HH_CMCEE_" + instanceName + "_Convergence_Trace.csv");

    std::ofstream outfile(result_file);
    outfile << "Iteration\tHeuristic\tEfficiency\tDiversity\tIterTime\tDelta\n";

    std::ofstream trace(trace_file);
    trace << "Iteration,Efficiency,Diversity,BestEfficiency,BestDiversity,Reward,Runtime\n";

    // ------------------------------------------------------------
    // MAIN LOOP
    // ------------------------------------------------------------
    clock_t total_start_time = clock();
    int iteration = 0;
    int selected_heuristic = -1;
    int no_improvement_count = 0;
    const int NO_IMPROVEMENT_LIMIT = 100;

    while ((static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC < max_time)) {
        iteration++;
        clock_t iteration_start = clock();

        // Pick heuristic
        if (iteration <= (int)heuristics.size())
            selected_heuristic = heuristics[iteration - 1];
        else
            selected_heuristic = std::max_element(
                heuristic_scores.begin(), heuristic_scores.end(),
                [](const auto &a, const auto &b) { return a.second < b.second; })->first;

        // Apply heuristic
        clock_t h_start = clock();
        ApplyHeuristic(selected_heuristic, team);
        double heuristic_time = (double)(clock() - h_start) / CLOCKS_PER_SEC;

        // Evaluate
        objective_Function1(team);
        int new_cost_eff = f_cur;
        int new_cost_div = f_cur_div;
        double delta = new_cost_eff - cost_eff;
        double reward = (delta > 0 ? 1.0 : (delta < 0 ? -1.0 : 0.0));
        rewards_record.push_back(reward);

        // ACCEPTANCE
        bool improved = false;
        if (delta > 0) {
            improved = true;
            cost_eff = new_cost_eff;
            cost_div = new_cost_div;

            free_solution(current_solution, num_node, num_team, num_each_t);
            current_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
            heuristic_scores[selected_heuristic] += delta;

            // GLOBAL BEST
            if ((cost_eff > best_cost_eff) && (cost_div >= min_div)) {
                best_cost_eff = cost_eff;
                best_cost_div = cost_div;

                free_solution(best_solution1, num_node, num_team, num_each_t);
                best_solution1 = deep_copy_solution(team, num_node, num_team, num_each_t);
                best_eff       = cost_eff;
                best_div       = cost_div;
                time_taken     = heuristic_time;
                best_found_at  = iteration;

                // Copy solution arrays (restored)
                for (int m = 0; m < num_node; m++) {
                    fbest_solution[m] = best_solution[m];
                }
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        } else {
            // Revert
            for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = current_solution[i][j];
        }

        // Early stop if no improvement
        if (!improved)
            no_improvement_count++;
        else
            no_improvement_count = 0;

        if (no_improvement_count >= NO_IMPROVEMENT_LIMIT) {
            std::cout << "\nNo improvement in best for " << NO_IMPROVEMENT_LIMIT << " iterations.\n";
            break;
        }

        // Timing
        double iter_time = (double)(clock() - iteration_start) / CLOCKS_PER_SEC;
        double runtime   = (double)(clock() - total_start_time) / CLOCKS_PER_SEC;

        objective_values.push_back(new_cost_eff);
        diversity_values.push_back(new_cost_div);
        iteration_times.push_back(iter_time);

        // FILE OUTPUTS
        outfile << iteration << "\t" << selected_heuristic << "\t"
                << new_cost_eff << "\t" << new_cost_div << "\t"
                << iter_time << "\t" << delta << "\n";

        trace << iteration << "," << new_cost_eff << "," << new_cost_div << ","
              << best_cost_eff << "," << best_cost_div << ","
              << reward << "," << runtime << "\n";

        // Console output
        std::cout << "Iter: " << iteration
                  << " | Heuristic: " << selected_heuristic
                  << " | Eff: " << new_cost_eff
                  << " | Div: " << new_cost_div
                  << " | BestEff: " << best_cost_eff
                  << " | BestDiv: " << best_cost_div
                  << " | Delta=" << delta
                  << " | Reward=" << reward
                  << " | Time=" << iter_time << "s\n";
    }

    outfile.close();
    trace.close();

    // ------------------------------------------------------------
    // FINAL SUMMARY
    // ------------------------------------------------------------
    double total_time = (double)(clock() - total_start_time) / CLOCKS_PER_SEC;
    double avg_eff = 0, avg_div = 0, avg_cpu = 0;

    if (!objective_values.empty())
        avg_eff = std::accumulate(objective_values.begin(), objective_values.end(), 0.0)
                  / objective_values.size();
    if (!diversity_values.empty())
        avg_div = std::accumulate(diversity_values.begin(), diversity_values.end(), 0.0)
                  / diversity_values.size();
    if (!iteration_times.empty())
        avg_cpu = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0)
                  / iteration_times.size();

    std::cout << "\nFinal Summary\n"
              << "-----------------------------\n"
              << "Best Efficiency : " << best_cost_eff << "\n"
              << "Best Diversity  : " << best_cost_div << "\n"
              << "Average Eff     : " << avg_eff << "\n"
              << "Average Div     : " << avg_div << "\n"
              << "Average IterTime: " << avg_cpu << "s\n"
              << "Convergence Trace Saved: " << trace_file << "\n"
              << "==========================================\n";

    check_best_solution();

    std::cout <<
        "=============================================================================\n"
        "Greedy Selection Hyper-Heuristic Framework Finished.\n"
        "=============================================================================\n";
}

void Hyper_heuristic::Random_Selection_Hyperheuristic_CMCEE(int max_time) {
    std::cout <<
        "=============================================================================\n"
        "Random Selection Hyper-Heuristic Framework Start its Processes.\n"
        "=============================================================================\n";

    // ------------------------------------------------------------
    // INITIALIZATION
    // ------------------------------------------------------------
    //generate_initialrandom();

    int** previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** best_solution112  = deep_copy_solution(team, num_node, num_team, num_each_t);

    int* team_size = new int[num_team + 1];
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    objective_Function1(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    int best_cost_eff = cost_eff;
    int best_cost_div = cost_div;

    std::cout << "Initial objectives - Efficiency: " << cost_eff
              << ", Diversity: " << cost_div << std::endl;

    // ------------------------------------------------------------
    // HEURISTICS AND TRACKING
    // ------------------------------------------------------------
    std::vector<int> heuristics = {17,18,19,20,21,22,23};
    int max_iterations = 100;
    std::vector<int> selected_heuristics;

    std::map<int, std::vector<int>> heuristic_costs;
    std::map<int, std::vector<double>> heuristic_times;
    std::vector<int> objective_values;
    std::vector<int> diversity_values;
    std::vector<double> iteration_times;
    std::vector<double> rewards_record;

    std::set<int> S_imp, S_wrs, S_eq, S_ac, S_uq, S_nb;
    std::map<int,double> heuristic_total_time;
    std::vector<double> fitness_history;

    double total_elapsed_time = 0.0;

    // ------------------------------------------------------------
    // FILE SETUP
    // ------------------------------------------------------------
    std::filesystem::path folder =
        "D:/Datasets/RESULTS_OF_HH_MODELS/Random_Convergence/";

    if (!std::filesystem::exists(folder))
        std::filesystem::create_directories(folder);

    std::filesystem::path results_file =
        folder / ("Random_Selection_HH_CMCEE_" + instanceName + "_results.txt");
    std::filesystem::path features_file =
        folder / ("Random_Selection_HH_CMCEE_" + instanceName + "_features.csv");
    std::filesystem::path trace_file =
        folder / ("Random_Selection_HH_CMCEE_" + instanceName + "_Convergence_Trace.csv");

    std::ofstream outfile(results_file);
    outfile << "Iteration\tHeuristic\tOldEff\tOldDiv\tNewEff\tNewDiv\tTime\n";

    std::ofstream featfile(features_file);
    featfile << "Iteration";
    for (int i = 1; i <= 39; i++) featfile << ",F" << i;
    featfile << "\n";

    std::ofstream trace(trace_file);
    trace << "Iteration,Efficiency,Diversity,BestEfficiency,BestDiversity,Reward,Runtime\n";

    // ------------------------------------------------------------
    // MAIN LOOP
    // ------------------------------------------------------------
    clock_t total_start_time = clock();
    int iteration = 0;

    while ((double)(clock() - total_start_time) / CLOCKS_PER_SEC < max_time && iteration < max_iterations) {
        iteration++;
        int old_eff = cost_eff;
        int old_div = cost_div;

        // Random heuristic
        int selected_heuristic = heuristics[std::rand() % heuristics.size()];

        // Apply heuristic
        clock_t h_start = clock();
        ApplyHeuristic(selected_heuristic, team);
        double elapsed_time = (double)(clock() - h_start) / CLOCKS_PER_SEC;
        total_elapsed_time += elapsed_time;

        // Evaluate
        objective_Function1(team);
        int new_eff = f_cur;
        int new_div = f_cur_div;
        double reward = (new_eff > old_eff ? 1.0 : (new_eff < old_eff ? -1.0 : 0.0));
        rewards_record.push_back(reward);

        // --- Feature updates ---
        if (new_eff > old_eff) S_imp.insert(selected_heuristic);
        else if (new_eff < old_eff) S_wrs.insert(selected_heuristic);
        else S_eq.insert(selected_heuristic);
        S_nb.insert(selected_heuristic);
        heuristic_total_time[selected_heuristic] += elapsed_time;
        fitness_history.push_back(new_eff);

        // ACCEPTANCE
        if ((new_eff > cost_eff) && (new_div >= min_div)) {
            cost_eff = new_eff;
            cost_div = new_div;
            free_solution(previous_solution, num_node, num_team, num_each_t);
            previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);

            selected_heuristics.push_back(selected_heuristic);

            if (new_eff > best_cost_eff){
                best_cost_eff = new_eff;
                best_cost_div = new_div;
                free_solution(best_solution112, num_node, num_team, num_each_t);
                best_solution112 = deep_copy_solution(team, num_node, num_team, num_each_t);
                time_taken = elapsed_time;

                // Copy solution arrays (unchanged)
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        } else {
            // Revert
            for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = previous_solution[i][j];
        }

        // Logging
        heuristic_costs[selected_heuristic].push_back(new_eff);
        heuristic_times[selected_heuristic].push_back(elapsed_time);
        objective_values.push_back(cost_eff);
        diversity_values.push_back(cost_div);
        iteration_times.push_back(elapsed_time);

        // FILE OUTPUT
        outfile << iteration << "\t"
                << selected_heuristic << "\t"
                << old_eff << "\t" << old_div << "\t"
                << new_eff << "\t" << new_div << "\t"
                << elapsed_time << "\n";

        double runtime = (double)(clock() - total_start_time) / CLOCKS_PER_SEC;
        trace << iteration << "," << new_eff << "," << new_div << ","
              << best_cost_eff << "," << best_cost_div << ","
              << reward << "," << runtime << "\n";

        // FEATURES
        if (!fitness_history.empty()) {
            Features F = calculate_features(
                S_imp, S_wrs, S_eq, S_ac, S_uq, S_nb,
                fitness_history.front(),
                *std::max_element(fitness_history.begin(), fitness_history.end()),
                *std::min_element(fitness_history.begin(), fitness_history.end()),
                fitness_history,
                heuristic_total_time,
                heuristics.size(),
                iteration
            );

            featfile << iteration
                     << "," << F.F1 << "," << F.F2_nb_ge_1 << "," << F.F3_iterations
                     << "," << F.F4_total_nb << "," << F.F5_total_imp << "," << F.F6_total_wrs
                     << "," << F.F7_total_eq << "," << F.F8_total_ac << "," << F.F9_total_uq
                     << "," << F.F10 << "," << F.F11 << "," << F.F12 << "," << F.F13 << "," << F.F14
                     << "," << F.F15 << "," << F.F16 << "," << F.F17 << "," << F.F18 << "," << F.F19
                     << "," << F.F20 << "," << F.F21 << "," << F.F22 << "," << F.F23 << "," << F.F24
                     << "," << F.F25_max_th << "," << F.F26_min_th << "," << F.F27_avg_th << "," << F.F28_variance_th
                     << "," << F.F29 << "," << F.F30 << "," << F.F31 << "," << F.F32 << "," << F.F33
                     << "," << F.F34 << "," << F.F35 << "," << F.F36 << "," << F.F37 << "," << F.F38
                     << "," << F.F39 << "\n";
        }

        // Console
        std::cout << "Iteration: " << iteration
                  << " | Heuristic: " << selected_heuristic
                  << " | Efficiency: " << new_eff
                  << " | Diversity: " << new_div
                  << " | Best Eff: " << best_cost_eff
                  << " | Best Div: " << best_cost_div
                  << " | Reward=" << reward
                  << " | Time=" << elapsed_time << "s\n";
    }

    // ------------------------------------------------------------
    // CLOSE FILES AND SUMMARY
    // ------------------------------------------------------------
    outfile.close();
    featfile.close();
    trace.close();

    double average_objective = 0.0, average_diversity = 0.0, average_cpu_time = 0.0;
    if (!objective_values.empty())
        average_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0) / objective_values.size();
    if (!diversity_values.empty())
        average_diversity = std::accumulate(diversity_values.begin(), diversity_values.end(), 0.0) / diversity_values.size();
    if (!iteration_times.empty())
        average_cpu_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();

    std::cout << "\nFinal Summary\n"
              << "---------------------------\n"
              << "Best Efficiency : " << best_cost_eff << "\n"
              << "Best Diversity  : " << best_cost_div << "\n"
              << "Average Eff     : " << average_objective << "\n"
              << "Average Div     : " << average_diversity << "\n"
              << "Average IterTime: " << average_cpu_time << "s\n"
              << "Convergence Trace Saved: " << trace_file << "\n"
              << "===========================\n";

    check_best_solution();

    std::cout <<
        "=============================================================================\n"
        "Random Selection Hyper-Heuristic Framework Finished.\n"
        "=============================================================================\n";
}

// Implementation of Hyper-heuristic Based Multi-Armed Bandit (UCB) Selection Strategy for CMCEE
void Hyper_heuristic::MAB_Selection_Hyperheuristic_CMCEE(int max_time) {
    std::cout <<
        "=============================================================================\n"
        "Multi-Armed Bandit (UCB1) with Credit Assignment Selection Hyper-Heuristic Framework Start its Processes.\n"
        "=============================================================================\n";

    // ------------------------------------------------------------
    // INITIALIZATION
    // ------------------------------------------------------------
    //generate_initialrandom();
    int** previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** best_solution1    = deep_copy_solution(team, num_node, num_team, num_each_t);

    //objective_Function1(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    int best_cost_eff = cost_eff;
    int best_cost_div = cost_div;

    int* team_size = new int[num_team + 1];
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    // ------------------------------------------------------------
    // HEURISTICS AND MAB STRUCTURES
    // ------------------------------------------------------------
    std::vector<int> heuristics = {17,18,19,20,21,22,23};
    std::map<int, int> heuristic_selections;
    std::map<int, double> heuristic_credits;

    for (int h : heuristics) {
        heuristic_selections[h] = 0;
        heuristic_credits[h] = 0.0;
    }

    std::map<int, std::vector<int>> heuristic_costs;
    std::map<int, std::vector<double>> heuristic_times;
    std::vector<int> objective_values;
    std::vector<int> diversity_values;
    std::vector<double> iteration_times;
    std::vector<double> rewards_record;

    double total_elapsed_time = 0.0;
    int iteration = 0;

    // ------------------------------------------------------------
    // FILE SETUP
    // ------------------------------------------------------------
    std::filesystem::path folder =
        "D:/Datasets/RESULTS_OF_HH_MODELS/MAB_Convergence/";

    if (!std::filesystem::exists(folder))
        std::filesystem::create_directories(folder);

    std::filesystem::path results_file =
        folder / ("MAB_Selection_HH_CMCEE_" + instanceName + "_results.txt");
    std::filesystem::path trace_file =
        folder / ("MAB_Selection_HH_CMCEE_" + instanceName + "_Convergence_Trace.csv");

    std::ofstream outfile(results_file);
    outfile << "Iteration\tSelected_Heuristic\tOld_Eff\tOld_Div\tNew_Eff\tNew_Div\tTime_Taken(s)\tCreditDelta\n";

    std::ofstream trace(trace_file);
    trace << "Iteration,Efficiency,Diversity,BestEfficiency,BestDiversity,Reward,Runtime\n";

    clock_t total_start_time = clock();

    // ------------------------------------------------------------
    // MAIN LOOP
    // ------------------------------------------------------------
    while ((double)(clock() - total_start_time) / CLOCKS_PER_SEC < max_time) {
        iteration++;

        int old_eff = cost_eff;
        int old_div = cost_div;

        // ----- UCB1 SELECTION -----
        int selected_heuristic = -1;
        double best_ucb = -1.0;

        for (int h : heuristics) {
            double avg_credit = (heuristic_selections[h] > 0)
                ? (heuristic_credits[h] / heuristic_selections[h])
                : 0.0;
            double explore = std::sqrt(2.0 * std::log(iteration + 1) / (heuristic_selections[h] + 1e-6));
            double ucb_value = avg_credit + explore;
            if (ucb_value > best_ucb) {
                best_ucb = ucb_value;
                selected_heuristic = h;
            }
        }

        // ----- APPLY HEURISTIC -----
        clock_t h_start = clock();
        ApplyHeuristic(selected_heuristic, team);
        double elapsed_time = (double)(clock() - h_start) / CLOCKS_PER_SEC;
        total_elapsed_time += elapsed_time;

        // ----- EVALUATE -----
        //objective_Function1(team);
        int new_eff = f_cur;
        int new_div = f_cur_div;

        double f_c = static_cast<double>(old_eff);
        double f_n = static_cast<double>(new_eff);
        double eps = 1e-9;
        double frac = (f_c - f_n) / (f_c + f_n + eps);
        bool improved = (f_n > f_c);
        double reward = (improved ? 1.0 : (f_n < f_c ? -1.0 : 0.0));
        rewards_record.push_back(reward);

        // ----- CREDIT ASSIGNMENT -----
        if (improved) {
            heuristic_credits[selected_heuristic] += frac;
            if (heuristics.size() > 1) {
                double penalty = frac / (heuristics.size() - 1);
                for (int h : heuristics)
                    if (h != selected_heuristic) heuristic_credits[h] -= penalty;
            }
        } else {
            heuristic_credits[selected_heuristic] -= frac;
            if (heuristics.size() > 1) {
                double bonus = frac / (heuristics.size() - 1);
                for (int h : heuristics)
                    if (h != selected_heuristic) heuristic_credits[h] += bonus;
            }
        }

        heuristic_selections[selected_heuristic]++;

        // ----- ACCEPT/REJECT -----
        if ((new_eff > old_eff) && (new_div >= min_div)) {
            cost_eff = new_eff;
            cost_div = new_div;
            free_solution(previous_solution, num_node, num_team, num_each_t);
            previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);

            if (new_eff > best_cost_eff) {
                best_cost_eff = new_eff;
                best_cost_div = new_div;

                free_solution(best_solution1, num_node, num_team, num_each_t);
                best_solution1 = deep_copy_solution(team, num_node, num_team, num_each_t);
                time_taken = elapsed_time;

                // Copy best arrays
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        } else {
            // Revert
            for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = previous_solution[i][j];
        }

        // ----- RECORD RESULTS -----
        heuristic_costs[selected_heuristic].push_back(new_eff);
        heuristic_times[selected_heuristic].push_back(elapsed_time);
        objective_values.push_back(cost_eff);
        diversity_values.push_back(cost_div);
        iteration_times.push_back(elapsed_time);

        outfile << iteration << "\t"
                << selected_heuristic << "\t"
                << old_eff << "\t" << old_div << "\t"
                << new_eff << "\t" << new_div << "\t"
                << elapsed_time << "\t" << frac << "\n";

        double runtime = (double)(clock() - total_start_time) / CLOCKS_PER_SEC;
        trace << iteration << "," << new_eff << "," << new_div << ","
              << best_cost_eff << "," << best_cost_div << ","
              << reward << "," << runtime << "\n";

        // Console progress
        std::cout << "Iter: " << iteration
                  << " | LLH: " << selected_heuristic
                  << " | Eff: " << new_eff
                  << " | Div: " << new_div
                  << " | BestEff: " << best_cost_eff
                  << " | BestDiv: " << best_cost_div
                  << " | Reward=" << reward
                  << " | Delat_Credit=" << frac
                  << " | Time=" << elapsed_time << "s\n";
    }

    // ------------------------------------------------------------
    // WRAP-UP
    // ------------------------------------------------------------
    outfile.close();
    trace.close();

    double avg_eff = 0, avg_div = 0, avg_cpu = 0;
    if (!objective_values.empty())
        avg_eff = std::accumulate(objective_values.begin(), objective_values.end(), 0.0) / objective_values.size();
    if (!diversity_values.empty())
        avg_div = std::accumulate(diversity_values.begin(), diversity_values.end(), 0.0) / diversity_values.size();
    if (!iteration_times.empty())
        avg_cpu = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();

    std::cout << "\nFinal Summary\n"
              << "------------------------------------\n"
              << "Best Efficiency : " << best_cost_eff << "\n"
              << "Best Diversity  : " << best_cost_div << "\n"
              << "Average Eff     : " << avg_eff << "\n"
              << "Average Div     : " << avg_div << "\n"
              << "Average IterTime: " << avg_cpu << "s\n"
              << "Convergence Trace Saved: " << trace_file << "\n"
              << "------------------------------------\n";

    check_best_solution();

    std::cout <<
        "=============================================================================\n"
        "Multi-Armed Bandit (UCB1) Hyper-Heuristic Framework Finished.\n"
        "=============================================================================\n";
}

//Implementation of Hyper-heuristic Based Multi-Armed bandit Selection Strategies For Solving CMCEE
int fbest_solution_eff;

void Hyper_heuristic::HH_Choice_Function_Selection_CMCEE(int max_time) {
    // ==========================================================================
    // 1) PARAMETER & DATA STRUCTURE SETUP
    std::cout <<"============================================================================="<< std::endl;
    std::cout << "Choice Function Selection Hyper-heuristic Framework Start its Processes." << std::endl;
    std::cout <<"============================================================================="<< std::endl;

    double alpha = 1.0;
    double beta  = 0.5;
    double gamma = 0.5;

    std::map<int, double> heuristic_recent_perf;
    std::map<int, double> heuristic_total_perf;
    std::map<int, int>    heuristic_last_used_iter;
    std::map<int, int>    heuristic_usage_count;
    std::map<int, double> heuristic_total_time;
    std::map<int, int>    heuristic_improvement_count;
    std::map<int, double> LLH_score;

    std::vector<int>    objective_values;
    std::vector<double> iteration_times;
    std::vector<int>    selected_heuristics;

    // ==========================================================================
    // 2) INITIALIZE CURRENT SOLUTION & HEURISTICS
    // ==========================================================================
    //generate_initialrandom();
    objective_Function1(team);

    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    int best_iteration = 0;

    int** s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** s_best    = deep_copy_solution(team, num_node, num_team, num_each_t);
    int* team_size = new int[num_team + 1];

    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    int fbest_solution_eff = cost_eff;
    int fbest_solution_div = cost_div;

    std::vector<int> heuristics = {17, 18, 19, 20, 21, 22, 23};
    for (int h : heuristics) {
        heuristic_last_used_iter[h]   = 0;
        heuristic_recent_perf[h]      = 0.0;
        heuristic_total_perf[h]       = 0.0;
        heuristic_usage_count[h]      = 0;
        heuristic_total_time[h]       = 0.0;
        heuristic_improvement_count[h]= 0;
        LLH_score[h]                  = 0.0;
    }

    // ==========================================================================
    // 3) FILE & TIMER SETUP
    // ==========================================================================
    clock_t total_start_time = clock();
    int iteration = 0;

    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";
    if (!std::filesystem::exists(folder_path)) {
        try {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return;
        }
    }

    // Main result file
    std::filesystem::path results_file = folder_path / ("HH_Choice_Function_Selection_CMCEE_" + instanceName + "_results.txt");
    std::ofstream outfile(results_file);
    if (!outfile) {
        std::cerr << "Error: Unable to open results file: " << results_file << std::endl;
        return;
    }
    outfile << "Iteration\tSelected_Heuristic\tEfficiency\tDiversity\tTimeTaken(s)\tDelta\n";

    // NEW: Convergence Trace CSV
    std::filesystem::path trace_file = folder_path / ("HH_Choice_Function_Selection_CMCEE_" + instanceName + "_Convergence_Trace.csv");
    std::ofstream trace(trace_file);
    trace << "Iteration,Efficiency,Diversity,BestEfficiency,BestDiversity,Delta,Runtime\n";

    // ==========================================================================
    // 4) MAIN LOOP
    // ==========================================================================
    while (static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC < max_time) {
        iteration++;
        clock_t iteration_start_time = clock();

        // ---------------------- 4.1 Choice Function ----------------------
        std::map<int, double> choice_function_values;
        for (int h : heuristics) {
            double f1 = heuristic_recent_perf[h];
            double f2 = heuristic_total_perf[h];
            double f3 = static_cast<double>(iteration - heuristic_last_used_iter[h]);
            choice_function_values[h] = alpha * f1 + beta * f2 + gamma * f3 + LLH_score[h];
        }

        // ---------------------- 4.2 Select Heuristic ---------------------
        int selected_heuristic = -1;
        double max_cf_value = -std::numeric_limits<double>::infinity();
        for (int h : heuristics) {
            if (choice_function_values[h] > max_cf_value) {
                max_cf_value = choice_function_values[h];
                selected_heuristic = h;
            }
        }
        heuristic_last_used_iter[selected_heuristic] = iteration;

        // ---------------------- 4.3 Apply Heuristic ----------------------
        //int** s_temp = team;
        clock_t heuristic_start_time = clock();
        ApplyHeuristic(selected_heuristic, team);
        double heuristic_time = static_cast<double>(clock() - heuristic_start_time) / CLOCKS_PER_SEC;
        heuristic_total_time[selected_heuristic] += heuristic_time;
        heuristic_usage_count[selected_heuristic]++;

        objective_Function1(team);
        int cost_eff_temp = f_cur;
        int cost_div_temp = f_cur_div;

        // ---------------------- 4.4 Delta Calculation --------------------
        double f_c = static_cast<double>(cost_eff);
        double f_n = static_cast<double>(cost_eff_temp);
        if (f_c + f_n == 0.0) f_n += 1e-9;
        double Delta = (f_n - f_c) / (f_c + f_n);

        // ---------------------- 4.5 Credit Assignment --------------------
        if (Delta > 0) {
            LLH_score[selected_heuristic] += Delta;
            double punish_value = Delta / (heuristics.size() - 1);
            for (int h : heuristics)
                if (h != selected_heuristic) LLH_score[h] -= punish_value;

            heuristic_improvement_count[selected_heuristic]++;
            heuristic_recent_perf[selected_heuristic] = Delta;
            heuristic_total_perf[selected_heuristic] += Delta;
        } else {
            LLH_score[selected_heuristic] -= Delta;
            double reward_others_value = Delta / (heuristics.size() - 1);
            for (int h : heuristics)
                if (h != selected_heuristic) LLH_score[h] += reward_others_value;
            heuristic_recent_perf[selected_heuristic] = 0.0;
        }

        // ---------------------- 4.6 Accept/Reject ------------------------
        if ((cost_eff_temp > cost_eff) && (cost_div_temp >= min_div)) {
            free_solution(s_current, num_node, num_team, num_each_t);
            s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
            cost_eff  = cost_eff_temp;
            cost_div  = cost_div_temp;

            if ((cost_eff_temp > fbest_solution_eff) && (cost_div_temp >= min_div)) {
            free_solution(s_best, num_node, num_team, num_each_t);
            s_best = deep_copy_solution(team, num_node, num_team, num_each_t);

            fbest_solution_eff = cost_eff_temp;
            fbest_solution_div = cost_div_temp;
            best_iteration     = iteration;
            best_eff = cost_eff_temp;
            best_div = cost_div_temp;
            time_taken = heuristic_time;
            selected_heuristics.push_back(selected_heuristic);

            for (int m = 0; m < num_node; m++)
                fbest_solution[m] = best_solution[m];
            for (int m = 1; m <= num_team; m++) {
                eff_fbest[m] = eff_best[m];
                div_fbest[m] = div_best[m];
            }
        }
        }
        else {
            for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = s_current[i][j];
        }


        // ---------------------- 4.8 Logging ------------------------------
        objective_values.push_back(cost_eff_temp);
        double iteration_time = static_cast<double>(clock() - iteration_start_time) / CLOCKS_PER_SEC;
        iteration_times.push_back(iteration_time);

        outfile << iteration << "\t" << selected_heuristic << "\t"
                << cost_eff_temp << "\t" << cost_div_temp << "\t"
                << heuristic_time << "\t" << Delta << "\n";

        double runtime = static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC;
        trace << iteration << "," << cost_eff_temp << "," << cost_div_temp << ","
              << fbest_solution_eff << "," << fbest_solution_div << ","
              << Delta << "," << runtime << "\n";

        std::cout << "Iteration: " << iteration
                  << " | LLH: " << selected_heuristic
                  << " | Eff: " << cost_eff_temp
                  << " | Div: " << cost_div_temp
                  << " | BestEff: " << fbest_solution_eff
                  << " | BestDiv: " << fbest_solution_div
                  << " | Delta: " << Delta
                  << " | Time: " << heuristic_time << "s" << std::endl;
    }

    // ==========================================================================
    // 5) FINAL ANALYSIS & SUMMARY OUTPUT
    // ==========================================================================
    outfile.close();
    trace.close();

    double total_time = static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC;

    if (!objective_values.empty()) {
        average_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0)
                            / objective_values.size();
        worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    }

    double average_iteration_time = (!iteration_times.empty())
        ? std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size()
        : 0.0;

    int best_heuristic = -1, worst_heuristic = -1;
    double max_total_perf = -std::numeric_limits<double>::infinity();
    double min_total_perf =  std::numeric_limits<double>::infinity();

    for (int h : heuristics) {
        if (heuristic_total_perf[h] > max_total_perf) { max_total_perf = heuristic_total_perf[h]; best_heuristic = h; }
        if (heuristic_total_perf[h] < min_total_perf) { min_total_perf = heuristic_total_perf[h]; worst_heuristic = h; }
    }

    std::filesystem::path summary_file_path = folder_path / ("HH_Choice_Function_Selection_CMCEE_" + instanceName + "_summary.txt");
    std::ofstream summary_file(summary_file_path);
    if (summary_file) {
        summary_file << "Best Efficiency: " << fbest_solution_eff << "\n";
        summary_file << "Best Diversity: " << fbest_solution_div << "\n";
        summary_file << "Found at Iteration: " << best_iteration << "\n";
        summary_file << "Total Time: " << total_time << " seconds\n";
        summary_file << "Average Objective Function Value: " << average_objective << "\n";
        summary_file << "Worst Objective Function Value: " << worst_objective << "\n";
        summary_file << "Average Iteration Time: " << average_iteration_time << " seconds\n";
        summary_file << "Best Heuristic: " << best_heuristic << " (Total Performance: " << max_total_perf << ")\n";
        summary_file << "Worst Heuristic: " << worst_heuristic << " (Total Performance: " << min_total_perf << ")\n";
        summary_file.close();
        std::cout << "Summary results saved to " << summary_file_path << std::endl;
    }

    std::cout << "\n--- Summary ---\n";
    std::cout << "Best Efficiency: " << fbest_solution_eff << "\n";
    std::cout << "Best Diversity: " << fbest_solution_div << "\n";
    std::cout << "Total Time: " << total_time << " seconds\n";
    std::cout << "Average Objective Function Value: " << average_objective << "\n";
    std::cout << "Worst Objective Function Value: " << worst_objective << "\n";
    std::cout << "Average Iteration Time: " << average_iteration_time << " seconds\n";
    std::cout << "Convergence trace saved to: " << trace_file << "\n";
    check_best_solution();

    std::cout <<"============================================================================="<< std::endl;
    std::cout <<"Choice Function Selection Hyper-heuristic Framework Finished its Processes." << std::endl;
    std::cout <<"============================================================================="<< std::endl;
}

double calculate_mean(const std::vector<double>& values) {
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

double calculate_stddev(const std::vector<double>& values, double mean) {
    double sq_sum = std::accumulate(values.begin(), values.end(), 0.0,
                                    [mean](double acc, double val) {
                                        return acc + (val - mean) * (val - mean);
                                    });
    return std::sqrt(sq_sum / values.size());
}

// Implementation of Accept
bool Hyper_heuristic::Accept(int** Scurrent, int** Snew, double f_current, double f_new) {
    // Example acceptance criteria:
    // - If the new solution is better, accept it
    // - If worse, accept with a probability (simulated annealing-like)
    if(f_new > f_current) {
        return true;
    } else {
        double probability = exp((f_new - f_current) / 100.0); // Temperature-like factor
        double rand_val = (double)rand() / RAND_MAX;
        return rand_val < probability;
    }
}

// Implementation of updateBestSolution
void Hyper_heuristic::updateBestSolution(int** Scurrent, double f_current) {
    if(f_current > fbest_efficiency) {
        fbest_efficiency = f_current;
        fbest = fbest_efficiency;
        fbest_diversity = f_cur_div; // Update diversity as well

        // Update Sbest
        for(int i = 0; i <= num_team; i++) {
            for(int j = 0; j < num_node; j++) {
                Sbest[i][j] = Scurrent[i][j];
            }
        }
        team = Sbest;
        cout << "New Best Found: Efficiency = " << fbest_efficiency
             << ", Diversity = " << fbest_diversity << endl;
    }
}

// Implementation of TerminationCriterionSatisfied
bool Hyper_heuristic::TerminationCriterionSatisfied(int iter, int max_iter) {
    // Example criterion: maximum number of iterations
    return iter >= max_iter;
}

// Helper function to perform a deep copy of a 2D array
int** DeepCopySolution(int** original, int rows, int cols) {
    int** copy = new int*[rows];
    for (int i = 0; i < rows; ++i) {
        copy[i] = new int[cols];
        for (int j = 0; j < cols; ++j) {
            copy[i][j] = original[i][j];
        }
    }
    return copy;
}

void Hyper_heuristic::Q_Learning_Selection_Hyperheuristic_CMCEE(int max_time)
{
    std::cout <<
        "=============================================================================\n"
        "Q-Learning Selection Hyper-heuristic Framework Start its Processes.\n"
        "=============================================================================\n";

    // ------------------------------------------------------------
    // RNG SEEDING
    // ------------------------------------------------------------
    static bool seeded = false;

    // ------------------------------------------------------------
    // Q-LEARNING PARAMETERS
    // ------------------------------------------------------------
    const double alpha     = 0.30;
    const double gamma     = 0.50;
    double       epsilon   = 0.30;
    const double eps_decay = 0.99;
    const double eps_min   = 0.05;

    const int topK = 5;
    //const std::vector<int> heuristics = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23};
    const std::vector<int> heuristics = {17,19,20,21,22,23};

    // Q-table: Q[state][action] = value
    std::map<std::string, std::map<int,double>> Q_table;

    auto reward_from_delta = [](int d)->double {
        if (d > 0) return  1.0;
        if (d < 0) return -1.0;
        return 0.0;
    };

    // ------------------------------------------------------------
    // INITIAL SOLUTION
    // ------------------------------------------------------------
    //generate_initialrandom();
    display(team);
    objective_Function1(team);

    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    int** previous_solution =team;
    int** best_solution1 =team;
    //int** previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    //int** best_solution1    = deep_copy_solution(team, num_node, num_team, num_each_t);

    int best_cost_eff = cost_eff;
    int best_cost_div = cost_div;
    // Allocate team_size array globally
    team_size = new int[num_team + 1];

    // Assign team sizes
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;

    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;

    //std::cout << "Team 0 size = " << team_size[0] << "\n";
    //for (int i = 1; i <= num_team; ++i)
        //std::cout << "Team " << i << " size = " << team_size[i] << "\n";

    std::cout << "Initial objectives: Eff=" << cost_eff
              << "  Div=" << cost_div << "\n";

    std::vector<int>    Selected;
    std::vector<int>    objective_values;
    std::vector<int>    diversity_values;
    std::vector<double> iteration_times;

    std::map<int,double> heuristic_total_time;
    std::map<int,int>    heuristic_improvement_count;
    std::map<int,double> heuristic_rewards;
    std::map<int,int>    heuristic_usage_count;

    // ------------------------------------------------------------
    // STATE TRACKING
    // ------------------------------------------------------------
    int f_eff_max = 10000;
    int f_div_max = 50000;

    int accepted_moves = 0;
    int total_moves    = 0;

    std::vector<double> reward_hist;
    std::vector<int> div_values;
    std::vector<int> div_dummy;
    std::vector<int> eff_values;
    std::vector<int>rewards_record;

    double flex     = 0.0;
    double flex_max = 1.0;

    // ------------------------------------------------------------
    // FILE SETUP
    // ------------------------------------------------------------
    std::filesystem::path folder =
        "D:/Datasets/RESULTS_OF_QL_HH_MODELS/Q_learning_Convergence/";

    if (!std::filesystem::exists(folder))
        std::filesystem::create_directories(folder);

    std::filesystem::path result_file =
        folder / ("Q_Learning_CMCEE_" + instanceName + "_results.txt");
    std::filesystem::path trace_file =
        folder / ("Q_Learning_CMCEE_" + instanceName + "_Convergence_Trace.csv");

    std::ofstream outfile(result_file);
    outfile << "Iter\tHeuristic\tEff\tDiv\tDelta\tQvalue\tReward\tIterTime\n";

    std::ofstream trace(trace_file);
    trace << "Iteration,Efficiency,Diversity,BestEfficiency,BestDiversity,Reward,AvgQ,Runtime\n";


    // ------------------------------------------------------------
    // MAIN LOOP
    // ------------------------------------------------------------
    auto t_start = std::chrono::steady_clock::now();

    int iteration = 0;
    int itercount = 0;
    int max_iter  = 1000;
    double avg_Q_value = 0.0;
    int prev_eff = cost_eff;
    int prev_div = cost_div;

    while (true)
    {
        iteration++;
        auto t_iter_start = std::chrono::steady_clock::now();

        // time check
        double runtime =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        if (runtime >= max_time || itercount >= max_iter)
            break;

        // ------------------------------------------------------------
        // STATE
        // ------------------------------------------------------------
        StateFeatures s = compute_state_vector1(
           cost_eff, cost_div,
           prev_eff, prev_div,
           f_eff_max, f_div_max,
           iteration, max_iter,
           accepted_moves, total_moves,
           reward_hist,
           div_dummy,
           flex, flex_max);

        std::string key_state = discretize_state(s);

        if (!Q_table.count(key_state)) {
            for (int h : heuristics)
                Q_table[key_state][h] = 0.0;
        }

        // ------------------------------------------------------------
        // SELECT HEURISTIC (EPSILON-GREEDY)
        // ------------------------------------------------------------
        int i_next;
        double rv = static_cast<double>(rand()) / RAND_MAX;

        if (rv < epsilon)
            i_next = heuristics[rand() % heuristics.size()];
        else {
            double maxQ = -1e18;
            std::vector<int> ties;
            for (int h : heuristics) {
                double qv = Q_table[key_state][h];
                if (qv > maxQ) {
                    maxQ = qv;
                    ties = {h};
                }
                else if (fabs(qv - maxQ) < 1e-12) {
                    ties.push_back(h);
                }
            }
            i_next = ties[rand() % ties.size()];
        }

        // ------------------------------------------------------------
        // APPLY HEURISTIC
        // ------------------------------------------------------------
        auto t_hstart = std::chrono::steady_clock::now();
        ApplyHeuristic(i_next, team);
        double heuristic_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_hstart).count();

        heuristic_total_time[i_next] += heuristic_time;
        heuristic_usage_count[i_next]++;

        // ------------------------------------------------------------
        // EVALUATE
        // ------------------------------------------------------------
        objective_Function1(team);

        int new_eff = f_cur;
        int new_div = f_cur_div;

        objective_values.push_back(new_eff);
        diversity_values.push_back(new_div);

        int delta = new_eff - cost_eff;
        reward_hist.push_back(delta);
        div_values.push_back(new_div);

        double reward = reward_from_delta(delta);

        heuristic_rewards[i_next] += reward;
        if (delta > 0)
            heuristic_improvement_count[i_next]++;

        // ------------------------------------------------------------
        // NEXT STATE
        // ------------------------------------------------------------
        StateFeatures s_next = compute_state_vector1(
            new_eff, new_div,
            cost_eff, cost_div,
            f_eff_max, f_div_max,
            iteration, max_iter,
            accepted_moves, total_moves,
            reward_hist, div_values,
            flex, flex_max);

        std::string key_next = discretize_state(s_next);
        if (!Q_table.count(key_next)) {
            for (int h : heuristics) Q_table[key_next][h] = 0.0;
        }

        // ------------------------------------------------------------
        // Q UPDATE
        // ------------------------------------------------------------
        double maxQnext = -1e18;
        for (int h : heuristics)
            maxQnext = std::max(maxQnext, Q_table[key_next][h]);

        double &Qsa = Q_table[key_state][i_next];
        Qsa += alpha * (reward + gamma * maxQnext - Qsa);

        // ------------------------------------------------------------
        // ACCEPTANCE (Safe with Deep Copy)
        // ------------------------------------------------------------
         if ((new_eff > cost_eff) && (new_div >= min_div)) {
            // Accept the new solution
            cost_eff = new_eff;
            cost_div = new_div;
            previous_solution = team;
            accepted_moves++;
            // Optionally, deep copy team to previous_solution if needed
            Selected.push_back(i_next);
            if (new_eff > best_cost_eff) {
                // Update best solution
                best_cost_eff = new_eff;
                best_cost_div = new_div;
                best_solution1 = team;
                time_taken = heuristic_time;

                // Optionally, deep copy team to best_solution1 if needed
                // Update best solution arrays
                for (int m = 0; m < num_node; m++)
                     fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                     eff_fbest[m] = eff_best[m];
                     div_fbest[m] = div_best[m];
            }
            }
        }
        else {
            // Revert to previous_solution
            team = previous_solution;
        }

        total_moves++;

        // ------------------------------------------------------------
        // ITER TIME
        // ------------------------------------------------------------
        double iteration_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_iter_start).count();

        eff_values.push_back(new_eff);
        div_values.push_back(new_div);
        rewards_record.push_back(reward);
        iteration_times.push_back(iteration_time);

        // ------------------------------------------------------------
        // FILE OUTPUT
        // ------------------------------------------------------------
        if (outfile) {
            outfile << iteration << "\t"
                    << i_next << "\t"
                    << new_eff << "\t"
                    << new_div << "\t"
                    << heuristic_time << "\t"
                    << iteration_time << "\t"
                    << delta << "\t"
                    << Qsa << "\n";
        }
        trace << iteration << "," << new_eff << "," << new_div << ","
              << best_cost_eff << "," << best_cost_div << ","
              << reward << "," << avg_Q_value << "," << runtime << "\n";
        // ------------------------------------------------------------
        // CONSOLE OUTPUT (your preferred form)
        // ------------------------------------------------------------
        std::cout << "Iter: " << iteration
                  << " | Selected Heuristic: " << i_next
                  << " | Eff: " << new_eff
                  << " | Div: " << new_div
                  << " | Best Eff: " << best_cost_eff
                  << " | Best Div: " << best_cost_div
                  << " | Time Taken: " << heuristic_time
                  << " | Delta: " << delta
                  << " | Reward: " << reward
                  << " | Qsa[" << i_next << "]: " << Qsa
                  << std::endl;

        if (iteration % 50 == 0)
            //std::cout << "Epsilon decayed to: " << epsilon << std::endl;
        epsilon = std::max(epsilon * eps_decay, eps_min);
        itercount++;
    }

    outfile.close();
    trace.close();
    // ------------------------------------------------------------
    // FINAL STATISTICS
    // ------------------------------------------------------------
    double avg_eff = 0, avg_div = 0, avg_cpu = 0;

    if (!objective_values.empty())
        avg_eff = std::accumulate(objective_values.begin(),
                                  objective_values.end(), 0.0)
                  / objective_values.size();

    if (!diversity_values.empty())
        avg_div = std::accumulate(diversity_values.begin(),
                                  diversity_values.end(), 0.0)
                  / diversity_values.size();

    if (!iteration_times.empty())
        avg_cpu = std::accumulate(iteration_times.begin(),
                                  iteration_times.end(), 0.0)
                  / iteration_times.size();

    // ------------------------------------------------------------
    // FINAL PRINT
    // ------------------------------------------------------------
    std::cout <<
        "\nFinal Results\n"
        "--------------------------\n";

    std::cout << "Best Efficiency: " << best_cost_eff << "\n";
    std::cout << "Best Diversity : " << best_cost_div << "\n";
    std::cout << "Avg Efficiency : " << avg_eff << "\n";
    std::cout << "Avg Diversity  : " << avg_div << "\n";
    std::cout << "Avg Iter Time  : " << avg_cpu << "\n";

    std::cout << "Used LLH sequence: ";
    for (int h : Selected) std::cout << h << " ";
    std::cout << "\n";
    std::cout <<"Total Iterations: " << total_moves << "\n";
    std::cout <<"Convergence Trace Saved: " << trace_file << "\n";
    std::cout <<"=============================================================================\n";

    check_best_solution();

    std::cout <<
        "=============================================================================\n"
        "Q-Learning Selection Hyper-heuristic Framework Finished.\n"
        "=============================================================================\n";
}
// ==========================================================
//  SECTION 1: STATE FEATURE VECTOR AND DISCRETIZATION
// ==========================================================
// Table 3.5 implementation for s_t = [f_eff^norm, f_div^norm, Δeff, Δdiv, iter_ratio,
//                                    accept_ratio, reward_avg, div_std_norm, temp_norm, flex_norm]



// Rolling average helper
double Hyper_heuristic::rolling_average(const std::vector<double>& history, int window = 20) {
    if (history.empty()) return 0.0;
    int start = std::max(0, (int)history.size() - window);
    double sum = 0.0;
    for (int i = start; i < (int)history.size(); ++i) sum += history[i];
    return sum / (history.size() - start);
}

Hyper_heuristic::StateFeatures Hyper_heuristic::compute_state_vector1(
    int f_eff, int f_div,
    int prev_eff, int prev_div,
    int f_eff_max, int f_div_max,
    int iter, int max_iter,
    int accepted_moves, int total_moves,
    const std::vector<double>& reward_hist,
    const std::vector<int>& div_values,
    double flex, double flex_max)
{
    StateFeatures s;

    // Normalized efficiency and diversity
    s.f_eff_norm = (f_eff_max > 0) ? (double)f_eff / f_eff_max : 0.0;
    s.f_div_norm = (f_div_max > 0) ? (double)f_div / f_div_max : 0.0;

    // Normalized deltas
    s.delta_eff_norm =
        ((double)f_eff - prev_eff) / std::max(1.0, (double)f_eff_max);

    s.delta_div_norm =
        ((double)f_div - prev_div) / std::max(1.0, (double)f_div_max);

    // Progress ratio
    s.iter_ratio = (double)iter / std::max(1, max_iter);

    // Acceptance ratio
    s.accept_ratio =
        (total_moves > 0) ? (double)accepted_moves / total_moves : 0.0;

    // Rolling reward average for stability
    s.reward_avg = rolling_average(reward_hist, 20);

    // Diversity std-normalized
    s.div_std_norm = compute_div_std_norm(div_values, (double)f_div_max);

    // Flexibility normalisation
    s.flex_norm = (flex_max > 0.0) ? flex / flex_max : 0.0;

    return s;
}


Hyper_heuristic::StateFeatures Hyper_heuristic::compute_state_vector(
    int f_eff, int f_div,
    int prev_eff, int prev_div,
    int f_eff_max, int f_div_max,
    int iter, int max_iter,
    int accepted_moves, int total_moves,
    const std::vector<double>& reward_hist,
    const std::vector<int>& div_values,
    double temp, double temp0,
    double flex, double flex_max)
{
    StateFeatures s;
    s.f_eff_norm   = (f_eff_max > 0) ? (double)f_eff / f_eff_max : 0.0;
    s.f_div_norm   = (f_div_max > 0) ? (double)f_div / f_div_max : 0.0;
    s.delta_eff_norm = ((double)f_eff - prev_eff) / std::max(1.0, (double)f_eff_max);
    s.delta_div_norm = ((double)f_div - prev_div) / std::max(1.0, (double)f_div_max);
    s.iter_ratio     = (double)iter / std::max(1, max_iter);
    s.accept_ratio   = (total_moves > 0) ? (double)accepted_moves / total_moves : 0.0;
    s.reward_avg     = rolling_average(reward_hist, 20);
    s.div_std_norm   = compute_div_std_norm(div_values, (double)f_div_max);
    s.temp_norm      = (temp0 > 0) ? temp / temp0 : 0.0;
    s.flex_norm      = (flex_max > 0) ? flex / flex_max : 0.0;
    return s;
}

// Compute normalized standard deviation of diversity values
double Hyper_heuristic::compute_div_std_norm(const std::vector<int>& div_values, double div_max) {
    if (div_values.empty() || div_max <= 0.0) return 0.0;

    double mean = 0.0;
    for (int v : div_values) mean += v;
    mean /= div_values.size();

    double variance = 0.0;
    for (int v : div_values) variance += (v - mean) * (v - mean);
    variance /= div_values.size();

    double std_dev = std::sqrt(variance);
    return std_dev / div_max;  // Normalize by maximum diversity
}

// Convert the continuous feature vector into a discrete state key (Option A)
std::string Hyper_heuristic::discretize_state(const StateFeatures& s) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "S";
    std::vector<double> feats = {
        s.f_eff_norm, s.f_div_norm, s.delta_eff_norm, s.delta_div_norm,
        s.iter_ratio, s.accept_ratio, s.reward_avg,
        s.div_std_norm, s.temp_norm, s.flex_norm
    };
    for (double f : feats) {
        int bin = static_cast<int>(std::round(f * 10));  // discretize into 0–10 bins
        oss << "_" << std::clamp(bin, 0, 10);
    }
    return oss.str();
}

// ==========================================================
//  SECTION 2: LS / OP / MA ADAPTERS + REWARD & ACCEPTANCE UTILITIES
// ==========================================================

// ----------  LS Algorithm Adapters (Level 1)  ----------
enum LSAlgo { LS_SA = 1, LS_ILS, LS_TS, LS_GD, LS_LAHC, LS_BASIC };

// Connects LS selection to your actual algorithm implementations
/*int** Hyper_heuristic::run_LS(LSAlgo algo, int** team) {
    switch (algo) {
        case LS_SA:    return simulated_annealing();       // your SA implementation
        case LS_ILS:   return iterated_local_search();     // your ILS implementation
        case LS_TS:    return fits();     // use TS if available
        case LS_GD:    return great_deluge_algorithm();     // placeholder for Great Deluge
        case LS_LAHC:  return late_acceptance_hill_climbing();     // placeholder for LAHC
        case LS_BASIC: return local_search();     // basic local search
        default:       return team;
    }
}*/
int** Hyper_heuristic::Apply_LS_OP(int LSi, int OPj, int** Sstart)
{
    // Start from the provided solution
    team = Sstart;
    double SA_temp = 100.0;
    // SA parameters
    double T0 = SA_temp;
    double alpha = cooling_rate;

    // LAHC parameter
    int Lwindow = 50;

    // GD parameters
    double GD_level0     = f_cur;
    double GD_decayRate  = 0.98;

    // ---- Selection of LS Algorithm ----
    if (LSi == 1)  {
        // Simulated Annealing
        return SimulatedAnnealing(Sstart, OPj, T0, alpha);
    }
    else if (LSi == 2) {
        // Iterated Local Search
        return IteratedLocalSearch(Sstart, OPj);
    }
    else if (LSi == 3) {
        // Late Acceptance Hill Climbing
        return LateAcceptance(Sstart, OPj, Lwindow);
    }
    else if (LSi == 4) {
        // Great Deluge
        return GreatDeluge(Sstart, OPj, GD_level0, GD_decayRate);
    }
    else if (LSi == 5) {
        // Tabu-based Fits
        return TabuSearch(Sstart, OPj);
    }
    else {
        // Fallback: return Sstart unchanged
        return Sstart;
    }
}
// ----------  Operator Adapters (Level 2)  ----------
// LLH 1–14 → atomic neighborhood structures
int** Hyper_heuristic::apply_LLHop(int op_id, int** sol) {
    switch (op_id) {
        case  1: return LLH1(sol);
        case  2: return LLH2(sol);
        case  3: return LLH3(sol);
        case  4: return LLH4(sol);
        case  5: return LLH5(sol);
        case  6: return LLH6(sol);
        case  7: return LLH7(sol);
        case  8: return LLH8(sol);
        case  9: return LLH9(sol);
        case 10: return LLH10(sol);
        case 11: return LLH17(sol);
        case 12: return LLH12(sol);
        case 13: return LLH13(sol);
        case 14: return LLH14(sol);
        default: return sol;
    }
}

// ----------  Move-Acceptance (Level 3)  ----------
enum MA_Strategy { MA_ONLY_IMPROVE = 1, MA_ACCEPT_ALL, MA_SA, MA_R2R, MA_THRESHOLD };

// Hyper-parameters (can be adjusted dynamically in the main loop)
double SA_Temp       = 100.0;     // starting temperature
double SA_Cooling    = 0.995;
double R2R_Threshold = 5.0;
double TH_Threshold  = 0.15;
double decay_R2R     = 0.999;
double decay_TH      = 0.999;
int    R2R_record_eff = std::numeric_limits<int>::lowest();

// Decide whether to accept the new solution
bool Hyper_heuristic::accept_move(
        MA_Strategy ma,
        int cur_eff, int cur_div,
        int new_eff, int new_div,
        double min_div)
{
    if (ma == MA_ONLY_IMPROVE)
        return (new_eff > cur_eff) && (new_div >= min_div);

    if (ma == MA_ACCEPT_ALL)
        return (new_div >= min_div);

    if (ma == MA_SA) {
        if (new_div < min_div) return false;
        int diff = new_eff - cur_eff;
        if (diff >= 0) return true;
        double prob = std::exp((double)diff / std::max(1e-9, SA_Temp));
        return ((double)rand() / RAND_MAX) < prob;
    }

    if (ma == MA_R2R) {
        if (new_div < min_div) return false;
        if (R2R_record_eff == std::numeric_limits<int>::lowest())
            R2R_record_eff = cur_eff;
        return (new_eff >= R2R_record_eff - (int)std::floor(R2R_Threshold));
    }

    if (ma == MA_THRESHOLD) {
        if (new_div < min_div) return false;
        return (new_eff >= cur_eff - (int)std::floor(TH_Threshold));
    }

    return false;
}

// ----------  Reward & Delta Utilities  ----------
inline double Hyper_heuristic::compute_delta(int new_eff, int old_eff)
{
    return (double)new_eff - (double)old_eff;
}


// Reward derived from delta
inline double Hyper_heuristic::reward_from_delta(double delta)
{
    if (delta > 0)  return 1.0;
    if (delta == 0) return 0.0;
    return -1.0;
}

void Hyper_heuristic::TriLevel_HH_Qlearning_CMCEE(int max_time) {

    std::cout << "===================================================================\n";
    std::cout << "Tri-Level Hyper-Heuristic Framework (LS->OP->MA)\n";
    std::cout << "===================================================================\n";

    // ---------------- Initialization ----------------
    std::cout << "Initial Solution:\n";
    //generate_initialrandom();
    objective_Function(team);
    //display(team);
    std::cout << "\n===================================================================\n\n";
    std::cout << "Initial objective functions: Eff=" << f_cur << " | Div=" << f_cur_div << "\n\n";
    std::cout << "===================================================================\n\n";

    int cost_eff = f_cur, cost_div = f_cur_div;
    int best_eff = cost_eff, best_div = cost_div;
    int prev_eff = cost_eff, prev_div = cost_div;
    int f_eff_max = cost_eff, f_div_max = cost_div;

    //int** s_current = team;
    int** s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
    //int** best_sol  =  team;
    int** best_sol  = deep_copy_solution(team, num_node, num_team, num_each_t);

    team_size = new int[num_team + 1];
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    int best_LS = -1, best_OP = -1;
    MA_Strategy best_MA = MA_ONLY_IMPROVE;
    double time_best_found = 0.0;

    // Q-tables
    std::map<std::string, std::map<int,double>> Q_LS;
    std::map<std::string, std::map<int,double>> Q_OP;
    std::map<std::string, std::map<int,double>> Q_MA;

    std::vector<double> reward_hist;
    std::vector<int> div_values;
    int accepted_moves = 0, total_moves = 0;

    // Parameters
    double alpha = 0.7, gamma = 0.6;
    double eps_LS = 1.0, eps_OP = 1.0, eps_MA = 1.00;
    double eps_decay = 0.99, eps_min = 0.05;
	double F = 0.15;
    double SA_temp = 1.0, SA_init = 1.0;
    double flex = 1.0, flex_max = 1.0;
    // -------------------------------------------------
    // LS / OP / MA STATISTICS (SUMMARY ONLY)
    // -------------------------------------------------
    std::map<int,int>    LS_usage, OP_usage, MA_usage;
    std::map<int,int>    LS_improve, OP_improve, MA_improve;
    std::map<int,double> LS_reward, OP_reward, MA_reward;
    std::map<int,double> LS_time,   OP_time,   MA_time;


    // File setup
    std::filesystem::path folder_path =
        "D:/Datasets/TRI_LEVEL_HH_MODELS/InstanceSeparateHH_TriLevel/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path file_path =
        folder_path / ("TriLevel_Q_HH_CMCEE_" + instanceName + "_results.txt");
    std::ofstream outfile(file_path);
    outfile << "Iter\tLS\tOP\tMA\tEff\tDiv\tBestEff\tBestDiv\tDelta\tReward\tStepSec\n";

    // NEW: Convergence Trace
    std::filesystem::path conv_path =
        folder_path / ("TriLevel_Q_HH_CMCEE_" + instanceName + "_Convergence_Trace.csv");
    std::ofstream conv_file(conv_path);
    conv_file << "Iteration,Eff,Div,BestEff,BestDiv,Reward,Delta,TimeSec\n";

    auto t0 = std::chrono::steady_clock::now();
    double total_time = 0.0;
    int iteration = 0;
    int max_iter = 1000;

    static std::map<int,int> op_counts;
    static std::map<int,double> op_rewards;

    // ====================== MAIN LOOP ======================
    while (true) {
        iteration++;
        total_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        if (total_time >= max_time || iteration >= max_iter)
            break;

        // State representation
        StateFeatures s = compute_state_vector(
            cost_eff, cost_div, prev_eff, prev_div,
            f_eff_max, f_div_max, iteration, max_iter,
            accepted_moves, total_moves, reward_hist, div_values,
            SA_temp, SA_init, flex, flex_max);
        std::string key_state = discretize_state(s);

        // ---------------- Level 1: LS selection ----------------
        if (Q_LS.find(key_state) == Q_LS.end())
            for (int a = 1; a <= 5; a++)
                Q_LS[key_state][a] = 0.0;

        int LSi;
        double rand01 = (double)rand() / RAND_MAX;
        if (rand01 < eps_LS)
            LSi = 1 + (rand() % 5);
        else
            LSi = std::max_element(Q_LS[key_state].begin(), Q_LS[key_state].end(),
                [](auto &x, auto &y){return x.second < y.second;})->first;
        if (LSi < 1 || LSi > 5) LSi = 1;

        // ---------------- Level 2: OP selection (UCB) ----------------
        double total_counts = 0;
        for (auto &p : op_counts)
            total_counts += p.second;

        int selected_op = 1;
        double best_ucb = -1e18;
        double c = 1.0;
        for (int op = 1; op <= 8; op++) {
            double rew = op_rewards[op];
            double cnt = op_counts[op];
            double explore = c * std::sqrt((2.0 * std::log(total_counts + 1)) / (cnt + 1.0));
            double ucb = rew + explore;
            if (ucb > best_ucb) { best_ucb = ucb; selected_op = op; }
        }

        // ---------------- Level 3: MA selection ----------------
        if (Q_MA.find(key_state) == Q_MA.end())
            for (int a = 1; a <= 5; a++)
                Q_MA[key_state][a] = 0.0;

        MA_Strategy aMA;
        double randMA = (double)rand() / RAND_MAX;
        if (randMA < eps_MA)
            aMA = (MA_Strategy)(1 + rand() % 5);
        else
            aMA = (MA_Strategy)std::max_element(Q_MA[key_state].begin(),
                Q_MA[key_state].end(), [](auto &x, auto &y){return x.second < y.second;})->first;
        if (aMA < 1 || aMA > 5)
            aMA = MA_ONLY_IMPROVE;

        // ---------------- Apply LS + OP ----------------
        auto step_start = std::chrono::steady_clock::now();
        Apply_LS_OP(LSi, selected_op, team);
        objective_Function1(team);
        int new_eff = f_cur, new_div = f_cur_div;
        total_moves++;

        // ---------------- Q-Updates ----------------
        double delta = compute_delta(new_eff, prev_eff);
        double r = reward_from_delta(delta);
        reward_hist.push_back(r);

        // ---------------- Update Statistics ----------------
        LS_usage[LSi]++;
        OP_usage[selected_op]++;
        MA_usage[aMA]++;
        LS_reward[LSi] += r;
        OP_reward[selected_op] += r;
        MA_reward[aMA] += r;

        // ---------------- Acceptance ----------------
        bool accepted = accept_move(aMA, cost_eff, cost_div, new_eff, new_div, min_div);

        if (accepted) {
            prev_eff = cost_eff;
            prev_div = cost_div;
            cost_eff = new_eff;
            cost_div = new_div;
            accepted_moves++;
            LS_improve[LSi]++;
            OP_improve[selected_op]++;
            MA_improve[aMA]++;

            if (new_eff > best_eff && new_div >= min_div) {
                best_eff = new_eff;
                best_div = new_div;
                free_solution(s_current, num_node, num_team, num_each_t);
                s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
                free_solution(best_sol, num_node, num_team, num_each_t);
                best_sol = deep_copy_solution(team, num_node, num_team, num_each_t);
                //s_current = team;
                //best_sol = team;
                best_LS = LSi;
                best_OP = selected_op;
                best_MA = aMA;
                time_best_found = total_time;
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        } else {
            for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = s_current[i][j];
                    //team = s_current;
        }

        double step_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - step_start).count();
        LS_time[LSi] += step_sec;
        OP_time[selected_op] += step_sec;
        MA_time[aMA] += step_sec;



        std::string next_state = discretize_state(
            compute_state_vector(cost_eff, cost_div, prev_eff, prev_div, f_eff_max, f_div_max,
                                 iteration, max_iter, accepted_moves, total_moves,
                                 reward_hist, div_values, SA_temp, SA_init, flex, flex_max));

        double max_next_LS = -1e18;
        for (auto &p : Q_LS[next_state])
            max_next_LS = std::max(max_next_LS, p.second);
        Q_LS[key_state][LSi] += alpha * (r + gamma * max_next_LS - Q_LS[key_state][LSi]);

        double max_next_MA = -1e18;
        for (auto &p : Q_MA[next_state])
            max_next_MA = std::max(max_next_MA, p.second);
        Q_MA[key_state][aMA] += alpha * (r + gamma * max_next_MA - Q_MA[key_state][aMA]);

        op_counts[selected_op]++;
        op_rewards[selected_op] += r;

        eps_LS = std::max(eps_LS * eps_decay, eps_min);
        eps_MA = std::max(eps_MA * eps_decay, eps_min);

        outfile << iteration << "\t" << LSi << "\t" << selected_op << "\t" << aMA << "\t"
                << new_eff << "\t" << new_div << "\t" << best_eff << "\t" << best_div << "\t"
                << delta << "\t" << r << "\t" << step_sec << "\n";

        conv_file << iteration << "," << new_eff << "," << new_div << ","
                  << best_eff << "," << best_div << "," << r << "," << delta
                  << "," << total_time << "\n";

        std::cout << "Iter " << iteration
                  << " | LS=" << LSi
                  << " | OP=" << selected_op
                  << " | MA=" << aMA
                  << " | Eff=" << new_eff
                  << " | Div=" << new_div
                  << " | BestEff=" << best_eff
                  << " | BestDiv=" << best_div
                  << " | Delta=" << delta
                  << " | r=" << r
                  << " | t=" << total_time << "s\n";
    }

    // ---------------- Close Files ----------------
    outfile.close();
    conv_file.close();
    // ====================== SUMMARY FILE ======================
    double runtime = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    std::filesystem::path summary_file =
        folder_path / ("TriLevel_Q_HH_CMCEE_" + instanceName + "_summary.txt");
    std::ofstream summary(summary_file);

    summary << "Final Best Objectives:\n";
    summary << "Efficiency: " << best_eff << "\n";
    summary << "Diversity:  " << best_div << "\n\n";

    summary << "Best Solution Found By:\n";
    summary << "  LS: " << best_LS << "\n";
    summary << "  OP: " << best_OP << "\n";
    summary << "  MA: " << best_MA << "\n";
    summary << "  Time to Find: " << time_best_found << " seconds\n\n";

    summary << "LS Usage Summary:\n";
    for (auto &p : LS_usage)
        summary << "LS " << p.first
                << " | Used: " << p.second
                << " | Improve: " << LS_improve[p.first]
                << " | Reward: " << LS_reward[p.first]
                << " | Time: " << LS_time[p.first] << "\n";

    summary << "\nOP Usage Summary:\n";
    for (auto &p : OP_usage)
        summary << "OP " << p.first
                << " | Used: " << p.second
                << " | Improve: " << OP_improve[p.first]
                << " | Reward: " << OP_reward[p.first]
                << " | Time: " << OP_time[p.first] << "\n";

    summary << "\nMA Usage Summary:\n";
    for (auto &p : MA_usage)
        summary << "MA " << p.first
                << " | Used: " << p.second
                << " | Improve: " << MA_improve[p.first]
                << " | Reward: " << MA_reward[p.first]
                << " | Time: " << MA_time[p.first] << "\n";


    summary.close();
    check_best_solution();
    std::cout << "===================================================================\n";
    std::cout << "Tri-Level Q-Learning HH Finished.\n";
    std::cout << "Runtime: " << runtime << " sec\n";
    std::cout << "Results saved to: " << file_path << "\n";
    std::cout << "Convergence trace saved to: " << conv_path << "\n";
    std::cout << "Summary saved to: " << summary_file << "\n";
    std::cout << "===================================================================\n";

}
// Enumeration for menu options
enum MenuOptions {
    EXIT_OPTION = 0,
    // Existing Hyper-Heuristic Methods
    Q_LEARNING_SELECTION = 1,
    HH_CHOICE_FUNCTION_SELECTION,
    RANDOM_SELECTION,
    MAB_SELECTION,
    GREEDY_SELECTION,
    // Single Optimization Algorithms
    ITERATED_LOCAL_SEARCH = 7,
    SIMULATED_ANNEALING,
    TABU_SEARCH,
    FEASIBLE_LOCAL_SEARCH,
    INFEASIBLE_LOCAL_SEARCH,
    MEMETIC_ALGORITHM,
    GREAT_DELUGE,
    Late_Acceptance,
    Guided_Local,
    
};

void Hyper_heuristic::execute_algorithm(int runs, const std::string& algorithm_name) {
    d_min = 1.05;
    time_limit = num_node / 2;

    ffbest = std::numeric_limits<int>::min();
    double stat_best  = std::numeric_limits<double>::lowest();
    double stat_avg   = 0.0;
    double stat_worst = std::numeric_limits<double>::max();
    double avg_time   = 0.0;

    // ---------------------------------------------------------------------
    // Create results directory if not exists
    // ---------------------------------------------------------------------
    std::filesystem::path base_folder = "D:/Datasets/MHs_Algorithm_Results/";
    if (!std::filesystem::exists(base_folder))
        std::filesystem::create_directories(base_folder);

    // File paths
    std::filesystem::path results_path   = base_folder / ("Results_" + algorithm_name + ".csv");
    std::filesystem::path bestsol_path   = base_folder / ("Best_Solution_" + algorithm_name + ".txt");

    std::ofstream results_file(results_path, std::ios::app);
    std::ofstream best_sol_file(bestsol_path, std::ios::app);

    if (!results_file.is_open() || !best_sol_file.is_open()) {
        std::cerr << "Error opening output files in: " << base_folder << std::endl;
        return;
    }

    // Header for CSV (only if file empty)
    if (results_file.tellp() == 0)
        results_file << "Run,Algorithm,Best_Fitness,Avg_Fitness,Worst_Fitness,Avg_Time(sec)\n";

    int** solution = nullptr;

    // ---------------------------------------------------------------------
    // Main execution over runs
    // ---------------------------------------------------------------------
    for (int run = 0; run < runs; run++) {
        fbest      = std::numeric_limits<int>::min();
        f_best_inn = std::numeric_limits<int>::min();

        generate_initial();

        double t0 = static_cast<double>(clock());
        int fitness = std::numeric_limits<int>::min();

        if      (algorithm_name == "ILS")  { solution = iterated_local_search();            fitness = fbest; }
        else if (algorithm_name == "SA")   { solution = simulated_annealing();              fitness = fbest; }
        else if (algorithm_name == "TS")   { solution = fits();                             fitness = f_best_inn; }
        else if (algorithm_name == "FLS")  { fitness  = feasible_local_search(); }
        else if (algorithm_name == "IFLS") { solution = infeasible_local_search();          fitness = f_best_inn; }
        else if (algorithm_name == "MA")   { solution = memetic();                          fitness = fbest; }
        else if (algorithm_name == "GD")   { solution = great_deluge_algorithm();           fitness = fbest; }
        else if (algorithm_name == "LAHC") { solution = late_acceptance_hill_climbing();    fitness = fbest; }
        else if (algorithm_name == "GLS")  { fitness  = guided_local_search(); }
        else {
            std::cerr << "Unknown algorithm: " << algorithm_name << ". Skipping run.\n";
            continue;
        }

        double t1 = static_cast<double>(clock());
        double elapsed_time = (t1 - t0) / CLOCKS_PER_SEC;
        avg_time += elapsed_time;

        // Track global best
        if (fitness > ffbest) {
            ffbest = fitness;

            for (int m = 0; m < num_node; m++)
                fbest_solution[m] = best_solution[m];
            for (int m = 1; m <= num_team; m++) {
                eff_fbest[m] = eff_best[m];
                div_fbest[m] = div_best[m];
            }
        }

        if (fitness > stat_best)  stat_best  = fitness;
        if (fitness < stat_worst) stat_worst = fitness;
        stat_avg += fitness;

        std::cout << "Run " << run + 1
                  << ": f = " << fitness
                  << ", ffbest = " << ffbest
                  << ", time = " << elapsed_time << " sec\n";
    }

    stat_avg /= runs;
    avg_time /= runs;

    // ---------------------------------------------------------------------
    // Save summarized results to CSV
    // ---------------------------------------------------------------------
    results_file << runs << "," << algorithm_name << ","
                 << stat_best << "," << stat_avg << ","
                 << stat_worst << "," << avg_time << "\n";
    results_file.close();
        std::cout << "\n"; std::cout << "------------------------------------------------\n";
        std::cout << "The best solution:\n";
     for (int t = 1; t <= num_team; t++)
         std::cout << "eff[" << t << "] = " << eff_fbest[t] << " ";
         std::cout << "\n"; for (int t = 1; t <= num_team; t++)
         std::cout << "div[" << t << "] = " << div_fbest[t] << " ";
         std::cout << "\n------------------------------------------------\n";
         std::cout << "Best Solution Vector:\n";
     for (int i = 0; i < num_node; i++)
         std::cout << fbest_solution[i] << " ";
         std::cout << "\n";
    // ---------------------------------------------------------------------
    // Save best solution vector
    // ---------------------------------------------------------------------
    best_sol_file << "Algorithm: " << algorithm_name << "\n";
    best_sol_file << "Best Fitness: " << ffbest << "\n";
    best_sol_file << "Efficiency per Team: ";
    for (int t = 1; t <= num_team; t++) best_sol_file << eff_fbest[t] << " ";
    best_sol_file << "\nDiversity per Team: ";
    for (int t = 1; t <= num_team; t++) best_sol_file << div_fbest[t] << " ";
    best_sol_file << "\nBest Solution Vector:\n";
    for (int i = 0; i < num_node; i++) best_sol_file << fbest_solution[i] << " ";
    best_sol_file << "\n--------------------------------------------------\n";
    best_sol_file.close();

    // ---------------------------------------------------------------------
    // Display on console
    // ---------------------------------------------------------------------
    check_best_solution();

    std::cout << "==================================================\n";
    std::cout << "Algorithm: " << algorithm_name << " finished over "
              << runs << " runs.\n";
    std::cout << "Best: " << stat_best
              << " | Avg: " << stat_avg
              << " | Worst: " << stat_worst
              << " | Avg Time: " << avg_time << " sec\n";
    std::cout << "Results saved to: " << results_path << "\n";
    std::cout << "==================================================\n";
}

// Function to handle single optimization algorithm selection and execution
void Hyper_heuristic::runSingleOptimizationAlgorithm(int choice, int max_time)
    {

        std::string algorithm_name;
        switch(choice) {
            case ITERATED_LOCAL_SEARCH:
                std::cout << "Running Iterated Local Search (ILS)..." << std::endl;
                algorithm_name = "ILS";
                break;

            case SIMULATED_ANNEALING:
                std::cout << "Running Simulated Annealing (SA)..." << std::endl;
                algorithm_name = "SA";
                break;

            case TABU_SEARCH:
                std::cout << "Running Tabu Search (TS)..." << std::endl;
                algorithm_name = "TS";
                break;

            case FEASIBLE_LOCAL_SEARCH:
                std::cout << "Running Feasible Local Search (FLS)..." << std::endl;
                algorithm_name = "FLS";
                break;

            case INFEASIBLE_LOCAL_SEARCH:
                std::cout << "Running Infeasible Local Search (IFLS)..." << std::endl;
                algorithm_name = "IFLS";
                break;

            case MEMETIC_ALGORITHM:
                std::cout << "Running Memetic Algorithm (MA)..." << std::endl;
                algorithm_name = "MA";
                break;

           case GREAT_DELUGE:
                std::cout << "Running GREAT_DELUGE..." << std::endl;
                algorithm_name = "GD";
                break;
            case Late_Acceptance:
                std::cout << "Running Late_Acceptance_Hill_Climbing..." << std::endl;
                algorithm_name = "LAHC";
                break;
            case Guided_Local:
                std::cout << "Running Guided_Local_Search..." << std::endl;
                algorithm_name = "GLS";
                break;
            default:
                std::cout << "Invalid single optimization algorithm choice." << std::endl;
                return;
        }

        execute_algorithm(31, algorithm_name); // Example: 31 runs
    }

// Helper function to format the filename with leading zeros
string format_filename(int instance_number, const string &dataset_dir = "D:/Datasets/", const string &pattern = "test-n100m10t8.dat")
{
    // Convert instance number to string with leading zero if necessary
    string instance_str = to_string(instance_number);
    if(instance_str.length() < 2){
        instance_str = "0" + instance_str;
    }

    // Construct the full file path
    return dataset_dir + instance_str + pattern;
}

std::string parseInstanceName(const std::string &filename) {
    // Match and extract parts from the filename
    std::regex pattern(R"((\d{2})test-n(\d+)m(\d+)t(\d+)\.dat)");
    std::smatch match;

    if (std::regex_match(filename, match, pattern)) {
        std::string instance = match[1].str() + "-P" + match[2].str() + "T" + match[3].str() + "M" + match[4].str();
        return instance;
    } else {
        return ""; // Return an empty string if parsing fails
    }
}


int main(int argc, char *argv[])
{
	//srand((unsigned)time(NULL));
	if (argc < 1)	{
		cout << "usage: MA.exe input_file";
		exit(1);
	}

    std::string datasetDir   = "/workspaces/TL-HH/Datasets";
    std::string resultsDir   = "D:\\Datasets\\RESULTS_OF_HH_MODELSParameters\\";

    std::vector<std::string> datasetFiles;
    for (const auto &entry : fs::directory_iterator(datasetDir)) {
        if (entry.is_regular_file()) {
            datasetFiles.push_back(entry.path().string());
        }
    }

    Hyper_heuristic H;
    int max_time = 600; // time limit for all runs

    // ====================================================
    //              MAIN MENU LOOP
    // ====================================================
    while (true) {

        // ------------------ MENU -----------------------
        std::cout << "\n===========================================\n";
        std::cout << "         HYPER-HEURISTIC MAIN MENU\n";
        std::cout << "===========================================\n";
        std::cout << "Select ONE method to run on ALL instances:\n";
        std::cout << " 1: Q-Learning Selection HH\n";
        std::cout << " 2: Choice Function HH\n";
        std::cout << " 3: Random Selection HH\n";
        std::cout << " 4: Multi-Armed Bandit HH\n";
        std::cout << " 5: Greedy Selection HH\n";
        std::cout << " 6: Tri-Level HH (QL + MAB + QL)\n";
        std::cout << " 7: Single Optimization Algorithm\n";
        std::cout << " 0: EXIT PROGRAM\n";
        std::cout << "-------------------------------------------\n";
        std::cout << "Enter your choice: ";

        int choice = 0;
        std::cin >> choice;

        if (choice == 0) {
            std::cout << "Program terminated.\n";
            break;
        }

        // ---------------------------------------------
        // SELECT SINGLE OPTIMIZATION (OPTION 12)
        // ---------------------------------------------
        int single_choice = -1;
        if (choice == 7) {
            std::cout << "\nSelect ONE Optimization Algorithm:\n";
            std::cout << "7: Iterated Local Search (ILS)\n";
            std::cout << "8: Simulated Annealing (SA)\n";
            std::cout << "9: Tabu Search (TS)\n";
            std::cout << "10: Feasible Local Search (FLS)\n";
            std::cout << "11: Infeasible Local Search (IFLS)\n";
            std::cout << "12: Memetic Algorithm (MA)\n";
            std::cout << "13: Great Deluge Algorithm (GD)\n";
            std::cout << "14: Late Acceptance Hill Climbing (LAHC)\n";
            std::cout << "Enter (7 to 14): ";
            std::cin >> single_choice;

            if (single_choice < 7 || single_choice > 14) {
                std::cout << "Invalid selection. Returning to menu...\n";
                continue;
            }
        }

        // ====================================================
        //           RUN SELECTED METHOD ON ALL INSTANCESdatasetFiles.size()
        // ====================================================
        for (size_t idx = 0; idx < datasetFiles.size() ; idx++) {

            std::string current_file   = datasetFiles[idx];
            fs::path p(current_file);
            std::string rawName        = p.filename().string();
            instanceName               = parseInstanceName(rawName);

            if (instanceName.empty()) {
                std::cerr << "[WARNING] Could not parse instance: " << rawName << "\n";
                continue;
            }

            std::cout << "\nProcessing Instance " << (idx + 1)
                      << "/" << datasetFiles.size() << "\n";
            std::cout << "Instance Name: " << instanceName << "\n";

            std::string resultsPath = resultsDir + "Results_" + instanceName + ".txt";

            std::ofstream resultsFile(resultsPath);
            if (!resultsFile) {
                std::cerr << "Error writing to: " << resultsPath << "\n";
                continue;
            }

            resultsFile << "Method\tBestEff\tBestDiv\tTime(sec)\n";

            H.Parameters();
            H.initialization(current_file);
            std::cout<<"\n Initial Random Solution:\n";
            H.generate_initialrandom();
            H.display(team);

            // Display problem info
            std::cout << "num_node="  << num_node
                      << "  num_team=" << num_team
                      << "  num_each_t=" << num_each_t
                      << "  min_div="   << min_div << "\n";

            switch (choice) {
                case 1:
                    H.Q_Learning_Selection_Hyperheuristic_CMCEE(max_time);
                    resultsFile << "QL_HH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 2:
                    H.HH_Choice_Function_Selection_CMCEE(max_time);
                    resultsFile << "CF_HH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 3:
                    H.Random_Selection_Hyperheuristic_CMCEE(max_time);
                    resultsFile << "Random_HH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 4:
                    H.MAB_Selection_Hyperheuristic_CMCEE(max_time);
                    resultsFile << "MAB_HH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 5:
                    H.Greedy_Selection_Hyperheuristic_CMCEE(max_time);
                    resultsFile << "Greedy_HH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 6:
                    H.TriLevel_HH_Qlearning_CMCEE(max_time);
                    resultsFile << "TriLevel_HH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 7:
                    H.runSingleOptimizationAlgorithm(single_choice, max_time);
                    resultsFile << "Algorithm_" << single_choice
                                << "\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;
            }

            resultsFile.close();
            std::cout << "Saved: " << resultsPath << "\n";
            std::cout << "--------------------------------------------------------\n";
        }

        std::cout << "\nRun complete. Press ENTER to return to menu...";
        std::cin.ignore();
        std::cin.get();
    }

    free_memory();
    return 0;
}



