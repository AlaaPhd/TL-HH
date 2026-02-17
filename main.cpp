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
    std::vector<int> hyperHeuristics = {1, 2, 3, 4, 5};
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
    int    calculate_efficiency12(int*, int);
    std::pair<int,int> objective_after_LLH(int **);
    double calculate_diversity(const Inner&);
    double calculate_diversity12(int* , int );
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
    int**   GRASP();
    int     iterated_local_search12();
    int     Multi_O_Iterated_local_search();
    int**   simulated_annealing();
    int**   memetic();
    void    HH_RL_GD();
    int**   improvise_New_HM();

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
    int  Greedy_Randomized_list();
    int  generate_initial();
    void initial_HM();


    // -------------------------------------------------------------------------
    // Repair Functions
    // -------------------------------------------------------------------------
    void repairDiversity(Inner &team, double min_div, std::unordered_set<std::size_t> &visitedHashes);
    void repairDiversity(Inner &, double);
    void repairSolutions();
    void repair_solution();
    void repair_solution_restrict();
    void repair_solution12();
    void repair_duplicates(Outer& , int , int , int , Outer& );


    // -------------------------------------------------------------------------
    // Swap and Mutation Operators
    // -------------------------------------------------------------------------
    int** swap_LLH1(int &, int &);
    int** swap_LLH2(int &, int &);
    int** swap_LLH3(int **);
    int** swap_LLH4(int**);
    int** swap_LLH5(int**);
    int** swap_LLH6(int**);
    int** swap_LLH7(int **);
    int   shift(int , int &, int &);
    int   reassignment(int , int &, int &,int );
    int   ruin_and_recreate(int , int &, int &);
    int** ruin_and_recreate(int **);
    int   swap1(int , int &, int &);
    int   swap_ils(int , int &, int &);
    int   swap_min(int , int &, int &);
    int   swap_min_with_z(int , int &, int &);
    double swap_min12(int , int &, int &);
    void  inverse_operator();
    void  randomSwap();
    void  perturbRAssign(int**);

    // -------------------------------------------------------------------------
    // Crossover Operators
    // -------------------------------------------------------------------------
    void  cross_over2();
    int** SoftBackboneCrossover(int**, int**);
    int** PertubILS(int**);
    int** perturb(int **);
    int** perturbation(int **);

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
    int   getHeuristicsOfTypelocal();
    int   getHeuristicsOfTypeMutation();
    int   getHeuristicsOfTypeCROSSOVER();
    void  Random_Selection_Hyperheuristic_CMCEE11(int );
    void  Random_Selection_Hyperheuristic_CMCEE(int );
    void  HH_Choice_Function_Selection_CMCEE(int);
    void  SSHH_Selection_Hyperheuristic_CMCEE(int);
    void  Adaptive_SSHH_Selection_Hyperheuristic_CMCEE(int);

    std::tuple<int**, double, std::vector<std::vector<int>>,
           std::map<int, std::vector<double>>,
           std::map<int, std::vector<double>>,
           std::map<int, std::vector<int>>> ADSH_random(int, double );
    std::tuple<
        int**,                              // final team array
        double,                             // best efficiency
        std::vector<std::vector<int>>,      // L  (log of selections)
        std::map<int, std::vector<double>>, // cost     per LLH
        std::map<int, std::vector<double>>, // time     per LLH
        std::map<int, std::vector<int>>     // iterations per LLH
    >
    ADSH_pairHH(double optimal);
    void testingmethods_ADSH_R(double );
    void compare_acceptance_criteria(double );
    void MAB_Selection_Hyperheuristic_CMCEE(int);
    void MAHH_Algorithm(int );
    void MAHH_Selection_CMCEE(int);
    void MAHH_Selection_ThreeMAB(int max_time, MABStrategy mab_strategy);
    void MAHH_Algorithm12(int max_time, MABStrategy mab_strategy);
    void Greedy_Selection_Hyperheuristic_CMCEE(int);
    void Tabu_Search_Hyperheuristic_Adaptive_Acceptance(int);
    // Q-Learning Based Selection Function
    void Q_Learning_Selection_Hyperheuristic_CMCEE(int);
    void Q_Learning_Selection_Hyperheuristic_CMCEE1(int);
    void Q_Learning_Selection_Hyperheuristic_CMCEE_DHSS(int);
    // Example definition of Q_Learning_Refined
    void Q_Learning_Refined(int);
    void TriLevel_HH_Qlearning_CMCEE(int);
     // MAB Refined
   void MAB_Refined(int , const std::vector<int>& );
   // Greedy Refined
   void Greedy_Refined(int , const std::vector<int>& );

    // Choice Function Refined
    void ChoiceFunction_Refined(int, const std::vector<int>&);

    void execute_algorithm(int , const std::string&);
    void runSingleOptimizationAlgorithm(int , int );

    // -------------------------------------------------------------------------
    // Utility and Display
    // -------------------------------------------------------------------------
    void display(int**);
    void displayResults();
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
    objective_Function1(team);

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

int Hyper_heuristic::variable_neighborhood_search(int max_iter) {
    // Initialize best efficiency with the current best
    int best_efficiency = static_cast<int>(fbest);

    // Define the list of heuristics
    std::vector<Heuristic> heuristics = {
        Heuristic(1, "LLH1: Swap two randomly selected individuals between two randomly selected teams"),
        Heuristic(2, "LLH2: Swap four randomly selected individuals between two randomly selected teams"),
        Heuristic(3, "LLH3: From one randomly selected team (excluding team 0), randomly select two individuals and exchange them with two unallocated individuals from team 0"),
        Heuristic(4, "LLH4: Swap two consecutive individuals between two randomly selected teams"),
        Heuristic(5, "LLH5: Local Search")
    };

    const int kmax = heuristics.size();
    int total_iterations = 0;
    double total_selections = 0.0;

    // ---------- NEW: Convergence Setup ----------
    std::vector<int> convergence;
    convergence.push_back(best_efficiency);

    std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "variable_neighborhood_search_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open()) {
        conv_file << "Iteration,Fbest\n";
    }
    // --------------------------------------------

    // Main loop
    while (total_iterations < max_iter) {
        double best_ucb = -std::numeric_limits<double>::infinity();
        int selected_index = 0;

        // ---- UCB Heuristic Selection ----
        for (size_t i = 0; i < heuristics.size(); ++i) {
            if (heuristics[i].usage_count == 0) {
                best_ucb = std::numeric_limits<double>::infinity();
                selected_index = i;
                break;
            } else {
                double average_reward = heuristics[i].average_reward;
                double ucb = average_reward + std::sqrt((2.0 * std::log(total_selections + 1.0)) /
                                                        heuristics[i].usage_count);
                if (ucb > best_ucb) {
                    best_ucb = ucb;
                    selected_index = i;
                }
            }
        }

        Heuristic& selected_heuristic = heuristics[selected_index];
        int new_efficiency = 0;

        // ---- Apply Selected Heuristic ----
        switch (selected_heuristic.id) {
            case 1: {
                int** solution = LLH1(team);
                objective_Function1(solution);
                new_efficiency = static_cast<int>(f_cur);
                break;
            }
            case 2: {
                int** solution = LLH2(team);
                objective_Function1(solution);
                new_efficiency = static_cast<int>(f_cur);
                break;
            }
            case 3: {
                int** solution = LLH3(team);
                objective_Function1(solution);
                new_efficiency = static_cast<int>(f_cur);
                break;
            }
            case 4: {
                int** solution = LLH4(team);
                objective_Function1(solution);
                new_efficiency = static_cast<int>(f_cur);
                break;
            }
            case 5: {
                new_efficiency = feasible_local_search();
                break;
            }
            default:
                std::cerr << "Invalid heuristic ID: " << selected_heuristic.id << std::endl;
                return best_efficiency;
        }

        // ---- Reward Calculation ----
        double reward = 0.0;
        if (new_efficiency > best_efficiency) {
            best_efficiency = new_efficiency;
            fbest = static_cast<double>(best_efficiency);
            reward = 1.0;
            for (int m = 0; m < num_node; m++)
                fbest_solution[m] = best_solution[m];
            for (int m = 1; m <= num_team; m++) {
                eff_fbest[m] = eff_best[m];
                div_fbest[m] = div_best[m];
            }
        }

        // ---- Update Heuristic Stats ----
        selected_heuristic.usage_count += 1;
        selected_heuristic.total_reward += reward;
        selected_heuristic.average_reward = selected_heuristic.total_reward / selected_heuristic.usage_count;

        total_selections += 1.0;
        total_iterations++;

        // ---- Save Convergence ----
        convergence.push_back(best_efficiency);
        if (conv_file.is_open()) {
            conv_file << total_iterations << "," << best_efficiency << "\n";
        }
    }

    if (conv_file.is_open())
        conv_file.close();

    //std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return best_efficiency;
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
int** Hyper_heuristic::SoftBackboneCrossover(int** Si, int** Sj) {
    // Implementation of soft backbone crossover (Algorithm 3)
    int **arr1 = new int*[num_team + 1];
	int **arr2 = new int*[num_team + 1];
	int** offspring = new int*[num_node];
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
	offspring = team;
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
    return offspring;
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
void Hyper_heuristic::initial_HM() {
    // --- sanity ---
    std::cout << "HMS = " << HMS << "\n"
              << "num_node = " << num_node << "\n"
              << "num_team = " << num_team << "\n"
              << "num_each_t = " << num_each_t << "\n";

    const int leftovers_sz = std::max(0, num_node - num_team * num_each_t);

    // storage
    HM.HM1.assign(HMS, std::vector<Inner>(num_team + 1));
    HM3.HM1.assign(HMS, std::vector<Inner>(num_team + 1));
    HMEF.assign(HMS, std::vector<int>(num_team + 1, 0));
    HMDF.assign(HMS, std::vector<int>(num_team + 1, 0));
    SF.assign(HMS, 0);

    // pre-size
    for (int i = 0; i < HMS; ++i) {
        HM3.HM1[i][0].ind1.assign(leftovers_sz, 0);
        for (int j = 1; j <= num_team; ++j)
            HM.HM1[i][j].ind1.assign(num_each_t, 0);
    }

    // --- helper: make a harmony signature (order-invariant) ---
    auto make_signature = [&](const std::vector<std::vector<int>>& crews,
                              const std::vector<int>& leftovers) -> std::string {
        // crews: index 1..num_team used (0 ignored)
        std::vector<std::vector<int>> canon;
        canon.reserve(num_team);
        for (int t = 1; t <= num_team; ++t) {
            std::vector<int> v = crews[t];
            std::sort(v.begin(), v.end());          // order-invariant inside team
            canon.push_back(std::move(v));
        }
        // sort teams to be team-order invariant (by lexicographic)
        std::sort(canon.begin(), canon.end());

        std::vector<int> left = leftovers;
        std::sort(left.begin(), left.end());        // order-invariant leftovers

        // serialize to string
        std::ostringstream oss;
        oss << "T:";
        for (const auto& v : canon) {
            oss << "[";
            for (size_t k = 0; k < v.size(); ++k) {
                if (k) oss << ",";
                oss << v[k];
            }
            oss << "]";
        }
        oss << "|L:[";
        for (size_t k = 0; k < left.size(); ++k) {
            if (k) oss << ",";
            oss << left[k];
        }
        oss << "]";
        return oss.str();
    };

    std::unordered_set<std::string> seen;  // signatures we already stored
    seen.reserve(HMS * 4);

    int filled = 0;
    int guard_iters = 0; // safety to avoid infinite loops on tight spaces

    while (filled < HMS && guard_iters < 10000) {
        ++guard_iters;

        // 1) create a random feasible solution into global 'team'
        generate_initialrandom();  // fills: team[1..num_team][0..num_each_t-1] and team[0][] (optional)

        // optional: repair if needed here
        // repair_solution();

        // 2) rebuild leftovers to be safe
        std::vector<int> used(num_node, 0);
        for (int j = 1; j <= num_team; ++j)
            for (int k = 0; k < num_each_t; ++k)
                used[team[j][k]] = 1;

        std::vector<int> leftovers;
        leftovers.reserve(leftovers_sz);
        for (int id = 0; id < num_node; ++id)
            if (!used[id]) leftovers.push_back(id);

        // 3) build crews vec for signature
        std::vector<std::vector<int>> crews(num_team + 1);
        for (int j = 1; j <= num_team; ++j) {
            crews[j].assign(team[j], team[j] + num_each_t); // copy from int** team
        }

        // 4) get signature
        std::string sig = make_signature(crews, leftovers);

        // 5) if duplicate, skip and try again
        if (seen.find(sig) != seen.end()) {
            continue; // duplicate harmony → generate another
        }

        // 6) accept: copy into HM/HM3 and compute fitness
        int i = filled; // next row to fill

        for (int j = 1; j <= num_team; ++j) {
            for (int k = 0; k < num_each_t; ++k)
                HM.HM1[i][j].ind1[k] = team[j][k];
        }

        // compute eff/div and SF
        int min_eff = INT_MAX;
        for (int j = 1; j <= num_team; ++j) {
            int teff = 0;
            for (int k = 0; k < num_each_t; ++k) {
                const int id = HM.HM1[i][j].ind1[k];
                teff += eff[id];
            }
            HMEF[i][j] = teff;
            min_eff = std::min(min_eff, teff);

            int tdiv = 0;
            for (int a = 0; a < num_each_t; ++a)
                for (int b = a + 1; b < num_each_t; ++b) {
                    const int ia = HM.HM1[i][j].ind1[a];
                    const int ib = HM.HM1[i][j].ind1[b];
                    tdiv += div_in[ia][ib];
                }
            HMDF[i][j] = tdiv;
        }
        SF[i] = min_eff;

        // copy leftovers to HM3
        HM3.HM1[i][0].ind1 = leftovers;
        if ((int)HM3.HM1[i][0].ind1.size() != leftovers_sz)
            HM3.HM1[i][0].ind1.resize(leftovers_sz, 0); // pad/truncate if needed

        // mark signature and advance
        seen.insert(std::move(sig));
        ++filled;
    }

    if (filled < HMS) {
        std::cerr << "[warn] initial_HM: could not fill all " << HMS
                  << " unique harmonies (filled=" << filled << ").\n";
        // If you want, you can relax uniqueness at the end to fill remaining rows.
    }

    // --- debug print ---
    std::cout << "Initialization of Harmony Memory\n";
    for (int i = 0; i < filled; ++i) {
        std::cout << "S" << i << ":\n";
        for (int j = 1; j <= num_team; ++j) {
            std::cout << "Team " << j << ": ";
            for (int k = 0; k < (int)HM.HM1[i][j].ind1.size(); ++k)
                std::cout << HM.HM1[i][j].ind1[k] << "\t";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "Harmony Solutions Objective Fitness (min-eff per harmony):\n";
    for (int i = 0; i < filled; ++i) std::cout << "\t" << SF[i] << "\t";
    std::cout << "\n";
}

void Hyper_heuristic::opposite_Based_L() {
    int max1[num_team + 1] = {0}, min1[num_team + 1];
    int **team12;

    // Initialize arrays
    for (int i = 0; i <= num_team; i++) {
        w_div[i] = 0;
        w_eff[i] = 0;
        div_best[i] = 0;
        eff_best[i] = 0;
        min1[i] = INT_MAX; // Initialize min1 to maximum integer value
    }

    // Dynamically allocate memory for team12
    team12 = new int*[num_team + 1];
    for (int j = 0; j <= num_team; j++) {
        if (j == 0) {
            team12[j] = new int[num_node - (num_team * num_each_t)]; // For unallocated individuals
        } else {
            team12[j] = new int[num_each_t]; // For each team
        }
    }

    // Find max and min for unallocated individuals (team 0)
    for (int i = 0; i < num_node - (num_team * num_each_t); ++i) {
        if (team[0][i] > max1[0]) {
            max1[0] = team[0][i];
        }
        if (team[0][i] < min1[0]) {
            min1[0] = team[0][i];
        }
    }

    // Find max and min for each team
    for (int j = 1; j <= num_team; j++) {
        for (int k = 0; k < num_each_t; k++) {
            if (team[j][k] > max1[j]) {
                max1[j] = team[j][k];
            }
            if (team[j][k] < min1[j]) {
                min1[j] = team[j][k];
            }
        }
    }

    // Calculate opposite values for unallocated individuals (team 0)
    for (int i = 0; i < num_node - (num_team * num_each_t); ++i) {
        team12[0][i] = max1[0] + min1[0] - team[0][i];
    }

    // Calculate opposite values for each team
    for (int j = 1; j <= num_team; j++) {
        for (int k = 0; k < num_each_t; k++) {
            team12[j][k] = max1[j] + min1[j] - team[j][k];
        }
    }

    // Calculate efficiency and diversity for the opposite teams
    for (int t = 1; t <= num_team; t++) {
        for (int j = 0; j < num_each_t; j++) {
            w_eff[t] += eff[team12[t][j]];
            for (int k = j + 1; k < num_each_t; k++) {
                w_div[t] += div_in[team12[t][j]][team12[t][k]];
            }
        }
    }

    // Find the team with minimum efficiency
    int idx = min_func(w_eff, num_team);
    f_cur = w_eff[idx];

    // Output opposite values for unallocated individuals (team 0)
    cout << endl << "Team 0 (Unallocated Individuals): ";
    for (int i = 0; i < num_node - (num_team * num_each_t); ++i) {
        cout << team12[0][i] << " ";
    }
    cout << endl;

    // Output opposite values for each team along with their efficiency and diversity
    for (int t = 1; t <= num_team; t++) {
        cout << "Team " << t << ": ";
        for (int j = 0; j < num_each_t; j++) {
            cout << team12[t][j] << "\t";
        }
        cout << "eff = " << w_eff[t] << "\tdiv = " << w_div[t];
        cout << endl;
    }

    // Deallocate memory for team12
    for (int j = 0; j <= num_team; j++) {
        delete[] team12[j];
    }
    delete[] team12;
}
// In your Hyper_heuristic class or wherever you initialize HMEF and HMDF
std::vector<std::vector<int>> HMEF(HMS, std::vector<int>(num_team ));
std::vector<std::vector<int>> HMDF(HMS, std::vector<int>(num_team ));

int** Hyper_heuristic::improvise_New_HM()
{
    const int NR      = num_team;     // teams 1..NR (team 0 is leftovers)
    const int HMS     = 5;            // harmony memory size
    const double HMCR = 0.90;         // memory consideration rate
    const double PAR  = 0.30;         // pitch-adjustment rate
    const int maxiter = 1000;           // outer iterations
    const int max_no_improve = 10;   // early stop if no improvement

    // RNGs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_real(0.0, 1.0);
    std::uniform_int_distribution<>  dis_hms(0, HMS - 1);
    std::uniform_int_distribution<>  dis_node(0, std::max(0, num_node - 1));

    // candidate holder (HM6: index 0 for leftovers, 1..NR crews)
    Outer Newteam;
    Newteam.HM6.resize(NR + 1);
    for (int j = 0; j <= NR; ++j)
        Newteam.HM6[j].ind1.assign(j == 0 ? 0 : num_each_t, -1); // team0 size set later

    std::vector<int> w_eff(NR + 1, 0);
    std::vector<int> w_div(NR + 1, 0);

    // ensure HM/HM3/HMEF/HMDF/SF are ready
    initial_HM(); // SF[h] should hold min-eff per harmony (higher is better)

    auto in_bounds = [&](int id) -> bool {
        return id >= 0 && id < num_node;
    };

    // small tweak near original id (±1..±3)
    auto pitch_adjust_small_tweak = [&](int base,
                                        const std::unordered_set<int>& available) -> int {
        if (!in_bounds(base)) base = std::min(std::max(base, 0), num_node - 1);
        for (int d = 1; d <= 3; ++d) {
            int up   = base + d;
            int down = base - d;
            if (in_bounds(up)   && available.count(up))   return up;
            if (in_bounds(down) && available.count(down)) return down;
        }
        return -1;
    };

    // any free id from 0..num_node-1 not in 'used'
    auto pick_from_global_pool = [&](const std::unordered_set<int>& used) -> int {
        for (int tries = 0; tries < 30; ++tries) {
            int cand = dis_node(gen);
            if (in_bounds(cand) && !used.count(cand)) return cand;
        }
        for (int cand = 0; cand < num_node; ++cand)
            if (!used.count(cand)) return cand;
        return -1; // none left (should not happen if counts match)
    };

    int NI = 0, no_improve_cnt = 0;

    while (NI < maxiter && no_improve_cnt < max_no_improve)
    {   //cout<<"Iteration No.:"<<NI;
        // available IDs for this candidate
        std::unordered_set<int> available;
        available.reserve(num_node * 2);
        for (int id = 0; id < num_node; ++id) available.insert(id);

        bool improved = false;

        // ===== Build a new candidate harmony (teams 1..NR) =====
        for (int j = 1; j <= NR; ++j)
        {
            for (int k = 0; k < num_each_t; ++k)
            {
                int chosen = -1;
                double r1 = dis_real(gen);

                if (r1 < HMCR) {
                    // pick from HM
                    int h_pick = dis_hms(gen);
                    int cand = -1;

                    if (h_pick >= 0 && h_pick < HMS) {
                        int fromHM = HM.HM1[h_pick][j].ind1[k];
                        if (in_bounds(fromHM) && available.count(fromHM))
                            cand = fromHM;
                        else {
                            // try another member from same crew in that HM row
                            for (int alt : HM.HM1[h_pick][j].ind1) {
                                if (in_bounds(alt) && available.count(alt)) { cand = alt; break; }
                            }
                        }
                    }

                    // pitch adjust (small tweak near cand) if we had a valid pick
                    double r2 = dis_real(gen);
                    if (r2 < PAR && cand != -1) {
                        int alt = pitch_adjust_small_tweak(cand, available);
                        if (alt != -1) cand = alt;
                        else {
                            int g = pick_from_global_pool(available);
                            if (g != -1) cand = g;
                        }
                    }

                    // fallback repair if still invalid
                    if (cand == -1 || !available.count(cand))
                        cand = pick_from_global_pool(available);

                    chosen = cand;
                } else {
                    // random from remaining pool
                    chosen = pick_from_global_pool(available);
                }

                if (chosen == -1) chosen = pick_from_global_pool(available); // last resort

                Newteam.HM6[j].ind1[k] = chosen;
                available.erase(chosen);
            }

            // quick (optional) pre-repair eff/div
            w_eff[j] = 0;
            for (int id : Newteam.HM6[j].ind1)
                if (in_bounds(id)) w_eff[j] += eff[id];

            w_div[j] = 0;
            for (int a = 0; a < num_each_t; ++a)
                for (int b = a + 1; b < num_each_t; ++b) {
                    int ia = Newteam.HM6[j].ind1[a];
                    int ib = Newteam.HM6[j].ind1[b];
                    if (in_bounds(ia) && in_bounds(ib))
                        w_div[j] += div_in[ia][ib];
                }
        }

        // team 0 leftovers
        Newteam.HM6[0].ind1.clear();
        Newteam.HM6[0].ind1.reserve(available.size());
        for (int id : available) Newteam.HM6[0].ind1.push_back(id);
        // ===== De-duplicate across all teams (handles repeats after repair) =====
        std::vector<int> freq(num_node, 0);
        for (int j = 1; j <= NR; ++j)
            for (int id : Newteam.HM6[j].ind1)
                if (in_bounds(id)) ++freq[id];

        std::vector<int> unused;
        unused.reserve(num_node);
        for (int id = 0; id < num_node; ++id)
            if (freq[id] == 0) unused.push_back(id);

        auto take_unused = [&]() -> int {
            if (unused.empty()) return -1;
            int u = unused.back();
            unused.pop_back();
            return u;
        };

        for (int j = 1; j <= NR; ++j) {
            std::unordered_set<int> seen;
            for (int k = 0; k < num_each_t; ++k) {
                int id = Newteam.HM6[j].ind1[k];
                bool dup_in_team = !seen.insert(id).second;
                bool dup_global  = in_bounds(id) ? (freq[id] > 1) : true;

                if (dup_in_team || dup_global) {
                    if (in_bounds(id) && freq[id] > 0) --freq[id];
                    int repl = take_unused();
                    if (repl == -1) {
                        // fallback: linear scan
                        for (int cand = 0; cand < num_node; ++cand)
                            if (freq[cand] == 0) { repl = cand; break; }
                    }
                    if (repl != -1) {
                        Newteam.HM6[j].ind1[k] = repl;
                        ++freq[repl];
                        seen.insert(repl);
                    } else {
                        // if no replacement is possible, restore freq
                        if (in_bounds(id)) ++freq[id];
                    }
                }
            }
        }

        // rebuild team 0 after de-dup
        Newteam.HM6[0].ind1.clear();
        for (int id = 0; id < num_node; ++id)
            if (freq[id] == 0) Newteam.HM6[0].ind1.push_back(id);

        // ===== Repair whole candidate BEFORE scoring =====
        // copy into global team for your repair function
        for (int j = 1; j <= NR; ++j)
            for (int k = 0; k < num_each_t; ++k)
                team[j][k] = Newteam.HM6[j].ind1[k];
        // if your repair needs team0, copy it similarly (only if team[0] capacity allows)
        // for (size_t k = 0; k < Newteam.HM6[0].ind1.size() && k < team0_capacity; ++k) team[0][k] = Newteam.HM6[0].ind1[k];

        repair_solution();

        // copy repaired teams back
        for (int j = 1; j <= NR; ++j)
            for (int k = 0; k < num_each_t; ++k)
                Newteam.HM6[j].ind1[k] = team[j][k];


        // ===== Score candidate (max–min efficiency) =====
        for (int j = 1; j <= NR; ++j) {
            w_eff[j] = 0;
            for (int id : Newteam.HM6[j].ind1)
                if (in_bounds(id)) w_eff[j] += eff[id];

            w_div[j] = 0;
            for (int a = 0; a < num_each_t; ++a)
                for (int b = a + 1; b < num_each_t; ++b) {
                    int ia = Newteam.HM6[j].ind1[a];
                    int ib = Newteam.HM6[j].ind1[b];
                    if (in_bounds(ia) && in_bounds(ib))
                        w_div[j] += div_in[ia][ib];
                }
        }

        int min_eff_val = INT_MAX;
        for (int j = 1; j <= NR; ++j) min_eff_val = std::min(min_eff_val, w_eff[j]);
        f_cur = min_eff_val;

        // find worst harmony (smallest SF)
        int worst_HM = 0;
        for (int h = 1; h < HMS; ++h)
            if (SF[h] < SF[worst_HM]) worst_HM = h;

        // replacement (maximisation)
        if (f_cur > SF[worst_HM]) {
            improved = true;

            // store teams 1..NR
            for (int j = 1; j <= NR; ++j) {
                HM.HM1[worst_HM][j].ind1 = Newteam.HM6[j].ind1;
                HMEF[worst_HM][j] = w_eff[j];
                HMDF[worst_HM][j] = w_div[j];
            }
            // store team 0 leftovers into HM3
            HM3.HM1[worst_HM][0].ind1 = Newteam.HM6[0].ind1;

            SF[worst_HM] = f_cur;
        }

        no_improve_cnt = improved ? 0 : (no_improve_cnt + 1);
        ++NI;
    }

    // ===== Copy best harmony to global team[][] =====
    int best_HM = 0;
    for (int h = 1; h < HMS; ++h)
        if (SF[h] > SF[best_HM]) best_HM = h;

    for (int j = 1; j <= NR; ++j)
        for (int k = 0; k < num_each_t; ++k)
            team[j][k] = HM.HM1[best_HM][j].ind1[k];

    // If you want to also expose team 0 (leftovers), read from HM3.HM1[best_HM][0].ind1

    print_harmony_memory(HM, HMS, NR, HMEF, HMDF);
    display(team);
    return team;
}

// Function prototypes
void Hyper_heuristic::repair_duplicates(Outer& HM, int HMS, int NR, int num_each_t, Outer& HM3) {
    std::unordered_set<int> used_individuals;
    for (int i = 0; i < HMS; ++i) {
        for (int j = 1; j <= NR; ++j) {
            for (int k = 0; k < num_each_t; ++k) {
                int individual = HM.HM1[i][j].ind1[k];
                if (individual != 0) {
                    used_individuals.insert(individual);
                }
            }
        }
    }

    for (int i = 0; i < HMS; ++i) {
        for (int j = 1; j <= NR; ++j) {
            for (int k = 0; k < num_each_t; ++k) {
                int individual = HM.HM1[i][j].ind1[k];
                if (used_individuals.count(individual) > 1 || individual == 0) {
                    for (int m = 0; m < num_node - num_team * num_each_t; ++m) {
                        int candidate = HM3.HM1[0][j].ind1[m];
                        if (candidate != 0 && used_individuals.find(candidate) == used_individuals.end()) {
                            used_individuals.erase(individual);
                            HM.HM1[i][j].ind1[k] = candidate;
                            used_individuals.insert(candidate);
                            break;
                        }
                    }
                }
            }
        }
    }
}

void Hyper_heuristic::print_harmony_memory(const Outer &HM, int HMS, int NR,
                                           const std::vector<std::vector<int>>& HMEF,
                                           const std::vector<std::vector<int>>& HMDF) {
    cout << endl;
    cout << "------------------------------------------------------------------------" << endl;
    cout << "Harmony Memory after improvisation:\n";
    for (int i = 0; i < HMS; ++i) {
        cout << "S" << i << ": " << endl;
        for (int j = 1; j <= NR; ++j) {
            cout << "Team " << j << ": ";
            for (size_t k = 0; k < HM.HM1[i][j].ind1.size(); ++k) {
                cout << HM.HM1[i][j].ind1[k] << "\t";
            }
            cout << endl;
        }
        cout << "------------------------------------------------------------------------" << endl;
    }

    cout << "Harmony Efficiency Fitness after improvisation:\n";
    for (int i = 0; i < HMS; i++) {
        cout << "S" << i << ": ";
        for (int j = 1; j <= num_team; j++) {
            cout << "\t" << HMEF[i][j] << "\t";
        }
        cout << endl;
    }

    cout << "Harmony Diversity Fitness after improvisation:\n";
    for (int i = 0; i < HMS; i++) {
        cout << "S" << i << ": ";
        for (int j = 1; j <= num_team; j++) {
            cout << "\t" << HMDF[i][j] << "\t";
        }
        cout << endl;
    }

    cout << "Harmony Solutions Objective Fitness after improvisation:\n";
    for (int i = 0; i < HMS; i++) {
        // Since HMEF[i] is a vector, use iterators to find the minimum element starting from index 1
        if (HMEF[i].size() > NR) {
            auto min_it = std::min_element(HMEF[i].begin() + 1, HMEF[i].begin() + NR + 1);
            cout << "\t" << *min_it << "\t";
        } else {
            // Handle the case where HMEF[i] doesn't have enough elements
            cout << "\tN/A\t";
        }
    }
    cout << endl;
}

void Hyper_heuristic::Parameters()
{
    ffbest = 0;
	fbest = 0;
	tl = 0;
	//beta = 0.4
	tabu_tenure = 10;
	generations = 50;
	fls_depth = 4000;
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

/*
int** Hyper_heuristic::LLH1(int** team) {
    // LLH1: Swap one unallocated member (team 0) with one from a random allocated team.

    int poolSize = pool_size_unallocated();
    if (num_team < 1 || num_each_t <= 0 || poolSize <= 0)
        return team;

    int team0   = 0;
    int team2   = rand_alloc_team();
    int memIdx  = rand_member_idx();
    int poolIdx = rand_pool_idx(poolSize);

    // Nodes involved
    int node_in  = team[team0][poolIdx];
    int node_out = team[team2][memIdx];

    // ------------------------------
    // 1. Swap pool ↔ allocated node
    // ------------------------------
    std::swap(team[team0][poolIdx], team[team2][memIdx]);

    // ------------------------------
    // 2. Repair membership mapping
    // ------------------------------
    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? poolSize : num_each_t);   // ensure valid bounds

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // ------------------------------
    // 3. Recompute efficiencies
    // ------------------------------
    w_eff[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_eff = 0;
        for (int j = 0; j < num_each_t; j++)
            sum_eff += eff[team[t][j]];
        w_eff[t] = sum_eff;
    }

    // ------------------------------
    // 4. Recompute pairwise diversity
    // ------------------------------
    w_div[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;

        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++) {
                int b = team[t][j];
                sum_div += div_in[a][b];
            }
        }
        w_div[t] = sum_div;
    }

    // ------------------------------
    // 5. Determine new objective
    // ------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}


int** Hyper_heuristic::LLH2(int** team) {
    // LLH2: Swap TWO unallocated members (team 0) with TWO members in TWO random allocated teams.

    int poolSize = pool_size_unallocated();
    if (num_team < 2 || num_each_t <= 0 || poolSize <= 1)
        return team;

    int team0 = 0;

    // Pick two distinct allocated teams
    auto [t2, t3] = rand_two_distinct_alloc_teams();

    // Pick members from allocated teams
    int idx2 = rand_member_idx();
    int idx3 = rand_member_idx();

    // Pick two distinct members from pool
    auto [p1, p2] = rand_two_distinct_pool_idx(poolSize);

    // ------------------------------
    // 1. Swap pool ↔ allocated
    // ------------------------------
    std::swap(team[team0][p1], team[t2][idx2]);
    std::swap(team[team0][p2], team[t3][idx3]);

    // ------------------------------
    // 2. Repair membership mapping
    // ------------------------------
    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? poolSize : num_each_t);

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // ------------------------------
    // 3. Recompute efficiencies
    // ------------------------------
    w_eff[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_eff = 0;
        for (int j = 0; j < num_each_t; j++)
            sum_eff += eff[team[t][j]];
        w_eff[t] = sum_eff;
    }

    // ------------------------------
    // 4. Recompute pairwise diversity
    // ------------------------------
    w_div[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;

        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++) {
                int b = team[t][j];
                sum_div += div_in[a][b];
            }
        }
        w_div[t] = sum_div;
    }

    // ------------------------------
    // 5. Determine new objective
    // ------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}


int** Hyper_heuristic::LLH3(int** team) {
    // LLH3: Swap one member from Team A with one member from Team B (allocated teams only).

    if (num_team < 2 || num_each_t <= 0)
        return team;

    // Pick two distinct allocated teams
    auto [t1, t2] = rand_two_distinct_alloc_teams();

    // Pick one member from each team
    int idx1 = rand_member_idx();
    int idx2 = rand_member_idx();

    // ------------------------------
    // 1. Swap nodes between teams
    // ------------------------------
    std::swap(team[t1][idx1], team[t2][idx2]);

    // ------------------------------
    // 2. Repair membership mapping
    // ------------------------------
    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? pool_size_unallocated() : num_each_t);

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // ------------------------------
    // 3. Recompute efficiencies
    // ------------------------------
    w_eff[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_eff = 0.0;
        for (int j = 0; j < num_each_t; j++)
            sum_eff += eff[team[t][j]];
        w_eff[t] = sum_eff;
    }

    // ------------------------------
    // 4. Recompute pairwise diversity
    // ------------------------------
    w_div[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;
        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++) {
                int b = team[t][j];
                sum_div += div_in[a][b];
            }
        }
        w_div[t] = sum_div;
    }

    // ------------------------------
    // 5. Determine new objective
    // ------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH4(int** team) {
    // LLH4: Move one member from Team A to Team B by replacing B's worst-eff member.

    if (num_team < 2 || num_each_t <= 0)
        return team;

    int t1 = rand_alloc_team();
    int t2 = rand_alloc_team();
    while (t2 == t1) t2 = rand_alloc_team();

    int idx1 = rand_member_idx();  // from t1
    int idx2 = rand_member_idx();  // to replace in t2

    // ------------------------------
    // 1. Swap members between teams
    // ------------------------------
    std::swap(team[t1][idx1], team[t2][idx2]);

    // ------------------------------
    // 2. Repair membership mapping
    // ------------------------------
    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? pool_size_unallocated() : num_each_t);
        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n] = t;
            address[n] = j;
        }
    }

    // ------------------------------
    // 3. Recompute efficiencies
    // ------------------------------
    w_eff[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_eff = 0;
        for (int j = 0; j < num_each_t; j++)
            sum_eff += eff[team[t][j]];
        w_eff[t] = sum_eff;
    }

    // ------------------------------
    // 4. Recompute pairwise diversity
    // ------------------------------
    w_div[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;

        for (int x = 0; x < num_each_t; x++) {
            int a = team[t][x];
            for (int y = x + 1; y < num_each_t; y++) {
                int b = team[t][y];    // FIX: declare b safely
                sum_div += div_in[a][b];
            }
        }
        w_div[t] = sum_div;
    }


    // ------------------------------
    // 5. Update objective
    // ------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH5(int** team) {
    // LLH5: Swap three consecutive members between two distinct allocated teams.
    // Same clean structure as LLH1: perform move → repair mapping → recompute eff/div → update f_cur.

    if (num_team < 2 || num_each_t < 3)
        return team;

    // ------------------------------
    // 1. Select two distinct teams and starting index
    // ------------------------------
    auto [t1, t2] = rand_two_distinct_alloc_teams();
    std::uniform_int_distribution<int> distIdx(0, num_each_t - 3);
    int i = distIdx(hh_rng());   // ensures i, i+1, i+2 are valid

    // ------------------------------
    // 2. Identify involved nodes
    // ------------------------------
    int a1 = team[t1][i];
    int b1 = team[t1][i + 1];
    int c1 = team[t1][i + 2];

    int a2 = team[t2][i];
    int b2 = team[t2][i + 1];
    int c2 = team[t2][i + 2];

    // ------------------------------
    // 3. Swap the triplets
    // ------------------------------
    std::swap(team[t1][i],     team[t2][i]);
    std::swap(team[t1][i + 1], team[t2][i + 1]);
    std::swap(team[t1][i + 2], team[t2][i + 2]);

    // ------------------------------
    // 4. Repair membership mapping
    // ------------------------------
    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? pool_size_unallocated() : num_each_t);
        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // ------------------------------
    // 5. Recompute efficiencies
    // ------------------------------
    w_eff[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_eff = 0.0;
        for (int j = 0; j < num_each_t; j++)
            sum_eff += eff[team[t][j]];
        w_eff[t] = sum_eff;
    }

    // ------------------------------
    // 6. Recompute diversities
    // ------------------------------
    w_div[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;

        for (int x = 0; x < num_each_t; x++) {
            int a = team[t][x];
            for (int y = x + 1; y < num_each_t; y++) {
                int b = team[t][y];    // FIX: declare b safely
                sum_div += div_in[a][b];
            }
        }
        w_div[t] = sum_div;
    }

    // ------------------------------
    // 7. Update objective values
    // ------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}


int** Hyper_heuristic::LLH6(int** team) {
    // LLH6: Swap one member from the min-efficiency team with one member from the unallocated pool (team 0).

    int poolSize = pool_size_unallocated();
    if (num_team < 1 || num_each_t <= 0 || poolSize <= 0)
        return team;

    int team0 = 0;

    // --------------------------------
    // 1. Find team with minimum efficiency
    // --------------------------------
    int min_team = min_func(w_eff, num_team);

    // Pick random positions
    int poolIdx  = rand_pool_idx(poolSize);
    int memIdx   = rand_member_idx();

    // Nodes involved
    int node_in  = team[team0][poolIdx];     // pool → allocated team
    int node_out = team[min_team][memIdx];   // allocated team → pool

    // --------------------------------
    // 2. Perform swap
    // --------------------------------
    std::swap(team[team0][poolIdx], team[min_team][memIdx]);

    // --------------------------------
    // 3. Repair membership mapping
    // --------------------------------
    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? poolSize : num_each_t);

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // --------------------------------
    // 4. Recompute efficiency
    // --------------------------------
    w_eff[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_eff = 0.0;
        for (int j = 0; j < num_each_t; j++)
            sum_eff += eff[team[t][j]];
        w_eff[t] = sum_eff;
    }

    // --------------------------------
    // 5. Recompute diversity
    // --------------------------------
    w_div[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;

        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++) {
                int b = team[t][j];
                sum_div += div_in[a][b];
            }
        }

        w_div[t] = sum_div;
    }

    // --------------------------------
    // 6. Update objective values
    // --------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}


int** Hyper_heuristic::LLH7(int** team) {
    // LLH7: Swap one random member between min-efficiency and max-efficiency teams.
    // Full recomputation version (same style as LLH1)

    if (num_team < 2 || num_each_t <= 0)
        return team;

    // --------------------------------
    // 1. Identify min-eff and max-eff teams
    // --------------------------------
    int tmin = 1, tmax = 1;
    double min_eff = w_eff[1], max_eff = w_eff[1];

    for (int t = 2; t <= num_team; t++) {
        if (w_eff[t] < min_eff) { min_eff = w_eff[t]; tmin = t; }
        if (w_eff[t] > max_eff) { max_eff = w_eff[t]; tmax = t; }
    }

    // If equal, choose two random distinct teams
    if (tmin == tmax) {
        tmin = rand_alloc_team();
        do { tmax = rand_alloc_team(); } while (tmax == tmin);
    }

    // --------------------------------
    // 2. Pick random members
    // --------------------------------
    int idx_min = rand_member_idx();
    int idx_max = rand_member_idx();

    int node_min = team[tmin][idx_min];
    int node_max = team[tmax][idx_max];

    // --------------------------------
    // 3. Perform the swap
    // --------------------------------
    std::swap(team[tmin][idx_min], team[tmax][idx_max]);

    // --------------------------------
    // 4. Repair membership mapping
    // --------------------------------
    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? pool_size_unallocated() : num_each_t);

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // --------------------------------
    // 5. Recompute efficiencies
    // --------------------------------
    w_eff[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_eff = 0.0;
        for (int j = 0; j < num_each_t; j++)
            sum_eff += eff[team[t][j]];
        w_eff[t] = sum_eff;
    }

    // --------------------------------
    // 6. Recompute diversity
    // --------------------------------
    w_div[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;

        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++) {
                int b = team[t][j];
                sum_div += div_in[a][b];
            }
        }
        w_div[t] = sum_div;
    }

    // --------------------------------
    // 7. Update objective
    // --------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH8(int** team) {
    // LLH8: Swap the first individual from two randomly selected allocated teams.
    // Full recomputation version (same style as LLH1)

    if (num_team < 2 || num_each_t <= 0)
        return team;

    // --------------------------------
    // 1. Select two DISTINCT allocated teams
    // --------------------------------
    auto [t1, t2] = rand_two_distinct_alloc_teams();
    int idx = 0;   // always first member

    int node1 = team[t1][idx];
    int node2 = team[t2][idx];

    // --------------------------------
    // 2. Perform the swap
    // --------------------------------
    std::swap(team[t1][idx], team[t2][idx]);

    // --------------------------------
    // 3. Repair membership mapping
    // --------------------------------
    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? pool_size_unallocated() : num_each_t);

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // --------------------------------
    // 4. Recompute efficiencies
    // --------------------------------
    w_eff[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_eff = 0.0;
        for (int j = 0; j < num_each_t; j++)
            sum_eff += eff[team[t][j]];
        w_eff[t] = sum_eff;
    }

    // --------------------------------
    // 5. Recompute diversity
    // --------------------------------
    w_div[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;

        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++) {
                int b = team[t][j];
                sum_div += div_in[a][b];
            }
        }
        w_div[t] = sum_div;
    }

    // --------------------------------
    // 6. Update objective
    // --------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}


int** Hyper_heuristic::LLH9(int** team) {
    // LLH9: Swap the LAST individual from two randomly selected allocated teams.
    // Full recomputation version (same structure as LLH1)

    if (num_team < 2 || num_each_t <= 0)
        return team;

    // --------------------------------
    // 1. Pick two distinct teams
    // --------------------------------
    auto [t1, t2] = rand_two_distinct_alloc_teams();
    int idx = num_each_t - 1;  // last member of each team

    int node1 = team[t1][idx];
    int node2 = team[t2][idx];

    // --------------------------------
    // 2. Perform swap
    // --------------------------------
    std::swap(team[t1][idx], team[t2][idx]);

    // --------------------------------
    // 3. Repair membership mapping
    // --------------------------------
    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? pool_size_unallocated() : num_each_t);

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // --------------------------------
    // 4. Recompute efficiencies
    // --------------------------------
    w_eff[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_eff = 0.0;
        for (int j = 0; j < num_each_t; j++)
            sum_eff += eff[team[t][j]];
        w_eff[t] = sum_eff;
    }

    // --------------------------------
    // 5. Recompute diversity
    // --------------------------------
    w_div[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;

        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++) {
                int b = team[t][j];
                sum_div += div_in[a][b];
            }
        }
        w_div[t] = sum_div;
    }

    // --------------------------------
    // 6. Update the objective values
    // --------------------------------
    int idx_eff = min_func(w_eff, num_team);
    int idx_div = min_func(w_div, num_team);

    f_cur     = w_eff[idx_eff];
    f_cur_div = w_div[idx_div];
    repair_solution();
    return team;
}



int** Hyper_heuristic::LLH10(int** team) {
    // LLH10: Swap one member from the minimum-efficiency team with
    //        one random member from another allocated team.
    //        (Full recomputation version, same structure as LLH1)

    if (num_team < 2 || num_each_t <= 0)
        return team;

    // --------------------------------
    // 1. Find the minimum-efficiency team
    // --------------------------------
    int t_min = min_func(w_eff, num_team);

    // Pick a second team different from t_min
    int t2 = rand_alloc_team();
    while (t2 == t_min)
        t2 = rand_alloc_team();

    // --------------------------------
    // 2. Choose random members
    // --------------------------------
    int idx_min = rand_member_idx();
    int idx2    = rand_member_idx();

    int node_min = team[t_min][idx_min];
    int node_out = team[t2][idx2];

    // --------------------------------
    // 3. Swap members
    // --------------------------------
    std::swap(team[t_min][idx_min], team[t2][idx2]);

    // --------------------------------
    // 4. Repair membership: state[] and address[]
    // --------------------------------
    int poolSize = pool_size_unallocated();

    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? poolSize : num_each_t);

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // --------------------------------
    // 5. Recompute efficiencies
    // --------------------------------
    w_eff[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum = 0;
        for (int j = 0; j < num_each_t; j++)
            sum += eff[team[t][j]];
        w_eff[t] = sum;
    }

    // --------------------------------
    // 6. Recompute diversity (pairwise)
    // --------------------------------
    w_div[0] = 0;
    for (int t = 1; t <= num_team; t++) {
        double sum = 0.0;
        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++) {
                int b = team[t][j];
                sum += div_in[a][b];
            }
        }
        w_div[t] = sum;
    }

    // --------------------------------
    // 7. Update objective values
    // --------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}


int** Hyper_heuristic::LLH11(int** team) {
    // LLH11: Circularly shift d columns across all allocated teams (1..num_team)
    //        and fully recompute efficiency/diversity.

    int d = 4;
    if (num_team < 2 || num_each_t < d)
        return team;

    // --------------------------------
    // 1. Circular shift each column c
    // --------------------------------
    for (int c = 0; c < d; c++) {

        // Copy column members of allocated teams
        std::vector<int> col(num_team + 1);
        for (int t = 1; t <= num_team; t++)
            col[t] = team[t][c];

        // Backward circular shift
        for (int t = 1; t < num_team; t++)
            team[t][c] = col[t + 1];

        team[num_team][c] = col[1];   // wrap
    }

    // --------------------------------
    // 2. Repair membership mapping
    // --------------------------------
    int poolSize = pool_size_unallocated();

    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? poolSize : num_each_t);

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // --------------------------------
    // 3. Recompute efficiencies
    // --------------------------------
    w_eff[0] = 0;

    for (int t = 1; t <= num_team; t++) {
        double sum = 0;
        for (int j = 0; j < num_each_t; j++)
            sum += eff[team[t][j]];
        w_eff[t] = sum;
    }

    // --------------------------------
    // 4. Recompute pairwise diversity
    // --------------------------------
    w_div[0] = 0;

    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;

        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++) {
                int b = team[t][j];
                sum_div += div_in[a][b];
            }
        }

        w_div[t] = sum_div;
    }

    // --------------------------------
    // 5. Update global objective
    // --------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}

int** Hyper_heuristic::LLH12(int** team) {
    // LLH12: Swap 50% of the members between the least efficient team
    //        and the unallocated pool (team 0) using full recomputation.

    if (num_team < 1 || num_each_t <= 1)
        return team;

    // ------------------------------------------
    // 1. Find team with minimum efficiency
    // ------------------------------------------
    int min_t = 1;
    double min_eff = w_eff[1];

    for (int t = 2; t <= num_team; t++) {
        if (w_eff[t] < min_eff) {
            min_eff = w_eff[t];
            min_t = t;
        }
    }

    int poolSize = pool_size_unallocated();
    int swapCount = num_each_t / 2;
    if (swapCount > poolSize)
        swapCount = poolSize;

    if (swapCount <= 0)
        return team;

    // ------------------------------------------
    // 2. Perform the swaps
    // ------------------------------------------
    for (int i = 0; i < swapCount; i++) {
        std::swap(team[min_t][i], team[0][i]);
    }

    // ------------------------------------------
    // 3. Repair state[] and address[]
    // ------------------------------------------
    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? poolSize : num_each_t);

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // ------------------------------------------
    // 4. Recompute efficiencies
    // ------------------------------------------
    w_eff[0] = 0;

    for (int t = 1; t <= num_team; t++) {
        double sum = 0;
        for (int j = 0; j < num_each_t; j++)
            sum += eff[team[t][j]];
        w_eff[t] = sum;
    }

    // ------------------------------------------
    // 5. Recompute pairwise diversity
    // ------------------------------------------
    w_div[0] = 0;

    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;

        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++) {
                int b = team[t][j];
                sum_div += div_in[a][b];
            }
        }

        w_div[t] = sum_div;
    }

    // ------------------------------------------
    // 6. Determine new objective values
    // ------------------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}


int** Hyper_heuristic::LLH13(int** team) {
    // LLH13: Ruin 50% of ALL allocated members and recreate safely.
    // Rebuild team0 + all allocated teams from one shuffled list.

    int total_alloc = num_team * num_each_t;     // allocated nodes
    int pool_size   = pool_size_unallocated();   // existing pool capacity
    int total_nodes = total_alloc + pool_size;   // ALL nodes in system

    int ruin_percentage = 50;
    int num_ruin = (total_alloc * ruin_percentage) / 100;

    // -------------------------------------------
    // 1. Create a full list of ALL nodes
    // -------------------------------------------
    std::vector<int> full_list;
    full_list.reserve(total_nodes);

    // pool first
    for (int i = 0; i < pool_size; i++)
        full_list.push_back(team[0][i]);

    // allocated teams
    for (int t = 1; t <= num_team; t++)
        for (int j = 0; j < num_each_t; j++)
            full_list.push_back(team[t][j]);

    // safety check
    if ((int)full_list.size() != total_nodes) {
        std::cout << "LLH13 ERROR: list mismatch\n";
        return team;
    }

    // -------------------------------------------
    // 2. Shuffle the entire list
    // -------------------------------------------
    std::shuffle(full_list.begin(), full_list.end(), hh_rng());

    // -------------------------------------------
    // 3. Rebuild team0 (pool)
    // -------------------------------------------
    // Remove exactly num_ruin nodes into pool
    int new_pool_size = num_ruin;

    for (int i = 0; i < new_pool_size; i++)
        team[0][i] = full_list[i];

    // -------------------------------------------
    // 4. Rebuild allocated teams
    // -------------------------------------------
    int idx = new_pool_size;
    for (int t = 1; t <= num_team; t++) {
        for (int j = 0; j < num_each_t; j++) {
            team[t][j] = full_list[idx++];
        }
    }

    // -------------------------------------------
    // 5. Rebuild state[] and address[]
    // -------------------------------------------
    for (int i = 0; i < new_pool_size; i++) {
        int n = team[0][i];
        state[n]   = 0;
        address[n] = i;
    }

    for (int t = 1; t <= num_team; t++) {
        for (int j = 0; j < num_each_t; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // -------------------------------------------
    // 6. Recompute team metrics
    // -------------------------------------------
    for (int t = 1; t <= num_team; t++) {
        double eff_sum = 0;
        double div_sum = 0;

        // efficiency
        for (int j = 0; j < num_each_t; j++)
            eff_sum += eff[team[t][j]];

        // diversity
        for (int i = 0; i < num_each_t; i++)
            for (int j = i + 1; j < num_each_t; j++)
                div_sum += div_in[team[t][i]][team[t][j]];

        w_eff[t] = eff_sum;
        w_div[t] = div_sum;
    }

    // -------------------------------------------
    // 7. Update global objective
    // -------------------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}


int** Hyper_heuristic::LLH14(int** team) {
    // LLH15: Random perturbation—swap k members between allocated teams and pool
    // LLH1-style full recomputation (safe).

    int team0 = 0;
    int k = 2;  // number of random swaps
    int poolSize = pool_size_unallocated();

    if (num_team < 1 || num_each_t <= 0 || poolSize <= 0)
        return team;

    // ------------------------------------------
    // 1. Perform k random swaps
    // ------------------------------------------
    for (int s = 0; s < k; s++) {
        int t_alloc = rand_alloc_team();
        int idx_alloc = rand_member_idx();
        int idx_pool  = rand_pool_idx(poolSize);

        std::swap(team[t_alloc][idx_alloc], team[team0][idx_pool]);
    }

    // ------------------------------------------
    // 2. Repair state[] and address[]
    // ------------------------------------------
    poolSize = pool_size_unallocated();

    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? poolSize : num_each_t);

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // ------------------------------------------
    // 3. Recompute efficiencies
    // ------------------------------------------
    w_eff[0] = 0;

    for (int t = 1; t <= num_team; t++) {
        double sum = 0;
        for (int j = 0; j < num_each_t; j++)
            sum += eff[team[t][j]];
        w_eff[t] = sum;
    }

    // ------------------------------------------
    // 4. Recompute pairwise diversity
    // ------------------------------------------
    w_div[0] = 0;

    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;

        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++) {
                int b = team[t][j];
                sum_div += div_in[a][b];
            }
        }

        w_div[t] = sum_div;
    }

    // ------------------------------------------
    // 5. Determine new objective results
    // ------------------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}



int** Hyper_heuristic::LLH15(int** team) {
    // LLH16: Swap a small random subset of members between two allocated teams.
    // LLH1-style: Full recomputation for safety and correctness.

    if (num_team < 2 || num_each_t < 2)
        return team;

    int num_elements = 4;
    if (num_elements > num_each_t)
        num_elements = num_each_t / 2;

    // ------------------------------------------
    // 1. Pick two distinct allocated teams
    // ------------------------------------------
    int t1 = rand_alloc_team();
    int t2 = rand_alloc_team();
    while (t2 == t1)
        t2 = rand_alloc_team();

    // ------------------------------------------
    // 2. Swap first num_elements members
    // ------------------------------------------
    for (int i = 0; i < num_elements; i++) {
        std::swap(team[t1][i], team[t2][i]);
    }

    // ------------------------------------------
    // 3. Repair membership mapping
    // ------------------------------------------
    int poolSize = pool_size_unallocated();

    for (int t = 0; t <= num_team; t++) {
        int limit = (t == 0 ? poolSize : num_each_t);

        for (int j = 0; j < limit; j++) {
            int n = team[t][j];
            state[n]   = t;
            address[n] = j;
        }
    }

    // ------------------------------------------
    // 4. Recompute efficiencies
    // ------------------------------------------
    w_eff[0] = 0;

    for (int t = 1; t <= num_team; t++) {
        double sum = 0;
        for (int j = 0; j < num_each_t; j++)
            sum += eff[team[t][j]];
        w_eff[t] = sum;
    }

    // ------------------------------------------
    // 5. Recompute pairwise diversity
    // ------------------------------------------
    w_div[0] = 0;

    for (int t = 1; t <= num_team; t++) {
        double sum_div = 0.0;
        for (int i = 0; i < num_each_t; i++) {
            int a = team[t][i];
            for (int j = i + 1; j < num_each_t; j++)
                sum_div += div_in[a][team[t][j]];
        }
        w_div[t] = sum_div;
    }

    // ------------------------------------------
    // 6. Determine new objective values
    // ------------------------------------------
    int idx_e = min_func(w_eff, num_team);
    int idx_d = min_func(w_div, num_team);

    f_cur     = w_eff[idx_e];
    f_cur_div = w_div[idx_d];
    repair_solution();
    return team;
}
*/
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
        f_cur = feasible_local_search();
        // Metaheuristic: infeasible local search: Apply local hill-climbing / feasible_local_search() on current solution.
        break;

    case 1:
        Scurrent = infeasible_local_search();
        // Metaheuristic: infeasible local search
        break;

    case 2:
        Scurrent = fits();
        // Metaheuristic: Tabu Search (FITS)
        break;

    case 3:
        Scurrent = iterated_local_search();
        // Metaheuristic: Iterated Local Search (ILS)
        break;

    case 4:
        Scurrent = simulated_annealing();
        // Metaheuristic: Simulated Annealing (SA)
        break;

    case 5:
        Scurrent = memetic();
        // Metaheuristic: Memetic Algorithm
        break;

    case 6:
        Scurrent = great_deluge_algorithm();
        // Metaheuristic: Great Deluge
        break;

    case 7:
        fbest12 = guided_local_search();
        // Metaheuristic: Guided Local Search
        break;

    case 8:
        Scurrent = late_acceptance_hill_climbing();
        // Metaheuristic: LAHC
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
    std::vector<int> heuristics = {17,18,19,20,21,22,23};
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
/*
void Hyper_heuristic::Greedy_Selection_Hyperheuristic_CMCEE(int max_time) {
    // ==========================================================================
    // 1) INITIAL SOLUTION SETUP
    // ==========================================================================
    // Seed the random number generator
    //std::srand(static_cast<unsigned int>(std::time(0)));
    std::cout << "Greedy Hyper-heuristic Framework Start its Processes." << std::endl;

    // Evaluate the current 'team' solution (could be random or pre-generated)
    generate_initialrandom();
    objective_Function(team);
    int cost_eff = f_cur;      // Current solution efficiency
    int cost_div = f_cur_div;  // Current solution diversity

    std::cout << "\tInitial objectives: Efficiency = " << cost_eff
              << ", Diversity = " << cost_div << "\n";

    // Track the best solution found so far
    int best_cost_eff = cost_eff;
    int best_cost_div = cost_div;
    int best_found_at = 0;

    // Keep pointers to solutions for acceptance/reversion
    //int** best_solution1   = team;  // stores the best solution
    //int** current_solution = team;  // currently accepted solution
    // Store deep copies of current and best solutions
    int** current_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** best_solution1  = deep_copy_solution(team, num_node, num_team, num_each_t);
    int* team_size = new int[num_team + 1];

    // Assign team sizes
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;

    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;

    // ==========================================================================
    // 2) HEURISTICS & PERFORMANCE TRACKING
    // ==========================================================================
    // Pool of low-level heuristics
    std::vector<int> heuristics = {15, 16, 17, 18, 19, 20,21,22,23};
    //std::vector<int> heuristics = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 17, 18, 19, 20};

    // "heuristic_scores" tracks the total improvement each heuristic has achieved
    std::map<int, double> heuristic_scores;
    for (auto h : heuristics) {
        heuristic_scores[h] = 0.0;
    }

    // Data structures for analysis
    std::vector<int>    objective_values;
    std::vector<int>    diversity_values;
    std::vector<double> iteration_times;

    double total_elapsed_time = 0.0;

    // ==========================================================================
    // 3) FILE I/O SETUP
    // ==========================================================================
    // Create/log directory
    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";
    if (!std::filesystem::exists(folder_path)) {
        try {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return;
        }
    }

    // Define file path for iteration logs
    std::filesystem::path results_file = folder_path / ("Greedy_Selection_HH_CMCEE_" + instanceName + "_results.txt");
    std::ofstream iteration_outfile(results_file);
    if (!iteration_outfile) {
        std::cerr << "Error: Unable to open results file: " << results_file << "\n";
        return;
    }
    iteration_outfile << "Iteration\tSelected Heuristic\tEfficiency\tDiversity\tTime (s)\tDelta\n";

    // ==========================================================================
    // 4) MAIN LOOP
    // ==========================================================================
    clock_t total_start_time = clock();
    int iteration = 0;
    int selected_heuristic = -1;
    int iter = 0;  // an additional iteration counter

    // Track how many consecutive iterations the global best solution fails to improve
    int no_improvement_count = 0;
    const int NO_IMPROVEMENT_LIMIT = 100;  // Terminate if no improvement for 100 consecutive iterations

    // The loop runs as long as EITHER time < max_time OR iter < 1000
    while ((static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC < max_time))
    {
        iteration++;
        iter++;
        clock_t iteration_start_time = clock();

        // ------------------------------------------------
        // 4a) Pick a heuristic
        // ------------------------------------------------
        if (iteration <= static_cast<int>(heuristics.size())) {
            // First N iterations: apply each heuristic once in sequence
            selected_heuristic = heuristics[iteration - 1];
        } else {
            // After that, pick the best-performing heuristic so far
            selected_heuristic = std::max_element(
                heuristic_scores.begin(), heuristic_scores.end(),
                [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                    return a.second < b.second;
                })->first;
        }

        // ------------------------------------------------
        // 4b) Apply the selected heuristic & measure CPU time
        // ------------------------------------------------
        clock_t heuristic_start_time = clock();
        ApplyHeuristic(selected_heuristic, team);
        double heuristic_time = static_cast<double>(clock() - heuristic_start_time) / CLOCKS_PER_SEC;
        total_elapsed_time += heuristic_time;

        // ------------------------------------------------
        // 4c) Evaluate the new solution
        // ------------------------------------------------
        objective_Function(team);
        int new_cost_eff = f_cur;
        int new_cost_div = f_cur_div;

        // Compute delta = new_eff - old_eff for acceptance
        double delta = new_cost_eff - cost_eff;

        // ------------------------------------------------
        // 4d) Accept/Reject the new solution
        // ------------------------------------------------
        bool improved = false;
        if (delta > 0) {
            // Accept => Update the current solution
            cost_eff = new_cost_eff;
            cost_div = new_cost_div;
            //current_solution = team; // newly improved solution

            free_solution(current_solution , num_node, num_team, num_each_t);
            current_solution  = deep_copy_solution(team, num_node, num_team, num_each_t);

            // Accumulate improvement in heuristic_scores
            heuristic_scores[selected_heuristic] += delta;
            improved = true;

            // If it's the best so far & meets min_div
            if ((cost_eff > best_cost_eff) && (cost_div >= min_div)) {
                best_cost_eff = cost_eff;
                best_cost_div = cost_div;
                //best_solution1 = team;
                free_solution(best_solution1 , num_node, num_team, num_each_t);
                best_solution1  = deep_copy_solution(team, num_node, num_team, num_each_t);
                best_eff       = cost_eff;
                best_div       = cost_div;
                time_taken     = heuristic_time;
                best_found_at  = iteration;

                // Copy solution arrays if needed
                for (int m = 0; m < num_node; m++) {
                    fbest_solution[m] = best_solution[m];
                }
                for (int m = 1; m <= num_team; m++ ) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        } else {
            // No improvement => revert
            delta = 0.0;
            //team = current_solution;
            for (int i = 0; i <= num_team; ++i)
            for (int j = 0; j < team_size[i]; ++j)
                team[i][j] = current_solution[i][j];

        }

        // ------------------------------------------------
        // 4e) Check if best solution improved
        // ------------------------------------------------
        if (improved && (cost_eff >= best_cost_eff)) {
            // There's a global improvement => reset no_improvement_count
            no_improvement_count = 0;
        } else {
            // No global improvement => increment
            no_improvement_count++;
        }
        // Terminate early if no improvement for 100 consecutive iterations
        if (no_improvement_count >= NO_IMPROVEMENT_LIMIT) {
            std::cout << "\nNo improvement in the global best solution for "
                      << NO_IMPROVEMENT_LIMIT << " consecutive iterations.\n"
                      << "Terminating early.\n";
            break;
        }

        // ------------------------------------------------
        // 4f) Logging / Analysis
        // ------------------------------------------------
        objective_values.push_back(cost_eff);
        diversity_values.push_back(cost_div);

        // Console output for iteration
        std::cout << "Iteration: " << iteration
                  << " | Selected Heuristic: " << selected_heuristic
                  << " | Efficiency: " << new_cost_eff
                  << " | Diversity: " << new_cost_div
                  << " | Best Efficiency: " << best_cost_eff
                  << " | Best Diversity: " << best_cost_div
                  << " | Time Taken: " << heuristic_time << " seconds"
                  << " | Delta: " << delta
                  << std::endl;

        // Write iteration results to file
        double iteration_time = static_cast<double>(clock() - iteration_start_time) / CLOCKS_PER_SEC;
        iteration_times.push_back(iteration_time);

        iteration_outfile << iteration << "\t"
                          << selected_heuristic << "\t"
                          << cost_eff << "\t"
                          << cost_div << "\t"
                          << iteration_time << "\t"
                          << delta << "\n";
    } // end while

    // ==========================================================================
    // 5) FINAL SUMMARY
    // ==========================================================================
    double total_time = static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC;

    // Calculate statistics
    if (!objective_values.empty()) {
        double total_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0);
        average_objective = total_objective / objective_values.size();
    }

    if (!diversity_values.empty()) {
        double total_diversity = std::accumulate(diversity_values.begin(), diversity_values.end(), 0.0);
        average_diversity = total_diversity / diversity_values.size();
    }

    if (!objective_values.empty()) {
        worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    }

    double average_cpu_time = 0.0;
    if (!iteration_times.empty()) {
        double total_iteration_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
        average_cpu_time = total_iteration_time / iteration_times.size();
    }

    // Close iteration log file
    iteration_outfile.close();

    // Save best solution summary
    std::filesystem::path summary_path = folder_path / ("Greedy_Selection_HH_CMCEE_" + instanceName + "_Summary.txt");
    {
        std::ofstream summary_file(summary_path);
        if (summary_file) {
            summary_file << "Best selected heuristic: " << selected_heuristic << "\n";
            summary_file << "Best Efficiency: " << best_cost_eff << "\n";
            summary_file << "Best Diversity: " << best_cost_div << "\n";
            summary_file << "Found at Iteration: " << best_found_at << "\n";
            summary_file << "Total Time: " << total_time << " seconds\n";
            summary_file << "Average Objective Function Value: " << average_objective << "\n";
            summary_file << "Average Diversity Value: " << average_diversity << "\n";
            summary_file << "Worst Objective Function Value: " << worst_objective << "\n";
            summary_file << "Average CPU Time per Iteration: " << average_cpu_time << " seconds\n";
            summary_file.close();
        } else {
            std::cerr << "Error: Unable to save summary file.\n";
        }
    }

    // Print summary to console
    std::cout << "Best selected heuristic: " << selected_heuristic << "\n"
              << "Best Efficiency: " << best_cost_eff << "\n"
              << "Best Diversity: " << best_cost_div << "\n"
              << "Found at Iteration: " << best_found_at << "\n"
              << "Total Time: " << total_time << " seconds\n"
              << "Average Objective Function Value: " << average_objective << "\n"
              << "Average Diversity Value: " << average_diversity << "\n"
              << "Worst Objective Function Value: " << worst_objective << "\n"
              << "Average CPU Time per Iteration: " << average_cpu_time << " seconds\n";

    // Final check or post-processing
    check_best_solution();

    // Print final messages
    std::cout << "Results saved to " << results_file << "\n";
    std::cout << "Greedy Hyper-heuristic Framework Finished its Processes." << std::endl;
}




*/
/*
void Hyper_heuristic::Greedy_Selection_Hyperheuristic_CMCEE(int max_time) {
    // Initialize random seed for reproducibility
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Initialize solution
    generate_initialrandom();
    objective_Function(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    int worst_OF = f_cur;
    int sum_OF = 0, count_OF = 0;
    std::cout << "\t Initial objectives eff and div: " << cost_eff << ", " << cost_div << std::endl;
    int **best_solution1 = team;

    // Initialize best solution trackers
    int best_cost_eff = cost_eff, best_cost_div = cost_div;
    int best_found_at = 0;

    // Data structures for heuristic scores and performance tracking
    std::map<int, double> heuristic_scores;
    std::vector<int> heuristics = {15, 16, 17, 18, 19, 20};
    for (auto &h : heuristics) {
        heuristic_scores[h] = 0.0;
    }

    std::map<int, std::vector<int>> heuristic_costs;
    std::map<int, std::vector<double>> heuristic_times;
    std::vector<int> Selected;
    std::vector<int> objective_values;
    double total_elapsed_time = 0.0;

    int** previous_solution = team;
    best_solution1 = team;

    clock_t total_start_time = clock();
    int iteration = 0;

    // Open iteration-level results file with headers
    std::string folder_path = "D:/Result Hyper-heuristic Models/";
    std::string iteration_file_path = folder_path + "Greedy_Selection_Hyperheuristic_CMCEE_results.txt";
    std::ofstream iteration_outfile(iteration_file_path);
    if (!iteration_outfile) {
        std::cerr << "Error opening iteration results file: " << iteration_file_path << std::endl;
    } else {
        // Write headers
        iteration_outfile << "Iteration\tSelected Heuristic\tEfficiency\tDiversity\tTime Taken (seconds)\tDelta\n";
    }

    while (static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC < max_time) {
        iteration++;
        clock_t iteration_start_time = clock();

        // Check if all heuristic scores are 0
        bool all_scores_zero = true;
        for (auto &pair : heuristic_scores) {
            if (pair.second != 0.0) {
                all_scores_zero = false;
                break;
            }
        }

        if (all_scores_zero) {
            // Invoke each heuristic using the current solution
            for (auto &h : heuristics) {
                std::cout << "  Iteration: " << iteration
                          << "\t, Invoking Heuristic LLH[" << h << "] with current cost_eff: "
                          << cost_eff << ", cost_div: " << cost_div << std::endl;

                int** temp_solution = team;

                clock_t heuristic_start_time = clock();
                ApplyHeuristic(h, temp_solution);
                double heuristic_time = static_cast<double>(clock() - heuristic_start_time) / CLOCKS_PER_SEC;
                heuristic_times[h].push_back(heuristic_time);
                total_elapsed_time += heuristic_time;

                objective_Function(temp_solution);
                int newcost_eff = f_cur;
                int newcost_div = f_cur_div;

                std::cout << "\t\tAfter Heuristic LLH[" << h << "], newcost_eff: " << newcost_eff
                          << ", newcost_div: " << newcost_div << std::endl;

                int cost_change_eff = newcost_eff - cost_eff;
                int cost_change_div = newcost_div - cost_div;

                heuristic_scores[h] = cost_change_eff + cost_change_div;

                // Reset score if no improvement
                if (cost_change_eff < 0 || cost_change_div < 0) {
                    heuristic_scores[h] = 0.0;
                    std::cout << "\t\t Heuristic LLH[" << h << "] did not improve the solution. Score reset to 0." << std::endl;
                }
            }
        }

        // Select heuristic with highest score
        int selected_heuristic = -1;
        double max_score = -std::numeric_limits<double>::infinity();
        for (auto &pair : heuristic_scores) {
            if (pair.second > max_score) {
                max_score = pair.second;
                selected_heuristic = pair.first;
            }
        }

        // Tie-breaking
        std::vector<int> top_heuristics;
        for (auto &pair : heuristic_scores) {
            if (pair.second == max_score && max_score > 0.0) {
                top_heuristics.push_back(pair.first);
            }
        }

        if (top_heuristics.size() > 1) {
            selected_heuristic = top_heuristics[std::rand() % top_heuristics.size()];
            std::cout << "\t\tTie detected. Selected Heuristic LLH[" << selected_heuristic << "] via tie-breaking." << std::endl;
        }

        if (selected_heuristic == -1) {
            selected_heuristic = heuristics[std::rand() % heuristics.size()];
            std::cout << "\t\tNo positive heuristic scores. Randomly selected Heuristic LLH[" << selected_heuristic << "]." << std::endl;
        }

        Selected.push_back(selected_heuristic);

        std::cout << "  Iteration: " << iteration
                  << "\t\t, Selected Heuristic LLH[" << selected_heuristic
                  << "], cost_eff: " << cost_eff << ", cost_div: " << cost_div << std::endl;

        clock_t selected_start_time = clock();
        ApplyHeuristic(selected_heuristic, team);
        double selected_elapsed_time = static_cast<double>(clock() - selected_start_time) / CLOCKS_PER_SEC;
        heuristic_times[selected_heuristic].push_back(selected_elapsed_time);
        heuristic_costs[selected_heuristic].push_back(cost_eff);
        total_elapsed_time += selected_elapsed_time;

        objective_Function(team);
        int newcost_eff_selected = f_cur;
        int newcost_div_selected = f_cur_div;

        std::cout << "\t\tAfter Selected Heuristic LLH[" << selected_heuristic
                  << "], newcost_eff: " << newcost_eff_selected
                  << ", newcost_div: " << newcost_div_selected << "\n";

        bool accepted = false;
        if ((newcost_eff_selected > cost_eff) && (newcost_div_selected >= cost_div)) {
            // Accept
            previous_solution = team;
            cost_eff = newcost_eff_selected;
            cost_div = newcost_div_selected;
            accepted = true;

            if ((newcost_eff_selected > best_cost_eff) && (newcost_div_selected >= best_cost_div)) {
                best_solution1 = team;
                best_cost_eff = newcost_eff_selected;
                best_cost_div = newcost_div_selected;
                best_found_at = iteration;
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++ ) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
                std::cout << "\t\tNew best solution found at iteration " << iteration << " with eff: "
                          << best_cost_eff << " and div: " << best_cost_div << std::endl;
            }
        } else {
            // Reject
            team = previous_solution;
            std::cout << "\t\tSolution not improved. Reverting to previous solution." << std::endl;
        }

        if (accepted) {
            heuristic_scores[selected_heuristic] += (newcost_eff_selected - cost_eff) + (newcost_div_selected - cost_div);
            std::cout << "\t\tHeuristic LLH[" << selected_heuristic << "] accepted. Score updated to "
                      << heuristic_scores[selected_heuristic] << std::endl;
        } else {
            heuristic_scores[selected_heuristic] = 0.0;
            std::cout << "\t\tHeuristic LLH[" << selected_heuristic << "] rejected. Score reset to 0." << std::endl;
        }

        objective_values.push_back(cost_eff);

        if (cost_eff < worst_OF) {
            worst_OF = cost_eff;
        }
        sum_OF += cost_eff;
        count_OF++;

        double iteration_time = static_cast<double>(clock() - iteration_start_time) / CLOCKS_PER_SEC;
        heuristic_times[selected_heuristic].push_back(iteration_time);

        // Compute a delta if needed, otherwise set to 0
        double delta = 0.0; // No delta calculation defined, just using 0.0 here

        // Write iteration results in desired format to iteration_outfile
        if (iteration_outfile) {
            iteration_outfile << iteration << "\t"
                              << selected_heuristic << "\t"
                              << cost_eff << "\t"
                              << cost_div << "\t"
                              << iteration_time << "\t"
                              << delta << "\n";
        }
    }

    double total_time = static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC;
    double average_OF = (count_OF > 0) ? (sum_OF / static_cast<double>(count_OF)) : 0.0;

    double average_computation_time = 0.0;
    {
        double total_time_all = 0.0;
        int total_iterations = 0;
        for (auto &pair : heuristic_times) {
            total_time_all += std::accumulate(pair.second.begin(), pair.second.end(), 0.0);
            total_iterations += (int)pair.second.size();
        }
        if (total_iterations > 0) {
            average_computation_time = total_time_all / (double)total_iterations;
        }
    }

    int worse_performing_heuristics = 0;
    for (auto &pair : heuristic_scores) {
        if (pair.second < 0.0) {
            worse_performing_heuristics++;
        }
    }

    int best_pair_heuristic1 = -1, best_pair_heuristic2 = -1;
    double best_pair_score = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < heuristics.size(); i++) {
        for (size_t j = i + 1; j < heuristics.size(); j++) {
            int h1 = heuristics[i];
            int h2 = heuristics[j];
            double combined_score = heuristic_scores[h1] + heuristic_scores[h2];
            if (combined_score > best_pair_score) {
                best_pair_score = combined_score;
                best_pair_heuristic1 = h1;
                best_pair_heuristic2 = h2;
            }
        }
    }

    // Close iteration-level results file
    if (iteration_outfile) {
        iteration_outfile.close();
    }

    // Save summary to file
    std::ofstream result_file("D:\\Result Hyper-heuristic Models\\Greedy_Selection_Hyperheuristic_CMCEE.txt");
    if (result_file.is_open()) {
        result_file << "Final best objectives:\n";
        result_file << "Efficiency: " << best_cost_eff << "\n";
        result_file << "Diversity: " << best_cost_div << "\n";
        result_file << "Found at iteration: " << best_found_at << "\n\n";

        result_file << "Statistics:\n";
        result_file << "Total time: " << total_time << " seconds\n";
        result_file << "Average objective function value: " << average_OF << "\n";
        result_file << "Worst objective function value: " << worst_OF << "\n";
        result_file << "Average computation time per iteration: " << average_computation_time << " seconds\n";
        if (best_found_at != 0) {
            result_file << "Time to Find Best Solution: Iteration " << best_found_at << "\n";
        } else {
            result_file << "Best solution was not updated during the optimization.\n";
        }
        result_file << "Number of Best Improving Solutions: " << (best_found_at != 0 ? 1 : 0) << "\n";
        result_file << "Number of Worse Performing Heuristics: " << worse_performing_heuristics << "\n";
        if (best_pair_heuristic1 != -1 && best_pair_heuristic2 != -1) {
            result_file << "Pair of Heuristics with Best Acceptance Move Strategy: "
                        << best_pair_heuristic1 << " and " << best_pair_heuristic2 << "\n";
        } else {
            result_file << "Not enough heuristics to determine a pair with the best acceptance move strategy.\n";
        }

        result_file << "\nHeuristic Performance:\n";
        for (auto &h : heuristics) {
            double total_h_time = 0.0;
            if (!heuristic_times[h].empty()) {
                total_h_time = std::accumulate(heuristic_times[h].begin(), heuristic_times[h].end(), 0.0);
            }

            result_file << "Heuristic " << h << ":\n";
            result_file << "  Total Time: " << std::fixed << std::setprecision(4) << total_h_time << " seconds\n";
            result_file << "  Usage Count: " << heuristic_costs[h].size() << "\n";
            result_file << "  Improvement Count: " << heuristic_costs[h].size() << "\n";
            result_file << "  Total Performance: " << heuristic_scores[h] << "\n\n";
        }

        result_file.close();
        std::cout << "Results saved to D:\\Result Hyper-heuristic Models\\Greedy_Selection_Hyperheuristic_CMCEE.txt" << std::endl;
    } else {
        std::cerr << "Error: Could not open the file for writing results.\n";
    }

    // Console output summary
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Statistical Analysis:" << std::endl;
    std::cout << "Best Objective Function (BOF): " << best_cost_eff << std::endl;
    std::cout << "Worst Objective Function (OF): " << worst_OF << std::endl;
    std::cout << "Average Objective Function (OF): " << average_OF << std::endl;
    std::cout << "Average Computation Time per Iteration: " << average_computation_time << " seconds" << std::endl;
    if (best_found_at != 0) {
        std::cout << "Time to Find Best Solution: Iteration " << best_found_at << std::endl;
    } else {
        std::cout << "Best solution was not updated during the optimization." << std::endl;
    }
    std::cout << "Number of Best Improving Solutions: " << (best_found_at != 0 ? 1 : 0) << std::endl;
    std::cout << "Number of Worse Performing Heuristics: " << worse_performing_heuristics << std::endl;
    if (best_pair_heuristic1 != -1 && best_pair_heuristic2 != -1) {
        std::cout << "Pair of Heuristics with Best Acceptance Move Strategy: "
                  << best_pair_heuristic1 << " and " << best_pair_heuristic2 << std::endl;
    } else {
        std::cout << "Not enough heuristics to determine a pair with the best acceptance move strategy." << std::endl;
    }

    check_best_solution();
    //free_memory();
}
*//*
void Hyper_heuristic::Random_Selection_Hyperheuristic_CMCEE(int max_time) {
    std::cout << "Random Selection Hyper-heuristic Framework Start its Processes." << std::endl;
    // Initialize solution
    generate_initialrandom();
    int** previous_solution = team; // Store the previous solution for reverting if needed
    int** best_solution112= team;
    // Calculate initial objectives
    objective_Function1(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    int best_cost_eff = cost_eff;
    int best_cost_div = cost_div;

    std::cout << "Initial objectives - Efficiency: " << cost_eff << ", Diversity: " << cost_div << std::endl;

    // Define heuristics and iteration count
    std::vector<int> heuristics = {15, 16, 17, 18, 19, 20,21,22,23}; // List of low-level heuristics
    //std::vector<int> heuristics = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 17, 18, 19, 20};

    int max_iterations = 100; // Maximum number of iterations
    std::vector<int> selected_heuristics; // Track selected heuristics

    // Data structures for analysis
    std::map<int, std::vector<int>> heuristic_costs; // Track costs for each heuristic
    std::map<int, std::vector<double>> heuristic_times; // Track times for each heuristic
    std::vector<int> objective_values; // Store efficiency values over iterations
    std::vector<int> diversity_values; // Store diversity values over iterations
    std::vector<double> iteration_times; // Store time taken for each iteration

    double total_elapsed_time = 0.0;

    // File output: Create or open file to save iteration-wise results
    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";


    // Check if the directory exists, create it if it doesn't
    if (!std::filesystem::exists(folder_path)) {
        try {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return; // Exit the function if directory creation fails
        }
    }

    // Define the full file path for saving the results
    std::filesystem::path results_file = folder_path / ("Random_Selection_HH_CMCEE_" + instanceName + "_results.txt");
    std::ofstream outfile(results_file);
    if (!outfile) {
        std::cerr << "Error: Unable to open results file: " << results_file << std::endl;
        return;
    }
    outfile << "Iteration\tSelected_Heuristic\tOld_Eff\tOld_Div\tNew_Eff\tNew_Div\tTime_Taken(seconds)\n";

    // Start timing
    clock_t total_start_time = clock();
    int iteration = 0;

    // Main loop
    while (static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC < max_time && iteration < max_iterations) {
        iteration++;

        // Store old values before applying heuristic
        int old_eff = cost_eff;
        int old_div = cost_div;

        // Select a random heuristic
        int selected_heuristic = heuristics[std::rand() % heuristics.size()];
        //std::cout << "Iteration " << iteration << ": Applying Heuristic LLH[" << selected_heuristic << "]\n";

        // Apply the selected heuristic
        clock_t heuristic_start_time = clock();
        ApplyHeuristic(selected_heuristic, team);
        double elapsed_time = static_cast<double>(clock() - heuristic_start_time) / CLOCKS_PER_SEC;
        total_elapsed_time += elapsed_time;

        // Evaluate the new solution
        objective_Function1(team);
        int new_cost_eff = f_cur;
        int new_cost_div = f_cur_div;

        //std::cout << "Iteration " << iteration << ": New Efficiency = " << new_cost_eff << ", New Diversity = " << new_cost_div << "\n";

        // Accept or reject the new solution
        if ((new_cost_eff > cost_eff) && (new_cost_div >= min_div)) {
            // Accept the new solution
            cost_eff = new_cost_eff;
            cost_div = new_cost_div;
            previous_solution = team;
            selected_heuristics.push_back(selected_heuristic);

            // Update the best solution if improved
            if (new_cost_eff > best_cost_eff) {
                best_cost_eff = new_cost_eff;
                best_cost_div = new_cost_div;
                best_solution112 = team;
                time_taken = elapsed_time;

                // Update best solution arrays
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        } else {
            // Revert to the previous solution
            team = previous_solution;
        }

        // Collect data for analysis
        heuristic_costs[selected_heuristic].push_back(new_cost_eff);
        heuristic_times[selected_heuristic].push_back(elapsed_time);
        objective_values.push_back(cost_eff);
        diversity_values.push_back(cost_div);
        iteration_times.push_back(elapsed_time);

        // Log iteration results to file
        outfile << iteration << "\t"
                << selected_heuristic << "\t"
                << old_eff << "\t"
                << old_div << "\t"
                << new_cost_eff << "\t"
                << new_cost_div << "\t"
                << elapsed_time << "\n";
      std::cout << "Iteration: " << iteration
                << " | Selected Heuristic: " << selected_heuristic
                << " | Efficiency: " << old_eff
                << " | Diversity: " << old_div
                << " | Best Efficiency: " << best_cost_eff
                << " | Best Diversity: " << best_cost_div
                << " | Time Taken: " << elapsed_time << " seconds"
                << std::endl;
    }

    // Close the iteration results file
    outfile.close();
    std::cout << "Iteration-wise results saved to " << results_file << std::endl;

    // Calculate statistical metrics
    double average_objective = 0.0;
    if (!objective_values.empty()) {
        double total_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0);
        average_objective = total_objective / objective_values.size();
    }

    double average_diversity = 0.0;
    if (!diversity_values.empty()) {
        double total_diversity = std::accumulate(diversity_values.begin(), diversity_values.end(), 0.0);
        average_diversity = total_diversity / diversity_values.size();
    }

    int worst_objective = *std::min_element(objective_values.begin(), objective_values.end());

    double average_cpu_time = 0.0;
    if (!iteration_times.empty()) {
        double total_iteration_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
        average_cpu_time = total_iteration_time / iteration_times.size();
    }

    // Save summary to a separate file
    std::filesystem::path summary_file_path = folder_path / ("Random_Selection_HH_CMCEE_" + instanceName + "_summary.txt");
    std::ofstream summary_file(summary_file_path);
    if (summary_file) {
        summary_file << "Best Efficiency: " << best_cost_eff << "\n";
        summary_file << "Best Diversity: " << best_cost_div << "\n";
        summary_file << "Total Time: " << total_elapsed_time << " seconds\n";
        summary_file << "Average Objective Function Value: " << average_objective << "\n";
        summary_file << "Average Diversity Value: " << average_diversity << "\n";
        summary_file << "Worst Objective Function Value: " << worst_objective << "\n";
        summary_file << "Average CPU Time per Iteration: " << average_cpu_time << " seconds\n";
        summary_file.close();
        std::cout << "Summary results saved to " << summary_file_path << std::endl;
    } else {
        std::cerr << "Error: Unable to save summary file.\n";
    }

    // Final console output
    std::cout << "\n--- Summary ---\n";
    std::cout << "Best Efficiency: " << best_cost_eff << "\n";
    std::cout << "Best Diversity: " << best_cost_div << "\n";
    std::cout << "Total Time: " << total_elapsed_time << " seconds\n";
    std::cout << "Selected Heuristics: ";
    for (const auto& h : selected_heuristics) std::cout << h << " ";
    std::cout << std::endl;

    check_best_solution();
    std::cout << "Random Selection Hyper-heuristic Framework Finished its Processes." << std::endl;

}
*/
void Hyper_heuristic::Random_Selection_Hyperheuristic_CMCEE(int max_time) {
    std::cout <<
        "=============================================================================\n"
        "Random Selection Hyper-Heuristic Framework Start its Processes.\n"
        "=============================================================================\n";

    // ------------------------------------------------------------
    // INITIALIZATION
    // ------------------------------------------------------------
    generate_initialrandom();

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
    generate_initialrandom();
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


void Hyper_heuristic::MAHH_Algorithm(int max_time) {
    // Initialize matrices for tracking heuristic performance
    std::map<std::vector<int>, double> reward_matrix;  // Maps sequences to rewards
    std::map<std::vector<int>, int> count_matrix;      // Tracks the number of times sequences are applied

    // File output: Create or open file to save iteration-wise results
    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";

    // Check if the directory exists, create it if it doesn't
    if (!std::filesystem::exists(folder_path)) {
        try {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return; // Exit the function if directory creation fails
        }
    }

    // Define the full file path for saving the results
    std::filesystem::path results_file = folder_path / ("MAHH_Selection_CMCEE_" + instanceName + "_results.txt");
    std::ofstream outfile(results_file);
    if (!outfile) {
        std::cerr << "Error: Unable to open results file: " << results_file << std::endl;
        return;
    }
    outfile << "Iteration\tSelected_Sequence\tOld_Eff\tOld_Div\tNew_Eff\tNew_Div\tTime_Taken(seconds)\tReward\n";

    // Initialize solution
    generate_initialrandom();
    objective_Function1(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;

    // Initialize solutions
    // Store deep copies of current and best solutions
    int** s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** s_best    = deep_copy_solution(team, num_node, num_team, num_each_t);
    // Assign team sizes
    // Allocate team_size array globally
    team_size = new int[num_team + 1];
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;

    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;

    int fbest_solution_eff = cost_eff;
    int fbest_solution_div = cost_div;
    int cost_eff_temp;
    int cost_div_temp;
    // Initialize sequence
    std::vector<int> sequence;

    // Randomly set the last heuristic
    std::vector<int> heuristics = {15, 16, 17, 18, 19, 20}; // Example low-level heuristics
    int i_last = heuristics[std::rand() % heuristics.size()];

    // Data structures for analysis
    std::vector<int> objective_values; // Store efficiency values over iterations
    std::vector<int> diversity_values; // Store diversity values over iterations
    std::vector<double> iteration_times; // Store time taken for each iteration

    // Start timing
    clock_t start_time = clock();
    double elapsed_time = 0.0; // To store elapsed time
    double reward= 0.0;
    int iteration = 0; // Track iteration count
    while (elapsed_time < max_time) {
        iteration++;

        // Sample the next low-level heuristic considering the last one
        int i_next = heuristics[std::rand() % heuristics.size()];
        sequence.push_back(i_next);

        // Decision to apply the sequence
        int u_next = std::rand() % 2; // Random binary decision (0 or 1)
        if (u_next == 1) {
            // Create a temporal solution
            int** s_temp = team; // Properly allocate memory

            ApplySequence(sequence, s_temp); // Apply the sequence to the temporal solution

            // Evaluate the new solution
            objective_Function(s_temp);
            cost_eff_temp = f_cur;
            cost_div_temp = f_cur_div;

            // Calculate reward
            reward = (cost_eff_temp - cost_eff) + (cost_div_temp - cost_div);
            reward = std::max(reward, 0.0); // Only consider positive rewards

            // Update matrices using MAB strategy
            reward_matrix[sequence] += reward;
            count_matrix[sequence]++;

            // Accept or reject the new solution
            if ((cost_eff_temp > cost_eff) && (cost_div_temp >= min_div)) {
                // Accept the temporal solution
                free_solution(s_current, num_node, num_team, num_each_t);
                s_current = deep_copy_solution(s_temp, num_node, num_team, num_each_t);
                cost_eff = cost_eff_temp;
                cost_div = cost_div_temp;
            }

            // If the temporal solution is better than the best solution
            if ((cost_eff_temp > fbest_solution_eff) && (cost_div_temp >= min_div)) {
                free_solution(s_best, num_node, num_team, num_each_t);
                s_best = deep_copy_solution(s_temp, num_node, num_team, num_each_t);

                fbest_solution_eff = cost_eff_temp;
                fbest_solution_div = cost_div_temp;
                best_eff = cost_eff_temp;
                best_div = cost_div_temp;

                // Update elapsed time
                elapsed_time = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
                time_taken = elapsed_time;

                // Update best solution arrays
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            } else {
                for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = s_current[i][j];

            }
        }

        // Record objective function values
        objective_values.push_back(cost_eff);
        diversity_values.push_back(cost_div);
        iteration_times.push_back(elapsed_time);

        // Output results for the current iteration
        std::cout << "Iteration: " << iteration
                  << " | Selected Heuristic: " << i_next
                  << " | Efficiency: " << cost_eff
                  << " | Diversity: " << cost_div
                  << " | Best Efficiency: " << fbest_solution_eff
                  << " | Best Diversity: " << fbest_solution_div
                  << " | Elapsed Time: " << elapsed_time << " seconds"
                  << std::endl;

        // Log iteration results to file
        outfile << iteration << "\t";
        for (int h : sequence) outfile << h << " ";
        outfile << "\t" << cost_eff << "\t" << cost_div << "\t"
                << cost_eff_temp << "\t" << cost_div_temp << "\t"
                << elapsed_time << "\t" << reward << "\n";

        // Reset sequence
        sequence.clear();
        i_last = i_next; // Update the last heuristic

        // Update elapsed time
        elapsed_time = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
    }

    // Close the iteration results file
    outfile.close();
    std::cout << "Iteration-wise results saved to " << results_file << std::endl;

    // Calculate statistical metrics
    double average_objective = 0.0;
    if (!objective_values.empty()) {
        double total_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0);
        average_objective = total_objective / objective_values.size();
    }

    double average_diversity = 0.0;
    if (!diversity_values.empty()) {
        double total_diversity = std::accumulate(diversity_values.begin(), diversity_values.end(), 0.0);
        average_diversity = total_diversity / diversity_values.size();
    }

    int worst_objective = *std::min_element(objective_values.begin(), objective_values.end());

    double average_cpu_time = 0.0;
    if (!iteration_times.empty()) {
        double total_iteration_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
        average_cpu_time = total_iteration_time / iteration_times.size();
    }

    // Save summary to a separate file
    std::filesystem::path summary_file_path = folder_path / ("MAHH_Selection_CMCEE_" + instanceName + "_summary.txt");
    std::ofstream summary_file(summary_file_path);
    if (summary_file) {
        summary_file << "Best Efficiency: " << fbest_solution_eff << "\n";
        summary_file << "Best Diversity: " << fbest_solution_div << "\n";
        summary_file << "Total Time: " << elapsed_time << " seconds\n";
        summary_file << "Average Objective Function Value: " << average_objective << "\n";
        summary_file << "Average Diversity Value: " << average_diversity << "\n";
        summary_file << "Worst Objective Function Value: " << worst_objective << "\n";
        summary_file << "Average CPU Time per Iteration: " << average_cpu_time << " seconds\n";
        summary_file << "Heuristic Sequence Rewards:\n";
        for (const auto& pair : reward_matrix) {
            const std::vector<int>& seq = pair.first;
            double reward = pair.second;

            summary_file << "Sequence: ";
            for (int h : seq) summary_file << h << " ";
            summary_file << "| Reward: " << reward
                         << " | Count: " << count_matrix[seq] << "\n";
        }
        summary_file.close();
        std::cout << "Summary results saved to " << summary_file_path << std::endl;
    } else {
        std::cerr << "Error: Unable to save summary file.\n";
    }

    // Final console output
    std::cout << "\n--- Summary ---\n";
    std::cout << "Best Efficiency: " << fbest_solution_eff << "\n";
    std::cout << "Best Diversity: " << fbest_solution_div << "\n";
    std::cout << "Total Time: " << elapsed_time << " seconds\n";
    std::cout << "Average Objective Function Value: " << average_objective << "\n";
    std::cout << "Average Diversity Value: " << average_diversity << "\n";
    std::cout << "Worst Objective Function Value: " << worst_objective << "\n";
    std::cout << "Average CPU Time per Iteration: " << average_cpu_time << " seconds\n";
    std::cout << "Heuristic Sequence Rewards:\n";
    for (const auto& pair : reward_matrix) {
        const std::vector<int>& seq = pair.first;
        double reward = pair.second;

        std::cout << "Sequence: ";
        for (int h : seq) std::cout << h << " ";
        std::cout << "| Reward: " << reward
                  << " | Count: " << count_matrix[seq] << "\n";
    }

    check_best_solution();
}

void Hyper_heuristic::ApplySequence(const std::vector<int>& sequence, int** solution) {
    for (int heuristic : sequence) {
        ApplyHeuristic(heuristic, solution); // Apply each heuristic in the sequence
    }
}

//Implementation of Hyper-heuristic Based Three Multi-Armed bandit Selection Strategies For Solving CMCEE
void Hyper_heuristic::MAHH_Selection_ThreeMAB(int max_time, MABStrategy mab_strategy) {

    std::cout << "Multi-Armed Bandit with Three reward UCB1, EPSILON GREEDY, THOMPSON_SAMPLING Selection Hyper-heuristic Framework Start its Processes." << std::endl;

    // Initialize variables
    std::map<int, double> heuristic_rewards;      // Total rewards per heuristic
    std::map<int, int> heuristic_counts;          // Number of times each heuristic is selected
    std::map<int, double> heuristic_estimates;    // Estimated value of each heuristic
    std::map<int, double> heuristic_alpha;        // Success counts for Thompson Sampling
    std::map<int, double> heuristic_beta;         // Failure counts for Thompson Sampling
    std::vector<int> objective_values;            // Stores objective function values over iterations
    std::vector<double> iteration_times;          // Time taken for each iteration
    std::vector<int> heuristics = {15, 16, 17, 18, 19, 20, 21, 22, 23}; // List of heuristics
    int total_iterations = 0;
    // Data structures for analysis
    std::vector<int> Selected;
    std::vector<int> heuristic_sequence; // To track the sequence of selected heuristics
    std::vector<int> diversity_values; // Stores diversity values over iterations

    // Parameters for epsilon-greedy
    double epsilon = 0.1; // Exploration rate for epsilon-greedy

    // Random number generator for Thompson Sampling
    std::mt19937 generator(std::random_device{}());

    // Initialize solution
    generate_initialrandom();
    objective_Function(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;

    // Initialize best solution
    // Store deep copies of current and best solutions
    int** s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** s_best    = deep_copy_solution(team, num_node, num_team, num_each_t);
    fbest_solution_eff = cost_eff;
    int best_cost_div = cost_div;
    int* team_size = new int[num_team + 1];

    // Assign team sizes
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;

    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;

    // Initialize heuristic statistics
    for (int h : heuristics) {
        heuristic_rewards[h] = 0.0;
        heuristic_counts[h] = 0;
        heuristic_estimates[h] = 0.0;
        heuristic_alpha[h] = 1.0; // For Thompson Sampling (Beta prior)
        heuristic_beta[h] = 1.0;  // For Thompson Sampling (Beta prior)
    }

    // Start timing
    auto start_time = std::chrono::steady_clock::now();
    double elapsed_time = 0.0; // To store elapsed time

    // Main loop
    while (elapsed_time < max_time) {
        total_iterations++;
        auto iteration_start = std::chrono::steady_clock::now();

        int selected_heuristic = -1;

        // Heuristic selection based on specified MAB strategy
        if (mab_strategy == UCB) {
            // UCB1 strategy
            double total_counts = std::accumulate(heuristic_counts.begin(), heuristic_counts.end(), 0.0,
                                                  [](double sum, const std::pair<int, int>& p) { return sum + p.second; });
            double log_total = std::log(std::max(1.0, total_counts));
            double max_ucb = -std::numeric_limits<double>::infinity();

            for (int h : heuristics) {
                double avg_reward = heuristic_counts[h] > 0 ? heuristic_rewards[h] / heuristic_counts[h] : 0.0;
                double ucb = avg_reward + std::sqrt(2.0 * log_total / (heuristic_counts[h] + 1e-5));
                if (ucb > max_ucb) {
                    max_ucb = ucb;
                    selected_heuristic = h;
                }
            }
        } else if (mab_strategy == EPSILON_GREEDY) {
            // Epsilon-Greedy strategy
            double rand_val = ((double) std::rand() / RAND_MAX);
            if (rand_val < epsilon) {
                // Exploration: select a random heuristic
                selected_heuristic = heuristics[std::rand() % heuristics.size()];
            } else {
                // Exploitation: select the best heuristic so far
                selected_heuristic = *std::max_element(heuristics.begin(), heuristics.end(),
                                                       [&](int h1, int h2) {
                                                           return heuristic_estimates[h1] < heuristic_estimates[h2];
                                                       });
            }
        } else if (mab_strategy == THOMPSON_SAMPLING) {
            // Thompson Sampling strategy
            double max_sample = -std::numeric_limits<double>::infinity();
            for (int h : heuristics) {
                // Sample from Beta distribution
                std::gamma_distribution<double> gamma_alpha(heuristic_alpha[h], 1.0);
                std::gamma_distribution<double> gamma_beta(heuristic_beta[h], 1.0);
                double sample_alpha = gamma_alpha(generator);
                double sample_beta = gamma_beta(generator);
                double theta = sample_alpha / (sample_alpha + sample_beta);
                if (theta > max_sample) {
                    max_sample = theta;
                    selected_heuristic = h;
                }
            }
        }

        // Apply the selected heuristic
        int **s_temp = team; // Use s_temp = team
        ApplyHeuristic(selected_heuristic, s_temp); // Apply heuristic to s_temp

        // Evaluate the new solution
        objective_Function(s_temp);
        int new_cost_eff = f_cur;
        int new_cost_div = f_cur_div;

        // Calculate reward (positive if improvement)
        double reward = 0.0;
        if ((new_cost_eff > cost_eff) || (new_cost_div >= min_div)) {
            reward = (new_cost_eff - cost_eff) + (new_cost_div - cost_div);
        }

        // Update heuristic statistics
        heuristic_rewards[selected_heuristic] += reward;
        heuristic_counts[selected_heuristic]++;

        if (mab_strategy == EPSILON_GREEDY) {
            // Update heuristic estimates
            heuristic_estimates[selected_heuristic] = heuristic_rewards[selected_heuristic] / heuristic_counts[selected_heuristic];
        } else if (mab_strategy == THOMPSON_SAMPLING) {
            // Update alpha and beta for Beta distribution
            heuristic_alpha[selected_heuristic] += reward;        // Successes
            heuristic_beta[selected_heuristic] += (1 - reward);   // Failures (assuming reward is 0 or 1)
        }

        // Accept or reject the new solution
        if ((new_cost_eff > cost_eff) && (new_cost_div >= min_div)) {
            // Accept the new solution
            // Assign s_current to s_temp
            free_solution(s_current, num_node, num_team, num_each_t);
            s_current = deep_copy_solution(s_temp, num_node, num_team, num_each_t);
            cost_eff = new_cost_eff;
            cost_div = new_cost_div;
        }

        // Update the best solution if improved
        if ((new_cost_eff > fbest_solution_eff) && (new_cost_div >= min_div)) {
            free_solution(s_best, num_node, num_team, num_each_t);
            s_best = deep_copy_solution(s_temp, num_node, num_team, num_each_t); // Assign s_best to s_temp
            fbest_solution_eff = new_cost_eff;
            best_cost_div = new_cost_div;
            best_eff = new_cost_eff;
            best_div = new_cost_div;

            // Update elapsed time
            elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            time_taken = elapsed_time;
            // Update best solution arrays
            for (int m = 0; m < num_node; m++)
                fbest_solution[m] = best_solution[m];
            for (int m = 1; m <= num_team; m++) {
                eff_fbest[m] = eff_best[m];
                div_fbest[m] = div_best[m];
            }
        } else {
            for (int i = 0; i <= num_team; ++i)
            for (int j = 0; j < team_size[i]; ++j)
                team[i][j] = s_current[i][j];
         // Revert to current solution
        }

        // Record objective function value
        objective_values.push_back(cost_eff);

        // Calculate iteration time
        double iteration_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - iteration_start).count();
        iteration_times.push_back(iteration_time);

        // Output results for the current iteration
        std::cout << "Iteration: " << total_iterations
                  << " | Strategy: " << (mab_strategy == UCB ? "UCB" : mab_strategy == EPSILON_GREEDY ? "Epsilon-Greedy" : "Thompson Sampling")
                  << " | Selected Heuristic: " << selected_heuristic
                  << " | Efficiency: " << cost_eff
                  << " | Diversity: " << cost_div
                  << " | Best Efficiency: " << fbest_solution_eff
                  << " | Best Diversity: " << best_cost_div
                  << " | Elapsed Time: " << elapsed_time << " seconds"
                  << std::endl;

        // Update elapsed time for the loop condition
        elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    }

    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";


    // Calculate statistics
    double total_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
    double average_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0) / objective_values.size();
    int worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    double average_iteration_time = total_time / total_iterations;

    // Save results to a file
    std::string strategy_name = (mab_strategy == UCB ? "UCB" : mab_strategy == EPSILON_GREEDY ? "Epsilon_Greedy" : "Thompson_Sampling");
    std::filesystem::path filename = folder_path / ("MAHH_Selection_CMCEE_" + instanceName + "__" + strategy_name + "_.txt");
    std::ofstream result_file(filename);
    if (result_file.is_open()) {
        result_file << "Final best objectives:\n";
        result_file << "Efficiency: " << fbest_solution_eff << "\n";
        result_file << "Diversity: " << best_cost_div << "\n\n";

        result_file << "Statistics:\n";
        result_file << "Total time: " << total_time << " seconds\n";
        result_file << "Average objective function value: " << average_objective << "\n";
        result_file << "Worst objective function value: " << worst_objective << "\n";
        result_file << "Average iteration time: " << average_iteration_time << " seconds\n\n";

        result_file << "Heuristic Performance:\n";
        for (int h : heuristics) {
            result_file << "Heuristic " << h << ":\n";
            result_file << "  Usage Count: " << heuristic_counts[h] << "\n";
            result_file << "  Total Reward: " << heuristic_rewards[h] << "\n";
            if (mab_strategy == EPSILON_GREEDY) {
                result_file << "  Estimated Value: " << heuristic_estimates[h] << "\n";
            } else if (mab_strategy == THOMPSON_SAMPLING) {
                result_file << "  Alpha (Successes): " << heuristic_alpha[h] << "\n";
                result_file << "  Beta (Failures): " << heuristic_beta[h] << "\n";
            }
            result_file << "\n";
        }

        result_file.close();
        std::cout << "Results saved to " << filename << std::endl;
    } else {
        std::cerr << "Error: Could not open the file for writing results.\n";
    }

    // Finalize and print results
    std::cout << "Final best objectives: Efficiency = " << fbest_solution_eff
              << ", Diversity = " << best_cost_div << std::endl;
    // Final console output
    std::cout << "\n--- Summary ---\n";
    std::cout << "Best Efficiency: " << fbest_solution_eff << "\n";
    std::cout << "Best Diversity: " << best_cost_div << "\n";
    std::cout << "Total Time: " << total_time << " seconds\n";
    std::cout << "Average Objective Function Value: " << average_objective << "\n";
    std::cout << "Worst Objective Function Value: " << worst_objective << "\n";
    std::cout << "Average Iteration Time: " << average_iteration_time << " seconds\n";
        // Clean up
    check_best_solution();
    std::cout << "Multi-Armed Bandit with Three reward UCB1, EPSILON GREEDY, THOMPSON_SAMPLING Selection Hyper-heuristic Framework Start its Processes." << std::endl;

}

void Hyper_heuristic::MAHH_Selection_CMCEE(int max_time) {
    // Initialize data structures for tracking heuristic performance
    std::map<int, double> heuristic_total_time;
    std::map<int, int> heuristic_improvement_count;
    std::map<int, double> heuristic_rewards;
    std::map<int, int> heuristic_usage_count;
    std::vector<int> objective_values;
    std::vector<double> iteration_times;

    // LLH_score map for credit assignment
    std::map<int, double> LLH_score;

    // Define heuristics
    std::vector<int> heuristics = {15, 16, 17, 18, 19, 20,21,22,23};
    int L = (int)heuristics.size();

    // Initialize LLH scores
    for (int h : heuristics) {
        LLH_score[h] = 0.0;
    }

    // Initialize solution
    generate_initialrandom();
    objective_Function1(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    int best_iteration = 0;

    // Current and best solutions
    // Store deep copies of current and best solutions
    int** s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** s_best    = deep_copy_solution(team, num_node, num_team, num_each_t);

    int cost_eff_temp = cost_eff;
    int cost_div_temp = cost_div;
    int fbest_solution_eff = cost_eff;
    int fbest_solution_div = cost_div;
    int* team_size = new int[num_team + 1];

    // Assign team sizes
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;

    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;

    // Start timing
    clock_t total_start_time = clock();
    int iteration = 0;

    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";


    // Define the full file path for saving the results
    std::filesystem::path file_path = folder_path / ("MAHH_Selection_CMCEE_" + instanceName + "_results.txt");

    // Open output file to log iteration-level results
    std::ofstream outfile(file_path);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << file_path << std::endl;
    } else {
        // Write headers
         outfile << "Iteration\tSelected Heuristic\tEfficiency\tDiversity\tTime Taken (seconds)\titeration time (seconds)\tDelta\tLLH_Score\n";
    }

    while (static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC < max_time) {
        iteration++;
        clock_t iteration_start_time = clock();

        // Select a heuristic randomly
        int i_next = heuristics[std::rand() % heuristics.size()];

        // Store old costs before applying heuristic
        int old_cost_eff = cost_eff;
        int old_cost_div = cost_div;

        // Apply heuristic
        //int **s_temp = team;
        clock_t heuristic_start_time = clock();
        ApplyHeuristic(i_next, team);
        double heuristic_time = static_cast<double>(clock() - heuristic_start_time) / CLOCKS_PER_SEC;
        heuristic_total_time[i_next] += heuristic_time;
        heuristic_usage_count[i_next]++;

        // Evaluate new solution
        objective_Function1(team);
        cost_eff_temp = f_cur;
        cost_div_temp = f_cur_div;

        // Compute f_current and f_new
        double f_c = static_cast<double>(old_cost_eff);
        double f_n = static_cast<double>(cost_eff_temp);

        // Compute delta
        double delta = 0.0;
        if ((f_c + f_n) != 0.0) {
            delta = (f_n - f_c) / (f_n + f_c);
        }

        // Update LLH scores
        if (delta > 0) {
            // Improvement
            LLH_score[i_next] = LLH_score[i_next] + delta; // (5.1)
            double punish_value = delta / (L - 1);
            for (int h : heuristics) {
                if (h != i_next) {
                    LLH_score[h] = LLH_score[h] - punish_value; // (5.2)
                }
            }
            heuristic_improvement_count[i_next]++;
        } else {
            // No improvement
            LLH_score[i_next] = LLH_score[i_next] - delta; // (5.3)
            double reward_others_value = delta / (L - 1);
            for (int h : heuristics) {
                if (h != i_next) {
                    LLH_score[h] = LLH_score[h] + reward_others_value; // (5.4)
                }
            }
        }

        // Accept or reject new solution
        if ((cost_eff_temp > old_cost_eff) && (cost_div_temp >= min_div)) {

            free_solution(s_current, num_node, num_team, num_each_t);
            s_current = deep_copy_solution(team, num_node, num_team, num_each_t);

            cost_eff = cost_eff_temp;
            cost_div = cost_div_temp;
        } else {
            for (int i = 0; i <= num_team; ++i)
            for (int j = 0; j < team_size[i]; ++j)
                team[i][j] = s_current[i][j];

        }

        // Update best solution if improved
        if ((cost_eff_temp > fbest_solution_eff) && (cost_div_temp >= min_div)) {
            free_solution(s_best, num_node, num_team, num_each_t);
            s_best = deep_copy_solution(team, num_node, num_team, num_each_t);
            fbest_solution_eff = cost_eff_temp;
            fbest_solution_div = cost_div_temp;
            best_iteration = iteration;
            best_eff = cost_eff_temp;
            best_div = cost_div_temp;
            time_taken = heuristic_time;

            for (int m = 0; m < num_node; m++)
                fbest_solution[m] = best_solution[m];
            for (int m = 1; m <= num_team; m++) {
                eff_fbest[m] = eff_best[m];
                div_fbest[m] = div_best[m];
            }
        }

        // Record objective function value
        objective_values.push_back(cost_eff_temp);

        // Calculate iteration time
        double iteration_time = static_cast<double>(clock() - iteration_start_time) / CLOCKS_PER_SEC;
        iteration_times.push_back(iteration_time);

        // Print iteration results to console
        // Output results for the current iteration to the console
       std::cout << "Iter: " << iteration
                 << " | Selected Heuristic: " << i_next
                 << " | Eff: " << cost_eff_temp
                 << " | Div: " << cost_div_temp
                 << " | Best Eff: " << fbest_solution_eff
                 << " | Best Div: " << fbest_solution_div
                 << " | Time Taken: " << heuristic_time << " seconds"
                 << " | Delta: " << delta
                 << " | LLH["<< i_next << "] Score: " << LLH_score[i_next]
                 << std::endl;


        // Write iteration results to file if open
        if (outfile) {
        outfile << iteration << "\t"
                << i_next << "\t"
                << cost_eff_temp << "\t"
                << cost_div_temp << "\t"
                << heuristic_time << "\t"
                << iteration_time << "\t"
                << delta << "\t"
                << LLH_score[i_next] << "\n";
        }

    }

    double total_time = static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC;

    // Calculate statistics
    //double average_objective = 0.0;
    if (!objective_values.empty()) {
        average_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0) / objective_values.size();
    }
   // int worst_objective = 0;
    if (!objective_values.empty()) {
        worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    }
    double average_iteration_time = 0.0;
    if (!iteration_times.empty()) {
        average_iteration_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    }

    // Identify best and worst heuristics based on LLH scores
    int best_heuristic = -1;
    int worst_heuristic = -1;
    double max_score = -std::numeric_limits<double>::infinity();
    double min_score = std::numeric_limits<double>::infinity();

    for (int h : heuristics) {
        if (LLH_score[h] > max_score) {
            max_score = LLH_score[h];
            best_heuristic = h;
        }
        if (LLH_score[h] < min_score) {
            min_score = LLH_score[h];
            worst_heuristic = h;
        }
    }

    // Save summary results to a file
    std::ofstream result_file(folder_path /("MAHH_Selection_CMCEE_" + instanceName + "_results.txt"));
    if (result_file.is_open()) {
        result_file << "Final best objectives:\n";
        result_file << "Efficiency: " << fbest_solution_eff << "\n";
        result_file << "Diversity: " << fbest_solution_div << "\n";
        result_file << "Found at iteration: " << best_iteration << "\n\n";

        result_file << "Statistics:\n";
        result_file << "Total time: " << total_time << " seconds\n";
        result_file << "Average objective function value: " << average_objective << "\n";
        result_file << "Worst objective function value: " << worst_objective << "\n";
        result_file << "Average iteration time: " << average_iteration_time << " seconds\n";
        result_file << "Best heuristic (based on LLH_score): " << best_heuristic << " with score " << LLH_score[best_heuristic] << "\n";
        result_file << "Worst heuristic (based on LLH_score): " << worst_heuristic << " with score " << LLH_score[worst_heuristic] << "\n\n";

        result_file << "Heuristic Performance:\n";
        for (int h : heuristics) {
            result_file << "Heuristic " << h << ":\n";
            result_file << "  Total Time: " << heuristic_total_time[h] << " seconds\n";
            result_file << "  Usage Count: " << heuristic_usage_count[h] << "\n";
            result_file << "  Improvement Count: " << heuristic_improvement_count[h] << "\n";
            result_file << "  LLH Score: " << LLH_score[h] << "\n";
            result_file << "\n";
        }

        result_file.close();
        std::cout << "Results saved to D:\\Result Hyper-heuristic Models\\MAHH_Results.txt" << std::endl;
    } else {
        std::cerr << "Error: Could not open the file for writing summary results.\n";
    }

    // Close the iteration-level results file
    if (outfile) {
        outfile.close();
    }

    std::cout << "Final best objectives: Efficiency = " << fbest_solution_eff
              << ", Diversity = " << fbest_solution_div << std::endl;

    // Clean up if necessary
    check_best_solution();
    //free_memory();
}
/*
void Hyper_heuristic::HH_Choice_Function_Selection_CMCEE(int max_time) {
    // ==========================================================================
    // 1) PARAMETER & DATA STRUCTURE SETUP
    std::cout <<"============================================================================="<< std::endl;
    std::cout << "Choice Function Selection Hyper-heuristic Framework Start its Processes." << std::endl;
    std::cout <<"============================================================================="<< std::endl;
    // Choice Function parameters:
    //   alpha (α): weight for recent performance f1
    //   beta  (β): weight for historical performance f2
    //   gamma (γ): weight for recency of use f3 (time since last used)
    double alpha = 1.0;
    double beta  = 0.5;
    double gamma = 0.5;

    // Data structures to track each heuristic's performance:
    std::map<int, double> heuristic_recent_perf;     // f1: recent performance
    std::map<int, double> heuristic_total_perf;      // f2: historical performance
    std::map<int, int>    heuristic_last_used_iter;  // for f3: iteration last used
    std::map<int, int>    heuristic_usage_count;     // how many times used
    std::map<int, double> heuristic_total_time;      // total time spent by each heuristic
    std::map<int, int>    heuristic_improvement_count; // #times each heuristic improved solution

    // For storing iteration-wise data:
    std::vector<int>    objective_values;  // store (efficiency) objective values
    std::vector<double> iteration_times;   // time per iteration
    std::vector<int>    selected_heuristics; // track which heuristics were chosen

    // "LLH_score" is where we apply Formulas (5.1–5.4):
    std::map<int, double> LLH_score;  // credit for each low-level heuristic (LLH)

    // ==========================================================================
    // 2) INITIALIZE CURRENT SOLUTION & HEURISTICS
    // ==========================================================================

    // Generate initial solution
    generate_initialrandom();
    objective_Function1(team);

    // Current solution’s efficiency & diversity
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    int best_iteration = 0; // iteration at which best solution is found

    // Store pointers to current and best solutions
    // Store deep copies of current and best solutions
    int** s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** s_best    = deep_copy_solution(team, num_node, num_team, num_each_t);
    int* team_size = new int[num_team + 1];

    // Assign team sizes
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;

    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;


    // Keep track of the best solution's efficiency and diversity
    int fbest_solution_eff = cost_eff;
    int fbest_solution_div = cost_div;

    // Define the pool of low-level heuristics
    std::vector<int> heuristics = {17, 18, 19, 20, 21, 22, 23};
    //const std::vector<int> heuristics = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 17, 18, 19, 20};

    // Initialize usage stats and performance measures for each heuristic
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

    // Start global timing
    clock_t total_start_time = clock();
    int iteration = 0;

    // Create folder if it doesn't exist
    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";
    if (!std::filesystem::exists(folder_path)) {
        try {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return;
        }
    }

    // Define file path for iteration-wise logs
    std::filesystem::path results_file = folder_path / ("HH_Choice_Function_Selection_CMCEE_" + instanceName + "_results.txt");
    std::ofstream outfile(results_file);
    if (!outfile) {
        std::cerr << "Error: Unable to open results file: " << results_file << std::endl;
        return;
    }

    // Write header to log file
    outfile << "Iteration\tSelected_Heuristic\tEfficiency\tDiversity\tTimeTaken(s)\tDelta\n";

    // ==========================================================================
    // 4) MAIN LOOP
    // ==========================================================================

    while (static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC < max_time) {
        iteration++;
        clock_t iteration_start_time = clock();

        // ------------------------------------------------
        // 4.1) Compute Choice Function for each heuristic
        // ------------------------------------------------
        std::map<int, double> choice_function_values;
        for (int h : heuristics) {
            // f1: recent performance
            double f1 = heuristic_recent_perf[h];
            // f2: historical performance
            double f2 = heuristic_total_perf[h];
            // f3: how long since we used this heuristic
            double f3 = static_cast<double>(iteration - heuristic_last_used_iter[h]);

            // Combined Choice Function:
            //   CF(h) = α*f1 + β*f2 + γ*f3 + LLH_score[h]
            choice_function_values[h] = alpha * f1
                                      + beta  * f2
                                      + gamma * f3
                                      + LLH_score[h];
        }

        // ------------------------------------------------
        // 4.2) Select the best heuristic (max CF value)
        // ------------------------------------------------
        int selected_heuristic = -1;
        double max_cf_value = -std::numeric_limits<double>::infinity();
        for (int h : heuristics) {
            if (choice_function_values[h] > max_cf_value) {
                max_cf_value = choice_function_values[h];
                selected_heuristic = h;
            }
        }

        // Mark last used iteration
        heuristic_last_used_iter[selected_heuristic] = iteration;

        // ------------------------------------------------
        // 4.3) Apply selected heuristic & measure time
        // ------------------------------------------------
        int** s_temp = team;  // apply heuristic to 'team'
        clock_t heuristic_start_time = clock();
        ApplyHeuristic(selected_heuristic, s_temp);
        double heuristic_time = static_cast<double>(clock() - heuristic_start_time) / CLOCKS_PER_SEC;
        heuristic_total_time[selected_heuristic] += heuristic_time;
        heuristic_usage_count[selected_heuristic]++;

        // Evaluate the new solution
        objective_Function1(s_temp);
        int cost_eff_temp = f_cur;
        int cost_div_temp = f_cur_div;

        // ------------------------------------------------
        // 4.4) Calculate improvement measure Δ
        //      (for Formulas 5.1–5.4; see "Delta" usage)
        // ------------------------------------------------
        double f_c = static_cast<double>(cost_eff);       // old solution measure
        double f_n = static_cast<double>(cost_eff_temp); // new solution measure

        if (f_c + f_n == 0.0) {
            f_n += 1e-9; // avoid division by zero
        }

        // For maximization: Delta = (f_n - f_c) / (f_c + f_n)
        double Delta = (f_n - f_c) / (f_c + f_n);

        // ------------------------------------------------
        // 4.5) Credit Assignment (Formulas 5.1–5.4)
        // ------------------------------------------------
        if (Delta > 0) {
            // ===== Improved solution =====
            // (5.1): LLH_i += Delta
            LLH_score[selected_heuristic] += Delta;

            // (5.2): LLH_j -= Delta/(L-1)  for j != i
            double punish_value = Delta / (heuristics.size() - 1);
            for (int h : heuristics) {
                if (h == selected_heuristic) continue;
                LLH_score[h] -= punish_value;
            }

            // Update performance stats
            heuristic_improvement_count[selected_heuristic]++;
            heuristic_recent_perf[selected_heuristic] = Delta;
            heuristic_total_perf[selected_heuristic] += Delta;
        } else {
            // ===== No improvement =====
            // (5.3): LLH_i -= Delta
            LLH_score[selected_heuristic] -= Delta;

            // (5.4): LLH_j += Delta/(L-1) for j != i
            double reward_others_value = Delta / (heuristics.size() - 1);
            for (int h : heuristics) {
                if (h == selected_heuristic) continue;
                LLH_score[h] += reward_others_value;
            }

            // If no improvement, we reset recent performance
            heuristic_recent_perf[selected_heuristic] = 0.0;
        }

        // ------------------------------------------------
        // 4.6) Accept/Reject new solution
        // ------------------------------------------------
        // For example, accept if new solution has better efficiency & meets min_div
        if ((cost_eff_temp > cost_eff) && (cost_div_temp >= min_div)) {
            // Accept

            free_solution(s_current, num_node, num_team, num_each_t);
            s_current = deep_copy_solution(s_temp, num_node, num_team, num_each_t);
            cost_eff  = cost_eff_temp;
            cost_div  = cost_div_temp;
             // Copy arrays if necessary
            for (int m = 0; m < num_node; m++)
                fbest_solution[m] = best_solution[m];
            for (int m = 1 ; m <= num_team ; m++) {
                eff_fbest[m] = eff_best[m];
                div_fbest[m] = div_best[m];
            }
        } else {
            // Reject => revert
            for (int i = 0; i <= num_team; ++i)
            for (int j = 0; j < team_size[i]; ++j)
                team[i][j] = s_current[i][j];
        }

        // ------------------------------------------------
        // 4.7) Update best solution if improved
        // ------------------------------------------------
        if ((cost_eff_temp > fbest_solution_eff) && (cost_div_temp >= min_div)) {

            free_solution(s_best, num_node, num_team, num_each_t);
            s_best = deep_copy_solution(s_temp, num_node, num_team, num_each_t);

            fbest_solution_eff = cost_eff_temp;
            fbest_solution_div = cost_div_temp;
            best_iteration     = iteration;
            best_eff = cost_eff_temp;
            best_div = cost_div_temp;
            time_taken = heuristic_time;
            selected_heuristics.push_back(selected_heuristic);

            // Copy arrays if necessary
            for (int m = 0; m < num_node; m++)
                fbest_solution[m] = best_solution[m];
            for (int m = 1; m <= num_team; m++) {
                eff_fbest[m] = eff_best[m];
                div_fbest[m] = div_best[m];
            }
        } else {
            // Keep the currently accepted solution
            // (already handled above)
            // Reject => revert
            // Reject => revert
            for (int i = 0; i <= num_team; ++i)
            for (int j = 0; j < team_size[i]; ++j)
                team[i][j] = s_current[i][j];
            //team = s_current;
        }

        // ------------------------------------------------
        // 4.8) Logging & Console Output for this iteration
        // ------------------------------------------------
        objective_values.push_back(cost_eff_temp);

        double iteration_time = static_cast<double>(clock() - iteration_start_time) / CLOCKS_PER_SEC;
        iteration_times.push_back(iteration_time);

        // Log to file
        outfile << iteration << "\t"
                << selected_heuristic << "\t"
                << cost_eff_temp << "\t"
                << cost_div_temp << "\t"
                << heuristic_time << "\t"
                << Delta << "\n";

        // Console output
        std::cout << "Iteration: " << iteration
                  << " | Selected Heuristic: " << selected_heuristic
                  << " | Efficiency: " << cost_eff_temp
                  << " | Diversity: " << cost_div_temp
                  << " | Best Efficiency: " << fbest_solution_eff
                  << " | Best Diversity: " << fbest_solution_div
                  << " | Time Taken: " << heuristic_time << "s"
                  << " | Delta: " << Delta
                  << std::endl;
    } // end while

    // ==========================================================================
    // 5) FINAL ANALYSIS & SUMMARY OUTPUT
    // ==========================================================================

    outfile.close(); // close iteration log
    double total_time = static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC;

    // Compute average objective
    //double average_objective = 0.0;
    if (!objective_values.empty()) {
        average_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0)
                            / objective_values.size();
    }

    // Worst objective found
    //int worst_objective = 0;
    if (!objective_values.empty()) {
        worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    }

    // Average iteration time
    double average_iteration_time = 0.0;
    if (!iteration_times.empty()) {
        average_iteration_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0)
                                 / iteration_times.size();
    }

    // Identify best/worst heuristics (by total performance)
    int best_heuristic   = -1;
    int worst_heuristic  = -1;
    double max_total_perf = -std::numeric_limits<double>::infinity();
    double min_total_perf =  std::numeric_limits<double>::infinity();

    for (int h : heuristics) {
        if (heuristic_total_perf[h] > max_total_perf) {
            max_total_perf = heuristic_total_perf[h];
            best_heuristic = h;
        }
        if (heuristic_total_perf[h] < min_total_perf) {
            min_total_perf = heuristic_total_perf[h];
            worst_heuristic = h;
        }
    }

    // Save summary in a separate file
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
        summary_file << "Best Heuristic: " << best_heuristic
                     << " (Total Performance: " << max_total_perf << ")\n";
        summary_file << "Worst Heuristic: " << worst_heuristic
                     << " (Total Performance: " << min_total_perf << ")\n";
        summary_file << "Heuristic Performance:\n";
        for (int h : heuristics) {
            summary_file << "Heuristic " << h << ":\n";
            summary_file << "  Total Time: " << heuristic_total_time[h] << " seconds\n";
            summary_file << "  Usage Count: " << heuristic_usage_count[h] << "\n";
            summary_file << "  Improvement Count: " << heuristic_improvement_count[h] << "\n";
            summary_file << "  Total Performance: " << heuristic_total_perf[h] << "\n";
            summary_file << "  LLH Score: " << LLH_score[h] << "\n\n";
        }
        summary_file.close();
        std::cout << "Summary results saved to " << summary_file_path << std::endl;
    } else {
        std::cerr << "Error: Unable to save summary file.\n";
    }

    // Final console output
    std::cout << "\n--- Summary ---\n";
    std::cout << "Best Efficiency: " << fbest_solution_eff << "\n";
    std::cout << "Best Diversity: " << fbest_solution_div << "\n";
    std::cout << "Total Time: " << total_time << " seconds\n";
    std::cout << "Average Objective Function Value: " << average_objective << "\n";
    std::cout << "Worst Objective Function Value: " << worst_objective << "\n";
    std::cout << "Average Iteration Time: " << average_iteration_time << " seconds\n";
    std::cout << "Best Heuristic: " << best_heuristic
              << " (Total Performance: " << max_total_perf << ")\n";
    std::cout << "Worst Heuristic: " << worst_heuristic
              << " (Total Performance: " << min_total_perf << ")\n";
    // Final check or post-processing
    check_best_solution();
   std::cout <<"============================================================================="<< std::endl;
   std::cout <<"Choice Function Selection Hyper-heuristic Framework Finished its Processes." << std::endl;
   std::cout <<"============================================================================="<< std::endl;
}
*/
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
    generate_initialrandom();
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

void Hyper_heuristic::Random_Selection_Hyperheuristic_CMCEE11(int max_time) {
    std::cout << "=============================================================================\n";
    std::cout << "Random Selection Hyper-heuristic Framework Start its Processes.\n";
    std::cout << "=============================================================================\n";

    // Initialization and setup
    int** previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** best_solution1    = deep_copy_solution(team, num_node, num_team, num_each_t);

    int* team_size = new int[num_team + 1];
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    generate_initialrandom();
    objective_Function1(team);
    display(team);

    int cost_eff = f_cur, cost_div = f_cur_div;
    int best_cost_eff = cost_eff, best_cost_div = cost_div;
    fbest_solution_eff = cost_eff;

    std::cout << "Initial objectives eff and div: " << cost_eff << ", " << cost_div << std::endl;

    std::vector<int> Selected;
    std::vector<int> L = {17, 18, 19, 20, 21, 22, 23};
    int T = 100;

    std::map<int, std::vector<int>> heuristic_costs;
    std::map<int, std::vector<double>> heuristic_times;
    std::map<int, int> heuristic_usage_count;
    std::vector<int> objective_values;
    std::vector<double> iteration_times;
    std::vector<double> rewards_record;

    double total_elapsed_time = 0.0;

    // Output directory
    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/Random_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    // Output files
    std::filesystem::path file_path = folder_path / ("Random_Selection_Hyperheuristic_CMCEE_" + instanceName + "_results.txt");
    std::filesystem::path trace_path = folder_path / ("Random_Selection_Hyperheuristic_CMCEE_" + instanceName + "_Convergence_Trace.csv");

    std::ofstream outfile(file_path);
    std::ofstream trace(trace_path);

    outfile << "Iteration\tSelected_Heuristic\tCost Efficiency\tCost Diversity\tTime Taken (s)\n";
    trace << "Iteration,Efficiency,Diversity,BestEfficiency,BestDiversity,Reward,Runtime\n";

    // Random heuristic selection
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, L.size() - 1);

    clock_t total_start = clock();

    for (int i = 0; i < T && ((double)(clock() - total_start) / CLOCKS_PER_SEC < max_time); ++i) {
        int selected_heuristic = L[dis(gen)];
        std::cout << "Iteration: " << i << ", Selected_heuristic LLH[" << selected_heuristic
                  << "], cost_eff: " << cost_eff << ", cost_div: " << cost_div << std::endl;

        clock_t start_time = clock();
        ApplyHeuristic(selected_heuristic, team);
        double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        total_elapsed_time += elapsed_time;

        objective_Function1(team);
        int newcost_eff = f_cur, newcost_div = f_cur_div;
        std::cout << "Iteration: " << i << ", After LLH[" << selected_heuristic
                  << "], newcost_eff: " << newcost_eff << ", newcost_div: " << newcost_div << "\n";

        double reward = (newcost_eff > cost_eff ? 1.0 : (newcost_eff < cost_eff ? -1.0 : 0.0));
        rewards_record.push_back(reward);

        // Acceptance
        if ((newcost_eff > cost_eff) && (newcost_div >= min_div)) {
            cost_eff = newcost_eff;
            cost_div = newcost_div;
            free_solution(previous_solution, num_node, num_team, num_each_t);
            previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);

            //best_eff = newcost_eff;
            //best_div = newcost_div;
            time_taken = elapsed_time;
            Selected.push_back(selected_heuristic);
            heuristic_usage_count[selected_heuristic]++;

            if ((newcost_eff > fbest_solution_eff) && (newcost_div >= min_div)){
                free_solution(best_solution1, num_node, num_team, num_each_t);
                best_solution1 = deep_copy_solution(team, num_node, num_team, num_each_t);

                fbest_solution_eff = newcost_eff;
                best_cost_div = newcost_div;
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        } else {
            // Revert solution
            for (int ti = 0; ti <= num_team; ++ti)
                for (int j = 0; j < team_size[ti]; ++j)
                    team[ti][j] = previous_solution[ti][j];
        }

        // Data collection
        heuristic_costs[selected_heuristic].push_back(newcost_eff);
        heuristic_times[selected_heuristic].push_back(elapsed_time);
        objective_values.push_back(newcost_eff + newcost_div);
        iteration_times.push_back(elapsed_time);

        // Write iteration data
        outfile << i << "\t" << selected_heuristic << "\t" << newcost_eff
                << "\t" << newcost_div << "\t" << elapsed_time << "\n";

        double runtime = (double)(clock() - total_start) / CLOCKS_PER_SEC;
        trace << i << "," << newcost_eff << "," << newcost_div << ","
              << best_cost_eff << "," << best_cost_div << ","
              << reward << "," << runtime << "\n";
    }

    outfile.close();
    trace.close();

    // ======= Summary and Statistics =======
    double avg_obj = 0.0, avg_time = 0.0;
    int worst_obj = 0;

    if (!objective_values.empty())
        avg_obj = std::accumulate(objective_values.begin(), objective_values.end(), 0.0) / objective_values.size();
    if (!iteration_times.empty())
        avg_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    if (!objective_values.empty())
        worst_obj = *std::min_element(objective_values.begin(), objective_values.end());

    std::cout << "\n--- Summary ---\n";
    std::cout << "Best Efficiency: " << fbest_solution_eff << "\n";
    std::cout << "Best Diversity: " << best_cost_div << "\n";
    std::cout << "Average Objective: " << avg_obj << "\n";
    std::cout << "Worst Objective: " << worst_obj << "\n";
    std::cout << "Average Iteration Time: " << avg_time << "s\n";
    std::cout << "Convergence trace saved to: " << trace_path << "\n";
    std::cout << "=============================================\n";

    check_best_solution();
    std::cout << "=============================================================================\n";
    std::cout << "Random Selection Hyper-heuristic Framework Finished its Processes.\n";
    std::cout << "=============================================================================\n";
}

/*
void Hyper_heuristic::Random_Selection_Hyperheuristic_CMCEE11(int max_time) {
    // Parameters and variables
    int **previous_solution;
    int **best_solution1;

    // Initialize solution
    generate_initialrandom();
    previous_solution = team;

    // Calculate initial total costs
    objective_Function(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    std::cout << "Initial objectives eff and div: " << cost_eff << ", " << cost_div << std::endl;

    int best_cost_eff = cost_eff, best_cost_div = cost_div;
    fbest_solution_eff = cost_eff;
    // Data structures for analysis
    std::vector<int> Selected;
    std::vector<int> L = {12,17,18,19,20}; // List of heuristics
    int T = 100; // Number of iterations
    std::map<int, std::vector<int>> heuristic_costs;
    std::map<int, std::vector<double>> heuristic_times;

    double total_elapsed_time = 0.0;

    // Define the folder path
    std::string folder_path = "D:/Result Hyper-heuristic Models/";

    //   define the full file path for saving the results
    std::string file_path = folder_path + ("Random_Selection_Hyperheuristic_CMCEE_" + instanceName + "_results.txt");

    // Create an output file stream to save the results to a text file
    std::ofstream outfile(file_path);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << file_path << std::endl;
        return;
    }

    // Write headers to the text file
    outfile << "Iteration\tSelected Heuristic\tCost Efficiency\tCost Diversity\tTime Taken (seconds)\n";

    // Random engine for heuristic selection
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, L.size() - 1);

    // Perform the iterations
    for (int i = 0; i < T; ++i) {
        // Randomly select a heuristic
        int selected_heuristic = L[dis(gen)];
        std::cout << "Iteration: " << i << ", Selected_heuristic LLH[" << selected_heuristic
                  << "], cost_eff: " << cost_eff << ", cost_div: " << cost_div << std::endl;

        clock_t start_time = clock();

        // Apply heuristic
        ApplyHeuristic(selected_heuristic, team);

        double elapsed_time = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
        total_elapsed_time += elapsed_time;

        // Calculate new costs
        objective_Function(team);
        int newcost_eff = f_cur, newcost_div = f_cur_div;
        std::cout << "Iteration: " << i << ", After Selected_heuristic LLH[" << selected_heuristic
                  << "], newcost_eff: " << newcost_eff << ", newcost_div: " << newcost_div << "\n";

        // Accept or reject the new solution
        if ((newcost_eff > cost_eff) && (newcost_div >= min_div)) {
            cost_eff = newcost_eff;
            cost_div = newcost_div;
            previous_solution = team;
            best_eff = newcost_eff;
            best_div = newcost_div;
            time_taken = elapsed_time;
            Selected.push_back(selected_heuristic);
            if (newcost_eff > fbest_solution_eff) {
                best_solution1 = team;
                fbest_solution_eff = newcost_eff;
            }
        } else {
            team = previous_solution;
        }

        // Collect data for text file
        heuristic_costs[selected_heuristic].push_back(newcost_eff);
        heuristic_times[selected_heuristic].push_back(elapsed_time);

        // Save data for each iteration to the text file
        outfile << i << "\t"  // Iteration
                << selected_heuristic << "\t"  // Selected Heuristic
                << newcost_eff << "\t"  // Efficiency
                << newcost_div << "\t"  // Diversity
                << elapsed_time << "\n";  // Time taken
    }

    // Close the output file
    outfile.close();

    // Results and analysis
    std::cout << "Total time taken to find the solution: " << std::fixed << std::setprecision(4)
              << total_elapsed_time << " seconds" << std::endl;
    std::cout << "Selected heuristic sequence: ";
    for (const auto& h : Selected) std::cout << h << " ";
    std::cout << std::endl;

    // Final results
    check_best_solution();
    //free_memory();
}

*/
std::tuple<int**, double, std::vector<std::vector<int>>,
           std::map<int, std::vector<double>>,
           std::map<int, std::vector<double>>,
           std::map<int, std::vector<int>>>
Hyper_heuristic::ADSH_random(int accept, double optimal) {
    // Setting parameters and initializing solution
    generate_initial();
    //int** previous_solution;
    //int** best_solution1;
    //int** solution = team;
    // Store deep copies of current and best solutions
    int** previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** best_solution1  = deep_copy_solution(team, num_node, num_team, num_each_t);

    //previous_solution = solution;
    //best_solution1 = solution;

    // Data structures for collecting data
    std::map<int, std::vector<double>> heuristic_costs;       // Cost efficiency per heuristic
    std::map<int, std::vector<double>> heuristic_times;       // Time taken per heuristic
    std::map<int, std::vector<int>> heuristic_iterations;     // Iterations when the heuristic was applied

    double total_elapsed_time = 0.0;
    int newcost_eff, newcost_div;
    int size = 100;
    objective_Function1(team);
    int cost_eff = f_cur, cost_div = f_cur_div;
    double best_cost_eff = cost_eff;
    double best_cost_div = cost_div;
    double F = 1;

    // Perform swaps
    std::vector<std::vector<int>> L = {{15}, {16}, {17}, {18}, {19},{20},{21},{22},{23}};
    int T = 1 * size;
    // Define the folder path
     double elapsed_time = 0.0;
    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";


    // Define the full file path for saving the results
    std::filesystem::path file_path = folder_path / ("Adaptive_Random_Selection_Strategies_For_Solving_CMCEE_" + instanceName + "_results.txt");

    // Create an output file stream to save the results to a text file
    std::ofstream outfile(file_path);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << file_path << std::endl;
    }
    else {  outfile << "Iteration\tSelected_Heuristics\tOld_Cost_Eff\tOld_Cost_Div\tNew_Cost_Eff\tNew_Cost_Div\tTime(sec)\n";
    }

    for (int j = 0; j < T; ++j) {
        // Select a random heuristic
        std::vector<int> selected_heuristic = L[rand() % L.size()];
        std::vector<int> actually_selected; // To log the selected heuristics in this iteration

        // Apply the first chosen heuristic(s)
        for (int i : selected_heuristic) {
            std::cout << "Iteration: " << j << ", Selected_heuristic LLH[" << i
                      << "], cost_eff: " << cost_eff << ", cost_div: " << cost_div << std::endl;
            start_time = clock();
            ApplyHeuristic(i, team);
            elapsed_time = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
            total_elapsed_time += elapsed_time;

            // Collect time data and iteration info
            heuristic_times[i].push_back(elapsed_time);
            heuristic_iterations[i].push_back(j);
            actually_selected.push_back(i);
        }

        // Perform additional random heuristic with 50% probability
        if (rand() % 2 == 0) {
            std::vector<int> selected_heuristic2 = L[rand() % L.size()];
            for (int i : selected_heuristic2) {
                start_time = clock();
                ApplyHeuristic(i, team);
                double elapsed_time = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
                total_elapsed_time += elapsed_time;
                heuristic_times[i].push_back(elapsed_time);
                heuristic_iterations[i].push_back(j);
                actually_selected.push_back(i);
            }
        }

        objective_Function1(team);
        newcost_eff = f_cur;
        newcost_div = f_cur_div;

        // Collect cost data for each heuristic involved this iteration
        for (int i : actually_selected) {
            heuristic_costs[i].push_back(newcost_eff);
        }

        // Save iteration details to file if open
        if (outfile) {
            outfile << j << "\t";
            // Print all selected heuristics in this iteration
            for (size_t idx = 0; idx < actually_selected.size(); ++idx) {
                outfile << actually_selected[idx];
                if (idx < actually_selected.size() - 1) outfile << ",";
            }
            outfile << "\t" << cost_eff << "\t" << cost_div << "\t"
                    << newcost_eff << "\t" << newcost_div << "\t"
                    << total_elapsed_time << "\n";
        }

        // Acceptance criteria
        if (accept == 0) {
            std::cout << "IE: Improve or Equal\n";
            if (newcost_eff >= cost_eff && newcost_div >= cost_div) {
                cost_eff = newcost_eff;
                cost_div = newcost_div;
                //previous_solution = team;
                free_solution(previous_solution , num_node, num_team, num_each_t);
                previous_solution  = deep_copy_solution(team, num_node, num_team, num_each_t);

                if (newcost_eff >= best_cost_eff && newcost_div >= min_div) {
                    //best_solution1 = team;
                    free_solution(best_solution1 , num_node, num_team, num_each_t);
                    best_solution1  = deep_copy_solution(team, num_node, num_team, num_each_t);
                    best_cost_eff = newcost_eff;
                    best_cost_div = newcost_div;
                    best_eff = newcost_eff;
                    best_div = newcost_div;
                    time_taken = elapsed_time;
                    for (int m = 0; m < num_node; m++)
                        fbest_solution[m] = best_solution[m];
                    for (int m = 1; m <= num_team; m++) {
                        eff_fbest[m] = eff_best[m];
                        div_fbest[m] = div_best[m];
                    }
                }
                L.push_back(actually_selected);
            } else {
                //team = previous_solution;
                for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = previous_solution[i][j];

            }
        } else if (accept == 1) {
            std::cout << "OI: Only Improve \n";
            if (newcost_eff > cost_eff && newcost_div > cost_div) {
                cost_eff = newcost_eff;
                cost_div = newcost_div;
                //previous_solution = team;
                free_solution(previous_solution , num_node, num_team, num_each_t);
                previous_solution  = deep_copy_solution(team, num_node, num_team, num_each_t);

                if (newcost_eff > best_cost_eff && newcost_div > min_div) {
                    //best_solution1 = team;
                    free_solution(best_solution1 , num_node, num_team, num_each_t);
                    best_solution1  = deep_copy_solution(team, num_node, num_team, num_each_t);
                    best_cost_eff = newcost_eff;
                    best_cost_div = newcost_div;
                    best_eff = newcost_eff;
                    best_div = newcost_div;
                    time_taken = elapsed_time;
                    for (int m = 0; m < num_node; m++)
                        fbest_solution[m] = best_solution[m];
                    for (int m = 1; m <= num_team; m++) {
                        eff_fbest[m] = eff_best[m];
                        div_fbest[m] = div_best[m];
                    }
                }
                L.push_back(actually_selected);
            } else {
                //team = previous_solution;
                for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = previous_solution[i][j];
            }
        } else if (accept == 2) {
            std::cout << "RR: Record-to-Record \n";
            double t = 0.09 / std::pow(size, 3);
            if (newcost_eff >= cost_eff || newcost_eff >= best_cost_eff + t * best_cost_eff) {
                cost_eff = newcost_eff;
                //previous_solution = team;
                free_solution(previous_solution , num_node, num_team, num_each_t);
                previous_solution  = deep_copy_solution(team, num_node, num_team, num_each_t);
                if (newcost_eff > best_cost_eff) {
                    //best_solution1 = team;
                    free_solution(best_solution1 , num_node, num_team, num_each_t);
                    best_solution1  = deep_copy_solution(team, num_node, num_team, num_each_t);
                    best_cost_eff = cost_eff;
                    best_cost_div = cost_div;
                    best_eff = newcost_eff;
                    best_div = cost_div;
                    time_taken = elapsed_time;
                    for (int m = 0; m < num_node; m++)
                        fbest_solution[m] = best_solution[m];
                    for (int m = 1; m <= num_team; m++) {
                        eff_fbest[m] = eff_best[m];
                        div_fbest[m] = div_best[m];
                    }
                }
                L.push_back(actually_selected);
            } else {
                //team = previous_solution;
                for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = previous_solution[i][j];
            }
        } else if (accept == 3) {
            std::cout << "Other criteria \n";
            if (newcost_eff >= cost_eff || newcost_eff >= optimal + F * (1 - (j / static_cast<double>(T)))) {
                cost_eff = newcost_eff;
                //previous_solution = team;
                free_solution(previous_solution , num_node, num_team, num_each_t);
                previous_solution  = deep_copy_solution(team, num_node, num_team, num_each_t);
                if (newcost_eff > best_cost_eff) {
                    //best_solution1 = team;
                    free_solution(best_solution1 , num_node, num_team, num_each_t);
                    best_solution1  = deep_copy_solution(team, num_node, num_team, num_each_t);
                    best_cost_eff = cost_eff;
                    best_cost_div = cost_div;
                    best_eff = newcost_eff;
                    best_div = cost_div;
                    time_taken = elapsed_time;
                    for (int m = 0; m < num_node; m++)
                        fbest_solution[m] = best_solution[m];
                    for (int m = 1; m <= num_team; m++) {
                        eff_fbest[m] = eff_best[m];
                        div_fbest[m] = div_best[m];
                    }
                }
                L.push_back(actually_selected);
            } else {
                //team = previous_solution;
                for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = previous_solution[i][j];
            }
        } else if (accept == 4) {
            std::cout << "Simulated Annealing-like \n";
            double p;
            try {
                p = std::exp((cost_eff - newcost_eff) / (F * (1 - (j / static_cast<double>(T)))));
            } catch (...) {
                p = -std::numeric_limits<double>::infinity();
            }
            if (newcost_eff >= cost_eff) {
                cost_eff = newcost_eff;
                //previous_solution = team;
                free_solution(previous_solution , num_node, num_team, num_each_t);
                previous_solution  = deep_copy_solution(team, num_node, num_team, num_each_t);
                if (newcost_eff > best_cost_eff) {
                    //best_solution1 = team;
                    free_solution(best_solution1 , num_node, num_team, num_each_t);
                    best_solution1  = deep_copy_solution(team, num_node, num_team, num_each_t);
                    best_cost_eff = cost_eff;
                    best_cost_div = cost_div;
                    best_eff = newcost_eff;
                    best_div = cost_div;
                    time_taken = elapsed_time;
                    for (int m = 0; m < num_node; m++)
                        fbest_solution[m] = best_solution[m];
                    for (int m = 1; m <= num_team; m++) {
                        eff_fbest[m] = eff_best[m];
                        div_fbest[m] = div_best[m];
                    }
                }
                L.push_back(actually_selected);
            } else if (p > static_cast<double>(rand()) / RAND_MAX) {
                cost_eff = newcost_eff;
                previous_solution = team;
            } else {
                //team = previous_solution;
                for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = previous_solution[i][j];
            }
        }
    }

    // If outfile was open, write final summary
    if (outfile) {
        outfile << "\nBest Cost Efficiency: " << best_cost_eff << "\n";
        outfile << "Best Cost Diversity: " << best_cost_div << "\n\n";

        // Write heuristic performance data
        outfile << "Heuristic Performance Data:\n";
        for (auto &kv : heuristic_costs) {
            int h = kv.first;
            outfile << "Heuristic " << h << ":\n";
            outfile << "  Costs: ";
            for (auto c : kv.second)
                outfile << c << " ";
            outfile << "\n  Times: ";
            for (auto t : heuristic_times[h])
                outfile << t << " ";
            outfile << "\n  Iterations: ";
            for (auto it : heuristic_iterations[h])
                outfile << it << " ";
            outfile << "\n\n";
        }

        outfile.close();
        std::cout << "Results saved to " << file_path << std::endl;
    }

    return std::make_tuple(team, best_cost_eff, L, heuristic_costs, heuristic_times, heuristic_iterations);
}


// Function to compare different acceptance criteria
void Hyper_heuristic::compare_acceptance_criteria(double optimal) {
    for (int accept = 0; accept <= 4; ++accept) {
        std::cout << "Running ADSH_random with accept = " << accept << std::endl;
        ADSH_random(accept, optimal);
    }
}

int** Hyper_heuristic::AdaptiveHeuristicSelection(int accept, double optimal) {
     std::cout <<"============================================================================="<< std::endl;
     std::cout <<"Adaptive Selection Hyper-heuristic Framework Start its Processes." << std::endl;
     std::cout <<"============================================================================="<< std::endl;
    // Setting parameters and initializing solution
    generate_initialrandom();
    //int **previous_solution = team;
    //int **best_solution1 = team;
    // Store deep copies of current and best solutions
    int** previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** best_solution1     = deep_copy_solution(team, num_node, num_team, num_each_t);
    int* team_size = new int[num_team + 1];

    // Assign team sizes
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;


    // Data structures for collecting data
    std::map<int, std::vector<double>> heuristic_costs;       // Cost efficiency per heuristic
    std::map<int, std::vector<double>> heuristic_times;       // Time taken per heuristic
    std::map<int, std::vector<int>> heuristic_iterations;     // Iterations when the heuristic was applied

    double total_elapsed_time = 0.0;
    int newcost_eff, newcost_div;
    int size = 100;
    objective_Function1(team);
    int cost_eff = f_cur, cost_div = f_cur_div;
    double best_cost_eff = cost_eff;
    double best_cost_div = cost_div;
    double F = 1.0;

    // Initialize heuristics as an array of arrays
    const int num_heuristics = 9;
    int L_array[num_heuristics][1] = {{15}, {16}, {17}, {18}, {19},{20},{21},{22},{23}};

    int T = size; // 1 * size

    // Initialize heuristic performance metrics
    std::map<int, double> heuristic_total_time_map;
    std::map<int, int> heuristic_improvement_count_map;
    std::map<int, double> heuristic_rewards_map;
    std::map<int, int> heuristic_usage_count_map;

    // Initialize selection probability map (for adaptive selection)
    std::map<int, double> heuristic_selection_prob;
    double total_reward = 0.0;
    for (int h = 0; h < num_heuristics; h++) {
        int heuristic_id = L_array[h][0];
        heuristic_selection_prob[heuristic_id] = 1.0;
        heuristic_rewards_map[heuristic_id] = 0.0;
        heuristic_usage_count_map[heuristic_id] = 0;
        heuristic_improvement_count_map[heuristic_id] = 0;
        heuristic_total_time_map[heuristic_id] = 0.0;
    }

    // Random number generation
    std::mt19937 gen(static_cast<unsigned int>(time(NULL)));
    std::uniform_real_distribution<> dis_real(0.0, 1.0);
    std::uniform_int_distribution<> dis_heuristic(0, num_heuristics - 1);

    // Statistical variables
    double worst_OF = std::numeric_limits<double>::max();
    double sum_OF = 0.0;
    int count_OF = 0;
    double average_OF = 0.0;
    int best_found_at = -1;
    int total_best_improvements = 0;
    int worse_performing_heuristics = 0;
    int best_pair_heuristic1 = -1, best_pair_heuristic2 = -1;
    double best_pair_reward = -std::numeric_limits<double>::infinity();

    clock_t total_start_time = clock();

    // Define the folder and file paths for saving results
    std::string folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";

    std::string file_path = folder_path + ("AdaptiveHeuristicSelection_results_" + instanceName + "_results.txt");

    // Open output file
    std::ofstream outfile(file_path);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << file_path << std::endl;
    } else {
        outfile << "Iteration\tSelected_Heuristic\tOld_Cost_Eff\tOld_Cost_Div\tNew_Cost_Eff\tNew_Cost_Div\tIteration_Time(sec)\n";
    }

    for (int j = 0; j < T; ++j) {
        clock_t iteration_start_time = clock();

        // Heuristic Selection: Performance-Based Probabilistic Selection
        total_reward = 0.0;
        for (int h = 0; h < num_heuristics; h++) {
            int heuristic_id = L_array[h][0];
            total_reward += heuristic_rewards_map[heuristic_id];
        }

        // Update selection probabilities
        for (int h = 0; h < num_heuristics; h++) {
            int heuristic_id = L_array[h][0];
            if (total_reward > 0.0) {
                heuristic_selection_prob[heuristic_id] = heuristic_rewards_map[heuristic_id] / total_reward;
            } else {
                heuristic_selection_prob[heuristic_id] = 1.0 / num_heuristics;
            }
        }

        // Create a cumulative distribution for selection
        double cumulative_prob[num_heuristics];
        cumulative_prob[0] = heuristic_selection_prob[L_array[0][0]];
        for (int h = 1; h < num_heuristics; h++) {
            cumulative_prob[h] = cumulative_prob[h - 1] + heuristic_selection_prob[L_array[h][0]];
        }

        // Select heuristic based on probabilities
        double rand_val = dis_real(gen);
        int selected_index = 0;
        while (selected_index < num_heuristics - 1 && rand_val > cumulative_prob[selected_index]) {
            selected_index++;
        }
        int selected_heuristic = L_array[selected_index][0];

        // Store old values before applying heuristic
        int old_cost_eff = cost_eff;
        int old_cost_div = cost_div;

        // Apply the selected heuristic
        clock_t start_time = clock();
        ApplyHeuristic(selected_heuristic, team);
        double elapsed_time = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
        total_elapsed_time += elapsed_time;

        heuristic_times[selected_heuristic].push_back(elapsed_time);
        heuristic_iterations[selected_heuristic].push_back(j);
        heuristic_usage_count_map[selected_heuristic]++;
        heuristic_total_time_map[selected_heuristic] += elapsed_time;

        // Possibly apply a second heuristic with 50% probability
        if (dis_real(gen) < 0.5) {
            int selected_index2 = dis_heuristic(gen);
            int selected_heuristic2 = L_array[selected_index2][0];
            clock_t start_time2 = clock();
            ApplyHeuristic(selected_heuristic2, team);
            double elapsed_time2 = static_cast<double>(clock() - start_time2) / CLOCKS_PER_SEC;
            total_elapsed_time += elapsed_time2;

            heuristic_times[selected_heuristic2].push_back(elapsed_time2);
            heuristic_iterations[selected_heuristic2].push_back(j);
            heuristic_usage_count_map[selected_heuristic2]++;
            heuristic_total_time_map[selected_heuristic2] += elapsed_time2;
        }

        // Evaluate new solution
        objective_Function1(team);
        newcost_eff = f_cur;
        newcost_div = f_cur_div;

        // Collect cost data
        heuristic_costs[selected_heuristic].push_back(newcost_eff);

        // Update worst and sum OF
        if (newcost_eff < worst_OF) {
            worst_OF = newcost_eff;
        }
        sum_OF += newcost_eff;
        count_OF++;

        // Acceptance criteria
        bool accepted = false;
        if (accept == 0) { // IE: Improve or Equal
            if (newcost_eff >= cost_eff && newcost_div >= min_div) {
                accepted = true;
            }
        } else if (accept == 1) { // OI: Only Improve
            if (newcost_eff > cost_eff && newcost_div > min_div) {
                accepted = true;
            }
        } else if (accept == 2) { // RR: Record-to-Record
            double t = 0.09 / std::pow(static_cast<double>(size), 3);
            if (newcost_eff >= cost_eff || newcost_eff >= best_cost_eff + t * best_cost_eff) {
                accepted = true;
            }
        } else if (accept == 3) { // Other criteria
            if (newcost_eff >= cost_eff || newcost_eff >= optimal + F * (1.0 - (static_cast<double>(j) / static_cast<double>(T)))) {
                accepted = true;
            }
        } else if (accept == 4) { // SA-like
            double p;
            try {
                p = std::exp((static_cast<double>(cost_eff) - static_cast<double>(newcost_eff)) / (F * (1.0 - (static_cast<double>(j) / static_cast<double>(T)))));
            } catch (...) {
                p = -std::numeric_limits<double>::infinity();
            }
            if (newcost_eff >= cost_eff && newcost_div >= min_div) {
                accepted = true;
            } else if (p > dis_real(gen)) {
                accepted = true;
            }
        }

        // Apply acceptance decision
        if (accepted) {
            cost_eff = newcost_eff;
            cost_div = newcost_div;
            //previous_solution = team;
            free_solution(previous_solution, num_node, num_team, num_each_t);
            previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);

            // Update best solution if improved
            if (newcost_eff >= best_cost_eff && newcost_div >= min_div) {
                //best_solution1 = team;
                free_solution(best_solution1, num_node, num_team, num_each_t);
                best_solution1 = deep_copy_solution(team, num_node, num_team, num_each_t);
                best_cost_eff = cost_eff;
                best_cost_div = cost_div;
                best_eff = newcost_eff;
                best_div = newcost_div;
                time_taken = elapsed_time;
                best_found_at = j;
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                    }
            }

            // Update rewards
            heuristic_rewards_map[selected_heuristic] += 1.0;
            total_best_improvements += 1;

        } else {
            //team = previous_solution;
            for (int i = 0; i <= num_team; ++i)
            for (int j = 0; j < team_size[i]; ++j)
                team[i][j] = previous_solution[i][j];

            // Penalize heuristic if not accepted
            heuristic_rewards_map[selected_heuristic] -= 0.5;
        }

        // Adaptive parameter adjustments
        double average_reward = 0.0;
        int heuristic_count = 0;
        for (auto &pair : heuristic_rewards_map) {
            average_reward += pair.second;
            heuristic_count++;
        }
        if (heuristic_count > 0) {
            average_reward /= heuristic_count;
            if (average_reward > 1.0) {
                F *= 0.95;
            } else {
                F *= 1.05;
            }
        }

        double iteration_time = static_cast<double>(clock() - iteration_start_time) / CLOCKS_PER_SEC;

        // Write iteration results to file
        if (outfile) {
            outfile << j << "\t"
                    << selected_heuristic << "\t"
                    << old_cost_eff << "\t" << old_cost_div << "\t"
                    << newcost_eff << "\t" << newcost_div << "\t"
                    << iteration_time << "\n";
        }
    }

    // Perform statistical analysis
    if (count_OF > 0) {
        average_OF = sum_OF / static_cast<double>(count_OF);
    }

    double average_computation_time = total_elapsed_time / static_cast<double>(T);

    // Count worse performing heuristics
    worse_performing_heuristics = 0;
    for (auto &pair : heuristic_rewards_map) {
        if (pair.second < 0.0) {
            worse_performing_heuristics++;
        }
    }

    // Find best pair of heuristics
    for (int h1 = 0; h1 < num_heuristics; h1++) {
        for (int h2 = h1 + 1; h2 < num_heuristics; h2++) {
            int heuristic1 = L_array[h1][0];
            int heuristic2 = L_array[h2][0];
            double combined_reward = heuristic_rewards_map[heuristic1] + heuristic_rewards_map[heuristic2];
            if (combined_reward > best_pair_reward) {
                best_pair_reward = combined_reward;
                best_pair_heuristic1 = heuristic1;
                best_pair_heuristic2 = heuristic2;
            }
        }
    }

    // Print and save final summary
    std::cout << "Statistical Analysis:\n";
    std::cout << "Best Objective Function (BOF): " << best_cost_eff << "\n";
    std::cout << "Worst Objective Function (OF): " << worst_OF << "\n";
    std::cout << "Average Objective Function (OF): " << average_OF << "\n";
    std::cout << "Average Computation Time per Iteration: " << average_computation_time << " seconds\n";
    if (best_found_at != -1) {
        std::cout << "Time to Find Best Solution: Iteration " << best_found_at << "\n";
    } else {
        std::cout << "Best solution was not updated during the optimization.\n";
    }
    std::cout << "Number of Best Improving Solutions: " << total_best_improvements << "\n";
    std::cout << "Number of Worse Performing Heuristics: " << worse_performing_heuristics << "\n";
    if (best_pair_heuristic1 != -1 && best_pair_heuristic2 != -1) {
        std::cout << "Pair of Heuristics with Best Acceptance Move Strategy: "
                  << best_pair_heuristic1 << " and " << best_pair_heuristic2 << "\n";
    } else {
        std::cout << "No best pair found.\n";
    }

    // Save final summary to file
    if (outfile) {
        outfile << "\nFinal Summary:\n";
        outfile << "Best Cost Efficiency: " << best_cost_eff << "\n";
        outfile << "Best Cost Diversity: " << best_cost_div << "\n";
        outfile << "Worst Objective Function (OF): " << worst_OF << "\n";
        outfile << "Average Objective Function (OF): " << average_OF << "\n";
        outfile << "Average Computation Time: " << average_computation_time << " seconds\n";
        if (best_found_at != -1) {
            outfile << "Best Found At Iteration: " << best_found_at << "\n";
        } else {
            outfile << "Best solution never improved beyond initial.\n";
        }
        outfile << "Number of Best Improving Solutions: " << total_best_improvements << "\n";
        outfile << "Number of Worse Performing Heuristics: " << worse_performing_heuristics << "\n";
        if (best_pair_heuristic1 != -1 && best_pair_heuristic2 != -1) {
            outfile << "Best Heuristic Pair: " << best_pair_heuristic1 << " and " << best_pair_heuristic2 << "\n";
        } else {
            outfile << "No best pair found.\n";
        }

        // Heuristic performance data
        outfile << "\nHeuristic Performance Data:\n";
        for (auto &kv : heuristic_costs) {
            int h = kv.first;
            outfile << "Heuristic " << h << ":\n";
            outfile << "  Costs: ";
            for (auto c : kv.second) outfile << c << " ";
            outfile << "\n  Times: ";
            for (auto t : heuristic_times[h]) outfile << t << " ";
            outfile << "\n  Iterations: ";
            for (auto it : heuristic_iterations[h]) outfile << it << " ";
            outfile << "\n  Total Reward: " << heuristic_rewards_map[h]
                    << "\n  Usage Count: " << heuristic_usage_count_map[h]
                    << "\n  Total Time: " << heuristic_total_time_map[h] << "\n\n";
        }

        outfile.close();
        std::cout << "Results saved to " << file_path << std::endl;
    }
    check_best_solution();
    //free_memory();
    return team;
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

void Hyper_heuristic::testingmethods_ADSH_R(double optimal = 495.0) {
    for (int accept = 0; accept < 3; ++accept) { // Accept values from 0 to 4
        std::vector<double> costs;
        std::vector<std::map<int, std::vector<double>>> heuristic_costs_runs;
        std::vector<std::map<int, std::vector<double>>> heuristic_times_runs;
        std::vector<std::map<int, std::vector<int>>> heuristic_iterations_runs;

        for (int i = 0; i < 5; ++i) { // Run 50 iterations
            srand(i); // Set seed for reproducibility
            auto [sol, obj, L, heuristic_costs, heuristic_times, heuristic_iterations] =
                ADSH_random(accept, optimal);

            costs.push_back(obj);
            heuristic_costs_runs.push_back(heuristic_costs);
            heuristic_times_runs.push_back(heuristic_times);
            heuristic_iterations_runs.push_back(heuristic_iterations);
        }

        // Calculate statistics
        double mean_cost = calculate_mean(costs);
        double stddev_cost = calculate_stddev(costs, mean_cost);
        double max_cost = *std::max_element(costs.begin(), costs.end());
        switch(accept){
        case 0:
            cout<<"\n\n -----------------------------------------------\n";
            std::cout << "Accept Criterion: " << accept << " :=(IE) Improving or Equal.\n";
            std::cout << "Mean Objective Value: " << mean_cost << "\n";
            std::cout << "Standard Deviation: " << stddev_cost << "\n";
            std::cout << "Maximum Objective Value: " << max_cost << "\n";
            std::cout << "All Objective Values: ";
        for (const auto& cost : costs) {
            std::cout << cost << " ";
        }
        std::cout << "\n\n";

        // Optionally, print heuristic-related data
        for (size_t run = 0; run < heuristic_costs_runs.size(); ++run) {
            std::cout << "Run " << run + 1 << " Heuristic Costs:\n";
            for (const auto& [heuristic, values] : heuristic_costs_runs[run]) {
                std::cout << "  Heuristic " << heuristic << ": ";
                for (const auto& value : values) {
                    std::cout << value << " ";
                }
                std::cout << "\n";
            }
        }
        check_best_solution();
        break;
        case 1:cout<<"\n\n -----------------------------------------------\n";
        std::cout << "Accept Criterion: " << accept << " :=(OI) Only Improving.\n";
        // Print results

        std::cout << "Mean Objective Value: " << mean_cost << "\n";
        std::cout << "Standard Deviation: " << stddev_cost << "\n";
        std::cout << "Maximum Objective Value: " << max_cost << "\n";
        std::cout << "All Objective Values: ";
        for (const auto& cost : costs) {
            std::cout << cost << " ";
        }
        std::cout << "\n\n";

        // Optionally, print heuristic-related data
        for (size_t run = 0; run < heuristic_costs_runs.size(); ++run) {
            std::cout << "Run " << run + 1 << " Heuristic Costs:\n";
            for (const auto& [heuristic, values] : heuristic_costs_runs[run]) {
                std::cout << "  Heuristic " << heuristic << ": ";
                for (const auto& value : values) {
                    std::cout << value << " ";
                }
                std::cout << "\n";
            }
        }
        check_best_solution();
        break;
        case 2:
        cout<<"\n\n -----------------------------------------------\n";
        std::cout << "Accept Criterion: " << accept << " := RR: Record to Record.\n";

        // Print results
        std::cout << "Accept Criterion: " << accept << "\n";
        std::cout << "Mean Objective Value: " << mean_cost << "\n";
        std::cout << "Standard Deviation: " << stddev_cost << "\n";
        std::cout << "Maximum Objective Value: " << max_cost << "\n";
        std::cout << "All Objective Values: ";
        for (const auto& cost : costs) {
            std::cout << cost << " ";
        }
        std::cout << "\n\n";

        // Optionally, print heuristic-related data
        for (size_t run = 0; run < heuristic_costs_runs.size(); ++run) {
            std::cout << "Run " << run + 1 << " Heuristic Costs:\n";
            for (const auto& [heuristic, values] : heuristic_costs_runs[run]) {
                std::cout << "  Heuristic " << heuristic << ": ";
                for (const auto& value : values) {
                    std::cout << value << " ";
                }
                std::cout << "\n";
            }
        }
        check_best_solution();
        break;
        default:
            cout<<" No acceptance criterion Chosen\n.";
        }
    }

    free_memory();
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

// Function to run MAHH_Algorithm with specified strategy
void run_strategy1(int max_time, MABStrategy strategy) {
    Hyper_heuristic hh_instance; // Create a new instance for each thread
    hh_instance.MAHH_Selection_ThreeMAB(max_time, strategy);
}

void Hyper_heuristic::MAHH_Algorithm12(int max_time, MABStrategy mab_strategy) {
     std::cout <<"============================================================================="<< std::endl;
     std::cout <<"Multi-armed Bandit Selection Hyper-heuristic Framework Start its Processes." << std::endl;
     std::cout <<"============================================================================="<< std::endl;
    // Initialize variables
    std::map<int, double> heuristic_rewards;      // Total rewards per heuristic
    std::map<int, int> heuristic_counts;          // Number of times each heuristic is selected
    std::map<int, double> heuristic_estimates;    // Estimated value of each heuristic
    std::map<int, double> heuristic_alpha;        // Success counts for Thompson Sampling
    std::map<int, double> heuristic_beta;         // Failure counts for Thompson Sampling
    std::vector<int> objective_values;            // Stores objective function values over iterations
    std::vector<double> iteration_times;          // Time taken for each iteration
    std::vector<int> heuristics = {15, 16, 17, 18, 19, 20,21,22,23}; // List of heuristics
    int total_iterations = 0;

    // Parameters for epsilon-greedy
    double epsilon = 0.1; // Exploration rate for epsilon-greedy

    // Initialize heuristic statistics
    for (int h : heuristics) {
        heuristic_rewards[h] = 0.0;
        heuristic_counts[h] = 0;
        heuristic_estimates[h] = 0.0;
        heuristic_alpha[h] = 1.0; // For Thompson Sampling (Beta prior)
        heuristic_beta[h] = 1.0;  // For Thompson Sampling (Beta prior)
    }

    // Initialize solution
    generate_initialrandom();
    objective_Function1(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;

    // Initialize best solution
    int **s_current = team; // Use s_current = team
    int **s_best = team;    // Use s_best = team
    int best_cost_eff = cost_eff;
    int best_cost_div = cost_div;

    // Initialize best solution arrays
    // Assuming best_solution, eff_best, div_best, etc., are already allocated and properly initialized
    std::string folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";

    // Declare the folder path
    //std::filesystem::path folder_path = "D:\\Result Hyper-heuristic Models\\";

    // Check if the directory exists, create it if it doesn't
    if (!std::filesystem::exists(folder_path)) {
        try {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return; // Exit the function if directory creation fails
        }
    }

    // Start timing
    auto start_time = std::chrono::steady_clock::now();

    // Random number generator for Thompson Sampling
    std::mt19937 rng_thompson(std::random_device{}());

    // Main loop
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() < max_time) {
        total_iterations++;
        auto iteration_start = std::chrono::steady_clock::now();

        int selected_heuristic = -1;

        // Heuristic selection based on specified MAB strategy
        if (mab_strategy == UCB) {
            // UCB1 strategy
            double total_counts = std::accumulate(heuristic_counts.begin(), heuristic_counts.end(), 0.0,
                                                  [](double sum, const std::pair<int, int>& p) { return sum + p.second; });
            double log_total = std::log(std::max(1.0, total_counts));
            double max_ucb = -std::numeric_limits<double>::infinity();

            for (int h : heuristics) {
                double avg_reward = heuristic_counts[h] > 0 ? heuristic_rewards[h] / heuristic_counts[h] : 0.0;
                double ucb = avg_reward + std::sqrt(2.0 * log_total / (heuristic_counts[h] + 1e-5));
                if (ucb > max_ucb) {
                    max_ucb = ucb;
                    selected_heuristic = h;
                }
            }
        }
        else if (mab_strategy == EPSILON_GREEDY) {
            // Epsilon-Greedy strategy
            double rand_val = ((double) std::rand() / RAND_MAX);
            if (rand_val < epsilon) {
                // Exploration: select a random heuristic
                selected_heuristic = heuristics[std::rand() % heuristics.size()];
            }
            else {
                // Exploitation: select the best heuristic so far
                selected_heuristic = *std::max_element(heuristics.begin(), heuristics.end(),
                                                       [&](int h1, int h2) {
                                                           return heuristic_estimates[h1] < heuristic_estimates[h2];
                                                       });
            }
        }
        else if (mab_strategy == THOMPSON_SAMPLING) {
            // Thompson Sampling strategy
            double max_sample = -std::numeric_limits<double>::infinity();
            for (int h : heuristics) {
                // Sample from Beta distribution using Gamma distribution
                std::gamma_distribution<double> gamma_alpha(heuristic_alpha[h], 1.0);
                std::gamma_distribution<double> gamma_beta(heuristic_beta[h], 1.0);
                double sample_alpha = gamma_alpha(rng_thompson);
                double sample_beta = gamma_beta(rng_thompson);
                double theta = sample_alpha / (sample_alpha + sample_beta);
                if (theta > max_sample) {
                    max_sample = theta;
                    selected_heuristic = h;
                }
            }
        }

        // Apply the selected heuristic
        int **s_temp = team; // Use s_temp = team

        // Measure heuristic application time
        auto heuristic_start_time = std::chrono::steady_clock::now();
        ApplyHeuristic(selected_heuristic, s_temp); // Apply heuristic to s_temp
        double heuristic_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - heuristic_start_time).count();

        // Evaluate the new solution
        objective_Function1(s_temp);
        int new_cost_eff = f_cur;
        int new_cost_div = f_cur_div;

        // Calculate reward (positive if improvement)
        double reward = 0.0;
        if ((new_cost_eff > cost_eff) || (new_cost_div >= min_div)) {
            reward = (new_cost_eff - cost_eff) + (new_cost_div - cost_div);
        }

        // Update heuristic statistics
        heuristic_rewards[selected_heuristic] += reward;
        heuristic_counts[selected_heuristic]++;

        // Update heuristic estimates for Epsilon-Greedy
        if (mab_strategy == EPSILON_GREEDY) {
            heuristic_estimates[selected_heuristic] = heuristic_rewards[selected_heuristic] / heuristic_counts[selected_heuristic];
        }
        // Update alpha and beta for Thompson Sampling
        else if (mab_strategy == THOMPSON_SAMPLING) {
            heuristic_alpha[selected_heuristic] += reward;        // Successes
            heuristic_beta[selected_heuristic] += (1 - reward);   // Failures (assuming reward is 0 or 1)
        }

        // Accept or reject the new solution
        if ((new_cost_eff > cost_eff) && (new_cost_div >= min_div)) {
            // Accept the new solution
            s_current = s_temp; // Assign s_current to s_temp
            cost_eff = new_cost_eff;
            cost_div = new_cost_div;
        }

        // Update the best solution if improved
        if ((new_cost_eff > best_cost_eff) && (new_cost_div >= min_div)) {
            s_best = s_temp; // Assign s_best to s_temp
            best_cost_eff = new_cost_eff;
            best_cost_div = new_cost_div;
            best_eff = new_cost_eff;
            best_div = new_cost_div;
            time_taken = heuristic_time;

            // Update best solution arrays
            for (int m = 0; m < num_node; m++)
                fbest_solution[m] = best_solution[m];
            for (int m = 1; m <= num_team; m++) {
                eff_fbest[m] = eff_best[m];
                div_fbest[m] = div_best[m];
            }
        }
        else {
            team = s_current; // Revert to current solution
        }

        // Record objective function value
        objective_values.push_back(cost_eff);

        // Calculate iteration time
        double iteration_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - iteration_start).count();
        iteration_times.push_back(iteration_time);

        // Output results for the current iteration
        std::cout << "Iteration: " << total_iterations
                  << " | Strategy: " << (mab_strategy == UCB ? "UCB" : mab_strategy == EPSILON_GREEDY ? "Epsilon-Greedy" : "Thompson Sampling")
                  << " | Selected Heuristic: " << selected_heuristic
                  << " | Efficiency: " << cost_eff
                  << " | Diversity: " << cost_div
                  << " | Best Efficiency: " << best_cost_eff
                  << " | Best Diversity: " << best_cost_div
                  << " | Heuristic Time: " << heuristic_time << " seconds"
                  << std::endl;
    }

    // Calculate statistics
    double total_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
    double average_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0) / objective_values.size();
    int worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    double average_iteration_time = total_time / total_iterations;

    // Identify best and worst heuristics based on rewards
    int best_heuristic = -1;
    int worst_heuristic = -1;
    double max_reward = -std::numeric_limits<double>::infinity();
    double min_reward = std::numeric_limits<double>::infinity();

    for (int h : heuristics) {
        if (heuristic_rewards[h] > max_reward) {
            max_reward = heuristic_rewards[h];
            best_heuristic = h;
        }
        if (heuristic_rewards[h] < min_reward) {
            min_reward = heuristic_rewards[h];
            worst_heuristic = h;
        }
    }
    // Define the folder and file paths for saving results


    // Construct the result file path
    std::string strategy_name = (mab_strategy == UCB ? "UCB" : mab_strategy == EPSILON_GREEDY ? "Epsilon_Greedy" : "Thompson_Sampling");
    std::string result_file_path = folder_path + ("MAHH_Results_" + instanceName +"_"+ strategy_name + ".txt");

    // Save results to a file
    std::ofstream result_file(result_file_path);
    if (result_file.is_open()) {
        result_file << "Final best objectives:\n";
        result_file << "Efficiency: " << best_cost_eff << "\n";
        result_file << "Diversity: " << best_cost_div << "\n\n";

        result_file << "Statistics:\n";
        result_file << "Total time: " << total_time << " seconds\n";
        result_file << "Average objective function value: " << average_objective << "\n";
        result_file << "Worst objective function value: " << worst_objective << "\n";
        result_file << "Average iteration time: " << average_iteration_time << " seconds\n";
        result_file << "Best heuristic: " << best_heuristic << " with total reward " << heuristic_rewards[best_heuristic] << "\n";
        result_file << "Worst heuristic: " << worst_heuristic << " with total reward " << heuristic_rewards[worst_heuristic] << "\n\n";

        result_file << "Heuristic Performance:\n";
        for (int h : heuristics) {
            result_file << "Heuristic " << h << ":\n";
            result_file << "  Usage Count: " << heuristic_counts[h] << "\n";
            result_file << "  Total Reward: " << heuristic_rewards[h] << "\n";
            if (mab_strategy == EPSILON_GREEDY) {
                result_file << "  Estimated Value: " << heuristic_estimates[h] << "\n";
            }
            else if (mab_strategy == THOMPSON_SAMPLING) {
                result_file << "  Alpha (Successes): " << heuristic_alpha[h] << "\n";
                result_file << "  Beta (Failures): " << heuristic_beta[h] << "\n";
            }
            result_file << "\n";
        }

        result_file.close();
        std::cout << "Results saved to " << result_file_path << std::endl;
    }
    else {
        std::cerr << "Error: Could not open the file for writing results.\n";
    }

    // Finalize and print results
    std::cout << "Final best objectives: Efficiency = " << best_cost_eff
              << ", Diversity = " << best_cost_div << std::endl;

    // Clean up
    check_best_solution();
    //free_memory();
     std::cout <<"============================================================================="<< std::endl;
     std::cout <<"Multi-armed Bandit Selection Hyper-heuristic Framework Finished its Processes." << std::endl;
     std::cout <<"============================================================================="<< std::endl;
}

// Function to run MAHH_Algorithm with specified strategy
void run_strategy(int max_time, MABStrategy strategy) {
    Hyper_heuristic hh_instance; // Create a new instance for each thread
    hh_instance.MAHH_Algorithm12(max_time, strategy);
}

/*// Q-Learning Based Selection Function Implementation
void Hyper_heuristic::Q_Learning_Selection_Hyperheuristic_CMCEE(int max_time) {
    // Parameters and variables
    // Q-Learning parameters
    double alpha = 0.1;   // Learning rate
    double gamma = 0.9;   // Discount factor
    double epsilon = 0.1; // Exploration rate

    // Initialize Q-table: Map heuristic_id to Q-value
    std::map<int, double> Q_table;
    std::vector<int> heuristics = {12, 17, 18, 19, 20}; // List of heuristics

    // Initialize Q-values to 0
    for (int h : heuristics) {
        Q_table[h] = 0.0;
    }
    int **previous_solution;
    int **best_solution1;

    // Initialize solution
    generate_initialrandom();
    objective_Function(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    std::cout << "Initial objectives eff and div: " << cost_eff << ", " << cost_div << std::endl;

    int best_cost_eff = cost_eff, best_cost_div = cost_div;

    // Data structures for analysis
    std::vector<int> Selected;
    int T = 100; // Number of iterations (can be adjusted based on max_time)
    std::map<int, std::vector<int>> heuristic_costs;
    std::map<int, std::vector<double>> heuristic_times;

    double total_elapsed_time = 0.0;

    // Define the folder path using std::filesystem
    std::filesystem::path folder_path = "D:/Result Hyper-heuristic Models/";

    // Check if the directory exists, create it if it doesn't
    if (!std::filesystem::exists(folder_path)) {
        try {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return; // Exit the function if directory creation fails
        }
    }

    // Define the full file path for saving the results
    std::filesystem::path file_path = folder_path / "Q_Learning_Selection_Hyperheuristic_CMCEE_results.txt";

    // Create an output file stream to save the results to a text file
    std::ofstream outfile(file_path);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << file_path << std::endl;
        return;
    }

    // Write headers to the text file
    outfile << "Iteration\tSelected Heuristic\tCost Efficiency\tCost Diversity\tTime Taken (seconds)\tReward\tQ-Value\n";

    // Initialize random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_real(0.0, 1.0);
    std::uniform_int_distribution<> dis_int(0, heuristics.size() - 1);

    // Data structures for statistical analysis
    std::map<int, double> heuristic_total_time;       // Total time taken by each heuristic
    std::map<int, int> heuristic_improvement_count;   // Number of times each heuristic improved the solution
    std::map<int, double> heuristic_rewards;          // Total rewards accumulated by each heuristic
    std::map<int, int> heuristic_usage_count;         // Number of times each heuristic was used
    std::vector<int> objective_values; // Stores efficiency values over iterations
    std::vector<double> iteration_times;
    // Start timing
    auto total_start_time = std::chrono::steady_clock::now();

    // Iterate until max_time is reached
    int iteration = 0; // Track iteration count
    while (true) {
        iteration++;
        auto iteration_start_time = std::chrono::steady_clock::now();

        // Check if max_time is exceeded
        double elapsed_time_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
        if (elapsed_time_total >= max_time) {
            break;
        }

        // Decide whether to explore or exploit
        double rand_val = dis_real(gen);
        int selected_heuristic = -1;
        if (rand_val < epsilon) {
            // Exploration: select a random heuristic
            selected_heuristic = heuristics[dis_int(gen)];
        }
        else {
            // Exploitation: select the heuristic with the highest Q-value
            double max_Q = -std::numeric_limits<double>::infinity();
            for (int h : heuristics) {
                if (Q_table[h] > max_Q) {
                    max_Q = Q_table[h];
                    selected_heuristic = h;
                }
            }
        }

        std::cout << "Iteration: " << iteration << ", Selected Heuristic LLH[" << selected_heuristic
                  << "], cost_eff: " << cost_eff << ", cost_div: " << cost_div << std::endl;

        // Start timing for heuristic application
        auto start_time = std::chrono::steady_clock::now();

        // Apply heuristic
        ApplyHeuristic(selected_heuristic, team);

        // Calculate elapsed time
        double elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        total_elapsed_time += elapsed_time;

        // Update heuristic time tracking
        heuristic_total_time[selected_heuristic] += elapsed_time;
        heuristic_usage_count[selected_heuristic]++;

        // Calculate new costs
        objective_Function(team);
        int newcost_eff = f_cur, newcost_div = f_cur_div;
        std::cout << "Iteration: " << iteration << ", After Selected Heuristic LLH[" << selected_heuristic
                  << "], newcost_eff: " << newcost_eff << ", newcost_div: " << newcost_div << "\n";

        // Calculate reward based on improvement
        double reward = 0.0;
        if ((newcost_eff > cost_eff) || (newcost_div > cost_div)) {
            reward = (newcost_eff - cost_eff) + (newcost_div - cost_div);
            heuristic_improvement_count[selected_heuristic]++;
        }
        heuristic_rewards[selected_heuristic] += reward;

        // Update Q-value
        // Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s,a') - Q(s,a))
        double max_Q_next = -std::numeric_limits<double>::infinity();
        for (int h : heuristics) {
            if (Q_table[h] > max_Q_next) {
                max_Q_next = Q_table[h];
            }
        }
        Q_table[selected_heuristic] += alpha * (reward + gamma * max_Q_next - Q_table[selected_heuristic]);

        // Record objective function value
        // Assuming single state; otherwise, extend to include states
        // For simplicity, using a single state approach here
        // If multiple states are involved, modify accordingly
        // Currently, state is implicit and not tracked

        // Accept or reject the new solution
        if ((newcost_eff > cost_eff) && (newcost_div >= cost_div)){
            // Accept the new solution
            cost_eff = newcost_eff;
            cost_div = newcost_div;
            previous_solution = team;
            Selected.push_back(selected_heuristic);
            if (newcost_eff > best_cost_eff) {
                best_solution1 = team;
                best_cost_eff = newcost_eff;
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
            team = previous_solution;
        }

        // Record objective function value
        objective_values.push_back(cost_eff);

        // Calculate iteration time
        double iteration_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - iteration_start_time).count();
        iteration_times.push_back(iteration_time);

        // Save data for each iteration to the text file
        outfile << iteration << "\t"  // Iteration
                << selected_heuristic << "\t"  // Selected Heuristic
                << newcost_eff << "\t"  // Efficiency
                << newcost_div << "\t"  // Diversity
                << elapsed_time << "\t"  // Time taken
                << reward << "\t"         // Reward
                << Q_table[selected_heuristic] << "\n"; // Updated Q-Value
    }

    // Close the output file
    outfile.close();

    // Statistical Analysis
    // Calculate average objective function value
    double average_objective = 0.0;
    if (!objective_values.empty()) {
        double total_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0);
        average_objective = total_objective / objective_values.size();
    }

    // Identify worst objective function value
    int worst_objective = std::numeric_limits<int>::max();
    if (!objective_values.empty()) {
        worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    }

    // Calculate average CPU time per iteration
    double average_cpu_time = 0.0;
    if (!iteration_times.empty()) {
        double total_iteration_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
        average_cpu_time = total_iteration_time / iteration_times.size();
    }

    // Identify best and worst heuristics based on rewards
    int best_heuristic = -1;
    int worst_heuristic = -1;
    double max_reward = -std::numeric_limits<double>::infinity();
    double min_reward = std::numeric_limits<double>::infinity();

    for (int h : heuristics) {
        if (heuristic_rewards[h] > max_reward) {
            max_reward = heuristic_rewards[h];
            best_heuristic = h;
        }
        if (heuristic_rewards[h] < min_reward) {
            min_reward = heuristic_rewards[h];
            worst_heuristic = h;
        }
    }

    // Save additional statistics to a separate summary file
    std::filesystem::path summary_file_path = folder_path / "Q_Learning_Selection_Hyperheuristic_CMCEE_summary.txt";
    std::ofstream summary_file(summary_file_path);
    if (summary_file.is_open()) {
        summary_file << "Final Best Objectives:\n";
        summary_file << "Efficiency: " << best_cost_eff << "\n";
        summary_file << "Diversity: " << best_cost_div << "\n\n";

        summary_file << "Statistical Analysis:\n";
        summary_file << "Total Time Taken: " << std::fixed << std::setprecision(4)
                     << total_elapsed_time << " seconds\n";
        summary_file << "Average Objective Function Value: " << average_objective << "\n";
        summary_file << "Worst Objective Function Value: " << worst_objective << "\n";
        summary_file << "Average CPU Time per Iteration: " << average_cpu_time << " seconds\n\n";

        summary_file << "Heuristic Performance:\n";
        for (int h : heuristics) {
            summary_file << "Heuristic " << h << ":\n";
            summary_file << "  Usage Count: " << heuristic_usage_count[h] << "\n";
            summary_file << "  Total Reward: " << heuristic_rewards[h] << "\n";
            summary_file << "  Improvement Count: " << heuristic_improvement_count[h] << "\n";
            summary_file << "  Total Time: " << heuristic_total_time[h] << " seconds\n\n";
        }

        summary_file << "Best Heuristic: " << best_heuristic << " with Total Reward: " << max_reward << "\n";
        summary_file << "Worst Heuristic: " << worst_heuristic << " with Total Reward: " << min_reward << "\n";

        summary_file.close();
        std::cout << "Summary results saved to " << summary_file_path << std::endl;
    }
    else {
        std::cerr << "Error: Could not open the summary file for writing results.\n";
    }

    // Additional Console Output
    std::cout << "Total time taken to find the solution: " << std::fixed << std::setprecision(4)
              << total_elapsed_time << " seconds" << std::endl;
    std::cout << "Selected heuristic sequence: ";
    for (const auto& h : Selected) std::cout << h << " ";
    std::cout << std::endl;

    // Final results
    check_best_solution();
    free_memory();
}*/

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
// ============================================================================
// Q-Learning + UCB-Based Hyper-Heuristic for CMCEE Problem
// ============================================================================
// This function implements a Q-learning-based heuristic selection mechanism
// integrated with UCB (Upper Confidence Bound) for balancing exploration and exploitation.
// It iteratively applies heuristics to improve efficiency and diversity objectives.
// ============================================================================

void Hyper_heuristic::Q_Learning_Selection_Hyperheuristic_CMCEE1(int max_time) {

    // ------------------------------------------------------------
    // 1. PARAMETER INITIALIZATION
    // ------------------------------------------------------------
    // Learning and exploration parameters
    double alpha_initial = 0.1;    // Initial learning rate for Q-value updates
    double gamma = 0.9;            // Discount factor controlling future reward influence
    double epsilon_initial = 1.0;  // Initial exploration rate for epsilon-greedy
    double alpha = alpha_initial;
    double epsilon = epsilon_initial;

    // UCB exploration parameter (used in action selection)
    double c = sqrt(2);

    // List of available low-level heuristics (LLHs)
    std::vector<int> heuristics = {15,16,17,18,19,20,21,22,23};

    // Q-learning and reward tracking tables
    std::map<int, double> Q_table;                 // Stores Q-values for each heuristic
    std::map<int, double> heuristic_rewards;       // Tracks accumulated rewards
    std::map<int, double> heuristic_total_time;    // Execution time for each heuristic
    std::map<int, int> heuristic_usage_count;      // Number of times each heuristic used
    std::map<int, int> heuristic_improvement_count;// Count of successful improvements

    // Initialize Q-values and counters to zero
    for (int h : heuristics) {
        Q_table[h] = 0.0;
        heuristic_rewards[h] = 0.0;
        heuristic_usage_count[h] = 0;
        heuristic_improvement_count[h] = 0;
    }

    // ------------------------------------------------------------
    // 2. INITIAL SOLUTION SETUP
    // ------------------------------------------------------------
    generate_initialrandom();  // Create an initial feasible team configuration
    objective_Function1(team); // Evaluate the initial solution

    int cost_eff = f_cur;      // Initial efficiency
    int cost_div = f_cur_div;  // Initial diversity
    int best_cost_eff = cost_eff, best_cost_div = cost_div;

    // Store deep copies of initial and best solutions
    int** previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** best_solution1    = deep_copy_solution(team, num_node, num_team, num_each_t);

    std::cout << "Initial objectives eff and div: " << cost_eff << ", " << cost_div << "\n";

    // ------------------------------------------------------------
    // 3. STATISTICAL AND LOGGING STRUCTURES
    // ------------------------------------------------------------
    std::vector<int> Selected;              // Record of selected heuristics
    std::vector<int> objective_values;      // Efficiency values per iteration
    std::vector<double> iteration_times;    // CPU time per iteration
    std::vector<double> rewards_record;     // Reward values per iteration
    double total_elapsed_time = 0.0;

    // ------------------------------------------------------------
    // 4. FILE OUTPUT SETUP
    // ------------------------------------------------------------
    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/QL_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    // Text log for iteration results
    std::filesystem::path file_path  = folder_path / ("Q_Learning_HH_CMCEE_" + instanceName + "_results.txt");
    std::ofstream outfile(file_path);
    outfile << "Iteration\tSelected Heuristic\tEfficiency\tDiversity\tTime(s)\tReward\tQ-Value\n";

    // CSV log for convergence trace (used in plotting)
    std::filesystem::path trace_path = folder_path / ("Q_Learning_HH_CMCEE_" + instanceName + "_Convergence_Trace.csv");
    std::ofstream trace(trace_path);
    trace << "Iteration,Efficiency,Diversity,BestEfficiency,BestDiversity,Reward,Runtime\n";

    // ------------------------------------------------------------
    // 5. RANDOMIZATION SETUP
    // ------------------------------------------------------------
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_real(0.0, 1.0);
    std::uniform_int_distribution<> dis_int(0, heuristics.size() - 1);

    // ------------------------------------------------------------
    // 6. MAIN ITERATIVE LEARNING LOOP
    // ------------------------------------------------------------
    clock_t total_start = clock();
    int iteration = 0;

    // Continue until maximum CPU time is reached
    while ((double)(clock() - total_start) / CLOCKS_PER_SEC < max_time) {
        iteration++;
        clock_t iter_start = clock();

        // --- Adaptive parameter decay (exploration vs exploitation) ---
        epsilon = std::max(0.01, epsilon_initial * exp(-0.001 * iteration));
        alpha   = std::max(0.01, alpha_initial * exp(-0.001 * iteration));

        // ------------------------------------------------------------
        // 6.1 HEURISTIC SELECTION: ε-greedy + UCB1
        // ------------------------------------------------------------
        int selected_heuristic = -1;
        double rand_val = dis_real(gen);

        // Count total usage across all heuristics
        int total_usage = 0;
        for (auto& kv : heuristic_usage_count) total_usage += kv.second;

        if (rand_val < epsilon) {
            // Exploration: random heuristic selection
            selected_heuristic = heuristics[dis_int(gen)];
        } else {
            // Exploitation: use UCB1 rule to balance reward and exploration
            double best_ucb = -1e9;
            for (int h : heuristics) {
                double avg_reward = (heuristic_usage_count[h] > 0)
                    ? heuristic_rewards[h] / heuristic_usage_count[h]
                    : 0.0;
                double explore = c * sqrt(log(total_usage + 1) / (heuristic_usage_count[h] + 1));
                double ucb_val = avg_reward + explore;
                if (ucb_val > best_ucb) {
                    best_ucb = ucb_val;
                    selected_heuristic = h;
                }
            }
        }

        // Print iteration summary to console
        std::cout << "Iteration " << iteration << " | LLH[" << selected_heuristic
                  << "] | Eff: " << cost_eff << " | Div: " << cost_div << "\n";

        // ------------------------------------------------------------
        // 6.2 APPLY SELECTED HEURISTIC
        // ------------------------------------------------------------
        clock_t h_start = clock();
        ApplyHeuristic(selected_heuristic, team); // Apply the chosen LLH to modify the solution
        double elapsed_time = (double)(clock() - h_start) / CLOCKS_PER_SEC;
        total_elapsed_time += elapsed_time;

        // Update usage statistics
        heuristic_total_time[selected_heuristic] += elapsed_time;
        heuristic_usage_count[selected_heuristic]++;

        // ------------------------------------------------------------
        // 6.3 EVALUATE NEW SOLUTION
        // ------------------------------------------------------------
        objective_Function1(team);
        int new_eff = f_cur, new_div = f_cur_div;

        // Reward = improvement in efficiency + diversity
        double reward = 0.0;
        if ((new_eff > cost_eff) || (new_div > cost_div)) {
            reward = (new_eff - cost_eff) + (new_div - cost_div);
            heuristic_improvement_count[selected_heuristic]++;
        }

        // Clip negative rewards to zero
        reward = std::max(0.0, reward);
        heuristic_rewards[selected_heuristic] += reward;
        rewards_record.push_back(reward);

        // ------------------------------------------------------------
        // 6.4 Q-LEARNING UPDATE RULE
        // ------------------------------------------------------------
        double max_Q_next = -1e9;
        for (int h : heuristics) max_Q_next = std::max(max_Q_next, Q_table[h]);
        Q_table[selected_heuristic] += alpha * (reward + gamma * max_Q_next - Q_table[selected_heuristic]);

        // ------------------------------------------------------------
        // 6.5 ACCEPTANCE / REJECTION
        // ------------------------------------------------------------
        if ((new_eff > cost_eff) && (new_div >= cost_div)) {
            // Accept the new solution and update history
            cost_eff = new_eff;
            cost_div = new_div;

            free_solution(previous_solution, num_node, num_team, num_each_t);
            previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);

            // Update global best
            if (new_eff > best_cost_eff) {
                best_cost_eff = new_eff;
                best_cost_div = new_div;

                free_solution(best_solution1, num_node, num_team, num_each_t);
                best_solution1 = deep_copy_solution(team, num_node, num_team, num_each_t);

                for (int m = 0; m < num_node; m++) fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        } else {
            // Revert to previous solution if not improved
            for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < num_each_t; ++j)
                    team[i][j] = previous_solution[i][j];
        }

        // ------------------------------------------------------------
        // 6.6 LOGGING RESULTS
        // ------------------------------------------------------------
        double iter_time = (double)(clock() - iter_start) / CLOCKS_PER_SEC;
        iteration_times.push_back(iter_time);
        objective_values.push_back(cost_eff);

        outfile << iteration << "\t" << selected_heuristic << "\t"
                << new_eff << "\t" << new_div << "\t"
                << elapsed_time << "\t" << reward << "\t"
                << Q_table[selected_heuristic] << "\n";

        double runtime = (double)(clock() - total_start) / CLOCKS_PER_SEC;
        trace << iteration << "," << new_eff << "," << new_div << ","
              << best_cost_eff << "," << best_cost_div << ","
              << reward << "," << runtime << "\n";
    }

    outfile.close();
    trace.close();

    // ------------------------------------------------------------
    // 7. STATISTICAL SUMMARY
    // ------------------------------------------------------------
    double avg_obj = 0.0, avg_time = 0.0;
    int worst_obj = 0;
    if (!objective_values.empty())
        avg_obj = std::accumulate(objective_values.begin(), objective_values.end(), 0.0) / objective_values.size();
    if (!iteration_times.empty())
        avg_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    if (!objective_values.empty())
        worst_obj = *std::min_element(objective_values.begin(), objective_values.end());

    // Print summary to console
    std::cout << "\nFinal Summary\n";
    std::cout << "--------------------------------------------\n";
    std::cout << "Best Efficiency : " << best_cost_eff << "\n";
    std::cout << "Best Diversity  : " << best_cost_div << "\n";
    std::cout << "Average Objective: " << avg_obj << "\n";
    std::cout << "Worst Objective : " << worst_obj << "\n";
    std::cout << "Average IterTime: " << avg_time << "s\n";
    std::cout << "Convergence Trace Saved: " << trace_path << "\n";
    std::cout << "--------------------------------------------\n";

    // Verify final best solution
    check_best_solution();

    std::cout << "=============================================================================\n";
    std::cout << "Q-Learning Selection Hyper-heuristic Framework Finished its Processes.\n";
    std::cout << "=============================================================================\n";
}
/*
void Hyper_heuristic::Q_Learning_Selection_Hyperheuristic_CMCEE1(int max_time) {
    // Parameters and variables
    // Initial Q-Learning parameters
    double alpha_initial = 0.1;    // Initial learning rate
    double gamma = 0.9;            // Discount factor
    double epsilon_initial = 1.0;  // Initial exploration rate

    double alpha = alpha_initial;
    double epsilon = epsilon_initial;
    int **previous_solution;
    int **best_solution1;
    // MAB parameter for UCB
    double c = sqrt(2); // Exploration parameter for UCB1

    // Initialize Q-table: Map heuristic_id to Q-value
    std::map<int, double> Q_table;
    std::vector<int> heuristics = {15,16, 17, 18, 19, 20,21,22,23}; // List of heuristics

    // Initialize Q-values to 0
    for (int h : heuristics) {
        Q_table[h] = 0.0;
    }

    // Initialize heuristic rewards and counts for MAB
    std::map<int, double> heuristic_rewards;          // Total rewards accumulated by each heuristic
    std::map<int, int> heuristic_usage_count;         // Number of times each heuristic was used

    // Initialize solution
    generate_initialrandom();
    objective_Function1(team);
    previous_solution = team;
    best_solution1 = team;
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    std::cout << "Initial objectives eff and div: " << cost_eff << ", " << cost_div << std::endl;

    int best_cost_eff = cost_eff, best_cost_div = cost_div;

    // Data structures for analysis
    std::vector<int> Selected;
    std::vector<int> objective_values; // Stores efficiency values over iterations
    std::vector<double> iteration_times;

    double total_elapsed_time = 0.0;

    // Define the folder path using std::filesystem
    std::filesystem::path folder_path = "D:/Result Hyper-heuristic Models/";

    // Check if the directory exists, create it if it doesn't
    if (!std::filesystem::exists(folder_path)) {
        try {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return; // Exit the function if directory creation fails
        }
    }

    // Define the full file path for saving the results
    std::filesystem::path file_path = folder_path / "Q_Learning_Selection_Hyperheuristic_CMCEE_results.txt";

    // Create an output file stream to save the results to a text file
    std::ofstream outfile(file_path);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << file_path << std::endl;
        return;
    }

    // Write headers to the text file
    outfile << "Iteration\tSelected Heuristic\tCost Efficiency\tCost Diversity\tTime Taken (seconds)\tReward\tQ-Value\n";

    // Initialize random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_real(0.0, 1.0);
    std::uniform_int_distribution<> dis_int(0, heuristics.size() - 1);

    // Data structures for statistical analysis
    std::map<int, double> heuristic_total_time;       // Total time taken by each heuristic
    std::map<int, int> heuristic_improvement_count;   // Number of times each heuristic improved the solution

    // Start timing
    auto total_start_time = std::chrono::steady_clock::now();

    // Iterate until max_time is reached
    int iteration = 0; // Track iteration count
    while (true) {
        iteration++;
        auto iteration_start_time = std::chrono::steady_clock::now();

        // Check if max_time is exceeded
        double elapsed_time_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
        if (elapsed_time_total >= max_time) {
            break;
        }

        // Update adaptive parameters (decaying over time)
        epsilon = epsilon_initial * exp(-0.001 * iteration);
        if (epsilon < 0.01) epsilon = 0.01; // Minimum exploration rate

        alpha = alpha_initial * exp(-0.001 * iteration);
        if (alpha < 0.01) alpha = 0.01; // Minimum learning rate

        // Decide whether to explore or exploit
        int selected_heuristic = -1;

        // Use UCB1 for action selection
        int total_heuristic_usage = 0;
        for (int h : heuristics) {
            total_heuristic_usage += heuristic_usage_count[h];
        }

        double rand_val = dis_real(gen);
        if (rand_val < epsilon) {
            // Exploration: select a random heuristic
            selected_heuristic = heuristics[dis_int(gen)];
        }
        else {
            // Exploitation: use UCB1 to select heuristic
            double max_UCB = -std::numeric_limits<double>::infinity();
            for (int h : heuristics) {
                double average_reward = 0.0;
                if (heuristic_usage_count[h] > 0) {
                    average_reward = heuristic_rewards[h] / heuristic_usage_count[h];
                }
                else {
                    // Assign zero average reward for untried heuristics
                    average_reward = 0.0;
                }
                double ucb_value = average_reward + c * sqrt(log(total_heuristic_usage + 1) / (heuristic_usage_count[h] + 1));
                if (ucb_value > max_UCB) {
                    max_UCB = ucb_value;
                    selected_heuristic = h;
                }
            }
        }

        std::cout << "Iteration: " << iteration << ", Selected Heuristic LLH[" << selected_heuristic
                  << "], cost_eff: " << cost_eff << ", cost_div: " << cost_div << std::endl;

        // Start timing for heuristic application
        auto start_time = std::chrono::steady_clock::now();

        // Apply heuristic
        ApplyHeuristic(selected_heuristic, team);

        // Calculate elapsed time
        double elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        total_elapsed_time += elapsed_time;

        // Update heuristic time tracking
        heuristic_total_time[selected_heuristic] += elapsed_time;
        heuristic_usage_count[selected_heuristic]++;

        // Calculate new costs
        objective_Function1(team);
        int newcost_eff = f_cur, newcost_div = f_cur_div;
        std::cout << "Iteration: " << iteration << ", After Selected Heuristic LLH[" << selected_heuristic
                  << "], newcost_eff: " << newcost_eff << ", newcost_div: " << newcost_div << "\n";

        // Calculate reward based on improvement
        double reward = 0.0;
        if ((newcost_eff > cost_eff) || (newcost_div >= cost_div)) {
            reward = (newcost_eff - cost_eff) + (newcost_div - cost_div);
            heuristic_improvement_count[selected_heuristic]++;
        }
        if (reward < 0){
            reward = 0;
            heuristic_rewards[selected_heuristic] += reward;
        }
        else{
            heuristic_rewards[selected_heuristic] += reward;
        }

        // Update Q-value using Q-Learning update rule
        double max_Q_next = -std::numeric_limits<double>::infinity();
        for (int h : heuristics) {
            if (Q_table[h] > max_Q_next) {
                max_Q_next = Q_table[h];
            }
        }
        Q_table[selected_heuristic] += alpha * (reward + gamma * max_Q_next - Q_table[selected_heuristic]);

        // Record objective function value
        objective_values.push_back(cost_eff);

        // Accept or reject the new solution
        if ((newcost_eff > cost_eff) && (newcost_div >= cost_div)) {
            // Accept the new solution
            cost_eff = newcost_eff;
            cost_div = newcost_div;
            previous_solution = team;

            // Optionally, deep copy team to previous_solution if needed
            Selected.push_back(selected_heuristic);
            if (newcost_eff > best_cost_eff) {
                // Update best solution
                best_cost_eff = newcost_eff;
                best_cost_div = newcost_div;
                best_solution1 = team;
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
            // Revert to previous_solution (implement deep copy if necessary)
            // For now, assuming team is restored from previous_solution
            team = previous_solution;
        }

        // Calculate iteration time
        double iteration_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - iteration_start_time).count();
        iteration_times.push_back(iteration_time);

        // Save data for each iteration to the text file
        outfile << iteration << "\t"  // Iteration
                << selected_heuristic << "\t"  // Selected Heuristic
                << newcost_eff << "\t"  // Efficiency
                << newcost_div << "\t"  // Diversity
                << elapsed_time << "\t"  // Time taken
                << reward << "\t"         // Reward
                << Q_table[selected_heuristic] << "\n"; // Updated Q-Value
    }

    // Close the output file
    outfile.close();

    // Statistical Analysis
    // Calculate average objective function value
    double average_objective = 0.0;
    if (!objective_values.empty()) {
        double total_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0);
        average_objective = total_objective / objective_values.size();
    }

    // Identify worst objective function value
    int worst_objective = std::numeric_limits<int>::max();
    if (!objective_values.empty()) {
        worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    }

    // Calculate average CPU time per iteration
    double average_cpu_time = 0.0;
    if (!iteration_times.empty()) {
        double total_iteration_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
        average_cpu_time = total_iteration_time / iteration_times.size();
    }

    // Identify best and worst heuristics based on rewards
    int best_heuristic = -1;
    int worst_heuristic = -1;
    double max_reward = -std::numeric_limits<double>::infinity();
    double min_reward = std::numeric_limits<double>::infinity();

    for (int h : heuristics) {
        if (heuristic_rewards[h] > max_reward) {
            max_reward = heuristic_rewards[h];
            best_heuristic = h;
        }
        if (heuristic_rewards[h] < min_reward) {
            min_reward = heuristic_rewards[h];
            worst_heuristic = h;
        }
    }

    // Save additional statistics to a separate summary file
    std::filesystem::path summary_file_path = folder_path / "Q_Learning_Selection_Hyperheuristic_CMCEE_summary.txt";
    std::ofstream summary_file(summary_file_path);
    if (summary_file.is_open()) {
        summary_file << "Final Best Objectives:\n";
        summary_file << "Efficiency: " << best_cost_eff << "\n";
        summary_file << "Diversity: " << best_cost_div << "\n\n";

        summary_file << "Statistical Analysis:\n";
        summary_file << "Total Time Taken: " << std::fixed << std::setprecision(4)
                     << total_elapsed_time << " seconds\n";
        summary_file << "Average Objective Function Value: " << average_objective << "\n";
        summary_file << "Worst Objective Function Value: " << worst_objective << "\n";
        summary_file << "Average CPU Time per Iteration: " << average_cpu_time << " seconds\n\n";

        summary_file << "Heuristic Performance:\n";
        for (int h : heuristics) {
            summary_file << "Heuristic " << h << ":\n";
            summary_file << "  Usage Count: " << heuristic_usage_count[h] << "\n";
            summary_file << "  Total Reward: " << heuristic_rewards[h] << "\n";
            summary_file << "  Improvement Count: " << heuristic_improvement_count[h] << "\n";
            summary_file << "  Total Time: " << heuristic_total_time[h] << " seconds\n\n";
        }

        summary_file << "Best Heuristic: " << best_heuristic << " with Total Reward: " << max_reward << "\n";
        summary_file << "Worst Heuristic: " << worst_heuristic << " with Total Reward: " << min_reward << "\n";

        summary_file.close();
        std::cout << "Summary results saved to " << summary_file_path << std::endl;
    }
    else {
        std::cerr << "Error: Could not open the summary file for writing results.\n";
    }

    // Additional Console Output
    std::cout << "Total time taken to find the solution: " << std::fixed << std::setprecision(4)
              << total_elapsed_time << " seconds" << std::endl;
    std::cout << "Selected heuristic sequence: ";
    for (const auto& h : Selected) std::cout << h << " ";
    std::cout << std::endl;

    // Final results
    check_best_solution();
    //free_memory();
}
*/

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
    const std::vector<int> heuristics = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23};
    //const std::vector<int> heuristics = {17,19,20,21,22,23};

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
    generate_initialrandom();
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

/*
// Implementation of the updated Q-Learning Selection Hyperheuristic
//// Q-Learning Based Selection Function Implementation (log-improvement state & SYMMETRIC tiered reward)
// State s ∈ {0..9} from Δ = E' − E  (Eq. 17)

void Hyper_heuristic::Q_Learning_Selection_Hyperheuristic_CMCEE(int max_time) {
    std::cout <<"=============================================================================\n";
    std::cout <<"Q-Learning Selection Hyper-heuristic Framework Start its Processes.\n";
    std::cout <<"=============================================================================\n";

     // -------------------------------
    // 1) Q-LEARNING PARAMETERS
    // -------------------------------
    const double alpha          = 0.30;   // learning rate
    const double gamma          = 0.50;   // discount factor
    double       epsilon        = 0.30;   // initial exploration rate
    const double epsilon_decay  = 0.99;   // decay factor
    const double MIN_EPS        = 0.05;   // floor for ε
    const int    topK           = 5;      // # of top heuristics to keep later

    // Set of LLHs
    const std::vector<int> heuristics = {17,18,19,20,21,22,23};
    //const std::vector<int> heuristics  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23};
    // Q-table: Map<StateKey, Map<Heuristic, Q-value>>
    std::map<std::string, std::map<int, double>> Q_table;

    // Credits for formulas (5.1–5.4) (kept same style / optional)
    std::map<int, double> heuristic_credits;
    for (int h : heuristics)
        heuristic_credits[h] = 0.0;

    // ---- Helpers for Eq.(17) & Reward (symmetric Eq. 18)
    auto pos_or_eps = [](double x, double eps=1e-12) -> double {
        return (x > eps ? x : eps);
    };
    // State mapping (Eq. 17)
    auto state_from_delta = [](double delta) -> int {
        const double tol = 1e-12;
        if (delta < -tol) return 0;                 // Δ < 0
        if (std::abs(delta) <= tol) return 1;       // Δ ≈ 0
        for (int z = 1; z <= 6; ++z) {              // z=1..6 → state=z+1 (2..7)
            double lo = std::pow(10.0, -z);
            double hi = std::pow(10.0, -(z-1));
            if (delta >= lo && delta < hi) return z + 1;
        }
        if (delta >= 1.0) return 8;                 // Δ ≥ 1
        return 9;                                   // otherwise (small positive)
    };
    // Reward (symmetric, graded penalties for Δ<0; graded rewards for Δ>0)
    auto reward_from_delta = [](double delta) -> double {
        const double tol = 1e-12;

        // Large positive improvement
        if (delta >= 1e-1 - tol) return 10.0;

        // Positive tiers: z = 1..5  → 10^{-(z+1)} ≤ Δ ≤ 10^{-z}  → 9..5
        for (int z = 1; z <= 5; ++z) {
            double lo = std::pow(10.0, -(z+1));
            double hi = std::pow(10.0, -z);
            if (delta + tol >= lo && delta - tol <= hi) return 10.0 - z; // 9..5
        }

        // Tiny/no gain (|Δ| < 1e-6) → 0
        if (std::fabs(delta) < 1e-6 + tol) return 0.0;

        // Large negative degradation
        if (delta <= -1e-1 + tol) return -10.0;

        // Negative tiers: mirror of positive → -5..-9
        for (int z = 1; z <= 5; ++z) {
            double hi = -std::pow(10.0, -(z+1)); // more negative
            double lo = -std::pow(10.0, -z);     // less negative
            if (delta - tol >= hi && delta + tol <= lo) return -(10.0 - z); // -9..-5
        }

        // Fallback penalty for small negatives not caught by tol
        return -5.0;
    };

    // 1a) Initialize Q-values for all discrete states s∈{0..9} and actions
    for (int s = 0; s <= 9; ++s) {
        std::string key = "s" + std::to_string(s);
        for (int h : heuristics) {

            Q_table[key][h] = 0.0;
        }
    }

    // -------------------------------
    // 2) INITIAL SOLUTION
    // -------------------------------
    generate_initialrandom();
    display(team);
    objective_Function1(team);
    int cost_eff = static_cast<int>(f_cur);
    int cost_div = static_cast<int>(f_cur_div);
    int** previous_solution = team;
    int** best_solution1    = team;

    std::cout << "Initial objectives eff and div: "
              << cost_eff << ", " << cost_div << std::endl;

    int best_cost_eff = cost_eff;
    int best_cost_div = cost_div;

    // Data for analysis
    std::vector<int>    Selected;
    std::vector<int>    objective_values;
    std::vector<int>    diversity_values;
    std::vector<double> iteration_times;

    double total_elapsed_time = 0.0;

    // -------------------------------
    // 3) FILE & FOLDER SETUP
    // -------------------------------
    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results12/";
    if (!std::filesystem::exists(folder_path)) {
        try {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return;
        }
    }

    // Results file
    std::filesystem::path file_path =
        folder_path / ("Q_Learning_Selection_Hyperheuristic_CMCEE_" + instanceName + "_results.txt");
    std::ofstream outfile(file_path);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << file_path << std::endl;
        return;
    }

    // Write headers
    outfile << "Iteration\tSelected Heuristic\tCost Efficiency\tCost Diversity\t"
            << "Time Taken (seconds)\tReward\tQ-Value\n";

    // Usage stats
    std::map<int, double> heuristic_total_time;
    std::map<int, int>    heuristic_improvement_count;
    std::map<int, double> heuristic_rewards;  // total reward for each LLH

    // -------------------------------
    // 4) TIME CONTROL & MAIN LOOP
    // -------------------------------
    auto total_start_time = std::chrono::steady_clock::now();
    int iteration = 0;
    int itercount = 0;
    int maxiter   = 1000; // stop if iteration hits 1000

    // NEW: state carried across iterations from Δ
    double last_delta = 0.0;
    int    s          = state_from_delta(last_delta);

    while (true) {
        iteration++;
        auto iteration_start_time = std::chrono::steady_clock::now();

        // 4a) Check time or iteration limit
        double elapsed_time_total = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - total_start_time
        ).count();
        if ((elapsed_time_total >= max_time) || (itercount >= maxiter)) {
            break;
        }

        // 4b) DETERMINE CURRENT STATE KEY & ENSURE Q-TABLE
        std::string current_state_key = "s" + std::to_string(s);
        if (Q_table.find(current_state_key) == Q_table.end()) {
            for (int h : heuristics)
                Q_table[current_state_key][h] = 0.0;
        }

        // 4c) EPSILON-GREEDY CHOICE (random tie-break on argmax)
        int selected_heuristic = -1;
        double random_val = (double)std::rand() / RAND_MAX;
        if (random_val < epsilon) {
            // Explore: pick random LLH
            selected_heuristic = heuristics[ std::rand() % heuristics.size() ];
        } else {
            // Exploit: pick LLH with highest Q-value in current state (random tie-break)
            double max_Q = -std::numeric_limits<double>::infinity();
            std::vector<int> best_actions;
            for (int h : heuristics) {
                double q = Q_table[current_state_key][h];
                if (q > max_Q + 1e-12) {
                    max_Q = q;
                    best_actions.clear();
                    best_actions.push_back(h);
                } else if (std::fabs(q - max_Q) <= 1e-12) {
                    best_actions.push_back(h);
                }
            }
            selected_heuristic = best_actions[ std::rand() % best_actions.size() ];
        }

        // 4d) APPLY SELECTED HEURISTIC
        auto start_time = std::chrono::steady_clock::now();
        ApplyHeuristic(selected_heuristic, team);

        double elapsed_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time
        ).count();
        total_elapsed_time += elapsed_time;

        // Track usage
        heuristic_total_time[selected_heuristic] += elapsed_time;
        heuristic_usage_count[selected_heuristic]++;

        // Evaluate new solution
        objective_Function1(team);
        int newcost_eff = static_cast<int>(f_cur);
        int newcost_div = static_cast<int>(f_cur_div);

        // Record objective function values
        objective_values.push_back(newcost_eff);
        diversity_values.push_back(newcost_div);

        // -----------------------------
        // 4e) CREDIT ASSIGNMENT (5.1–5.4) + SYMMETRIC REWARD
        // -----------------------------
        // Keep previous normalized change for credits (optional)
        double f_c   = static_cast<double>(cost_eff);
        double f_n   = static_cast<double>(newcost_eff);
        double denom = (f_c + f_n + 1e-9);
        double frac  = (f_c - f_n) / denom;          // previous normalized change
        bool   improved_eff = (f_n > f_c);

        if (improved_eff) {
            heuristic_credits[selected_heuristic] += frac;            // 5.1
            if (heuristics.size() > 1) {
                double penalty = frac / (heuristics.size() - 1);      // 5.2
                for (int h : heuristics) if (h != selected_heuristic) heuristic_credits[h] -= penalty;
            }
        } else {
            heuristic_credits[selected_heuristic] -= frac;            // 5.3
            if (heuristics.size() > 1) {
                double bonus = frac / (heuristics.size() - 1);        // 5.4
                for (int h : heuristics) if (h != selected_heuristic) heuristic_credits[h] += bonus;
            }
        }

        // --- Logarithmic improvement Δ and reward r (symmetric)
        double E       = pos_or_eps(static_cast<double>(cost_eff));
        double Ep      = pos_or_eps(static_cast<double>(newcost_eff));
        double delta   = std::log10(Ep) - std::log10(E);              // Δ = log10(E') − log10(E)
        double r       = reward_from_delta(delta);

        if (improved_eff) heuristic_improvement_count[selected_heuristic]++;
        heuristic_rewards[selected_heuristic] += r;

        // 4f) DETERMINE NEXT STATE & UPDATE Q
        int s_next = state_from_delta(delta);
        std::string next_state_key = "s" + std::to_string(s_next);
        if (Q_table.find(next_state_key) == Q_table.end()) {
            for (int h : heuristics) Q_table[next_state_key][h] = 0.0;
        }

        double max_Q_next = -std::numeric_limits<double>::infinity();
        for (int h : heuristics) {
            if (Q_table[next_state_key][h] > max_Q_next) max_Q_next = Q_table[next_state_key][h];
        }

        double& old_Q = Q_table[current_state_key][selected_heuristic];
        old_Q += alpha * (r + gamma * max_Q_next - old_Q);

        // 4g) ACCEPT/REJECT new solution (CMCEE constraint on diversity)
        if ((newcost_eff > cost_eff) && (newcost_div >= min_div)) {
            cost_eff = newcost_eff;
            cost_div = newcost_div;
            previous_solution = team;
            Selected.push_back(selected_heuristic);

            // Update best if further improvement
            if (newcost_eff > best_cost_eff) {
                best_cost_eff = newcost_eff;
                best_cost_div = newcost_div;
                best_eff       = newcost_eff;
                best_div       = newcost_div;
                best_solution1 = team;
                time_taken     = elapsed_time;

                for (int m = 0 ; m < num_node ; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1 ; m <= num_team ; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        } else {
            // Revert
            team = previous_solution;
        }

        // 4h) ITERATION LOGGING
        double iteration_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - iteration_start_time
        ).count();
        iteration_times.push_back(iteration_time);

        std::cout << "Iteration: " << iteration
                  << " | Selected Heuristic: " << selected_heuristic
                  << " | Efficiency: " << newcost_eff
                  << " | Diversity: " << newcost_div
                  << " | Best Efficiency: " << best_cost_eff
                  << " | Best Diversity: " << best_cost_div
                  << " | Time Taken: " << elapsed_time << " s"
                  << " | QL Reward: " << r
                  << std::endl;

        outfile << iteration << "\t"
                << selected_heuristic << "\t"
                << newcost_eff << "\t"
                << newcost_div << "\t"
                << elapsed_time << "\t"
                << r << "\t"
                << Q_table[current_state_key][selected_heuristic] << "\n";

        // Decay epsilon (with floor) and advance state
        epsilon = std::max(epsilon * epsilon_decay, MIN_EPS);
        s = s_next;
        last_delta = delta;
        itercount++;
    } // end while

    outfile.close(); // close results file

        // -------------------------------
    // 5) STATISTICAL ANALYSIS
    // -------------------------------
    if (!objective_values.empty()) {
        double sum_obj = std::accumulate(objective_values.begin(), objective_values.end(), 0.0);
        average_objective = sum_obj / objective_values.size();
    }
    if (!diversity_values.empty()) {
        double sum_div = std::accumulate(diversity_values.begin(), diversity_values.end(), 0.0);
        average_diversity = sum_div / diversity_values.size();
    }
    if (!objective_values.empty()) {
        worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    }
    if (!iteration_times.empty()) {
        double total_iteration_time = std::accumulate(
            iteration_times.begin(), iteration_times.end(), 0.0
        );
        average_cpu_time = total_iteration_time / iteration_times.size();
    }

    // -------------------------------
    // 5.5) FILTER LOW-PERFORMING HEURISTICS AND LOG
    // -------------------------------

        std::filesystem::path filter_log_path =
            folder_path / ("Q_Learning_Filtered_Heuristics_" + instanceName + ".txt");
        std::ofstream filter_log(filter_log_path);
        if (!filter_log) {
            std::cerr << "Error opening heuristic filter log file: " << filter_log_path << std::endl;
        } else {
            filter_log << "Heuristic\tAvgReward\tImprovements\tUsage\tStatus\n";
        }

        std::vector<int> filtered_heuristics;
        double reward_threshold = -5.0;   // filter criterion

        for (int h : heuristics) {
            double avg_reward = 0.0;
            if (heuristic_usage_count[h] > 0)
                avg_reward = heuristic_rewards[h] / static_cast<double>(heuristic_usage_count[h]);

            std::string status;
            if (avg_reward >= reward_threshold && heuristic_improvement_count[h] > 0) {
                filtered_heuristics.push_back(h);
                status = "Included";
            } else {
                status = "Excluded";
                std::cout << "❌ Excluding LLH " << h
                          << " (avg reward=" << avg_reward
                          << ", improvements=" << heuristic_improvement_count[h] << ")\n";
            }

            if (filter_log)
                filter_log << h << "\t" << std::fixed << std::setprecision(3)
                           << avg_reward << "\t" << heuristic_improvement_count[h]
                           << "\t" << heuristic_usage_count[h]
                           << "\t" << status << "\n";
        }

        if (filter_log) {
            filter_log.close();
            std::cout << "Heuristic filtering log saved to: " << filter_log_path << std::endl;
        }

        if (!filtered_heuristics.empty()) {
            std::cout << "✔ Retained " << filtered_heuristics.size()
                      << " heuristics after filtering.\n";
        } else {
            std::cout << "⚠ All heuristics excluded — restoring original set.\n";
        }


    // Identify best & worst heuristics based on total Q-Learning reward
    int best_heuristic = -1;
    int worst_heuristic = -1;
    double max_reward = -std::numeric_limits<double>::infinity();
    double min_reward =  std::numeric_limits<double>::infinity();
    for (int h : heuristics) {
        if (heuristic_rewards[h] > max_reward) { max_reward = heuristic_rewards[h]; best_heuristic = h; }
        if (heuristic_rewards[h] < min_reward) { min_reward = heuristic_rewards[h]; worst_heuristic = h; }
    }

    // Save summary to file
    std::filesystem::path summary_file_path =
        folder_path / ("Q_Learning_Selection_Hyperheuristic_CMCEE_" + instanceName + "_summary.txt");
    {
        std::ofstream summary_file(summary_file_path);
        if (summary_file.is_open()) {
            summary_file << "Final Best Objectives:\n";
            summary_file << "Efficiency: " << best_cost_eff << "\n";
            summary_file << "Diversity: " << best_cost_div << "\n\n";

            summary_file << "Statistical Analysis:\n";
            summary_file << "Total Time Taken: " << std::fixed << std::setprecision(4)
                         << total_elapsed_time << " seconds\n";
            summary_file << "Average Objective Function Value: " << average_objective << "\n";
            summary_file << "Average Diversity Value: " << average_diversity << "\n";
            summary_file << "Worst Objective Function Value: " << worst_objective << "\n";
            summary_file << "Average CPU Time per Iteration: " << average_cpu_time << " seconds\n\n";

            summary_file << "Heuristic Performance:\n";
            for (int h : heuristics) {
                summary_file << "Heuristic " << h << ":\n";
                summary_file << "  Usage Count: " << heuristic_usage_count[h] << "\n";
                summary_file << "  Total Q-Learning Reward: " << heuristic_rewards[h] << "\n";
                summary_file << "  Improvement Count: " << heuristic_improvement_count[h] << "\n";
                summary_file << "  Total Time: " << heuristic_total_time[h] << " seconds\n\n";
            }

            summary_file << "Best Heuristic: " << best_heuristic
                         << " with Total Reward: " << max_reward << "\n";
            summary_file << "Worst Heuristic: " << worst_heuristic
                         << " with Total Reward: " << min_reward << "\n";

            summary_file.close();
            std::cout << "Summary results saved to " << summary_file_path << std::endl;
        } else {
            std::cerr << "Error: Could not open the summary file for writing results.\n";
        }
    }

    // -------------------------------
    // 6) EXTRACT topK heuristics
    // -------------------------------
    best_heuristics.clear();  // a class-level vector
    {
        struct HeurInfo { int id; double avg; };
        std::vector<HeurInfo> rankVec;
        rankVec.reserve(heuristics.size());

        for (int h : heuristics) {
            double avg = 0.0;
            if (heuristic_usage_count[h] > 0) {
                avg = heuristic_rewards[h] / static_cast<double>(heuristic_usage_count[h]);
            }
            rankVec.push_back({h, avg});
        }
        std::sort(rankVec.begin(), rankVec.end(),
                  [](auto &a, auto &b) { return a.avg > b.avg; });

        int limit = std::min<int>(topK, static_cast<int>(rankVec.size()));
        for (int i = 0; i < limit; ++i) best_heuristics.push_back(rankVec[i].id);
    }

    // -------------------------------
    // 7) FINAL OUTPUT
    // -------------------------------
    std::cout << "Final Best Objectives:\n"
              << "Efficiency: " << best_cost_eff << "\n"
              << "Diversity: " << best_cost_div << "\n\n"
              << "Statistical Analysis:\n"
              << "Total Time Taken: " << std::fixed << std::setprecision(4)
              << total_elapsed_time << " seconds\n"
              << "Average Objective Function Value: " << average_objective << "\n"
              << "Average Diversity Value: " << average_diversity << "\n"
              << "Worst Objective Function Value: " << worst_objective << "\n"
              << "Average CPU Time per Iteration: " << average_cpu_time << " seconds\n\n"
              << "Total time taken to find the solution: "
              << std::fixed << std::setprecision(4) << total_elapsed_time << " seconds\n";

    std::cout << "Selected heuristic sequence: ";
    for (int h : Selected) std::cout << h << " ";
    std::cout << "\n";

    check_best_solution(); // final check
    std::cout <<"=============================================================================\n";
    std::cout <<"Q-Learning Selection Hyper-heuristic Framework Finished its Processes.\n";
    std::cout <<"=============================================================================\n";
}
*//*
void Hyper_heuristic::Q_Learning_Selection_Hyperheuristic_CMCEE(int max_time) {
    // Q-Learning parameters
    double alpha = 0.1;   // Learning rate
    double gamma = 0.9;   // Discount factor
    double epsilon = 0.1; // Exploration rate

    // Define heuristics
    std::vector<int> heuristics = {12, 17, 18, 19, 20}; // List of heuristics

    // Q-table: Map from state representation (string) to a map of (heuristic -> Q-value)
    std::map<std::string, std::map<int, double>> Q_table;

    // Initialize solution
    generate_initialrandom();
    objective_Function(team);
    double cost_eff = static_cast<double>(f_cur);
    double cost_div = static_cast<double>(f_cur_div);

    double best_cost_eff = cost_eff;
    double best_cost_div = cost_div;
    int **previous_solution = team;
    int **best_solution1 = team;

    std::cout << "Initial objectives: Efficiency = " << cost_eff << ", Diversity = " << cost_div << std::endl;

    // Data structures for logging
    std::vector<int> Selected;
    std::vector<double> iteration_times;

    // Create directory and output file
    std::filesystem::path folder_path = "D:/Result Hyper-heuristic Models/";
    if (!std::filesystem::exists(folder_path)) {
        try {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return;
        }
    }

    std::filesystem::path file_path = folder_path / "Q_Learning_Selection_Hyperheuristic_CMCEE_results.txt";
    std::ofstream outfile(file_path);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << file_path << std::endl;
        return;
    }

    outfile << "Iteration\tSelected Heuristic\tEfficiency\tDiversity\tTime Taken (seconds)\tReward\tQ-Value\n";

    // Random generators
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_real(0.0, 1.0);
    std::uniform_int_distribution<> dis_int(0, static_cast<int>(heuristics.size()) - 1);

    // Tracking performance
    std::map<int, double> heuristic_total_time;
    std::map<int, int> heuristic_improvement_count;
    std::map<int, double> heuristic_rewards;
    std::map<int, int> heuristic_usage_count;

    auto total_start_time = std::chrono::steady_clock::now();
    int iteration = 0;
    double total_elapsed_time = 0.0;

    // Helper lambda to encode state using the given features
    auto get_state_key = [&](double cost_eff, double cost_div, double best_cost_eff, double best_cost_div, int iteration, bool changed, bool unseen, bool last_action_sign, int last_action, double temp, double cs, int no_improvement) {
        // Compute features as described:
        double reduced_cost = 0.0; // difference between previous and current solutions (assume previous known)
        double cost_from_min = cost_eff - best_cost_eff;
        double cost_current = cost_eff;
        double min_cost = best_cost_eff;
        double temperature = temp;    // Assume you have a method or variable
        double cooling_schedule = cs; // Assume you have a method or variable
        int no_improve = no_improvement;
        int index_step = iteration;
        int was_changed = changed ? 1 : 0;
        int is_unseen = unseen ? 1 : 0;
        int last_action_s = last_action_sign ? 1 : 0;

        // Encode last_action in one-hot as a string (for simplicity, just include the action ID)
        // Real one-hot encoding might store multiple bits, but here we just store action ID.
        std::ostringstream oss;
        oss << reduced_cost << "," << cost_from_min << "," << cost_current << "," << min_cost << ","
            << temperature << "," << cooling_schedule << "," << no_improve << "," << index_step << ","
            << was_changed << "," << is_unseen << "," << last_action_s << "," << last_action;
        return oss.str();
    };

    // Acceptance criterion
    auto accept = [&](double new_eff, double new_div, double old_eff, double old_div) {
        // For simplicity, accept if it improves efficiency and does not reduce diversity
        return (new_eff > old_eff && new_div >= old_div);
    };

    // Track some auxiliary variables for state representation
    double temp = 1.0;      // Example temperature
    double cs = 0.99;       // Cooling schedule (just a placeholder)
    int no_improvement = 0; // Count how many iterations since last improvement
    bool last_action_sign = false;
    int last_action_id = -1;
    bool changed = false;
    bool unseen = true;  // Assume first solution is unseen

    // main loop
    while (true) {
        iteration++;
        auto iteration_start_time = std::chrono::steady_clock::now();

        double elapsed_time_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
        if (elapsed_time_total >= max_time) {
            break;
        }

        // Get current state
        std::string current_state_key = get_state_key(cost_eff, cost_div, best_cost_eff, best_cost_div, iteration, changed, unseen, last_action_sign, last_action_id, temp, cs, no_improvement);

        // Ensure Q-table entry
        if (Q_table.find(current_state_key) == Q_table.end()) {
            for (int h : heuristics) {
                Q_table[current_state_key][h] = 0.0;
            }
        }

        // Epsilon-greedy action selection
        double rand_val = dis_real(gen);
        int selected_heuristic = -1;
        if (rand_val < epsilon) {
            // Exploration
            selected_heuristic = heuristics[dis_int(gen)];
        } else {
            // Exploitation
            double max_Q = -std::numeric_limits<double>::infinity();
            for (int h : heuristics) {
                double qv = Q_table[current_state_key][h];
                if (qv > max_Q) {
                    max_Q = qv;
                    selected_heuristic = h;
                }
            }
        }

        std::cout << "Iteration: " << iteration << ", Selected Heuristic: " << selected_heuristic
                  << ", cost_eff: " << cost_eff << ", cost_div: " << cost_div << std::endl;

        auto start_time = std::chrono::steady_clock::now();

        // Apply heuristic
        ApplyHeuristic(selected_heuristic, team);

        double elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        total_elapsed_time += elapsed_time;

        heuristic_total_time[selected_heuristic] += elapsed_time;
        heuristic_usage_count[selected_heuristic]++;

        // Evaluate new solution
        objective_Function(team);
        double newcost_eff = static_cast<double>(f_cur);
        double newcost_div = static_cast<double>(f_cur_div);

        std::cout << "Iteration: " << iteration << ", After Heuristic: "
                  << selected_heuristic << ", new_eff: " << newcost_eff << ", new_div: " << newcost_div << "\n";

        // Determine reward using R_5310
        // R_5310:
        // 5 if f(x') > f(x_best)
        // 3 if f(x') > f(x)
        // 1 if accept(x', x)
        // 0 otherwise
        double reward = 0.0;
        bool accepted = false;

        if (newcost_eff > best_cost_eff) {
            reward = 5.0;
        } else if (newcost_eff > cost_eff) {
            reward = 3.0;
        } else if (accept(newcost_eff, newcost_div, cost_eff, cost_div)) {
            reward = 1.0;
        } else {
            reward = 0.0;
        }

        // Accept or reject the new solution based on accept function
        if (accept(newcost_eff, newcost_div, cost_eff, cost_div)) {
            cost_eff = newcost_eff;
            cost_div = newcost_div;
            previous_solution = team;
            accepted = true;
            Selected.push_back(selected_heuristic);

            if (newcost_eff > best_cost_eff) {
                best_cost_eff = newcost_eff;
                best_eff = newcost_eff;
                best_div =  newcost_div;
                best_cost_div = newcost_div;
                best_solution1 = team;
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
                no_improvement = 0; // Found a better solution
                last_action_sign = true;
            } else {
                // Improved over current (but not best) or accepted as is.
                if (newcost_eff > cost_eff) {
                    no_improvement = 0;
                    last_action_sign = true;
                } else {
                    // Accepted but not improved
                    no_improvement++;
                    last_action_sign = false;
                }
            }
            changed = true;
        } else {
            // Revert to previous solution
            team = previous_solution;
            no_improvement++;
            last_action_sign = false;
            changed = false;
        }

        heuristic_rewards[selected_heuristic] += reward;
        if (reward >= 3) {
            heuristic_improvement_count[selected_heuristic]++;
        }

        // Next state
        int next_iter = iteration + 1;
        bool next_unseen = true; // This would be determined by checking if solution was visited before
        std::string next_state_key = get_state_key(cost_eff, cost_div, best_cost_eff, best_cost_div, next_iter, changed, next_unseen, last_action_sign, selected_heuristic, temp, cs, no_improvement);

        if (Q_table.find(next_state_key) == Q_table.end()) {
            for (int h : heuristics) {
                Q_table[next_state_key][h] = 0.0;
            }
        }

        // Max Q-value of next state
        double max_Q_next = -std::numeric_limits<double>::infinity();
        for (int h : heuristics) {
            if (Q_table[next_state_key][h] > max_Q_next) {
                max_Q_next = Q_table[next_state_key][h];
            }
        }

        // Update Q-value
        double old_Q = Q_table[current_state_key][selected_heuristic];
        Q_table[current_state_key][selected_heuristic] = old_Q + alpha * (reward + gamma * max_Q_next - old_Q);

        // Update last_action
        last_action_id = selected_heuristic;

        double iteration_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - iteration_start_time).count();
        iteration_times.push_back(iteration_time);

        outfile << iteration << "\t"
                << selected_heuristic << "\t"
                << cost_eff << "\t"
                << cost_div << "\t"
                << elapsed_time << "\t"
                << reward << "\t"
                << Q_table[current_state_key][selected_heuristic] << "\n";
    }

    outfile.close();

    // Statistical analysis
    double total_iteration_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
    double average_iteration_time = iteration_times.empty() ? 0.0 : (total_iteration_time / iteration_times.size());

    // Find best/worst heuristic based on total reward
    int best_heuristic = -1;
    int worst_heuristic = -1;
    double max_reward = -std::numeric_limits<double>::infinity();
    double min_reward = std::numeric_limits<double>::infinity();
    for (int h : heuristics) {
        if (heuristic_rewards[h] > max_reward) {
            max_reward = heuristic_rewards[h];
            best_heuristic = h;
        }
        if (heuristic_rewards[h] < min_reward) {
            min_reward = heuristic_rewards[h];
            worst_heuristic = h;
        }
    }

    // Save summary
    std::filesystem::path summary_file_path = folder_path / "Q_Learning_Selection_Hyperheuristic_CMCEE_summary.txt";
    std::ofstream summary_file(summary_file_path);
    if (summary_file.is_open()) {
        summary_file << "Final Best Objectives:\n";
        summary_file << "Efficiency: " << best_cost_eff << "\n";
        summary_file << "Diversity: " << best_cost_div << "\n\n";

        summary_file << "Total Time Taken: " << std::fixed << std::setprecision(4)
                     << total_elapsed_time << " seconds\n";
        summary_file << "Average CPU Time per Iteration: " << average_iteration_time << " seconds\n\n";

        summary_file << "Heuristic Performance:\n";
        for (int h : heuristics) {
            summary_file << "Heuristic " << h << ":\n";
            summary_file << "  Usage Count: " << heuristic_usage_count[h] << "\n";
            summary_file << "  Total Reward: " << heuristic_rewards[h] << "\n";
            summary_file << "  Improvement Count: " << heuristic_improvement_count[h] << "\n";
            summary_file << "  Total Time: " << heuristic_total_time[h] << " seconds\n\n";
        }

        summary_file << "Best Heuristic: " << best_heuristic << " with Total Reward: " << max_reward << "\n";
        summary_file << "Worst Heuristic: " << worst_heuristic << " with Total Reward: " << min_reward << "\n";

        summary_file.close();
        std::cout << "Summary results saved to " << summary_file_path << std::endl;
    } else {
        std::cerr << "Error: Could not open the summary file for writing results.\n";
    }

    // Print final results
    std::cout << "Total time taken: " << total_elapsed_time << " seconds" << std::endl;
    std::cout << "Selected heuristic sequence: ";
    for (const auto& h : Selected) std::cout << h << " ";
    std::cout << std::endl;

    check_best_solution();
    free_memory();
}
*/
/*// Q-Learning Based Selection Function Implementation with Feature Calculations and TXT Saving
void Hyper_heuristic::Q_Learning_Selection_Hyperheuristic_CMCEE(int max_time) {
    // Q-Learning parameters
    double alpha = 0.1;   // Learning rate
    double gamma = 0.9;   // Discount factor
    double epsilon = 0.1; // Exploration rate

    // Define heuristics
    std::vector<int> heuristics = {12, 17, 18, 19, 20}; // List of heuristics

    // Initialize Q-table: Map<State Key, Map<Heuristic, Q-value>>
    std::map<std::string, std::map<int, double>> Q_table;

    // Define state boundaries (Adjust based on your problem's requirements)
    const int LOW_EFF = 0;
    const int HIGH_EFF = 1000; // Example value
    const int LOW_DIV = 0;
    const int HIGH_DIV = 200;   // Example value

    // Initialize Q-values for all possible states and actions to 0
    // For simplicity, assuming a finite and manageable number of states
    // Adjust the ranges as per your problem's requirements

    int MAX_BIN = 10; // Define based on expected max iterations (e.g., 1000 iterations => 10 bins of 100)

    for (int eff = LOW_EFF; eff <= HIGH_EFF; eff += 10) { // Increment steps as needed
        for (int div = LOW_DIV; div <= HIGH_DIV; div += 5) { // Increment steps as needed
            for (int bin = 1; bin <= MAX_BIN; ++bin) {
                State state = { static_cast<EfficiencyLevel>(eff),
                                static_cast<DiversityLevel>(div),
                                bin };
                std::string state_key = encode_state(state);
                for (int h : heuristics) {
                    Q_table[state_key][h] = 0.0;
                }
            }
        }
    }

    // Initialize solution
    generate_initialrandom();
    //feasible_local_search();
    objective_Function(team);
    double cost_eff = static_cast<double>(f_cur);
    double cost_div = static_cast<double>(f_cur_div);
    int** previous_solution = team;
    int ** best_solution1 = team;
    std::cout << "Initial objectives eff and div: " << cost_eff << ", " << cost_div << std::endl;

    double best_cost_eff = cost_eff, best_cost_div = cost_div;

    // Data structures for analysis
    std::vector<int> Selected;
    std::vector<int> heuristic_sequence; // To track the sequence of selected heuristics
    std::vector<int> objective_values;    // Stores efficiency values over iterations
    std::vector<int> diversity_values;    // Stores diversity values over iterations
    std::vector<double> iteration_times;  // Time taken for each iteration

    double total_elapsed_time = 0.0;

    // Define the folder path using std::filesystem
    std::filesystem::path folder_path = "D:/Result Hyper-heuristic Models/";

    // Check if the directory exists, create it if it doesn't
    if (!std::filesystem::exists(folder_path)) {
        try {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        }
        catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return; // Exit the function if directory creation fails
        }
    }

    // Define the instance file name
    std::string instance_file = "D:/Datasets/01test-n100m10t8.dat";
    // Extract the base name without extension
    std::string base_name = std::filesystem::path(instance_file).stem().string();

    // Define the full file path for saving the main results
    std::filesystem::path main_results_path = folder_path / (base_name + "_Q_Learning_Selection_Hyperheuristic_CMCEE_results.txt");

    // Create an output file stream to save the main results to a TXT file
    std::ofstream main_results_file(main_results_path);
    if (!main_results_file) {
        std::cerr << "Error opening main results TXT file for writing: " << main_results_path << std::endl;
        return;
    }

    // Write headers to the main results TXT file
    main_results_file << "Iteration\tSelected Heuristic\tCost Efficiency\tCost Diversity\tTime Taken (seconds)\tReward\tQ-Value\n";

    // Define the full file path for saving the features
    std::filesystem::path features_path = folder_path / (base_name + "_Features.txt");

    // Create an output file stream to save the features to a TXT file
    std::ofstream features_file(features_path);
    if (!features_file) {
        std::cerr << "Error opening features TXT file for writing: " << features_path << std::endl;
        return;
    }

    // Write feature headers to the features TXT file
    std::vector<std::string> feature_headers = {
        "F1", "F2_nb_ge_1", "F3_iterations", "F4_total_nb", "F5_total_imp",
        "F6_total_wrs", "F7_total_eq", "F8_total_ac", "F9_total_uq",
        "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18",
        "F19", "F20", "F21", "F22", "F23", "F24", "F25_max_th", "F26_min_th",
        "F27_avg_th", "F28_variance_th", "F29", "F30", "F31", "F32",
        "F33", "F34", "F35", "F36", "F37", "F38", "F39"
    };

    // Write feature headers separated by tabs
    for (size_t i = 0; i < feature_headers.size(); ++i) {
        features_file << feature_headers[i];
        if (i != feature_headers.size() - 1) features_file << "\t";
    }
    features_file << "\n";

    // Initialize random number generators
    std::random_device rd;
    std::mt19937 gen_rand(rd());
    std::uniform_real_distribution<> dis_real_dist(0.0, 1.0);
    std::uniform_int_distribution<> dis_int_dist(0, heuristics.size() - 1);

    // Data structures for statistical analysis
    std::map<int, double> heuristic_total_time;       // Total time taken by each heuristic
    std::map<int, int> heuristic_improvement_count;   // Number of times each heuristic improved the solution
    std::map<int, double> heuristic_rewards;          // Total rewards accumulated by each heuristic
    std::map<int, int> heuristic_usage_count;         // Number of times each heuristic was used

    // Feature-related variables
    std::set<int> S_imp;  // Improving solutions
    std::set<int> S_wrs;  // Worsening solutions
    std::set<int> S_eq;   // Equal quality solutions
    std::set<int> S_ac;   // Accepted solutions
    std::set<int> S_uq;   // Unique solutions
    std::set<int> S_nb;   // New best solutions

    // Define initial states for Sfirst, Sbest, Sworst
    double Sfirst = cost_eff; // Assuming cost_eff represents fitness
    double Sbest = cost_eff;
    double Sworst = cost_eff;

    // Start timing
    auto total_start_time = std::chrono::steady_clock::now();

    // Iterate until max_time is reached
    int iteration = 0; // Track iteration count
    while (true) {
        iteration++;
        auto iteration_start_time = std::chrono::steady_clock::now();

        // Check if max_time is exceeded
        double elapsed_time_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
        if (elapsed_time_total >= max_time) {
            break;
        }

        // Determine current state
        State current_state = determine_current_state(cost_eff, cost_div, iteration);
        std::string current_state_key = encode_state(current_state);

        // Ensure the current state exists in the Q-table
        if (Q_table.find(current_state_key) == Q_table.end()) {
            // Initialize Q-values for unseen state
            for (int h : heuristics) {
                Q_table[current_state_key][h] = 0.0;
            }
        }

        // Decide whether to explore or exploit
        double rand_val = dis_real_dist(gen_rand);
        int selected_heuristic = -1;
        if (rand_val < epsilon) {
            // Exploration: select a random heuristic
            selected_heuristic = heuristics[dis_int_dist(gen_rand)];
        }
        else {
            // Exploitation: select the heuristic with the highest Q-value for the current state
            double max_Q = -std::numeric_limits<double>::infinity();
            for (int h : heuristics) {
                if (Q_table[current_state_key][h] > max_Q) {
                    max_Q = Q_table[current_state_key][h];
                    selected_heuristic = h;
                }
            }
        }

        std::cout << "Iteration: " << iteration << ", Selected Heuristic LLH[" << selected_heuristic
                  << "], cost_eff: " << cost_eff << ", cost_div: " << cost_div << std::endl;

        // Start timing for heuristic application
        auto start_time = std::chrono::steady_clock::now();

        // Apply heuristic
        ApplyHeuristic(selected_heuristic, team);

        // Calculate elapsed time
        double elapsed_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        total_elapsed_time += elapsed_time;

        // Update heuristic time tracking
        heuristic_total_time[selected_heuristic] += elapsed_time;
        heuristic_usage_count[selected_heuristic]++;

        // Calculate new costs
        objective_Function(team);
        double newcost_eff = static_cast<double>(f_cur);
        double newcost_div = static_cast<double>(f_cur_div);
        std::cout << "Iteration: " << iteration << ", After Selected Heuristic LLH[" << selected_heuristic
                  << "], newcost_eff: " << newcost_eff << ", newcost_div: " << newcost_div << "\n";

        // Calculate reward based on improvement
        double reward = 0.0;
        if ((newcost_eff > cost_eff) || (newcost_div >= cost_div)) {
            reward = (newcost_eff - cost_eff) + (newcost_div - cost_div);
            heuristic_improvement_count[selected_heuristic]++;
            // Update solution sets
            S_imp.insert(selected_heuristic); // Example: Add heuristic ID to improving set
            // Similarly, update S_wrs, S_eq, S_ac, S_uq, S_nb based on actual logic
        }
        heuristic_rewards[selected_heuristic] += reward;

        // Define next state based on some logic (Replace with actual state transition)
        State next_state = determine_current_state(newcost_eff, newcost_div, iteration + 1);
        std::string next_state_key = encode_state(next_state);

        // Ensure the next state exists in the Q-table
        if (Q_table.find(next_state_key) == Q_table.end()) {
            for (int h : heuristics) {
                Q_table[next_state_key][h] = 0.0;
            }
        }

        // Find the maximum Q-value for the next state
        double max_Q_next = -std::numeric_limits<double>::infinity();
        for (int h : heuristics) {
            if (Q_table[next_state_key][h] > max_Q_next) {
                max_Q_next = Q_table[next_state_key][h];
            }
        }

        // Update Q-value for the current state and selected heuristic
        Q_table[current_state_key][selected_heuristic] += alpha * (reward + gamma * max_Q_next - Q_table[current_state_key][selected_heuristic]);

        // Record objective function values
        objective_values.push_back(static_cast<int>(newcost_eff));
        diversity_values.push_back(static_cast<int>(newcost_div));

        // Accept or reject the new solution
        // Accept or reject the new solution
        if ((newcost_eff > cost_eff) && (newcost_div >= cost_div)) {
            // Accept the new solution
            cost_eff = newcost_eff;
            cost_div = newcost_div;
            previous_solution = team;
            // Optionally, deep copy team to previous_solution if needed
            Selected.push_back(selected_heuristic);
            if (newcost_eff > best_cost_eff) {
                // Update best solution
                best_cost_eff = newcost_eff;
                best_cost_div = newcost_div;
                best_solution1 = team;
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

        // Feature Calculations
        Features features = calculate_features(
            current_state,
            S_imp,
            S_wrs,
            S_eq,
            S_ac,
            S_uq,
            S_nb,
            Sfirst,
            Sbest,
            Sworst,
            heuristic_total_time,
            heuristics.size(),
            iteration
        );

        // Save iteration data to the main results TXT file
        main_results_file << iteration << "\t"  // Iteration
                         << selected_heuristic << "\t"  // Selected Heuristic
                         << newcost_eff << "\t"  // Cost Efficiency
                         << newcost_div << "\t"  // Cost Diversity
                         << elapsed_time << "\t"  // Time Taken
                         << reward << "\t"         // Reward
                         << Q_table[current_state_key][selected_heuristic] << "\n"; // Q-Value

        // Save features to the features TXT file
        // Convert all features to strings separated by tabs
        features_file << features.F1 << "\t"
                      << features.F2_nb_ge_1 << "\t"
                      << features.F3_iterations << "\t"
                      << features.F4_total_nb << "\t"
                      << features.F5_total_imp << "\t"
                      << features.F6_total_wrs << "\t"
                      << features.F7_total_eq << "\t"
                      << features.F8_total_ac << "\t"
                      << features.F9_total_uq << "\t"
                      << features.F10 << "\t"
                      << features.F11 << "\t"
                      << features.F12 << "\t"
                      << features.F13 << "\t"
                      << features.F14 << "\t"
                      << features.F15 << "\t"
                      << features.F16 << "\t"
                      << features.F17 << "\t"
                      << features.F18 << "\t"
                      << features.F19 << "\t"
                      << features.F20 << "\t"
                      << features.F21 << "\t"
                      << features.F22 << "\t"
                      << features.F23 << "\t"
                      << features.F24 << "\t"
                      << features.F25_max_th << "\t"
                      << features.F26_min_th << "\t"
                      << features.F27_avg_th << "\t"
                      << features.F28_variance_th << "\t"
                      << features.F29 << "\t"
                      << features.F30 << "\t"
                      << features.F31 << "\t"
                      << features.F32 << "\t"
                      << features.F33 << "\t"
                      << features.F34 << "\t"
                      << features.F35 << "\t"
                      << features.F36 << "\t"
                      << features.F37 << "\t"
                      << features.F38 << "\t"
                      << features.F39 << "\n";

        // Calculate iteration time
        double iteration_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - iteration_start_time).count();
        iteration_times.push_back(iteration_time);

        // Update Statistical Features
        // Update Sfirst, Sbest, Sworst as needed based on your solution logic
        if (newcost_eff > Sbest) {
            Sbest = newcost_eff;
        }
        if (newcost_eff < Sworst) {
            Sworst = newcost_eff;
        }
    }

    // Close the main results and features TXT files
    main_results_file.close();
    features_file.close();

    // Define the summary file path
    std::filesystem::path summary_path = folder_path / (base_name + "_Summary.txt");

    // Create an output file stream to save the summary to a TXT file
    std::ofstream summary_file(summary_path);
    if (!summary_file) {
        std::cerr << "Error opening summary TXT file for writing: " << summary_path << std::endl;
        return;
    }

    // Statistical Analysis
    // Calculate average objective function value
    double average_objective = 0.0;
    if (!objective_values.empty()) {
        double total_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0);
        average_objective = total_objective / objective_values.size();
    }

    // Calculate average diversity value
    double average_diversity = 0.0;
    if (!diversity_values.empty()) {
        double total_diversity = std::accumulate(diversity_values.begin(), diversity_values.end(), 0.0);
        average_diversity = total_diversity / diversity_values.size();
    }

    // Identify worst objective function value
    int worst_objective = std::numeric_limits<int>::max();
    if (!objective_values.empty()) {
        worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    }

    // Calculate average CPU time per iteration
    double average_cpu_time = 0.0;
    if (!iteration_times.empty()) {
        double total_iteration_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
        average_cpu_time = total_iteration_time / iteration_times.size();
    }

    // Identify best and worst heuristics based on rewards
    int best_heuristic = -1;
    int worst_heuristic = -1;
    double max_reward = -std::numeric_limits<double>::infinity();
    double min_reward = std::numeric_limits<double>::infinity();

    for (int h : heuristics) {
        if (heuristic_rewards[h] > max_reward) {
            max_reward = heuristic_rewards[h];
            best_heuristic = h;
        }
        if (heuristic_rewards[h] < min_reward) {
            min_reward = heuristic_rewards[h];
            worst_heuristic = h;
        }
    }

    // Write summary data to the summary TXT file
    summary_file << "Final Best Objectives:\n";
    summary_file << "Efficiency: " << best_cost_eff << "\n";
    summary_file << "Diversity: " << best_cost_div << "\n\n";

    summary_file << "Statistical Analysis:\n";
    summary_file << "Total Time Taken: " << std::fixed << std::setprecision(4)
                << total_elapsed_time << " seconds\n";
    summary_file << "Average Objective Function Value: " << average_objective << "\n";
    summary_file << "Average Diversity Value: " << average_diversity << "\n";
    summary_file << "Worst Objective Function Value: " << worst_objective << "\n";
    summary_file << "Average CPU Time per Iteration: " << average_cpu_time << " seconds\n\n";

    summary_file << "Heuristic Performance:\n";
    for (int h : heuristics) {
        summary_file << "Heuristic " << h << ":\n";
        summary_file << "  Usage Count: " << heuristic_usage_count[h] << "\n";
        summary_file << "  Total Reward: " << heuristic_rewards[h] << "\n";
        summary_file << "  Improvement Count: " << heuristic_improvement_count[h] << "\n";
        summary_file << "  Total Time: " << heuristic_total_time[h] << " seconds\n\n";
    }

    summary_file << "Best Heuristic: " << best_heuristic << " with Total Reward: " << max_reward << "\n";
    summary_file << "Worst Heuristic: " << worst_heuristic << " with Total Reward: " << min_reward << "\n";

    // Close the summary file
    summary_file.close();
    std::cout << "Summary results saved to " << summary_path << std::endl;

    // Additional Console Output
    std::cout << "Total time taken to find the solution: " << std::fixed << std::setprecision(4)
              << total_elapsed_time << " seconds" << std::endl;
    std::cout << "Selected heuristic sequence: ";
    for (const auto& h : Selected) std::cout << h << " ";
    std::cout << std::endl;

    // Final results
    check_best_solution();
    free_memory();

    // Done
}
*/

/**********************************************************************
 *  Q-Learning  +  Dynamic-Heuristic-Set-Selection (DHSS)
 *********************************************************************/
void Hyper_heuristic::Q_Learning_Selection_Hyperheuristic_CMCEE_DHSS(int max_time)
{
    std::cout << "=============================================================================\n";
    std::cout << "Q-Learning + DHSS Hyper-heuristic Framework Start its Processes.\n";
    std::cout << "=============================================================================\n";

    // -------------------------------------------------------------------------
    // 1) Q-LEARNING PARAMETERS
    // -------------------------------------------------------------------------
    const double alpha          = 0.1;   // learning rate
    const double gamma          = 0.9;   // discount factor
    double       epsilon        = 0.3;   // initial exploration rate
    const double epsilon_decay  = 0.99;  // decay factor
    const double MIN_EPS        = 0.05;  // ε floor
    const int    topK           = 5;     // # of top heuristics to keep later

    // Universal set of heuristics
    const std::vector<int> heuristics = {15, 16, 17, 18, 19, 20};

    // -------------------------------------------------------------------------
    // 1a)  DHSS PARAMETERS
    // -------------------------------------------------------------------------
    const double R_excl   = 0.30;   // Telp/Tmax threshold for one-shot removal
    const int    phaseMax = 200;    // hard upper-bound on phase length
    const double pf       = 1.20;   // patience factor (Nfail = pf × wait_max)

    // -------------------------------------------------------------------------
    // 2)  Q-TABLE & CREDIT ACCOUNTS
    // -------------------------------------------------------------------------
    std::map<std::string, std::map<int,double>> Q_table;   // state→{heur,Q}
    std::map<int,double> heuristic_credits;                // for formulas 5.1–5.4
    for (int h:heuristics) heuristic_credits[h] = 0.0;

    // Pre-initialise Q rows for a coarse grid of states
    const int MAX_BIN = 10;
    for (int e=LOW_EFF;e<=HIGH_EFF;++e)
        for (int d=LOW_DIV;d<=HIGH_DIV;++d)
            for (int b=1;b<=MAX_BIN;++b){
                State s{ static_cast<EfficiencyLevel>(e),
                         static_cast<DiversityLevel>(d), b };
                std::string k = encode_state(s);
                for (int h:heuristics) Q_table[k][h] = 0.0;
            }

    // -------------------------------------------------------------------------
    // 3) DHSS DATA STRUCTURES
    // -------------------------------------------------------------------------
    std::vector<int>          active_set  = heuristics;  // S
    std::unordered_set<int>   removed_set;               // S0
    bool removal_done = false;

    struct Perf {
        int    nPlus=0, nMinus=0;       // # improve / worsen
        double dPlus=0, dMinus=0;       // % improve / worsen
        double ms=0;                    // CPU-time (ms)
    };
    std::unordered_map<int,Perf> perf;
    for (int h:heuristics) perf[h] = {};

    int waitMax   = 0;  // longest streak without new best
    int sinceBest = 0;  // current streak
    int phaseLen  = 0;  // length of current phase

    // -------------------------------------------------------------------------
    // 4) INITIAL SOLUTION & BOOK-KEEPING
    // -------------------------------------------------------------------------
    generate_initialrandom();
    display(team);
    objective_Function1(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    int** previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** best_solution1    = deep_copy_solution(team, num_node, num_team, num_each_t);

    int best_cost_eff = cost_eff;
    int best_cost_div = cost_div;
    // Allocate team_size array globally
    team_size = new int[num_team + 1];

    // Assign team sizes
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;

    std::cout << "Initial objectives eff and div: "
              << cost_eff << ", " << cost_div << '\n';

    // Traces & statistics
    std::vector<int>    Selected;
    std::vector<int>    objective_values;
    std::vector<int>    diversity_values;
    std::vector<double> iteration_times;
    std::map<int,double> heuristic_total_time;
    std::map<int,int>    heuristic_improvement_count;
    std::map<int,int>    heuristic_usage_count;          // <-- added for completeness
    std::map<int,double> heuristic_rewards;
    double total_elapsed_time = 0.0;

    // -------------------------------------------------------------------------
    // 5) FILE & FOLDER SETUP
    // -------------------------------------------------------------------------
    std::filesystem::path folder_path =
        "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path file_path =
        folder_path / ("Q_Learning_Selection_Hyperheuristic_CMCEE_" + instanceName + "_results.txt");
    std::ofstream outfile(file_path);
    outfile << "Iteration\tSelected Heuristic\tCost Efficiency\tCost Diversity\t"
            << "Time Taken (seconds)\tReward\tQ-Value\n";

    // -------------------------------------------------------------------------
    // 6) MAIN LOOP CONTROL
    // -------------------------------------------------------------------------
    auto total_start_time = std::chrono::steady_clock::now();
    const int maxiter = 1000;
    int iteration = 0, itercount = 0;

    while (true)
    {
        ++iteration; ++phaseLen;
        auto iter_t0 = std::chrono::steady_clock::now();

        // 6a)  STOP CRITERIA
        double Telp = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - total_start_time).count();
        if (Telp >= max_time || itercount >= maxiter) break;

        // 6b)  CURRENT STATE
        State curState = determine_current_state(cost_eff, cost_div, iteration);
        std::string curKey = encode_state(curState);
        if (Q_table.find(curKey)==Q_table.end())    // ensure row
            for (int h:heuristics) Q_table[curKey][h]=0.0;

        // 6c)  EPSILON-GREEDY CHOICE (active_set)
        int chosen = -1;
        if (((double)std::rand()/RAND_MAX) < epsilon)
            chosen = active_set[ std::rand() % active_set.size() ];
        else {
            double bestQ = -1e100;
            for (int h:active_set)
                if (Q_table[curKey][h] > bestQ) { bestQ = Q_table[curKey][h]; chosen = h; }
        }

        // 6d)  APPLY HEURISTIC
        auto h_t0 = std::chrono::steady_clock::now();
        ApplyHeuristic(chosen, team);
        double h_sec = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - h_t0).count();
        total_elapsed_time               += h_sec;
        heuristic_total_time[chosen]     += h_sec;
        heuristic_usage_count[chosen]    += 1;

        // Evaluate new solution
        objective_Function1(team);
        int new_eff = static_cast<int>(f_cur);
        int new_div = static_cast<int>(f_cur_div);
        objective_values.push_back(new_eff);
        diversity_values .push_back(new_div);

        // 6e)  UPDATE DHSS PERFORMANCE HISTORY
        {
            Perf &p = perf[chosen];
            p.ms += h_sec * 1000.0;                  // store as ms
            double delta = std::abs(cost_eff - new_eff);
            bool improved = new_eff > cost_eff;      // lower is better
            if (improved){ p.nPlus++;  p.dPlus += delta; }
            else          { p.nMinus++; p.dMinus+= delta; }
        }

        // 6f)  CREDIT FORMULAE (5.1–5.4) & REWARD
        double eps_z = 1e-9;
        double frac  = (cost_eff - new_eff) /
                       (cost_eff + new_eff + eps_z);     // positive if improved
        bool   improved = new_eff > cost_eff;
        if (improved){
            heuristic_credits[chosen] += frac;
            double pen = frac / (heuristics.size()-1);
            for (int h:heuristics) if (h!=chosen) heuristic_credits[h] -= pen;
        } else {
            heuristic_credits[chosen] -= frac;
            double bon = frac / (heuristics.size()-1);
            for (int h:heuristics) if (h!=chosen) heuristic_credits[h] += bon;
        }
        double reward = improved ? frac : 0.0;
        if (improved) heuristic_improvement_count[chosen]++;

        heuristic_rewards[chosen] += reward;

        // 6g)  Q-LEARNING  UPDATE
        State nxtState = determine_current_state(new_eff, new_div, iteration+1);
        std::string nxtKey = encode_state(nxtState);
        if (Q_table.find(nxtKey)==Q_table.end())
            for (int h:heuristics) Q_table[nxtKey][h]=0.0;
        double maxQnext=-1e100; for(int h:heuristics)
            maxQnext = std::max(maxQnext, Q_table[nxtKey][h]);
        Q_table[curKey][chosen] +=
            alpha * (reward + gamma * maxQnext - Q_table[curKey][chosen]);

        // 6h)  ACCEPT / REJECT
        if (improved && new_div >= min_div){
            cost_eff = new_eff;
            cost_div = new_div;
            // Backup current team before overwriting
            free_solution(previous_solution, num_node, num_team, num_each_t);
            previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
            Selected.push_back(chosen);
            if (new_eff > best_cost_eff){
                best_cost_eff = new_eff;
                best_cost_div = new_div;
                best_solution1 = team;
                // Backup current team before overwriting
                free_solution(best_solution1, num_node, num_team, num_each_t);
                best_solution1 = deep_copy_solution(team, num_node, num_team, num_each_t);
            }
            waitMax    = std::max(waitMax, sinceBest);
            sinceBest  = 0;
        } else {
            ++sinceBest;
            // revert
            for (int i = 0; i <= num_team; ++i)
            for (int j = 0; j < team_size[i]; ++j)
                team[i][j] = previous_solution[i][j];
        }

        // 6i)  ------------------  DHSS  ------------------
        bool canUpdate=false, canRemove=false;
        int  Nfail = static_cast<int>(pf * std::max(1, waitMax));
        if (sinceBest >= Nfail || phaseLen >= phaseMax) canUpdate = true;
        if (!removal_done && Telp/static_cast<double>(max_time) >= R_excl) canRemove = true;

        // One-shot permanent removal
        if (canRemove){
            removal_done = true;

            int totPlus=0, totMinus=0; double totDPlus=0, totDMinus=0;
            for(auto &kv:perf){ totPlus+=kv.second.nPlus; totMinus+=kv.second.nMinus;
                                totDPlus+=kv.second.dPlus; totDMinus+=kv.second.dMinus; }
            std::unordered_map<int,double> find, fgp;
            double avgFind=0, avgFgp=0; int cnt=0;
            for (int h: heuristics){
                double nSum = perf[h].nPlus+perf[h].nMinus + eps_z;
                double dSum = perf[h].dPlus+perf[h].dMinus + eps_z;
                find[h]= 0.25*perf[h].nPlus / nSum
                       - 0.25*perf[h].nMinus/ nSum
                       + 0.25*perf[h].dPlus / dSum
                       - 0.25*perf[h].dMinus/ dSum;
                fgp[h] = 0.25*perf[h].nPlus /(totPlus +eps_z)
                       - 0.25*perf[h].nMinus/(totMinus+eps_z)
                       + 0.25*perf[h].dPlus /(totDPlus+eps_z)
                       - 0.25*perf[h].dMinus/(totDMinus+eps_z);
                avgFind+=find[h]; avgFgp+=fgp[h]; ++cnt;
            }
            avgFind/=cnt; avgFgp/=cnt;
            for (int h:heuristics)
                if (find[h] < avgFind && fgp[h] < avgFgp) removed_set.insert(h);
        }

        // Dominance-based refresh
        if (canUpdate){
            std::vector<int> newS;
            for (int hi: heuristics){
                if (removed_set.count(hi)) continue;
                double vi=(perf[hi].nPlus+perf[hi].nMinus)?
                          (double)perf[hi].nPlus /(perf[hi].nPlus+perf[hi].nMinus):0.0;
                double ti=(perf[hi].nPlus+perf[hi].nMinus)?
                          perf[hi].ms/(perf[hi].nPlus+perf[hi].nMinus):1e9;
                bool dominated=false;
                for (int hk: heuristics){
                    if (hi==hk || removed_set.count(hk)) continue;
                    double vk=(perf[hk].nPlus+perf[hk].nMinus)?
                              (double)perf[hk].nPlus/(perf[hk].nPlus+perf[hk].nMinus):0.0;
                    double tk=(perf[hk].nPlus+perf[hk].nMinus)?
                              perf[hk].ms/(perf[hk].nPlus+perf[hk].nMinus):1e9;
                    if (vk>vi && tk<ti){ dominated=true; break; }
                }
                if (!dominated) newS.push_back(hi);
            }
            if (newS.empty()) newS = active_set;   // never empty
            active_set.swap(newS);
            phaseLen = 0;
        }

        // 6j)  ITERATION LOGGING
        double iter_sec = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - iter_t0).count();
        iteration_times.push_back(iter_sec);

        outfile << iteration << '\t' << chosen << '\t'
                << new_eff << '\t' << new_div << '\t'
                << iter_sec << '\t' << reward << '\t'
                << Q_table[curKey][chosen] << '\n';

        std::cout << "It " << iteration
                  << " | Sel " << chosen
                  << " | Eff " << new_eff
                  << " | BestEff " << best_cost_eff
                  << " | A: " << active_set.size()
                  << " | R: " << removed_set.size() << '\n';

        // 6k)  ε DECAY
        epsilon = std::max(MIN_EPS, epsilon * epsilon_decay);
        ++itercount;
    } // while

    outfile.close();

    // -------------------------------------------------------------------------
    // 7) STATISTICS, SUMMARY, OUTPUT   (identical to your original section)
    // -------------------------------------------------------------------------
    double average_objective  = 0.0;
    double average_diversity  = 0.0;
    int    worst_objective    = std::numeric_limits<int>::max();
    double average_cpu_time   = 0.0;

    if (!objective_values.empty())
        average_objective = std::accumulate(objective_values.begin(),
                                            objective_values.end(), 0.0)
                            / objective_values.size();

    if (!diversity_values.empty())
        average_diversity = std::accumulate(diversity_values.begin(),
                                            diversity_values.end(), 0.0)
                            / diversity_values.size();

    if (!objective_values.empty())
        worst_objective = *std::min_element(objective_values.begin(),
                                            objective_values.end());

    if (!iteration_times.empty())
        average_cpu_time = std::accumulate(iteration_times.begin(),
                                           iteration_times.end(), 0.0)
                           / iteration_times.size();

    int best_heuristic  = -1, worst_heuristic = -1;
    double max_reward = -1e100, min_reward = 1e100;
    for (int h: heuristics) {
        if (heuristic_rewards[h] > max_reward){
            max_reward = heuristic_rewards[h]; best_heuristic = h;
        }
        if (heuristic_rewards[h] < min_reward){
            min_reward = heuristic_rewards[h]; worst_heuristic = h;
        }
    }

    std::filesystem::path summary_path =
        folder_path / ("Q_Learning_Selection_Hyperheuristic_CMCEE_" + instanceName + "_summary.txt");
    {
        std::ofstream sf(summary_path);
        sf << "Final Best Objectives:\n"
           << "Efficiency: " << best_cost_eff << '\n'
           << "Diversity:  " << best_cost_div << "\n\n"

           << "Statistical Analysis:\n"
           << "Total Time Taken: " << std::fixed << std::setprecision(4)
           << total_elapsed_time << " seconds\n"
           << "Average Objective Function Value: " << average_objective << '\n'
           << "Average Diversity Value:         " << average_diversity << '\n'
           << "Worst Objective Function Value:   " << worst_objective << '\n'
           << "Average CPU Time per Iteration:   " << average_cpu_time << " seconds\n\n"

           << "Heuristic Performance:\n";
        for (int h: heuristics){
            sf << "Heuristic " << h << ":\n"
               << "  Usage Count:         " << heuristic_usage_count[h] << '\n'
               << "  Total QL Reward:     " << heuristic_rewards[h] << '\n'
               << "  Improvement Count:   " << heuristic_improvement_count[h] << '\n'
               << "  Total Time:          " << heuristic_total_time[h] << " seconds\n\n";
        }
        sf << "Best Heuristic:  " << best_heuristic
           << "  (Reward " << max_reward << ")\n";
        sf << "Worst Heuristic: " << worst_heuristic
           << "  (Reward " << min_reward << ")\n";
    }

    // Extract top-K heuristics
    best_heuristics.clear();
    struct HInfo { int id; double avg; };
    std::vector<HInfo> hv;
    for (int h:heuristics){
        double avg=0.0;
        if (heuristic_usage_count[h]>0)
            avg = heuristic_rewards[h]/heuristic_usage_count[h];
        hv.push_back({h,avg});
    }
    std::sort(hv.begin(), hv.end(),
              [](const HInfo&a,const HInfo&b){return a.avg>b.avg;});
    int limit=std::min(topK,(int)hv.size());
    for (int i=0;i<limit;++i) best_heuristics.push_back(hv[i].id);

    // Console summary
    std::cout << "Final Best Objectives:  Eff=" << best_cost_eff
              << "  Div=" << best_cost_div << '\n'
              << "Total time: " << total_elapsed_time << " s\n"
              << "Active heuristics at end: ";
    for (int h:active_set) std::cout << h << ' ';
    std::cout << "\nRemoved heuristics: ";
    for (int h:removed_set) std::cout << h << ' ';
    std::cout << "\nSelected heuristic sequence: ";
    for (int h:Selected) std::cout << h << ' ';
    std::cout << "\n=============================================================================\n";
    std::cout << "Q-Learning + DHSS Hyper-heuristic Framework Finished its Processes.\n";
    std::cout << "=============================================================================\n";
}

void Hyper_heuristic::SSHH_Selection_Hyperheuristic_CMCEE(int max_time)
{
    std::cout << "SSHH Selection Hyper-heuristic Framework Starting..." << std::endl;

    // === 1) Initialize solution ===
    generate_initialrandom();
    objective_Function1(team);

    int f_S       = f_cur;      // Current efficiency
    int f_S_prime = f_cur;      // New efficiency
    int f_Sb      = f_cur;      // Best efficiency
    int div_cost  = f_cur_div;  // Diversity cost

    //int** s_current = team;
    //int** s_best    = team;

    // Store deep copies of current and best solutions
    int** s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** s_best    = deep_copy_solution(team, num_node, num_team, num_each_t);


    // Allocate team_size array globally
    team_size = new int[num_team + 1];

    // Assign team sizes
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;

    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;
    // === 2) Transition & sequence matrices ===
    std::map<int, std::map<int, int>> T_ran;
    std::map<int, int> Seq;

    int best_found_at = 0;

    // === 3) Heuristic performance ===
    std::map<int, double> heuristic_scores;
    std::map<int, int>    heuristic_usage_count;
    std::map<int, double> heuristic_rewards;
    std::map<int, double> heuristic_total_time;
    std::map<int, int>    heuristic_improvement_count;
    std::vector<double> fitness_history;


    std::vector<int> heuristics = {17, 18, 19, 20, 21, 22, 23};
    for (int h : heuristics) {
        heuristic_scores[h]            = 0.0;
        heuristic_usage_count[h]       = 0;
        heuristic_rewards[h]           = 0.0;
        heuristic_total_time[h]        = 0.0;
        heuristic_improvement_count[h] = 0;
    }

    std::vector<int> HeuristicSequence;

    // === 4) RNG ===
    std::mt19937 gen_rand(static_cast<unsigned int>(std::time(0)));
    std::uniform_int_distribution<> dis_int_dist(0, heuristics.size() - 1);

    int curr = heuristics[dis_int_dist(gen_rand)];
    heuristic_usage_count[curr]++;

    // === 5) Data logging ===
    std::vector<int> objective_values;
    std::vector<int> diversity_values;
    std::vector<double> iteration_times;
    double total_elapsed_time = 0.0;

    // === 6) File paths ===
    std::filesystem::path folder_path = "D:/Result Hyper-heuristic Models/";
    if (!std::filesystem::exists(folder_path)) {
        std::filesystem::create_directories(folder_path);
    }

    std::filesystem::path main_results_path = folder_path / ("SSHH_Selection_CMCEE_" + instanceName + "_results.txt");
    std::ofstream main_results_file(main_results_path);
    if (!main_results_file) { std::cerr << "Error opening results file!\n"; return; }
    main_results_file << "Iteration\tSelected Heuristic\tCost Efficiency\tCost Diversity\tTime Taken (s)\tReward\tScore\n";

    std::filesystem::path features_path = folder_path / ("SSHH_Selection_CMCEE_" + instanceName + "_features.csv");
    std::ofstream features_file(features_path);
    if (!features_file) { std::cerr << "Error opening features file!\n"; return; }

    // === 7) Feature headers ===
    features_file << "Iteration";
    for (int i = 1; i <= 39; i++) features_file << ",F" << i;
    features_file << "\n";

    // === 8) Feature sets ===
    std::set<int> S_imp, S_wrs, S_eq, S_ac, S_uq, S_nb;

    double Sfirst = f_S, Sbest = f_S, Sworst = f_S;
    auto total_start_time = std::chrono::steady_clock::now();
    int iteration_count = 0;

    // === 9) Main loop ===
    while (true) {
        iteration_count++;
        auto iteration_start_time = std::chrono::steady_clock::now();

        double elapsed_total = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - total_start_time).count();
        if (elapsed_total >= max_time) break;

        // (A) Select next heuristic
        int next = -1;
        if (T_ran.find(curr) != T_ran.end() && !T_ran[curr].empty()) {
            int max_count = -1;
            for (const auto& [h, count] : T_ran[curr]) {
                if (count > max_count) { max_count = count; next = h; }
            }
        }
        if (next == -1) next = heuristics[dis_int_dist(gen_rand)];
        heuristic_usage_count[next]++;
        HeuristicSequence.push_back(next);

        // (B) Apply full sequence
        if (HeuristicSequence.size() >= heuristics.size()) {
            int** temp_team = team;
            auto seq_start = std::chrono::steady_clock::now();
            for (int h_seq : HeuristicSequence) {
                auto h_start = std::chrono::steady_clock::now();
                ApplyHeuristic(h_seq, temp_team);
                auto h_end   = std::chrono::steady_clock::now();
                heuristic_total_time[h_seq] += std::chrono::duration<double>(h_end - h_start).count();
            }
            auto seq_end = std::chrono::steady_clock::now();
            double seq_time = std::chrono::duration<double>(seq_end - seq_start).count();

            objective_Function1(temp_team);
            f_S_prime = f_cur;
            int f_div_prime = f_cur_div;

            if (f_S_prime > f_Sb && f_div_prime >= min_div) {
                f_Sb = f_S_prime;
                div_cost = f_div_prime;
                //s_current = temp_team;
                free_solution(s_current, num_node, num_team, num_each_t);
                s_current = deep_copy_solution(temp_team, num_node, num_team, num_each_t);

                free_solution(s_best, num_node, num_team, num_each_t);
                s_best = deep_copy_solution(temp_team, num_node, num_team, num_each_t);

                best_eff  = f_S_prime;
                best_div  = f_div_prime;
                time_taken = seq_time;
                T_ran[curr][next]++;

                f_S = f_S_prime;
                best_found_at = iteration_count;
                heuristic_rewards[next] += 1.0;
                heuristic_improvement_count[next]++;
                heuristic_scores[next] += 1.0;

                for (int m = 0; m < num_node; m++) fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }

                // Update fitness history
                fitness_history.push_back(f_S_prime);

            } else {
                heuristic_scores[next] -= 0.5;
                fitness_history.push_back(f_S);  // keep history consistent
            }
            HeuristicSequence.clear();
        }

        // (C) Update current heuristic
        curr = next;

        // (D) Calculate features
        Features F = calculate_features(
        S_imp, S_wrs, S_eq, S_ac, S_uq, S_nb,
        Sfirst, Sbest, Sworst,
        fitness_history,        // ✅ vector<double>
        heuristic_total_time,   // ✅ map<int,double>
        heuristics.size(),
        iteration_count
        );

        // (E) Log results
        double iteration_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - iteration_start_time).count();
        total_elapsed_time += iteration_time;

        main_results_file << iteration_count << "\t" << next << "\t"
                          << f_S << "\t" << div_cost << "\t"
                          << iteration_time << "\t"
                          << heuristic_rewards[next] << "\t"
                          << heuristic_scores[next] << "\n";

        features_file << iteration_count
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

        objective_values.push_back(f_S);
        diversity_values.push_back(div_cost);
        iteration_times.push_back(iteration_time);
    }

    // === 10) Summary ===
    std::filesystem::path summary_path = folder_path / ("SSHH_Selection_CMCEE_" + instanceName + "_Summary.txt");
    std::ofstream summary_file(summary_path);
    if (!summary_file) { std::cerr << "Error opening summary file!\n"; return; }

    double avg_obj = objective_values.empty() ? 0.0 : std::accumulate(objective_values.begin(), objective_values.end(), 0.0) / objective_values.size();
    double avg_div = diversity_values.empty() ? 0.0 : std::accumulate(diversity_values.begin(), diversity_values.end(), 0.0) / diversity_values.size();
    double avg_cpu = iteration_times.empty() ? 0.0 : std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    int    worst_obj  = objective_values.empty() ? 0 : *std::min_element(objective_values.begin(), objective_values.end());

    int worse_performing = 0;
    for (auto &p : heuristic_scores) if (p.second < 0.0) worse_performing++;

    // Best/worst heuristic
    int best_h = -1, worst_h = -1;
    double max_reward = -1e9, min_reward = 1e9;
    for (int h : heuristics) {
        if (heuristic_rewards[h] > max_reward) { max_reward = heuristic_rewards[h]; best_h = h; }
        if (heuristic_rewards[h] < min_reward) { min_reward = heuristic_rewards[h]; worst_h = h; }
    }

    summary_file << "Final Best Objectives:\n";
    summary_file << "Candidate (f_S): " << f_S << "\n";
    summary_file << "New (f_S'): " << f_S_prime << "\n";
    summary_file << "Best (f_Sb): " << f_Sb << "\n\n";

    summary_file << "Statistics:\n";
    summary_file << "Total Time Taken: " << total_elapsed_time << " seconds\n";
    summary_file << "Average Objective: " << avg_obj << "\n";
    summary_file << "Average Diversity: " << avg_div << "\n";
    summary_file << "Worst Objective: " << worst_obj << "\n";
    summary_file << "Average CPU per Iteration: " << avg_cpu << " seconds\n\n";

    summary_file << "Heuristic Performance:\n";
    for (int h : heuristics) {
        summary_file << "Heuristic " << h << ":\n";
        summary_file << "  Usage Count: " << heuristic_usage_count[h] << "\n";
        summary_file << "  Total Reward: " << heuristic_rewards[h] << "\n";
        summary_file << "  Improvement Count: " << heuristic_improvement_count[h] << "\n";
        summary_file << "  Total Time: " << heuristic_total_time[h] << " seconds\n\n";
    }
    summary_file << "Best Heuristic: " << best_h << " (Reward: " << max_reward << ")\n";
    summary_file << "Worst Heuristic: " << worst_h << " (Reward: " << min_reward << ")\n";
    summary_file << "Worse Performing Heuristics Count: " << worse_performing << "\n";

    summary_file.close();
    main_results_file.close();
    features_file.close();

    std::cout << "Summary results saved to " << summary_path << std::endl;
    std::cout << "SSHH finished.\n";
    check_best_solution();
}


void Hyper_heuristic::Adaptive_SSHH_Selection_Hyperheuristic_CMCEE(int max_time)
{
    std::cout << "Adaptive SSHH Selection Hyper-heuristic Framework Starting..." << std::endl;

    // === 1) Initialize solution ===
    generate_initialrandom();
    objective_Function1(team);

    int f_S       = f_cur;      // Current efficiency
    int f_S_prime = f_cur;      // New efficiency
    int f_Sb      = f_cur;      // Best efficiency
    int div_cost  = f_cur_div;  // Diversity cost

   // int** s_current = team;
   // int** s_best    = team;
    // Store deep copies of current and best solutions
    int** s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** s_best    = deep_copy_solution(team, num_node, num_team, num_each_t);
    int* team_size = new int[num_team + 1];

    // Assign team sizes
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;

    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;

    // === 2) Transition & sequence matrices ===
    std::map<int, std::map<int, int>> T_ran;
    std::map<int, int> Seq;

    int best_found_at = 0;

    // === 3) Heuristic performance ===
    std::map<int, double> heuristic_scores;
    std::map<int, int>    heuristic_usage_count;
    std::map<int, double> heuristic_rewards;
    std::map<int, double> heuristic_total_time;
    std::map<int, int>    heuristic_improvement_count;
    std::vector<double> fitness_history;


    std::vector<int> heuristics = {17, 18, 19, 20, 21, 22, 23};
    for (int h : heuristics) {
        heuristic_scores[h]            = 0.0;
        heuristic_usage_count[h]       = 0;
        heuristic_rewards[h]           = 0.0;
        heuristic_total_time[h]        = 0.0;
        heuristic_improvement_count[h] = 0;
    }

    std::vector<int> HeuristicSequence;

    // === 4) RNG ===
    std::mt19937 gen_rand(static_cast<unsigned int>(std::time(0)));
    std::uniform_int_distribution<> dis_int_dist(0, heuristics.size() - 1);

    int curr = heuristics[dis_int_dist(gen_rand)];
    heuristic_usage_count[curr]++;

    // === 5) Data logging ===
    std::vector<int> objective_values;
    std::vector<int> diversity_values;
    std::vector<double> iteration_times;
    double total_elapsed_time = 0.0;

    // === 6) File paths ===
    std::filesystem::path folder_path = "D:/Result Hyper-heuristic Models/";
    if (!std::filesystem::exists(folder_path)) {
        std::filesystem::create_directories(folder_path);
    }

    std::filesystem::path main_results_path = folder_path / ("SSHH_Selection_CMCEE_" + instanceName + "_results.txt");
    std::ofstream main_results_file(main_results_path);
    if (!main_results_file) { std::cerr << "Error opening results file!\n"; return; }
    main_results_file << "Iteration\tSelected Heuristic\tCost Efficiency\tCost Diversity\tTime Taken (s)\tReward\tScore\n";

    std::filesystem::path features_path = folder_path / ("SSHH_Selection_CMCEE_" + instanceName + "_features.csv");
    std::ofstream features_file(features_path);
    if (!features_file) { std::cerr << "Error opening features file!\n"; return; }

    // === 7) Feature headers ===
    features_file << "Iteration";
    for (int i = 1; i <= 39; i++) features_file << ",F" << i;
    features_file << "\n";

    // === 8) Feature sets ===
    std::set<int> S_imp, S_wrs, S_eq, S_ac, S_uq, S_nb;

    double Sfirst = f_S, Sbest = f_S, Sworst = f_S;
    auto total_start_time = std::chrono::steady_clock::now();
    int iteration_count = 0;

    // === 9) Main loop ===
    while (true) {
        iteration_count++;
        auto iteration_start_time = std::chrono::steady_clock::now();

        double elapsed_total = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - total_start_time).count();
        if (elapsed_total >= max_time) break;

        // (A) Select next heuristic
        int next = -1;
        if (T_ran.find(curr) != T_ran.end() && !T_ran[curr].empty()) {
            int max_count = -1;
            for (const auto& [h, count] : T_ran[curr]) {
                if (count > max_count) { max_count = count; next = h; }
            }
        }
        if (next == -1) next = heuristics[dis_int_dist(gen_rand)];
        heuristic_usage_count[next]++;
        HeuristicSequence.push_back(next);

        // (B) Apply full sequence
        if (HeuristicSequence.size() >= heuristics.size()) {
            //int** temp_team = team;
            auto seq_start = std::chrono::steady_clock::now();
            for (int h_seq : HeuristicSequence) {
                auto h_start = std::chrono::steady_clock::now();
                ApplyHeuristic(h_seq, team);
                auto h_end   = std::chrono::steady_clock::now();
                heuristic_total_time[h_seq] += std::chrono::duration<double>(h_end - h_start).count();
            }
            auto seq_end = std::chrono::steady_clock::now();
            double seq_time = std::chrono::duration<double>(seq_end - seq_start).count();

            objective_Function1(team);
            f_S_prime = f_cur;
            int f_div_prime = f_cur_div;

            if (f_S_prime > f_Sb && f_div_prime >= min_div) {
                f_Sb = f_S_prime;
                div_cost = f_div_prime;
                //s_current = temp_team;
                //s_best    = temp_team;
                free_solution(s_current, num_node, num_team, num_each_t);
                s_current = deep_copy_solution(team, num_node, num_team, num_each_t);

                free_solution(s_best, num_node, num_team, num_each_t);
                s_best = deep_copy_solution(team, num_node, num_team, num_each_t);

                best_eff  = f_S_prime;
                best_div  = f_div_prime;
                time_taken = seq_time;
                T_ran[curr][next]++;

                f_S = f_S_prime;
                best_found_at = iteration_count;
                heuristic_rewards[next] += 1.0;
                heuristic_improvement_count[next]++;
                heuristic_scores[next] += 1.0;

                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }

                // Update fitness history
                fitness_history.push_back(f_S_prime);

            } else {
                heuristic_scores[next] -= 0.5;
                fitness_history.push_back(f_S);  // keep history consistent
                for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = s_current[i][j];
            }
            HeuristicSequence.clear();
        }

        // (C) Update current heuristic
        curr = next;

        // (D) Calculate features
        Features F = calculate_features(
        S_imp, S_wrs, S_eq, S_ac, S_uq, S_nb,
        Sfirst, Sbest, Sworst,
        fitness_history,        // ✅ vector<double>
        heuristic_total_time,   // ✅ map<int,double>
        heuristics.size(),
        iteration_count
        );

        // (E) Log results
        double iteration_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - iteration_start_time).count();
        total_elapsed_time += iteration_time;

        main_results_file << iteration_count << "\t" << next << "\t"
                          << f_S << "\t" << div_cost << "\t"
                          << iteration_time << "\t"
                          << heuristic_rewards[next] << "\t"
                          << heuristic_scores[next] << "\n";

        features_file << iteration_count
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

        objective_values.push_back(f_S);
        diversity_values.push_back(div_cost);
        iteration_times.push_back(iteration_time);
    }

    // === 10) Summary ===
    std::filesystem::path summary_path = folder_path / ("SSHH_Selection_CMCEE_" + instanceName + "_Summary.txt");
    std::ofstream summary_file(summary_path);
    if (!summary_file) { std::cerr << "Error opening summary file!\n"; return; }

    double avg_obj = objective_values.empty() ? 0.0 : std::accumulate(objective_values.begin(), objective_values.end(), 0.0) / objective_values.size();
    double avg_div = diversity_values.empty() ? 0.0 : std::accumulate(diversity_values.begin(), diversity_values.end(), 0.0) / diversity_values.size();
    double avg_cpu = iteration_times.empty() ? 0.0 : std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size();
    int worst_obj  = objective_values.empty() ? 0 : *std::min_element(objective_values.begin(), objective_values.end());

    int worse_performing = 0;
    for (auto &p : heuristic_scores) if (p.second < 0.0) worse_performing++;

    // Best/worst heuristic
    int best_h = -1, worst_h = -1;
    double max_reward = -1e9, min_reward = 1e9;
    for (int h : heuristics) {
        if (heuristic_rewards[h] > max_reward) { max_reward = heuristic_rewards[h]; best_h = h; }
        if (heuristic_rewards[h] < min_reward) { min_reward = heuristic_rewards[h]; worst_h = h; }
    }

    summary_file << "Final Best Objectives:\n";
    summary_file << "Candidate (f_S): " << f_S << "\n";
    summary_file << "New (f_S'): " << f_S_prime << "\n";
    summary_file << "Best (f_Sb): " << f_Sb << "\n\n";

    summary_file << "Statistics:\n";
    summary_file << "Total Time Taken: " << total_elapsed_time << " seconds\n";
    summary_file << "Average Objective: " << avg_obj << "\n";
    summary_file << "Average Diversity: " << avg_div << "\n";
    summary_file << "Worst Objective: " << worst_obj << "\n";
    summary_file << "Average CPU per Iteration: " << avg_cpu << " seconds\n\n";

    summary_file << "Heuristic Performance:\n";
    for (int h : heuristics) {
        summary_file << "Heuristic " << h << ":\n";
        summary_file << "  Usage Count: " << heuristic_usage_count[h] << "\n";
        summary_file << "  Total Reward: " << heuristic_rewards[h] << "\n";
        summary_file << "  Improvement Count: " << heuristic_improvement_count[h] << "\n";
        summary_file << "  Total Time: " << heuristic_total_time[h] << " seconds\n\n";
    }
    summary_file << "Best Heuristic: " << best_h << " (Reward: " << max_reward << ")\n";
    summary_file << "Worst Heuristic: " << worst_h << " (Reward: " << min_reward << ")\n";
    summary_file << "Worse Performing Heuristics Count: " << worse_performing << "\n";

    summary_file.close();
    main_results_file.close();
    features_file.close();

    std::cout << "Summary results saved to " << summary_path << std::endl;
    std::cout << "SSHH finished.\n";
    check_best_solution();
}

/*
class MultiStageHyperHeuristicFramework : public Hyper_heuristic {
public:
    // Available Hyper-Heuristic IDs
    std::vector<int> hyperHeuristics = {1, 2, 3, 4, 5, 6};

    // Q-Learning Parameters
    const double alpha = 0.1;    // Learning rate
    const double gamma = 0.9;    // Discount factor
    const double epsilon = 0.1;  // Exploration rate

    // Q-Table: 100 states (0-99) x 6 actions
    std::vector<std::vector<double>> Q_table;

    // Heuristic Performance: Maps heuristic ID to a vector of improvement scores
    std::map<int, std::vector<double>> heuristicPerformance;
    // 8) Data structure to store iteration-level data
    struct ConvergenceRecord {
        int iteration;
        int heuristicID;
        double improvement;
        double timeTaken;
        double efficiencyBefore;
        double efficiencyAfter;
    };
    std::vector<ConvergenceRecord> convergenceData;

    // Current State (encoded as a single integer)
    int currentState;

    // Random Number Generators
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    // Maximum expected improvement (set this based on your problem)
    const double maxExpectedImprovement = 100.0; // Adjust as needed

    // Helper function to encode a 2D state pair into a single integer
    int encodeState(int feature1, int feature2) {
        // Both feature1 and feature2 should be in [0,9].
        return feature1 * 10 + feature2;
    }

    // Initialize Q-Table with all state-action pairs set to 0.0
    void initializeQTable() {
        int numStates = 100; // 10 * 10 states
        int numActions = static_cast<int>(hyperHeuristics.size());
        Q_table.assign(numStates, std::vector<double>(numActions, 0.0));

        std::cout << "Initialized Q-table with " << numStates << " states and " << numActions << " actions:\n";
        std::cout << std::fixed << std::setprecision(2); // Set precision for Q-values
        // Optionally, we could print fewer states if large output is not desired
        int statesToPrint = 10;
        for (int state = 0; state < statesToPrint; ++state) {
            std::cout << "State " << state << ": ";
            for (int action = 0; action < numActions; ++action) {
                std::cout << Q_table[state][action] << " ";
            }
            std::cout << "\n";
        }
    }

    // Select hyper-heuristic using epsilon-greedy strategy
    std::vector<int> selectHyperHeuristics(int k) {
        std::vector<int> selectedHeuristics;
        std::uniform_real_distribution<> prob_dis(0.0, 1.0);
        double rand_val = prob_dis(gen);

        if (rand_val < epsilon) {
            // Exploration: Select k random hyper-heuristics without replacement
            std::vector<int> shuffled = hyperHeuristics;
            std::shuffle(shuffled.begin(), shuffled.end(), gen);
            if (k > static_cast<int>(shuffled.size())) k = static_cast<int>(shuffled.size());
            selectedHeuristics.assign(shuffled.begin(), shuffled.begin() + k);
        } else {
            // Exploitation: Select the top k hyper-heuristics with highest Q-values for current state
            std::vector<std::pair<int, double>> heuristicQ;
            for (size_t i = 0; i < hyperHeuristics.size(); ++i) {
                heuristicQ.emplace_back(hyperHeuristics[i], Q_table[currentState][i]);
            }

            // Sort heuristics based on Q-values in descending order
            std::sort(heuristicQ.begin(), heuristicQ.end(),
                      [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                          return a.second > b.second;
                      });

            // Select top k heuristics
            for (int i = 0; i < k && i < static_cast<int>(heuristicQ.size()); ++i) {
                selectedHeuristics.push_back(heuristicQ[i].first);
            }
        }

        return selectedHeuristics;
    }

    // Apply the selected hyper-heuristic based on ID
    void applyHyperHeuristic(int heuristicID, int max_time) {
    try {
        if (heuristicID == 1) {
            MAHH_Algorithm(max_time);
        } else if (heuristicID == 2) {
            MAHH_Selection_CMCEE(max_time);
        } else if (heuristicID == 3) {
            Greedy_Selection_Hyperheuristic_CMCEE(max_time);
        } else if (heuristicID == 4) {
            Random_Selection_Hyperheuristic_CMCEE(max_time);
        } else if (heuristicID == 5) {
            SSHH_Selection_Hyperheuristic_CMCEE(max_time);
        } else if (heuristicID == 6) {
            Q_Learning_Selection_Hyperheuristic_CMCEE(max_time);
        }
        else {
            std::cerr << "Invalid Hyper-Heuristic ID: " << heuristicID << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in Hyper-Heuristic ID " << heuristicID << ": " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception in Hyper-Heuristic ID " << heuristicID << std::endl;
    }
}


    // Update Q-Table based on the reward received
    void updateQTable(int actionID, double reward, int nextState) {
        auto it = std::find(hyperHeuristics.begin(), hyperHeuristics.end(), actionID);
        if (it == hyperHeuristics.end()) {
            std::cerr << "Action ID not found in hyperHeuristics." << std::endl;
            return;
        }
        int actionIndex = static_cast<int>(std::distance(hyperHeuristics.begin(), it));

        // Find the maximum Q-value for the next state
        double maxNextQ = *std::max_element(Q_table[nextState].begin(), Q_table[nextState].end());

        // Update Q(s, a)
        Q_table[currentState][actionIndex] += alpha * (reward + gamma * maxNextQ - Q_table[currentState][actionIndex]);

        // Update heuristic performance
        heuristicPerformance[actionID].push_back(reward);
    }

    // Calculate the current efficiency using f_cur
    double calculateCurrentEfficiency() {
        return f_cur;
    }

    // Constructor
    MultiStageHyperHeuristicFramework() : gen(std::random_device{}()), dis(0.0, 1.0) {
        initializeQTable();
        currentState = encodeState(0, 0); // Initial state (feature1=0, feature2=0)
    }

    // Run the framework with Q-learning-based selection
   void run(int max_time, int iterations = 10) {
    for (int iter = 0; iter < iterations; ++iter) {
        std::cout << "Iteration " << iter + 1 << ":\n";
         initialization();
        // Always select all heuristics
        std::vector<int> selectedHeuristics = hyperHeuristics;

        for (const auto& heuristicID : selectedHeuristics) {
            double efficiencyBefore = calculateCurrentEfficiency();

            // Apply the hyper-heuristic
            std::cout << "  Applying Hyper-Heuristic ID " << heuristicID << std::endl;
            applyHyperHeuristic(heuristicID, max_time);

            double efficiencyAfter = calculateCurrentEfficiency();

            // Calculate improvement (assuming max f_cur is better)
            double improvement =  efficiencyAfter - efficiencyBefore;
            std::cout << "  Heuristic ID " << heuristicID << " achieved improvement: " << improvement << "\n";

            // Define next state based on the improvement
            int feature1 = (currentState / 10 + 1) % 10;
            int feature2_val = static_cast<int>((improvement / maxExpectedImprovement) * 10);
            if (feature2_val < 0) feature2_val = 0;
            if (feature2_val > 9) feature2_val = 9;
            int feature2 = feature2_val;

            int nextState = encodeState(feature1, feature2);

            // Update Q-Table with the received reward
            updateQTable(heuristicID, improvement, nextState);

            // Transition to the next state
            currentState = nextState;

            // Log the transition
            std::cout << "  Transitioned to State: " << currentState
                      << " (Feature1: " << (currentState / 10)
                      << ", Feature2: " << (currentState % 10) << ")\n\n";
        }
    }

    // Perform statistical analysis
    performStatisticalAnalysis();
}

    // Perform statistical analysis
    void performStatisticalAnalysis() {
        std::cout << "\n--- Statistical Analysis ---\n";
        for (const auto &[id, scores] : heuristicPerformance) {
            if (scores.empty()) {
                std::cout << "Heuristic ID " << id << ": No improvements recorded.\n";
                continue;
            }
            double total = std::accumulate(scores.begin(), scores.end(), 0.0);
            double mean = total / scores.size();
            double variance = 0.0;
            for (const auto &s : scores) {
                variance += (s - mean) * (s - mean);
            }
            variance /= scores.size();
            std::cout << "Heuristic ID " << id << ": Total Improvement = " << total
                      << ", Mean Improvement = " << mean
                      << ", Variance = " << variance << "\n";
        }
    }
};*/

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
int** Hyper_heuristic::run_LS(LSAlgo algo, int** team) {
    switch (algo) {
        case LS_SA:    return simulated_annealing();       // your SA implementation
        case LS_ILS:   return iterated_local_search();     // your ILS implementation
        case LS_TS:    return fits();     // use TS if available
        case LS_GD:    return great_deluge_algorithm();     // placeholder for Great Deluge
        case LS_LAHC:  return late_acceptance_hill_climbing();     // placeholder for LAHC
        case LS_BASIC: return local_search();     // basic local search
        default:       return team;
    }
}
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
        case  1: return LLH24(sol);
        case  2: return LLH17(sol);
        case  3: return LLH18(sol);
        case  4: return LLH19(sol);
        case  5: return LLH20(sol);
        case  6: return LLH21(sol);
        case  7: return LLH22(sol);
        case  8: return LLH23(sol);
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
double TH_Threshold  = 3.0;
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
    generate_initialrandom();
    objective_Function(team);
    display(team);
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
/*
// ==========================================================
//  SECTION 3: MAIN TRI-LEVEL Q-LEARNING HYPER-HEURISTIC
// ==========================================================
void Hyper_heuristic::TriLevel_HH_Qlearning_CMCEE(int max_time) {
    std::cout << "===================================================================\n";
    std::cout << "Tri-Level Q-Learning Hyper-Heuristic Framework (LS->OP->MA)\n";
    std::cout << "===================================================================\n";

    // --- Initialization ---
    generate_initialrandom();
    objective_Function(team);
    display(team);

    int cost_eff = f_cur, cost_div = f_cur_div;
    int best_eff = cost_eff, best_div = cost_div;
    int prev_eff = cost_eff, prev_div = cost_div;
    int f_eff_max = cost_eff, f_div_max = cost_div;

    int **best_sol = team;
    int** previous_solution = team;
    double total_time = 0.0;

    // --- Structures for Q-tables ---
    std::map<std::string, std::map<int,double>> Q_LS;
    std::map<std::string, std::map<int,double>> Q_OP;
    std::map<std::string, std::map<int,double>> Q_MA;

    std::vector<double> reward_hist;
    std::vector<int> div_values;
    int accepted_moves = 0, total_moves = 0;

    // --- Parameters ---
    double alpha = 0.10, gamma = 0.90;
    double eps_LS = 0.9, eps_OP = 0.9, eps_MA = 0.9;
    double eps_decay = 0.99, eps_min = 0.05;

    double SA_temp = 1.0, SA_init = 1.0;
    double flex = 1.0, flex_max = 1.0;

    // --- File setup ---
    std::filesystem::path folder_path = "D:/Datasets/TRI_LEVEL_HH_MODELS/InstanceSeparateHH_TriLevel/";
    if (!std::filesystem::exists(folder_path)) std::filesystem::create_directories(folder_path);

    std::filesystem::path file_path = folder_path / ("TriLevel_Q_HH_CMCEE_" + instanceName + "_results.txt");
    std::ofstream outfile(file_path);
    outfile << "Iter\tLS\tOP\tMA\tEff\tDiv\tbest_Eff\tbest_Div\tDelta\tReward\tStepSec\n";

    // --- Timing ---
    auto t0 = std::chrono::steady_clock::now();
    int max_iter = 100;
    int iteration = 0;

    // --- Main Loop ---
    while (true) {
        iteration++;
        total_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (total_time >= max_time || iteration >= max_iter) break;

        // ---------------- STATE ----------------
        StateFeatures s = compute_state_vector(
            cost_eff, cost_div,
            prev_eff, prev_div,
            f_eff_max, f_div_max,
            iteration, max_iter,
            accepted_moves, total_moves,
            reward_hist, div_values,
            SA_temp, SA_init,
            flex, flex_max);

        std::string key_state = discretize_state(s);

        // ---------------------------------------------------------
        // LEVEL 1 : Select Local Search Algorithm (LSi)
        // ---------------------------------------------------------

        // Initialize Q-table row if state not present
         if (Q_LS.find(key_state) == Q_LS.end()) {
                for (int a = 1; a <= 5; a++)
                Q_LS[key_state][a] = 0.0;      // 5 LS algorithms
          }

         // Epsilon-greedy selection: exploration or exploitation
        int LSi;
        double rand01 = (double)rand() / RAND_MAX;

        // ------------------------
        // Exploration
        // ------------------------
        if (rand01 < eps_LS) {
                LSi = 1 + (rand() % 5);             // pick random LS in {1..5}
        }
        // ------------------------
        // Exploitation (greedy)
        // ------------------------
        else {
            auto &row = Q_LS[key_state];
            auto best = std::max_element(
            row.begin(), row.end(),
                        [](const auto &x, const auto &y) {
                        return x.second < y.second;
                        }
            );
            LSi = best->first;                  // best LS index (1..5)
        }

         // Safety check (NEVER allow invalid values)
         if (LSi < 1 || LSi > 5) LSi = 1;

        static std::map<int, int> op_counts;
        static std::map<int, double> op_rewards;

        double total_counts = 0;
        for (auto &p : op_counts)
            total_counts += p.second;
        std::vector<int> ops;
        for (int i = 1;i <= 5;i++)
            ops.push_back(i);
        double c = 1.0;
        int selected_op = -1;
        double best_ucb = -1e9;

        for (int op : ops) {
            double reward = op_rewards[op];
            double count1  = op_counts[op];

            double explore = c * std::sqrt(
              (2.0 * std::log(total_counts + 1)) / (count1 + 1.0)
             );

            double ucb = reward + explore;

            if (ucb > best_ucb) {
            best_ucb = ucb;
            selected_op = op;
            }
        }


        // ---------------- LEVEL 3 (MA) ----------------
        if (Q_MA.find(key_state) == Q_MA.end())
            for (int a = 1; a <= 5; a++)
                Q_MA[key_state][a] = 0.0;

         MA_Strategy aMA;

        if (((double)rand() / RAND_MAX) < eps_MA)
            aMA = (MA_Strategy)(1 + rand() % 5);    // random MA rule
        else
            aMA = (MA_Strategy)std::max_element(
                Q_MA[key_state].begin(),
                Q_MA[key_state].end(),
                [](auto &x, auto &y){ return x.second < y.second; }
               )->first;                        // best MA rule

        // ---------------- APPLY LS + OP ----------------
        auto step_start = std::chrono::steady_clock::now();


        int** S_new = Apply_LS_OP(LSi, selected_op, team);
        //ApplyMeta_Heuristic(LSi, team);
        objective_Function(S_new);
        int new_eff = f_cur, new_div =f_cur_div;
        total_moves++;

        // ---------------- ACCEPTANCE ----------------
        bool accepted = accept_move(aMA, cost_eff, cost_div, new_eff, new_div, min_div);
        if (accepted) {
            accepted_moves++;
            prev_eff = cost_eff;
            prev_div = cost_div;
            cost_eff = new_eff;
            cost_div = new_div;
            if (new_eff > best_eff && new_div >= min_div) {
                best_eff = new_eff;
                best_div = new_div;
                best_sol = team;
                previous_solution = team;
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++) {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        } else {
            // revert
            team = previous_solution;

        }


        double step_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count();

        // ---------------- REWARD & Q-UPDATES ----------------
        double delta = compute_delta(new_eff, prev_eff);
        double r = reward_from_delta(delta);
        reward_hist.push_back(r);

        std::string next_state = discretize_state(compute_state_vector(
            cost_eff, cost_div,
            prev_eff, prev_div,
            f_eff_max, f_div_max,
            iteration, max_iter,
            accepted_moves, total_moves,
            reward_hist, div_values,
            SA_temp, SA_init,
            flex, flex_max));

        double max_next_LS = -1e9;
        for (auto &p : Q_LS[next_state]) max_next_LS = std::max(max_next_LS, p.second);
        Q_LS[key_state][LSi] += alpha * (r + gamma * max_next_LS - Q_LS[key_state][LSi]);

        double max_next_MA = -1e9;
        for (auto &p : Q_MA[next_state]) max_next_MA = std::max(max_next_MA, p.second);
        Q_MA[key_state][aMA] += alpha * (r + gamma * max_next_MA - Q_MA[key_state][aMA]);

        op_counts[selected_op]++;
        op_rewards[selected_op] += r;

        // ---------------- DECAY & COOLING ----------------
        eps_LS = std::max(eps_LS * eps_decay, eps_min);
        eps_MA = std::max(eps_MA * eps_decay, eps_min);
        SA_temp *= SA_Cooling;
        R2R_Threshold *= decay_R2R;
        TH_Threshold  *= decay_TH;

        // ---------------- LOG OUTPUT ----------------
        outfile << iteration << "\t" << LSi << "\t" << selected_op << "\t" << aMA << "\t"
                << cost_eff << "\t" << cost_div<< "\t" << best_eff << "\t"
                << best_div  << "\t" << delta << "\t" << r << "\t" << step_sec << "\n";

        //if (iteration % 10 == 0) {
            std::cout << "Iter " << iteration
                      << " | LS=" << LSi
                      << " | OP=" << selected_op
                      << " | MA=" << aMA
                      << " | Eff=" << new_eff
                      << " | Div=" << new_div
                      << " | BEST_Eff=" << best_eff
                      << " | BEST_Div=" << best_div
                      << " | Delta=" << delta
                      << " | Reward=" << r
                      << " | Epsilon_LS=" << eps_LS
                      << " | Epsilon_MA=" << eps_MA
                      << std::endl;
        //}
    }

    outfile.close();

    // --- Summary ---
    double runtime = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::filesystem::path summary_file = folder_path / ("TriLevel_Q_HH_CMCEE_" + instanceName + "_summary.txt");
    std::ofstream summary(summary_file);
    summary << "Runtime (sec): " << runtime << "\n";
    summary << "Best Efficiency: " << best_eff << "\n";
    summary << "Best Diversity: " << best_div << "\n";
    summary << "Accepted Moves: " << accepted_moves << "\n";
    summary << "Total Moves: " << total_moves << "\n";
    summary << "Acceptance Ratio: " << ((double)accepted_moves / std::max(1, total_moves)) << "\n";
    summary.close();
    check_best_solution();
    std::cout << "===================================================================\n";
    std::cout << "Tri-Level Q-Learning HH Finished.\n";
    std::cout << "Runtime: " << runtime << " sec | Best (Eff,Div)=(" << best_eff << "," << best_div << ")\n";
    std::cout << "Results saved to: " << file_path << "\n";
    std::cout << "===================================================================\n";

}
*/
class MultiStageHyperHeuristicFramework : public Hyper_heuristic {
public:
    // --------------------------------------------------
    // 1) Available Hyper-Heuristic IDs
    // --------------------------------------------------
    std::vector<int> hyperHeuristics = {1, 2, 3, 4, 5};

    // --------------------------------------------------
    // 2) Q-Learning Parameters
    // --------------------------------------------------
    const double alpha   = 0.1;  // Learning rate
    const double gamma   = 0.9;  // Discount factor
    const double epsilon = 0.2;  // Exploration rate

    // --------------------------------------------------
    // 3) Q-Table: Suppose 100 states x (#hyperHeuristics) actions
    // --------------------------------------------------
    std::vector<std::vector<double>> Q_table;

    // --------------------------------------------------
    // 4) Track improvements for each hyper-heuristic
    // --------------------------------------------------
    std::map<int, std::vector<double>> heuristicPerformance;

    // --------------------------------------------------
    // 5) Current state in [0..99]
    // --------------------------------------------------
    int currentState;

    // --------------------------------------------------
    // 6) Random number generation
    // --------------------------------------------------
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    // --------------------------------------------------
    // 7) A scale factor for improvement
    // --------------------------------------------------
    const double maxExpectedImprovement = 100.0;

    // --------------------------------------------------
    // 8) Convergence data structure
    // --------------------------------------------------
    struct ConvergenceRecord {
        int    iteration;
        int    heuristicID;
        double improvement;
        double timeTaken;
        double efficiencyBefore;
        double efficiencyAfter;
    };
    std::vector<ConvergenceRecord> convergenceData;

    // --------------------------------------------------
    // 9) Suppose f_cur is the current objective (efficiency)
    // --------------------------------------------------
    double f_cur = 0.0;

    //---------------------------------------------------
    // Additional references to solution/team, if needed
    // e.g., int** team;
    // ...
    //---------------------------------------------------

    public:
    // --------------------------------------------------
    // Constructor
    // --------------------------------------------------
    MultiStageHyperHeuristicFramework()
        : gen(std::random_device{}()), dis(0.0, 1.0), currentState(0)
    {
        initializeQTable();
        currentState = encodeState(0, 0);  // start in (feature1=0, feature2=0) => state=0
    }

    // --------------------------------------------------
    // encodeState(feature1, feature2) => integer [0..99]
    // --------------------------------------------------
    int encodeState(int feature1, int feature2) {
        // Both in [0..9]
        return feature1 * 10 + feature2;
    }

    // --------------------------------------------------
    // Initialize Q-table
    // --------------------------------------------------
    void initializeQTable() {
        int numStates  = 100;
        int numActions = static_cast<int>(hyperHeuristics.size());
        Q_table.assign(numStates, std::vector<double>(numActions, 0.0));

        // (Optional) partial print for debugging
        std::cout << "Initialized Q-table with " << numStates
                  << " states and " << numActions << " actions.\n";
        /*for (int s = 0; s < 5; ++s) {
            std::cout << "State " << s << ": ";
            for (int a = 0; a < numActions; ++a) {
                std::cout << Q_table[s][a] << " ";
            }
            std::cout << "\n";
        }*/
    }

    // --------------------------------------------------
    // Q-Learning: epsilon-greedy selection of one action
    // --------------------------------------------------
    int selectHyperHeuristic() {
        double r = dis(gen);

        int numActions = static_cast<int>(hyperHeuristics.size());
        if (r < epsilon) {
            // Exploration => pick random
            std::uniform_int_distribution<> dist_int(0, numActions - 1);
            int index = dist_int(gen);
            return hyperHeuristics[index];
        } else {
            // Exploitation => pick arg max Q
            int bestHeuristic = -1;
            double bestQ = -std::numeric_limits<double>::infinity();
            // find the index with the largest Q in Q_table[currentState]
            for (int i = 0; i < numActions; ++i) {
                double qv = Q_table[currentState][i];
                if (qv > bestQ) {
                    bestQ = qv;
                    bestHeuristic = hyperHeuristics[i];
                }
            }
            return bestHeuristic;
        }
    }

    // --------------------------------------------------
    // applyHyperHeuristic(heuristicID, max_time)
    // uses your provided methods
    // --------------------------------------------------
    void applyHyperHeuristic(int heuristicID, int max_time)  {
        try {
            switch (heuristicID) {
                case 1: HH_Choice_Function_Selection_CMCEE(max_time);       break;
                case 2: MAB_Selection_Hyperheuristic_CMCEE(max_time);       break;
                case 3: Greedy_Selection_Hyperheuristic_CMCEE(max_time);    break;
                case 4: Random_Selection_Hyperheuristic_CMCEE(max_time);    break;
                case 5: Q_Learning_Selection_Hyperheuristic_CMCEE(max_time);break;
                default:
                    std::cerr << "Invalid Hyper-Heuristic ID: " << heuristicID << "\n";
                    break;
            }
        }
        catch (const std::exception &e) {
            std::cerr << "Exception in Heuristic " << heuristicID << ": " << e.what() << "\n";
        }
        catch (...) {
            std::cerr << "Unknown exception in Heuristic " << heuristicID << "\n";
        }
    }

    // --------------------------------------------------
    // Q-Update
    // --------------------------------------------------
    void updateQTable(int actionID, double reward, int nextState) {
        // find index of 'actionID' in hyperHeuristics
        auto it = std::find(hyperHeuristics.begin(), hyperHeuristics.end(), actionID);
        if (it == hyperHeuristics.end()) {
            std::cerr << "Action ID not found in hyperHeuristics.\n";
            return;
        }
        int index = static_cast<int>(std::distance(hyperHeuristics.begin(), it));

        // max Q of nextState row
        double maxNextQ = *std::max_element(Q_table[nextState].begin(), Q_table[nextState].end());

        double oldQ = Q_table[currentState][index];
        double newQ = oldQ + alpha * (reward + gamma * maxNextQ - oldQ);
        Q_table[currentState][index] = newQ;

        // track improvement => store in heuristicPerformance
        heuristicPerformance[actionID].push_back(reward);
    }

    // --------------------------------------------------
    // get current objective value => f_cur
    // --------------------------------------------------
    double calculateCurrentEfficiency() {
        return f_cur;  // you might have logic that returns the best so far
    }

    // --------------------------------------------------
    // the main run loop
    // we do 'iterations' picks
    // --------------------------------------------------
    void run(int max_time, int iterations = 10) {
        // Clear old data
        convergenceData.clear();

        // Create output directory
        std::filesystem::path outDir("D:\\Result Hyper-heuristic Models\\");
        if (!std::filesystem::exists(outDir)) {
            try {
                std::filesystem::create_directories(outDir);
            } catch(...) {
                std::cerr << "Cannot create directory: " << outDir << "\n";
            }
        }

        // For each iteration
        for (int iter = 0; iter < iterations; ++iter) {
            std::cout << "Iteration " << (iter + 1) << ":\n";

            double effBefore = calculateCurrentEfficiency();

            // (A) select hyper-heuristic from Q
            int chosenHeuristic = selectHyperHeuristic();

            // (B) measure time
            auto startTime = std::chrono::steady_clock::now();

            // (C) apply
            std::cout << "  Applying Hyper-Heuristic ID " << chosenHeuristic << "\n";
            applyHyperHeuristic(chosenHeuristic, max_time);

            // end time
            auto endTime   = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(endTime - startTime).count();

            // (D) measure improvement
            double effAfter   = calculateCurrentEfficiency();
            double improvement= effAfter - effBefore;

            // define next state logic
            // e.g. increment the row, encode improvement
            int feature1 = (currentState / 10 + 1) % 10; // just for example
            // scale improvement by maxExpectedImprovement
            int feature2_val = static_cast<int>((improvement / maxExpectedImprovement) * 10);
            if (feature2_val < 0)  feature2_val = 0;
            if (feature2_val > 9)  feature2_val = 9;

            int nextState = encodeState(feature1, feature2_val);

            // reward is just 'improvement' in this example
            double reward = improvement;  // or some function
            updateQTable(chosenHeuristic, reward, nextState);

            // move state
            currentState = nextState;

            // console debug
            std::cout << "  Improvement: " << improvement
                      << ", nextState=" << currentState << "\n\n";

            // record data
            ConvergenceRecord rec;
            rec.iteration        = (iter+1);
            rec.heuristicID      = chosenHeuristic;
            rec.improvement      = improvement;
            rec.timeTaken        = elapsed;
            rec.efficiencyBefore = effBefore;
            rec.efficiencyAfter  = effAfter;
            convergenceData.push_back(rec);
        }

        // final analysis
        performStatisticalAnalysis();
    }

    // --------------------------------------------------
    // final analysis + save results
    // --------------------------------------------------
    void performStatisticalAnalysis() {
        std::cout << "\n--- Statistical Analysis (Heuristic Performance) ---\n";
        // sum improvements per heuristic
        for (auto &kv : heuristicPerformance) {
            int hID = kv.first;
            auto &scores = kv.second;
            if (scores.empty()) {
                std::cout << "Heuristic " << hID << ": No improvements.\n";
                continue;
            }
            double total = std::accumulate(scores.begin(), scores.end(), 0.0);
            double mean  = total / scores.size();
            double var   = 0.0;
            for (auto &val : scores) {
                var += (val - mean) * (val - mean);
            }
            var /= scores.size();
            std::cout << "Heuristic " << hID
                      << ": total=" << total
                      << ", mean="  << mean
                      << ", var="   << var << "\n";
        }

        // gather final objective from convergenceData
        if (convergenceData.empty()) {
            std::cout << "No convergence data.\n";
            return;
        }

        std::vector<double> allObj;
        for(auto &r : convergenceData) {
            allObj.push_back(r.efficiencyAfter);
        }

        double bestObj  = *std::max_element(allObj.begin(), allObj.end());
        double worstObj = *std::min_element(allObj.begin(), allObj.end());
        double sumObj   = std::accumulate(allObj.begin(), allObj.end(), 0.0);
        double avgObj   = sumObj / (double)allObj.size();

        std::cout << "\n--- Final Objective Stats ---\n";
        std::cout << "BestObj="  << bestObj
                  << ", WorstObj=" << worstObj
                  << ", AvgObj="   << avgObj  << "\n";

                  // Define the folder and file paths for saving results
        std::string folder_path = "D:/Result Hyper-heuristic Models/";

        std::string statsFile = folder_path + ("Multi_Stage_Hyper-heuristic_Model_" + instanceName +"Stats_Results.txt");
        // save stats
        std::ofstream ofs(statsFile, std::ios::app);
        if(!ofs.is_open()) {
            std::cerr<<"Error opening file: "<< statsFile<<"\n";
            return;
        }
        ofs << "\n=== Hyper-Heuristic Stats ===\n";
        for (auto &kv : heuristicPerformance) {
            int hID = kv.first;
            auto &sc = kv.second;
            if(sc.empty()) continue;
            double total = std::accumulate(sc.begin(), sc.end(), 0.0);
            double mean  = total / sc.size();
            double var   = 0.0;
            for(auto &vv : sc) {
                var += (vv - mean) * (vv - mean);
            }
            var /= sc.size();

            ofs<<"Heuristic " << hID
               <<": Total="  << total
               <<", Mean="   << mean
               <<", Var="    << var <<"\n";
        }
        ofs<<"BestObj=" << bestObj
           <<", WorstObj="<< worstObj
           <<", AvgObj="  << avgObj<<"\n";
        ofs.close();

        std::cout<<"Saved best/worst/avg objective to: "<< statsFile <<"\n";
    }
};
class MultiStageHyperHeuristicFramework1 : public Hyper_heuristic {
public:
    // Low-level Hyper-heuristics IDs
    std::vector<int> heuristics = {1, 2, 3, 4, 5};

    // MAB data structures
    std::map<int, int>    heuristicUsageCount;
    std::map<int, double> heuristicTotalReward;
    std::map<int, double> heuristicAvgReward;

    // Epsilon-greedy parameter
    double epsilon = 0.1;

    // Stage probabilities
    double pStayRandomDescent = 0.50;
    double pGoGreedy          = 0.25;

    enum Stage { GREEDY_STAGE, RANDOM_DESCENT_STAGE };

    struct ConvergenceRecord {
        int    iteration;
        int    chosenHeuristic;
        double oldEff;
        double newEff;
        double reward;
        double timeTaken;
        Stage  stage;
    };
    std::vector<ConvergenceRecord> convergenceData;

    std::mt19937 gen;
    std::uniform_real_distribution<> distReal;

    Stage currentStage = GREEDY_STAGE;
    double f_cur = 0.0;       // current objective value
    int** team = nullptr;     // placeholder for solution

    // Constructor
    MultiStageHyperHeuristicFramework1()
        : gen(std::random_device{}()), distReal(0.0, 1.0)
    {
        for (int h : heuristics) {
            heuristicUsageCount[h]   = 0;
            heuristicTotalReward[h]  = 0.0;
            heuristicAvgReward[h]    = 0.0;
        }
    }

    // ε-greedy MAB
    int multiArmedBanditSelectHeuristic() {
        double r = distReal(gen);
        if (r < epsilon) {
            std::uniform_int_distribution<> d(0, (int)heuristics.size() - 1);
            return heuristics[d(gen)];
        } else {
            int bestH = -1;
            double bestVal = -std::numeric_limits<double>::infinity();
            for (int h : heuristics) {
                double val = heuristicAvgReward[h];
                if (val > bestVal) {
                    bestVal = val;
                    bestH   = h;
                }
            }
            return bestH;
        }
    }

    void updateMAB(int chosenH, double reward) {
        heuristicUsageCount[chosenH]++;
        heuristicTotalReward[chosenH] += reward;
        heuristicAvgReward[chosenH] =
            heuristicTotalReward[chosenH] / heuristicUsageCount[chosenH];
    }

    // Call heuristics and update f_cur
    void applyHeuristic(int heuristicID, int max_time) {
        switch (heuristicID) {
            case 1: HH_Choice_Function_Selection_CMCEE(max_time); break;
            case 2: MAHH_Selection_CMCEE(max_time);               break;
            case 3: Greedy_Selection_Hyperheuristic_CMCEE(max_time); break;
            case 4: Random_Selection_Hyperheuristic_CMCEE(max_time); break;
            case 5: Q_Learning_Selection_Hyperheuristic_CMCEE(max_time); break;
            default:
                std::cerr << "Unknown heuristic: " << heuristicID << "\n";
                break;
        }
        // Update objective after heuristic modifies 'team'
        //objective_Function(team);
        f_cur;
    }

    // Main run
    void run(int max_time, int iterations = 20) {
        std::filesystem::path folder("D:/MAB_MultiStage_Results/");
        if(!std::filesystem::exists(folder)) {
            std::filesystem::create_directories(folder);
        }

        std::ofstream ofsLog((folder / "run_log.txt").string());
        std::ofstream ofsCSV((folder / "convergence.csv").string());
        ofsCSV << "Iteration,Stage,Heuristic,OldEff,NewEff,Reward,Time\n";

        convergenceData.clear();

        // Initialize solution and f_cur
        //initialize_solution();   // assume you have this
        //f_cur = objective_Function(team);
        double oldEff = f_cur;

        auto totalStart = std::chrono::steady_clock::now();

        for (int iter = 1; iter <= iterations; ++iter) {
            auto now = std::chrono::steady_clock::now();
            double elapsedRun = std::chrono::duration<double>(now - totalStart).count();
            if (elapsedRun >= max_time) break;

            std::cout << "Iteration " << iter
                      << " | Stage=" << (currentStage==GREEDY_STAGE?"GREEDY":"RANDOM_DESCENT")
                      << " | f_cur=" << f_cur << "\n";

            int chosenH = multiArmedBanditSelectHeuristic();

            auto itStart = std::chrono::steady_clock::now();
            applyHeuristic(chosenH, max_time);
            auto itEnd = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(itEnd - itStart).count();

            double newEff = f_cur;
            double improvement = newEff - oldEff;
            double reward = std::max(0.0, improvement); // non-negative reward
            updateMAB(chosenH, reward);

            ConvergenceRecord rec{iter, chosenH, oldEff, newEff, reward, elapsed, currentStage};
            convergenceData.push_back(rec);

            ofsLog << "Iter="<<iter
                   <<", Stage="<<(currentStage==GREEDY_STAGE?"GREEDY":"RANDOM_DESCENT")
                   <<", Heuristic="<<chosenH
                   <<", oldEff="<<oldEff
                   <<", newEff="<<newEff
                   <<", Reward="<<reward
                   <<", time="<<elapsed << "\n";

            ofsCSV << iter << ","
                   << (currentStage==GREEDY_STAGE?"GREEDY":"RANDOM_DESCENT") << ","
                   << chosenH << ","
                   << oldEff << ","
                   << newEff << ","
                   << reward << ","
                   << elapsed << "\n";

            bool improved = (improvement > 0.0);
            if (currentStage == GREEDY_STAGE) {
                if (!improved) currentStage = RANDOM_DESCENT_STAGE;
            } else {
                if (!improved) {
                    double r = distReal(gen);
                    if (r < pStayRandomDescent) {
                        // stay
                    } else if (r < (pStayRandomDescent + pGoGreedy)) {
                        currentStage = GREEDY_STAGE;
                    }
                }
            }

            oldEff = newEff;
        }

        ofsLog.close();
        ofsCSV.close();

        auto totalEnd = std::chrono::steady_clock::now();
        double totalTime = std::chrono::duration<double>(totalEnd - totalStart).count();
        std::cout << "Finished in " << totalTime << " seconds.\n";

        analyzeAndSave();
    }

    void analyzeAndSave() {
        std::cout << "\n--- MAB Stats ---\n";
        for (int h : heuristics) {
            std::cout <<"Heuristic "<< h
                      <<" | usage="<< heuristicUsageCount[h]
                      <<" | avgReward="<< heuristicAvgReward[h] <<"\n";
        }

        if (convergenceData.empty()) return;
        std::vector<double> finalObjs;
        for (auto &r : convergenceData) finalObjs.push_back(r.newEff);

        double bestObj  = *std::max_element(finalObjs.begin(), finalObjs.end());
        double worstObj = *std::min_element(finalObjs.begin(), finalObjs.end());
        double avgObj   = std::accumulate(finalObjs.begin(), finalObjs.end(), 0.0) / finalObjs.size();

        std::cout << "\n--- Final Obj Stats ---\n";
        std::cout <<"Best="<< bestObj <<", Worst="<< worstObj <<", Avg="<< avgObj <<"\n";

        std::ofstream ofsStats("D:/MAB_MultiStage_Results/Stats.txt", std::ios::app);
        ofsStats << "\n--- MAB MultiStage Stats ---\n";
        for (int h : heuristics) {
            ofsStats << "Heuristic "<< h
                     <<": usage="<< heuristicUsageCount[h]
                     <<", totalReward="<< heuristicTotalReward[h]
                     <<", avgReward="  << heuristicAvgReward[h] <<"\n";
        }
        ofsStats <<"BestObj="<< bestObj
                 <<", WorstObj="<< worstObj
                 <<", AvgObj="<< avgObj <<"\n";
        ofsStats.close();
    }
};

// Enumeration for menu options
enum MenuOptions {
    EXIT_OPTION = 0,
    // Existing Hyper-Heuristic Methods
    Q_LEARNING_SELECTION = 1,
    HH_CHOICE_FUNCTION_SELECTION,
    RANDOM_SELECTION,
    MAB_SELECTION,
    MAHH_SELECTION,
    GREEDY_SELECTION,
    COMPARE_ACCEPTANCE_CRITERIA,
    SSHH_SELECTION,
    ADAPTIVE_SSHH_SELECTION,
    MULTISTAGE_HYPERHEURISTIC,
    // Single Optimization Algorithms
    ITERATED_LOCAL_SEARCH = 12,
    SIMULATED_ANNEALING,
    TABU_SEARCH,
    FEASIBLE_LOCAL_SEARCH,
    INFEASIBLE_LOCAL_SEARCH,
    MEMETIC_ALGORITHM,
    GREAT_DELUGE,
    VARIABLE_NEIGHBORHOOD_SEARCH,
    Late_Acceptance,
    Guided_Local,
    HARMONY_SEARCH,
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
        else if (algorithm_name == "VNS")  { fitness  = variable_neighborhood_search(time_limit); }
        else if (algorithm_name == "LAHC") { solution = late_acceptance_hill_climbing();    fitness = fbest; }
        else if (algorithm_name == "GLS")  { fitness  = guided_local_search(); }
        else if (algorithm_name == "HSA")  {
            team = improvise_New_HM();
            int idx = min_func(w_eff, num_team);
            f_cur = w_eff[idx];
            fitness = f_cur;
        }
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

            case VARIABLE_NEIGHBORHOOD_SEARCH:
                std::cout << "Running Variable Neighborhood Search (VNS)..." << std::endl;
                algorithm_name = "VNS";
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
            case HARMONY_SEARCH:
                std::cout << "Running Harmony_Search..." << std::endl;
                algorithm_name = "HSA";
                break;
            default:
                std::cout << "Invalid single optimization algorithm choice." << std::endl;
                return;
        }

        execute_algorithm(31, algorithm_name); // Example: 31 runs
    }

// ===========================================================
// Parallel execution function for individual algorithms (GLOBAL VARIABLES)
// ===========================================================
void run_parallel_algorithm(const std::string& algorithm_name,
                            const std::string& datasetFile,
                            const std::string& instanceName,
                            const std::string& outputDir,
                            int max_time)
{
    Hyper_heuristic H;

    {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::cout << "[INFO] Starting " << algorithm_name
                  << " on instance " << instanceName << "...\n";
    }
    ffbest = std::numeric_limits<int>::min();
    double stat_best  = std::numeric_limits<double>::lowest();
    double stat_avg   = 0.0;
    double stat_worst = std::numeric_limits<double>::max();
    double avg_time   = 0.0;
    // Unique seed per thread
    srand(static_cast<unsigned int>(time(0)) + std::hash<std::thread::id>{}(std::this_thread::get_id()));

    H.Parameters();
    H.initialization(datasetFile);
    H.generate_initialrandom();

    auto start = std::chrono::high_resolution_clock::now();

    int fitness = std::numeric_limits<int>::min();
    if      (algorithm_name == "ILS")  { H.iterated_local_search();   fitness = fbest; }
    else if (algorithm_name == "SA")   { H.simulated_annealing();     fitness = fbest; }
    else if (algorithm_name == "TS")   { H.fits();                    fitness = f_best_inn; }
    else if (algorithm_name == "GD")   { H.great_deluge_algorithm();  fitness = fbest; }
    else if (algorithm_name == "LAHC") { H.late_acceptance_hill_climbing(); fitness = fbest; }
    else {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::cerr << "[ERROR] Unknown algorithm: " << algorithm_name << "\n";
        return;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // ---------------------------------------------------------------------
    //  GLOBAL-BEST UPDATE  (same as sequential version, but protected)
    // ---------------------------------------------------------------------
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::lock_guard<std::mutex> guard(global_data_mutex);

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
    }

    // ---------------------------------------------------------------------
    //  LOCAL SNAPSHOT (for independent log per algorithm)
    // ---------------------------------------------------------------------
    double local_eff = best_eff;
    double local_div = best_div;

    std::string resultsFilePath = outputDir + "Parallel_" + algorithm_name + "_" + instanceName + ".txt";
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::ofstream resultsFile(resultsFilePath, std::ios::app);
        if (resultsFile.is_open()) {
            resultsFile << "Algorithm\tBest Efficiency\tBest Diversity\tTime Taken (seconds)\n";
            resultsFile << algorithm_name << "\t" << local_eff << "\t" << local_div << "\t" << elapsed << "\n";
            resultsFile.close();
        }

        std::cout << "------------------------------------------------\n";
        std::cout << "[Thread: " << algorithm_name << "] Instance: " << instanceName << "\n";
        std::cout << "Best Efficiency: " << local_eff << "\n";
        std::cout << "Best Diversity : " << local_div << "\n";
        std::cout << "Time Taken (s) : " << elapsed << "\n";
        std::cout << "Saved: " << resultsFilePath << "\n";
        std::cout << "------------------------------------------------\n\n";
    }
        H.check_best_solution();

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
/*
int main(int argc, char *argv[]) {
    srand(static_cast<unsigned int>(time(0)));

    std::string datasetDir   = "D:/Datasets/";
    std::string resultsDir   = "D:/Datasets/RESULTS_OF_HH_MODELS/";

    // Collect dataset files
    std::vector<std::string> datasetFiles;
    for (const auto& entry : fs::directory_iterator(datasetDir)) {
        if (entry.is_regular_file()) datasetFiles.push_back(entry.path().string());
    }
    if (datasetFiles.empty()) {
        std::cerr << "No dataset files found in: " << datasetDir << "\n";
        return 0;
    }

    int max_time = 60;
    Hyper_heuristic H;

    std::cout << "================= MAIN MENU =================\n";
    std::cout << "1: Run Single Hyper-Heuristic (Sequential)\n";
    std::cout << "2: Run Parallel Metaheuristic Algorithms (SA, ILS, TS, GD, LAHC)\n";
    std::cout << "Enter your choice (0 to exit): ";
    int mode; std::cin >> mode;
    if (mode == 0) return 0;

    // =====================================================
    // MODE 1: SINGLE HYPER-HEURISTIC (Sequential Execution)
    // =====================================================
    if (mode == 1) {
        std::cout << "\nSelect ONE Hyper-Heuristic to run on ALL instances:\n";
        std::cout << "1: Q-Learning Selection HH\n";
        std::cout << "2: Choice Function HH\n";
        std::cout << "3: Random Selection HH\n";
        std::cout << "4: MAB Selection HH\n";
        std::cout << "5: MAHH Selection HH\n";
        std::cout << "6: Greedy Selection HH\n";
        std::cout << "7: MultiStage Hyper-Heuristic Framework\n";
        std::cout << "8: Execute Single Optimization Algorithm\n";
        std::cout << "Enter your choice: ";
        int choice; std::cin >> choice;

        int single_choice = -1;
        if (choice == 8) {
            std::cout << "Select Optimization Algorithm:\n";
            std::cout << "11: ILS\n12: SA\n13: TS\n14: GD\n15: LAHC\n";
            std::cin >> single_choice;
        }

        for (size_t idx = 0; idx < datasetFiles.size(); idx++) {
            std::string current_file = datasetFiles[idx];
            fs::path p(current_file);
            std::string instanceName = p.stem().string();

            std::string resultsFilePath = resultsDir + "Results_" + instanceName + ".txt";
            std::ofstream resultsFile(resultsFilePath);
            resultsFile << "Hyper-Heuristic Method\tBest Efficiency\tBest Diversity\tTime Taken (seconds)\n";

            H.Parameters();
            H.initialization(current_file);

            std::cout << "\n=================================================\n";
            std::cout << "Processing Instance " << (idx + 1) << "/" << datasetFiles.size() << "\n";
            std::cout << "Instance Name: " << instanceName << "\n";
            std::cout << "-------------------------------------------------\n";
            std::cout << "num_node= " << num_node
                      << ", num_team= " << num_team
                      << ", num_each_t= " << num_each_t
                      << ", min_div= " << min_div << "\n";
            std::cout << "-------------------------------------------------\n";

            auto start = std::chrono::high_resolution_clock::now();

            switch (choice) {
                case 1: H.Q_Learning_Selection_Hyperheuristic_CMCEE(max_time); break;
                case 2: H.HH_Choice_Function_Selection_CMCEE(max_time); break;
                case 3: H.Random_Selection_Hyperheuristic_CMCEE(max_time); break;
                case 4: H.MAB_Selection_Hyperheuristic_CMCEE(max_time); break;
                case 5: H.MAHH_Selection_CMCEE(max_time); break;
                case 6: H.Greedy_Selection_Hyperheuristic_CMCEE(max_time); break;
                case 7: {
                    MultiStageHyperHeuristicFramework1 HH1;
                    HH1.run(max_time, 20);
                    break;
                }
                case 8:
                    H.runSingleOptimizationAlgorithm(single_choice, max_time);
                    break;
                default:
                    std::cout << "Invalid selection.\n"; continue;
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();

            resultsFile << "Method_" << choice << "\t"
                        << best_eff << "\t" << best_div << "\t" << elapsed << "\n";
            resultsFile.close();

            std::cout << "Best Efficiency: " << best_eff
                      << ", Best Diversity: " << best_div
                      << ", Time: " << elapsed << "s\n";
            std::cout << "Saved: " << resultsFilePath << "\n";
            std::cout << "=================================================\n";
        }
    }

    // =====================================================
    // MODE 2: PARALLEL EXECUTION OF MULTIPLE METAHEURISTICS
    // =====================================================
    else if (mode == 2) {
        std::vector<std::string> algorithms = {"SA", "ILS", "TS", "GD", "LAHC"};

        for (const auto& dataset : datasetFiles) {
            fs::path p(dataset);
            std::string instanceName = p.stem().string();

            std::cout << "\n=================================================\n";
            std::cout << "Parallel Run: " << instanceName << "\n";
            std::cout << "-------------------------------------------------\n";

            std::vector<std::thread> threads;
            for (const auto& algo : algorithms) {
                threads.emplace_back(run_parallel_algorithm,
                                     algo, dataset, instanceName, resultsDir, max_time);
            }

            for (auto& t : threads)
                if (t.joinable()) t.join();

            std::cout << "All algorithms completed for: " << instanceName << "\n";
            std::cout << "=================================================\n\n";
        }
    }

    else {
        std::cout << "Invalid mode selection.\n";
    }

    free_memory();
    std::cout << "All processing completed successfully.\n";
    return 0;
}
*/
/*//corrected for single algorithm run
int main(int argc, char *argv[]) {
    // Initialize random seed
    srand(static_cast<unsigned int>(time(0)));

    // Directory containing dataset files
    std::string datasetDir = "D:/Datasets/Parameters test/";
    std::string datasetDirD1 = "D:\\Datasets\\RESULTS_OF_HH_MODELSParameters\\";

    // Collect all dataset files matching the pattern
    std::vector<std::string> datasetFiles;
    for (const auto &entry : fs::directory_iterator(datasetDir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            datasetFiles.push_back(entry.path().string());
        }
    }

    // Basic initialization
    Hyper_heuristic H;

    // Set time limit or max_time as needed
    int max_time = 120; // Example maximum time

    // Process each dataset file
    // Display menu for the user
    //std::string resultsFilePath = datasetDirD1 + "Results_" + instanceName + ".txt";
    //std::ofstream resultsFile(resultsFilePath);


    // -------- choose ONE algorithm once ----------
    std::cout << "Select ONE method to run on ALL instances:\n";
    std::cout << "1: Q-Learning Selection Hyperheuristic CMCEE\n";
    std::cout << "2: HH_Choice_Function_Selection_CMCEE\n";
    std::cout << "3: Random_Selection_Hyperheuristic_CMCEE\n";
    std::cout << "4: MAB_Selection_Hyperheuristic_CMCEE\n";
    std::cout << "5: MAHH_Selection_CMCEE\n";
    std::cout << "6: Greedy_Selection_Hyperheuristic_CMCEE\n";
    std::cout << "7: Compare Acceptance Criteria\n";
    std::cout << "8: Sequence based Selection Hyperheuristic CMCEE\n";
    std::cout << "9: Adaptive Sequence based Selection Hyperheuristic CMCEE\n";
    std::cout << "10: MultiStage HyperHeuristic Framework\n";
    std::cout << "11: Tri-Level Hyper-Heuristic (QL + MAB + QL)\n";
    std::cout << "12: Execute Single Optimization Algorithm\n";
    std::cout << "Enter your choice (0 to exit): ";

    int choice = 0;
    std::cin >> choice;
    if (choice == 0) {
        std::cout << "Exit.\n";
        return 0;
    }

    int single_choice = -1;
    if (choice == 12) {
        std::cout << "Select ONE Optimization Algorithm to run on ALL instances:\n";
        std::cout << "12: Iterated Local Search (ILS)\n";
        std::cout << "13: Simulated Annealing (SA)\n";
        std::cout << "14: Tabu Search (TS)\n";
        std::cout << "15: Feasible Local Search (FLS)\n";
        std::cout << "16: Infeasible Local Search (IFLS)\n";
        std::cout << "17: Memetic Algorithm (MA)\n";
        std::cout << "18: Great Deluge Algorithm (GD)\n";
        std::cout << "19: Variable Neighborhood Search (VNS)\n";
        std::cout << "20: Late Acceptance Hill Climbing (LAHC)\n";
        std::cout << "21: Guided Local Search (GLS)\n";
        std::cout << "22: Harmony Search (HS)\n";
        std::cout << "Enter your choice (12-22): ";
        std::cin >> single_choice;
        if (single_choice < 12 || single_choice > 22) {
            std::cout << "Invalid selection. Exiting.\n";
            return 0;
        }
    }

    // -------- run the selected method on EACH instance datasetFiles.size()----------
   // datasetFiles.size()
    for (size_t idx = 0; idx < datasetFiles.size() ; idx++) {
        std::string current_file = datasetFiles[idx];
        fs::path p(current_file);
        std::string rawInstanceName = p.filename().string();
        instanceName = parseInstanceName(rawInstanceName);

        if (instanceName.empty()) {
            std::cerr << "[WARNING] Failed to parse instance name for file: " << rawInstanceName << std::endl;
            continue;
        }

        std::cout << "\nProcessing Instance " << (idx + 1) << " of " << datasetFiles.size() << ":\n";
        std::cout << "Instance Name: " << instanceName << std::endl;

        // Create results file for this instance
        std::string resultsFilePath = datasetDirD1 + "Results_" + instanceName + ".txt";
        std::ofstream resultsFile(resultsFilePath); // overwrite per run
        if (!resultsFile) {
            std::cerr << "Error opening results file for writing: " << resultsFilePath << std::endl;
            continue;
        }
        // Header
        resultsFile << "Hyper-Heuristic Method\tBest Efficiency\tBest Diversity\tTime Taken (seconds)\n";

        // Load instance
        H.Parameters();
        H.initialization(current_file);

        std::cout << "num_node= " << num_node << "  "
                  << "num_team= " << num_team
                  << ",num_each_t= " << num_each_t << "  "
                  << "min_div= " << min_div << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;

            switch (choice) {

                case 1:
                    H.Q_Learning_Selection_Hyperheuristic_CMCEE(max_time);
                    resultsFile << "Q_Learning_Selection_Hyperheuristic_CMCEE\t"
                    << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

               case 2:
                   H.HH_Choice_Function_Selection_CMCEE(max_time);
                   resultsFile << "HH_Choice_Function_Selection_CMCEE\t"
                       << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                   break;
               case 3:
                   H.Random_Selection_Hyperheuristic_CMCEE(max_time);
                   resultsFile << "Random_Selection_Hyperheuristic_CMCEE\t"
                    << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                   break;
               case 4:
                   H.MAB_Selection_Hyperheuristic_CMCEE(max_time);
                   resultsFile << "MAB_Selection_Hyperheuristic_CMCEE\t"
                    << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                   break;
               case 5:
                   H.MAHH_Selection_CMCEE(max_time);
                   resultsFile << "MAHH_Selection_CMCEE\t"
                    << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                   break;
               case 6:
                   H.Greedy_Selection_Hyperheuristic_CMCEE(max_time);
                   resultsFile << "Greedy_Selection_Hyperheuristic_CMCEE\t"
                    << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                   break;
               case 7: {
                   double optimal_value = 500.0;
                   H.compare_acceptance_criteria(optimal_value);
                   resultsFile << "Compare_Acceptance_Criteria\t"
                    << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                   break;
                  }
               case 8:
                  H.SSHH_Selection_Hyperheuristic_CMCEE(max_time);
                  resultsFile << "SSHH_Selection_Hyperheuristic_CMCEE\t"
                    << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                  break;
               case 9:
                   H.Adaptive_SSHH_Selection_Hyperheuristic_CMCEE(max_time);
                   resultsFile << "Adaptive_SSHH_Selection_Hyperheuristic_CMCEE\t"
                    << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                   break;
               case 10: {
                   double optimal_value = 500.0;
                   MultiStageHyperHeuristicFramework1 HH1;
                   HH1.run(max_time, 20);
                   resultsFile << "MultiStage_HyperHeuristic_Framework\t"
                    << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                   break;
                  }
              case 11:
                 H.TriLevel_HH_Qlearning_CMCEE(max_time);
                 resultsFile << "TriLevel_HH_Qlearning_CMCEE\t"
                    << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                 break;
              case 12:
                   H.runSingleOptimizationAlgorithm(single_choice, max_time);
                   resultsFile << "Single_Optimization_Algorithm_" << single_choice << "\t"
                    << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                   break;
              default:
                   std::cout << "Invalid choice. Please try again." << std::endl;
                   break;
        }


            // Flush the output buffer to ensure data is written to the file
            resultsFile.flush();

        // Close the results file for the current instance
        resultsFile.close();
        std::cout << "Saved: " << resultsFilePath << std::endl;
        std::cout<<"\n";
        std::cout << "------------------------------------------------\n";
        }

    free_memory();
    return 0;
}
*/
/*
int main(int argc, char *argv[]) {
    // Initialize random seed
    srand(static_cast<unsigned int>(time(0)));

    // Directories
    std::string datasetDir = "D:/Datasets/";
    std::string datasetDirD1 = "D:\\Datasets\\RESULTS_OF_HH_MODELS\\";

    // Ensure the results directory exists
    if (!fs::exists(datasetDirD1)) {
        try {
            fs::create_directories(datasetDirD1);
            std::cout << "Created results directory: " << datasetDirD1 << std::endl;
        }
        catch (const fs::filesystem_error& e) {
            std::cerr << "Error creating results directory: " << e.what() << std::endl;
            return 1;
        }
    }

    // Collect all dataset files
    std::vector<std::string> datasetFiles;
    for (const auto &entry : fs::directory_iterator(datasetDir)) {
        if (entry.is_regular_file()) {
            datasetFiles.push_back(entry.path().string());
        }
    }

    if (datasetFiles.empty()) {
        std::cerr << "No dataset files found in directory: " << datasetDir << std::endl;
        return 1;
    }

    std::cout << "[INFO] Found " << datasetFiles.size() << " dataset files.\n";

    // Initialize Hyper_heuristic instance
    Hyper_heuristic H;
    // Set maximum time (in seconds) for each method
    int max_time = 120; // Example: 60 seconds

    // Initialize the combined results file
    std::string combinedResultsFilePath = datasetDirD1 + "Combined_Results178.txt";
    std::ofstream combinedResultsFile(combinedResultsFilePath, std::ios::out);
    if (!combinedResultsFile) {
        std::cerr << "Error opening combined results file for writing: " << combinedResultsFilePath << std::endl;
        return 1;
    }
    // Write headers to the combined results file
    combinedResultsFile << "Instance\tmin_div\t"
                        << "QL_best_eff\tQL_best_div\tQL_Taken_time\tQL_average_objective\tQL_worst_objective\t"
                        << "CF_best_eff\tCF_best_div\tCF_Taken_time\tCF_average_objective\tCF_worst_objective\t"
                        << "RS_best_eff\tRS_best_div\tRS_Taken_time\tRS_average_objective\tRS_worst_objective\t"
                        << "MAB_best_eff\tMAB_best_div\tMAB_Taken_time\tMAB_average_objective\tMAB_worst_objective\t"
                        //<< "MAHH_best_eff\tMAHH_best_div\tMAHH_Taken_time\tMAHH_average_objective\tMAHH_worst_objective\t"
                       // << "Greedy_best_eff\tGreedy_best_div\tGreedy_Taken_time\tGreedy_average_objective\tGreedy_worst_objective\t"
                       // << "Compare_best_eff\tCompare_best_div\tCompare_Taken_time\tCompare_average_objective\tCompare_worst_objective\t"
                        // << "SSHH_best_eff\tSSHH_best_div\tSSHH_Taken_time\tSSHH_average_objective\tSSHH_worst_objective\t" // Uncomment if implemented
                        // << "Adaptive_SSHH_best_eff\tAdaptive_SSHH_best_div\tAdaptive_SSHH_Taken_time\tAdaptive_SSHH_average_objective\tAdaptive_SSHH_worst_objective\t" // Uncomment if implemented
                        // << "MultiStage_best_eff\tMultiStage_best_div\tMultiStage_Taken_time\tMultiStage_average_objective\tMultiStage_worst_objective\t" // Uncomment if implemented
                        << "\n";
    // Write headers to the combined results file
    //combinedResultsFile << "Instance\tmin_div\t"
                        //<< "QL_best_eff\tQL_best_div\tQL_Taken_time\t"
                       // << "CF_best_eff\tCF_best_div\tCF_Taken_time\t"
                      //  << "RS_best_eff\tRS_best_div\tRS_Taken_time\t"
                       // << "MAB_best_eff\tMAB_best_div\tMAB_Taken_time\t"
                       // << "MAHH_best_eff\tMAHH_best_div\tMAHH_Taken_time\t"<<"\n";
                       // << "Greedy_best_eff\tGreedy_best_div\tGreedy_Taken_time\t"
                       // << "Compare_best_eff\tCompare_best_div\tCompare_Taken_time\t"
                        //<< "SSHH_best_eff\tSSHH_best_div\tSSHH_Taken_time\t"
                       // << "Adaptive_SSHH_best_eff\tAdaptive_SSHH_best_div\tAdaptive_SSHH_Taken_time\t"
                       // << "MultiStage_best_eff\tMultiStage_best_div\tMultiStage_Taken_time\n";*/

    // Optional: Write a hardcoded data line for testing

    //combinedResultsFile << "TestInstance\t100\t80.5\t75.3\t30.2\t78.0\t73.5\t29.8\t82.1\t76.4\t30.0\t79.5\t74.0\t29.9\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n";
  //*  std::cout << "Wrote a test data line to the results file.\n";

//datasetFiles.size()
    // Process each dataset file
 /*  for (size_t idx = 63; idx < 81 ; idx++) {
        std::string current_file = datasetFiles[idx];
        fs::path p(current_file);
        std::string rawInstanceName = p.filename().string();
        std::string instanceName = parseInstanceName(rawInstanceName);

        if (instanceName.empty()) {
            std::cerr << "[WARNING] Failed to parse instance name for file: " << rawInstanceName << std::endl;
            continue;
        }

        std::cout << "\nProcessing Instance " << (idx + 1) << " of " << datasetFiles.size() << ":\n";
        std::cout << "Instance Name: " << instanceName << std::endl;

        // Initialize results for the current instance by resetting global variables
        best_eff = 0.0;
        best_div = 0.0;
        time_taken = 0.0;

        // Initialize hyper-heuristic parameters and setup
        H.Parameters();
        H.initialization(current_file);
        std::cout << "num_node= " << num_node << "  "
                      << "num_team= " << num_team
                      << ",num_each_t=" << num_each_t << "  "
                      << "min_div= " << min_div << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << std::endl << std::endl;
        //H.generate_initial();
        //H.objective_Function(team);
        //H.display(team);
        cout<<"\n\n";
        // Variables to store results for each method
        //double QLbest_eff = 0.0, QLbest_div = 0.0, QLtime_taken = 0.0;
       // double CFbest_eff = 0.0, CFbest_div = 0.0, CFtime_taken = 0.0;
       // double RSbest_eff = 0.0, RSbest_div = 0.0, RStime_taken = 0.0;
        //double MABbest_eff = 0.0, MABbest_div = 0.0, MABtime_taken = 0.0;
        /*double MAHHbest_eff = 0.0, MAHHbest_div = 0.0, MAHHtime_taken = 0.0;
        double Greedybest_eff = 0.0, Greedybest_div = 0.0, Greedytime_taken = 0.0;
        double Comparebest_eff = 0.0, Comparebest_div = 0.0, Comparetime_taken = 0.0;
        double SSHHbest_eff = 0.0, SSHHbest_div = 0.0, SSHHtime_taken = 0.0;
        double Adaptive_SSHHbest_eff = 0.0, Adaptive_SSHHbest_div = 0.0, Adaptive_SSHHtime_taken = 0.0;
        double MultiStagebest_eff = 0.0, MultiStagebest_div = 0.0, MultiStagetime_taken = 0.0;*/
        // Variables to store results for each method
   /*     double QLbest_eff = 0.0, QLbest_div = 0.0, QLtime_taken = 0.0, QLaverage_objective = 0.0, QLworst_objective = 0.0;
        double CFbest_eff = 0.0, CFbest_div = 0.0, CFtime_taken = 0.0, CFaverage_objective = 0.0, CFworst_objective = 0.0;
        double RSbest_eff = 0.0, RSbest_div = 0.0, RStime_taken = 0.0, RSaverage_objective = 0.0, RSworst_objective = 0.0;
        double MABbest_eff = 0.0, MABbest_div = 0.0, MABtime_taken = 0.0, MABaverage_objective = 0.0, MABworst_objective = 0.0;
        double Greedybest_eff = 0.0, Greedybest_div = 0.0, Greedytime_taken = 0.0, Greedyaverage_objective = 0.0, Greedyworst_objective = 0.0;
        double MAHHbest_eff = 0.0, MAHHbest_div = 0.0, MAHHtime_taken = 0.0, MAHHaverage_objective = 0.0, MAHHworst_objective = 0.0;
        /*double Greedybest_eff = 0.0, Greedybest_div = 0.0, Greedytime_taken = 0.0, Greedyaverage_objective = 0.0, Greedyworst_objective = 0.0;
        double Comparebest_eff = 0.0, Comparebest_div = 0.0, Comparetime_taken = 0.0, Compareaverage_objective = 0.0, Compareworst_objective = 0.0;*/
        // Run all hyper-heuristic methods

       /* // 1. Q_Learning_Selection_Hyperheuristic_CMCEE
        H.Q_Learning_Selection_Hyperheuristic_CMCEE(max_time);
        QLbest_eff = best_eff;
        QLaverage_objective = average_objective;
        QLworst_objective = worst_objective;
        QLbest_div = best_div;
        QLtime_taken = time_taken;
        std::cout << "QL Results: Eff=" << QLbest_eff
                  << ", Div=" << QLbest_div
                  << ", QLaverage_objective=" << QLaverage_objective
                  << ", QLworst_objective=" << QLworst_objective
                  << ", Time=" << QLtime_taken << "s\n";

        // 2. HH_Choice_Function_Selection_CMCEE
        H.HH_Choice_Function_Selection_CMCEE(max_time);
        CFbest_eff = best_eff;
        CFaverage_objective = average_objective;
        CFworst_objective = worst_objective;
        CFbest_div = best_div;
        CFtime_taken = time_taken;
        std::cout << "CF Results: Eff=" << CFbest_eff
                  << ", Div=" << CFbest_div
                  << ", CFaverage_objective=" << CFaverage_objective
                  << ", CFworst_objective=" << CFworst_objective
                  << ", Time=" << CFtime_taken << "s\n";

        // 3. Random_Selection_Hyperheuristic_CMCEE
        H.Random_Selection_Hyperheuristic_CMCEE(max_time);
        RSbest_eff = best_eff;
        RSbest_div = best_div;
        RSaverage_objective = average_objective;
        RSworst_objective = worst_objective;
        RStime_taken = time_taken;
        std::cout << "RS Results: Eff=" << RSbest_eff
                  << ", Div=" << RSbest_div
                  << ", RSaverage_objective=" << RSaverage_objective
                  << ", RSworst_objective=" << RSworst_objective
                  << ", Time=" << RStime_taken << "s\n";

        // 4. MAB_Selection_Hyperheuristic_CMCEE
        H.MAB_Selection_Hyperheuristic_CMCEE(max_time);
        MABbest_eff = best_eff;
        MABbest_div = best_div;
        MABaverage_objective = average_objective;
        MABworst_objective = worst_objective;
        MABtime_taken = time_taken;
         std::cout << "MAB Results: Eff=" << MABbest_eff
                  << ", Div=" << MABbest_div
                  << ", MABaverage_objective=" << MABaverage_objective
                  << ", MABworst_objective=" << MABworst_objective
                  << ", Time=" << MABtime_taken << "s\n";


        // 5. MAHH_Selection_CMCEE (Uncomment if implemented)

        H.MAHH_Selection_CMCEE(max_time);
        MAHHbest_eff = best_eff;
        MAHHaverage_objective = average_objective;
        MAHHworst_objective = worst_objective;
        MAHHbest_div = best_div;
        MAHHtime_taken = time_taken;
        std::cout << "MAHH Results: Eff=" << MAHHbest_eff
                  << ", Div=" << MAHHbest_div
                  << ", MAHHaverage_objective=" << MAHHaverage_objective
                  << ", MAHHworst_objective=" << MAHHworst_objective
                  << ", Time=" << MAHHtime_taken << "s\n";


        // 6. Greedy_Selection_Hyperheuristic_CMCEE (Uncomment if implemented)

        H.Greedy_Selection_Hyperheuristic_CMCEE(max_time);
        Greedybest_eff = best_eff;
        Greedyaverage_objective = average_objective;
        Greedyworst_objective = worst_objective;
        Greedybest_div = best_div;
        Greedytime_taken = time_taken;
        std::cout << "Greedy Results: Eff=" << Greedybest_eff
                  << ", Div=" << MABbest_div
                  << ", MAHHaverage_objective=" << Greedyaverage_objective
                  << ", MAHHworst_objective=" << Greedyworst_objective
                  << ", Time=" << Greedytime_taken << "s\n";

        // 7. Compare Acceptance Criteria (Uncomment if implemented)
/*
        double optimal_value = 500.0; // Example value
        H.compare_acceptance_criteria(optimal_value);
        Comparebest_eff = best_eff;
        Compareaverage_objective = average_objective;
        Compareworst_objective = worst_objective;
        Comparebest_div = best_div;
        Comparetime_taken = time_taken;
        std::cout << "Compare Results: Eff=" << Comparebest_eff
                  << ", Div=" << Comparebest_div
                  << ", Compareaverage_objective=" << Compareaverage_objective
                  << ", Compareworst_objective=" << Compareworst_objective
                  << ", Time=" << Comparetime_taken << "s\n";


        // 8. SSHH_Selection_Hyperheuristic_CMCEE (Uncomment if implemented)
        /*
        H.SSHH_Selection_Hyperheuristic_CMCEE(max_time);
        SSHHbest_eff = best_eff;
        SSHHbest_div = best_div;
        SSHHtime_taken = time_taken;
        std::cout << "SSHH Results: Eff=" << SSHHbest_eff
                  << ", Div=" << SSHHbest_div
                  << ", Time=" << SSHHtime_taken << "s\n";
        */

        // 9. Adaptive_SSHH_Selection_Hyperheuristic_CMCEE (Uncomment if implemented)
        /*
        H.Adaptive_SSHH_Selection_Hyperheuristic_CMCEE(max_time);
        Adaptive_SSHHbest_eff = best_eff;
        Adaptive_SSHHbest_div = best_div;
        Adaptive_SSHHtime_taken = time_taken;
        std::cout << "Adaptive_SSHH Results: Eff=" << Adaptive_SSHHbest_eff
                  << ", Div=" << Adaptive_SSHHbest_div
                  << ", Time=" << Adaptive_SSHHtime_taken << "s\n";
        */

        // 10. MultiStage_HyperHeuristic_Framework
        /*
        H.runSingleOptimizationAlgorithm(0, max_time); // Assuming 0 corresponds to MultiStage
        MultiStagebest_eff = 0.0;
        MultiStagebest_div = 0.0;
        MultiStagetime_taken = 0.0;
        std::cout << "MultiStage Results: Eff=" << MultiStagebest_eff
                  << ", Div=" << MultiStagebest_div
                  << ", Time=" << MultiStagetime_taken << "s\n";
        */

    // Write results for the current instance
       // combinedResultsFile << instanceName << "\t" << min_div << "\t"
          //                 << std::fixed << std::setprecision(2)
                 //          << QLbest_eff << "\t"
                   //        << QLbest_div << "\t"
                     //      << QLtime_taken << "\t"
                       //    << CFbest_eff << "\t"
                         //  << CFbest_div << "\t"
                           //<< CFtime_taken << "\t"
                         //  << RSbest_eff << "\t"
                           //<< RSbest_div << "\t"
                           //<< RStime_taken << "\t"
                           //<< MABbest_eff << "\t"
                           //<< MABbest_div << "\t"
                           //<< MABtime_taken << "\t"
                           //<<"\n";
                           /*<< MAHHbest_eff << "\t"
                           << MAHHbest_div << "\t"
                           << MAHHtime_taken << "\t"
                           << Greedybest_eff << "\t"
                           << Greedybest_div << "\t"
                           << Greedytime_taken << "\t"
                           << Comparebest_eff << "\t"
                           << Comparebest_div << "\t"
                           << Comparetime_taken << "\t"
                           << SSHHbest_eff << "\t"
                           << SSHHbest_div << "\t"
                           << SSHHtime_taken << "\t"
 /*                          << Adaptive_SSHHbest_eff << "\t"
                           << Adaptive_SSHHbest_div << "\t"
                           << Adaptive_SSHHtime_taken << "\t"
                           << MultiStagebest_eff << "\t"
                           << MultiStagebest_div << "\t"
 /*                          << MultiStagetime_taken << "\n";*/
      /*   combinedResultsFile << instanceName << "\t" << min_div << "\t"
                           << std::fixed << std::setprecision(2)
                           << QLbest_eff << "\t"
                           << QLbest_div << "\t"
                           << QLtime_taken << "\t"
                           << QLaverage_objective << "\t"
                           << QLworst_objective << "\t"
                           << CFbest_eff << "\t"
                           << CFbest_div << "\t"
                           << CFtime_taken << "\t"
                           << CFaverage_objective << "\t"
                           << CFworst_objective << "\t"
                           << RSbest_eff << "\t"
                           << RSbest_div << "\t"
                           << RStime_taken << "\t"
                           << RSaverage_objective << "\t"
                           << RSworst_objective << "\t"
                           << MABbest_eff << "\t"
                           << MABbest_div << "\t"
                           << MABtime_taken << "\t"
                           << MABaverage_objective << "\t"
                           << MABworst_objective << "\t"
                           << Greedybest_eff << "\t"
                           << Greedybest_div << "\t"
                           << Greedytime_taken << "\t"
                           << Greedyaverage_objective << "\t"
                           << Greedyworst_objective << "\t"
                           << MAHHbest_eff << "\t"
                           << MAHHbest_div << "\t"
                           << MAHHtime_taken << "\t"
                           << MAHHaverage_objective << "\t"
                           << MAHHworst_objective << "\t"
                           /*<< Greedybest_eff << "\t"
                           << Greedybest_div << "\t"
                           << Greedytime_taken << "\t"
                           << Greedyaverage_objective << "\t"
                           << Greedyworst_objective << "\t"
                           << Comparebest_eff << "\t"
                           << Comparebest_div << "\t"
                           << Comparetime_taken << "\t"
                           << Compareaverage_objective << "\t"
                           << Compareworst_objective << "\t"
                           // << SSHHbest_eff << "\t"
                           // << SSHHbest_div << "\t"
                           // << SSHHtime_taken << "\t"
                           // << SSHHaverage_objective << "\t"
                           // << SSHHworst_objective << "\t"
                           // << Adaptive_SSHHbest_eff << "\t"
                           // << Adaptive_SSHHbest_div << "\t"
                           // << Adaptive_SSHHtime_taken << "\t"
                           // << Adaptive_SSHHaverage_objective << "\t"
                           // << Adaptive_SSHHworst_objective << "\t"
                           // << MultiStagebest_eff << "\t"
                           // << MultiStagebest_div << "\t"
                           // << MultiStagetime_taken << "\t"
                           // << MultiStageaverage_objective << "\t"
                           // << MultiStageworst_objective << "\t"*/
                         //  << "\n";
        // Add a newline after the last method for readability
       // combinedResultsFile << "\n";

        // Optional: Print confirmation
   /*     std::cout << "Successfully wrote results for instance: " << instanceName << std::endl;
    }

    // Close the combined results file
    combinedResultsFile.close();

    // Clean up any dynamically allocated memory if necessary
    free_memory();
    return 0;
}

/*

int main(int argc, char *argv[]) {
    // Initialize random seed
    srand(static_cast<unsigned int>(time(0)));

    // Directories
    std::string datasetDir = "D:/Datasets/";
    std::string datasetDirD1 = "D:\\Datasets\\RESULTS_OF_HH_MODELSlastsatified\\";

    // Ensure the results directory exists
    if (!fs::exists(datasetDirD1)) {
        try {
            fs::create_directories(datasetDirD1);
            std::cout << "Created results directory: " << datasetDirD1 << std::endl;
        }
        catch (const fs::filesystem_error& e) {
            std::cerr << "Error creating results directory: " << e.what() << std::endl;
            return 1;
        }
    }

    // Collect all dataset files
    std::vector<std::string> datasetFiles;
    for (const auto &entry : fs::directory_iterator(datasetDir)) {
        if (entry.is_regular_file()) {
            datasetFiles.push_back(entry.path().string());
        }
    }

    if (datasetFiles.empty()) {
        std::cerr << "No dataset files found in directory: " << datasetDir << std::endl;
        return 1;
    }

    std::cout << "[INFO] Found " << datasetFiles.size() << " dataset files.\n";

    // Initialize Hyper_heuristic instance
    Hyper_heuristic H;

    // Set maximum time (in seconds) for each method
    int max_time = 600; // Example: 60 seconds

    // Initialize the combined results file
    std::string combinedResultsFilePath = datasetDirD1 + "Combined_Results_last.txt";
    std::ofstream combinedResultsFile(combinedResultsFilePath, std::ios::out);
    if (!combinedResultsFile) {
        std::cerr << "Error opening combined results file for writing: " << combinedResultsFilePath << std::endl;
        return 1;
    }

    // Write headers to the combined results file
    combinedResultsFile << "Instance\tmin_div\t"
                        << "QL_best_eff\tQL_best_div\tQL_Taken_time\tQL_average_objective\tQL_worst_objective\t"
                        << "MAB_best_eff\tMAB_best_div\tMAB_Taken_time\tMAB_average_objective\tMAB_worst_objective\t"
                        << "Greedy_best_eff\tGreedy_best_div\tGreedy_Taken_time\tGreedy_average_objective\tGreedy_worst_objective\t"
                        << "CF_best_eff\tCF_best_div\tCF_Taken_time\tCF_average_objective\tCF_worst_objective\t"
                        << "RS_best_eff\tRS_best_div\tRS_Taken_time\tRS_average_objective\tRS_worst_objective\t"
                        << "\n";

    // Initialize Overall Metrics (Optional)
    double overall_average_objective = 0.0;
    double overall_average_diversity = 0.0;
    int overall_worst_objective = INT_MAX;
    std::map<int, int> overall_heuristic_usage_count;

    // Process each dataset file
    //(size_t idx = 64; idx <  datasetFiles.size() ; idx++)
    for (size_t idx = 0; idx < datasetFiles.size() ; idx++) { // Adjust indices as needed
        // Safety check to prevent out-of-range access
        if (idx >= datasetFiles.size()) {
            std::cerr << "[WARNING] Index " << idx << " is out of range for datasetFiles.\n";
            break;
        }

        std::string current_file = datasetFiles[idx];
        fs::path p(current_file);
        std::string rawInstanceName = p.filename().string();
        instanceName = parseInstanceName(rawInstanceName);

        if (instanceName.empty()) {
            std::cerr << "[WARNING] Failed to parse instance name for file: " << rawInstanceName << std::endl;
            continue;
        }

        std::cout << "\nProcessing Instance " << (idx + 1) << " of " << datasetFiles.size() << ":\n";
        std::cout << "Instance Name: " << instanceName << std::endl;


        // Initialize hyper-heuristic parameters and setup
        H.Parameters();
        H.initialization(current_file);
        std::cout << "num_node= " <<num_node << "  " // Replace with actual variables if available
                  << "num_team= " << num_team
                  << ",num_each_t=" << num_each_t << "  " // Replace with actual variables if available
                  << "min_div= " << min_div << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << std::endl << std::endl;
        //H.generate_initial();
        //H.compute_mindiv();
        // Variables to store results for each method
        double QLbest_eff = 0.0, QLbest_div = 0.0, QLtime_taken = 0.0, QLaverage_objective = 0.0, QLworst_objective = 0.0;
        double CFbest_eff = 0.0, CFbest_div = 0.0, CFtime_taken = 0.0, CFaverage_objective = 0.0, CFworst_objective = 0.0;
        double RSbest_eff = 0.0, RSbest_div = 0.0, RStime_taken = 0.0, RSaverage_objective = 0.0, RSworst_objective = 0.0;
        double MABbest_eff = 0.0, MABbest_div = 0.0, MABtime_taken = 0.0, MABaverage_objective = 0.0, MABworst_objective = 0.0;
        //double MAHHbest_eff = 0.0, MAHHbest_div = 0.0, MAHHtime_taken = 0.0, MAHHaverage_objective = 0.0, MAHHworst_objective = 0.0;
        double Greedybest_eff = 0.0, Greedybest_div = 0.0, Greedytime_taken = 0.0, Greedyaverage_objective = 0.0, Greedyworst_objective = 0.0;
        //double Comparebest_eff = 0.0, Comparebest_div = 0.0, Comparetime_taken = 0.0, Compareaverage_objective = 0.0, Compareworst_objective = 0.0;
        // Similarly declare for SSHH, Adaptive_SSHH, MultiStage if implemented

        // Execute HH Methods and Capture Metrics

        // 1. Q_Learning_Selection_Hyperheuristic_CMCEE (Heuristic ID: 11)
        H.Q_Learning_Selection_Hyperheuristic_CMCEE(max_time);
        QLbest_eff = best_eff;
        QLbest_div = best_div;
        QLtime_taken = time_taken;
        QLaverage_objective = average_objective;
        QLworst_objective = worst_objective;

        std::cout << "QL Results: Eff=" << QLbest_eff
                  << ", Div=" << QLbest_div
                  << ", Time=" << QLtime_taken << "s\n";

        std::cout <<"----------------------------------------------s\n\n";

        // 2. MAB_Selection_Hyperheuristic_CMCEE (Heuristic ID: 18)
        H.MAB_Selection_Hyperheuristic_CMCEE(max_time);
        MABbest_eff = best_eff;
        MABbest_div = best_div;
        MABtime_taken = time_taken;
        MABaverage_objective = average_objective;
        MABworst_objective = worst_objective;

        std::cout << "MAB Results: Eff=" << MABbest_eff
                  << ", Div=" << MABbest_div
                  << ", Time=" << MABtime_taken << "s\n";
        std::cout <<"----------------------------------------------s\n\n";

        // 3. Greedy_Selection_Hyperheuristic_CMCEE (Heuristic ID: 20)
        H.Greedy_Selection_Hyperheuristic_CMCEE(max_time);
        Greedybest_eff = best_eff;
        Greedybest_div = best_div;
        Greedytime_taken = time_taken;
        Greedyaverage_objective = average_objective;
        Greedyworst_objective = worst_objective;

        std::cout << "Greedy Results: Eff=" << Greedybest_eff
                  << ", Div=" << Greedybest_div
                  << ", Time=" << Greedytime_taken << "s\n";
        std::cout <<"----------------------------------------------s\n\n";

        // 4. HH_Choice_Function_Selection_CMCEE (Heuristic ID: 12)
        H.HH_Choice_Function_Selection_CMCEE(max_time);
        CFbest_eff = best_eff;
        CFbest_div = best_div;
        CFtime_taken = time_taken;
        CFaverage_objective = average_objective;
        CFworst_objective = worst_objective;

        std::cout << "CF Results: Eff=" << CFbest_eff
                  << ", Div=" << CFbest_div
                  << ", Time=" << CFtime_taken << "s\n";
        std::cout <<"----------------------------------------------s\n\n";

        // 3. Random_Selection_Hyperheuristic_CMCEE (Heuristic ID: 17)
        H.Random_Selection_Hyperheuristic_CMCEE(max_time);
        RSbest_eff = best_eff;
        RSbest_div = best_div;
        RStime_taken = time_taken;
        RSaverage_objective = average_objective;
        RSworst_objective = worst_objective;

        std::cout << "RS Results: Eff=" << RSbest_eff
                  << ", Div=" << RSbest_div
                  << ", Time=" << RStime_taken << "s\n";

        std::cout <<"----------------------------------------------s\n\n";


        // Write results for the current instance
        combinedResultsFile << instanceName << "\t" << min_div << "\t"
                           << std::fixed << std::setprecision(2)
                           << QLbest_eff << "\t"
                           << QLbest_div << "\t"
                           << QLtime_taken << "\t"
                           << QLaverage_objective << "\t"
                           << QLworst_objective << "\t"
                           << MABbest_eff << "\t"
                           << MABbest_div << "\t"
                           << MABtime_taken << "\t"
                           << MABaverage_objective << "\t"
                           << MABworst_objective << "\t"
                           << Greedybest_eff << "\t"
                           << Greedybest_div << "\t"
                           << Greedytime_taken << "\t"
                           << Greedyaverage_objective << "\t"
                           << Greedyworst_objective << "\t"
                           << CFbest_eff << "\t"
                           << CFbest_div << "\t"
                           << CFtime_taken << "\t"
                           << CFaverage_objective << "\t"
                           << CFworst_objective << "\t"
                           << RSbest_eff << "\t"
                           << RSbest_div << "\t"
                           << RStime_taken << "\t"
                           << RSaverage_objective << "\t"
                           << RSworst_objective << "\t"
                           << "\n";
        // Add a newline after the last method for readability (optional)
        // combinedResultsFile << "\n";

        // Optional: Print confirmation
        std::cout << "Successfully wrote results for instance: " << instanceName << std::endl;
    }

    // Close the combined results file
    combinedResultsFile.close();

    // Clean up any dynamically allocated memory if necessary
    free_memory();

    return 0;
}


// forward decls you already have:
// std::string parseInstanceName(const std::string& raw);
// extern std::string instanceName;
// extern int num_node, num_team, num_each_t;
// extern double min_div, best_eff, best_div, time_taken;
/*
int main(int argc, char *argv[]) {
    // Initialize random seed
    srand(static_cast<unsigned int>(time(0)));

    // Directories
    std::string datasetDir   = "D:/Datasets/";
    std::string datasetDirD1 = "D:\\Datasets\\RESULTS_OF_HH_MODELS_Single_MHs\\";

    // Ensure results folder exists
    try { fs::create_directories(datasetDirD1); }
    catch (...) { std::cerr << "Cannot create results dir: " << datasetDirD1 << std::endl; }

    // Collect all dataset files
    std::vector<std::string> datasetFiles;
    for (const auto &entry : fs::directory_iterator(datasetDir)) {
        if (entry.is_regular_file()) {
            datasetFiles.push_back(entry.path().string());
        }
    }
    if (datasetFiles.empty()) {
        std::cerr << "No dataset files found in: " << datasetDir << std::endl;
        return 0;
    }

    // Basic initialization
    Hyper_heuristic H;

    // Set time limit or max_time as needed
    int max_time = 600;

    // -------- choose ONE algorithm once ----------
    std::cout << "Select ONE method to run on ALL instances:\n";
    std::cout << "1: Q-Learning Selection Hyperheuristic CMCEE\n";
    std::cout << "2: HH_Choice_Function_Selection_CMCEE\n";
    std::cout << "3: Random_Selection_Hyperheuristic_CMCEE\n";
    std::cout << "4: MAB_Selection_Hyperheuristic_CMCEE\n";
    std::cout << "5: MAHH_Selection_CMCEE\n";
    std::cout << "6: Greedy_Selection_Hyperheuristic_CMCEE\n";
    std::cout << "7: Compare Acceptance Criteria\n";
    std::cout << "8: Sequence based Selection Hyperheuristic CMCEE\n";
    std::cout << "9: Adaptive Sequence based Selection Hyperheuristic CMCEE\n";
    std::cout << "10: MultiStage HyperHeuristic Framework\n";
    std::cout << "11: Execute Single Optimization Algorithm\n";
    std::cout << "Enter your choice (0 to exit): ";

    int choice = 0;
    std::cin >> choice;
    if (choice == 0) {
        std::cout << "Exit.\n";
        return 0;
    }

    int single_choice = -1;
    if (choice == 11) {
        std::cout << "Select ONE Optimization Algorithm to run on ALL instances:\n";
        std::cout << "11: Iterated Local Search (ILS)\n";
        std::cout << "12: Simulated Annealing (SA)\n";
        std::cout << "13: Tabu Search (TS)\n";
        std::cout << "14: Feasible Local Search (FLS)\n";
        std::cout << "15: Infeasible Local Search (IFLS)\n";
        std::cout << "16: Memetic Algorithm (MA)\n";
        std::cout << "17: Great Deluge Algorithm (GD)\n";
        std::cout << "18: Variable Neighborhood Search (VNS)\n";
        std::cout << "19: Late Acceptance Hill Climbing (LAHC)\n";
        std::cout << "20: Guided Local Search (GLS)\n";
        std::cout << "21: Harmony Search (HS)\n";
        std::cout << "Enter your choice (11-21): ";
        std::cin >> single_choice;
        if (single_choice < 11 || single_choice > 21) {
            std::cout << "Invalid selection. Exiting.\n";
            return 0;
        }
    }

    // Optional constant for case 7
    double optimal_value = 500.0;

    // -------- run the selected method on EACH instance ----------
    for (size_t idx = 0; idx < datasetFiles.size(); idx++) {
        std::string current_file = datasetFiles[idx];
        fs::path p(current_file);
        std::string rawInstanceName = p.filename().string();
        instanceName = parseInstanceName(rawInstanceName);

        if (instanceName.empty()) {
            std::cerr << "[WARNING] Failed to parse instance name for file: " << rawInstanceName << std::endl;
            continue;
        }

        std::cout << "\nProcessing Instance " << (idx + 1) << " of " << datasetFiles.size() << ":\n";
        std::cout << "Instance Name: " << instanceName << std::endl;

        // Create results file for this instance
        std::string resultsFilePath = datasetDirD1 + "Results_" + instanceName + ".txt";
        std::ofstream resultsFile(resultsFilePath); // overwrite per run
        if (!resultsFile) {
            std::cerr << "Error opening results file for writing: " << resultsFilePath << std::endl;
            continue;
        }

        // Header
        resultsFile << "Hyper-Heuristic Method\tBest Efficiency\tBest Diversity\tTime Taken (seconds)\n";

        // Load instance
        H.Parameters();
        H.initialization(current_file);

        std::cout << "num_node= " << num_node << "  "
                  << "num_team= " << num_team
                  << ",num_each_t= " << num_each_t << "  "
                  << "min_div= " << min_div << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;

        // Run the chosen method once on this instance
        switch (choice) {
            case 1:
                H.Q_Learning_Selection_Hyperheuristic_CMCEE(max_time);
                resultsFile << "Q_Learning_Selection_Hyperheuristic_CMCEE\t"
                            << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                break;
            case 2:
                H.HH_Choice_Function_Selection_CMCEE(max_time);
                resultsFile << "HH_Choice_Function_Selection_CMCEE\t"
                            << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                break;
            case 3:
                H.Random_Selection_Hyperheuristic_CMCEE(max_time);
                resultsFile << "Random_Selection_Hyperheuristic_CMCEE\t"
                            << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                break;
            case 4:
                H.MAB_Selection_Hyperheuristic_CMCEE(max_time);
                resultsFile << "MAB_Selection_Hyperheuristic_CMCEE\t"
                            << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                break;
            case 5:
                H.MAHH_Selection_CMCEE(max_time);
                resultsFile << "MAHH_Selection_CMCEE\t"
                            << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                break;
            case 6:
                H.Greedy_Selection_Hyperheuristic_CMCEE(max_time);
                resultsFile << "Greedy_Selection_Hyperheuristic_CMCEE\t"
                            << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                break;
            case 7:
                H.compare_acceptance_criteria(optimal_value);
                resultsFile << "Compare_Acceptance_Criteria\t"
                            << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                break;
            case 8:
                H.SSHH_Selection_Hyperheuristic_CMCEE(max_time);
                resultsFile << "SSHH_Selection_Hyperheuristic_CMCEE\t"
                            << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                break;
            case 9:
                H.Adaptive_SSHH_Selection_Hyperheuristic_CMCEE(max_time);
                resultsFile << "Adaptive_SSHH_Selection_Hyperheuristic_CMCEE\t"
                            << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                break;
            case 10: {
                MultiStageHyperHeuristicFramework1 HH1;
                HH1.run(max_time, 20);
                resultsFile << "MultiStage_HyperHeuristic_Framework\t"
                            << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                break;
            }
            case 11:
                H.runSingleOptimizationAlgorithm(single_choice, max_time);
                resultsFile << "Single_Optimization_Algorithm_" << single_choice << "\t"
                            << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                break;
            default:
                std::cout << "Invalid choice.\n";
                break;
        }

        resultsFile.flush();
        resultsFile.close();
        std::cout << "Saved: " << resultsFilePath << std::endl;
    }

    return 0;
}

*/

/*
int main(int argc, char *argv[]) {

    srand(static_cast<unsigned int>(time(0)));


    // --------------------------------------------------
    //               DIRECTORIES
    // --------------------------------------------------
    std::string datasetDir = "D:/Datasets/";
    std::string resultsDir = "D:/Datasets/LLH_RESULTS/";

    // Load dataset files
    std::vector<std::string> datasetFiles;
    for (const auto &entry : fs::directory_iterator(datasetDir)) {
        if (entry.is_regular_file())
            datasetFiles.push_back(entry.path().string());
    }

    if (datasetFiles.empty()) {
        std::cout << "No dataset files found.\n";
        return 0;
    }

    // Create hyper-heuristic object
    Hyper_heuristic H;

    // ======================================================
    //                 MAIN LOOP (MENU)
    // ======================================================
    while (true) {

        std::cout << "\n=========================================\n";
        std::cout << "       LOW-LEVEL HEURISTIC TEST MENU     \n";
        std::cout << "=========================================\n";

        std::cout << "Choose LLH to run (1–17), or 0 to EXIT:\n"
                  << "  1  -> LLH1  : Swap 1 pool member with 1 team member\n"
                  << "  2  -> LLH2  : Swap 2 pool members with 2 team members\n"
                  << "  3  -> LLH3  : Swap 1 member between two teams\n"
                  << "  4  -> LLH4  : Move a member between teams\n"
                  << "  5  -> LLH5  : Swap 3 consecutive members\n"
                  << "  6  -> LLH6  : Swap pool member with weakest team\n"
                  << "  7  -> LLH7  : Weakest vs strongest team swap\n"
                  << "  8  -> LLH8  : Swap first member of 2 teams\n"
                  << "  9  -> LLH9  : Swap last member of 2 teams\n"
                  << " 10  -> LLH10 : Swap weakest-team member with another\n"
                  << " 11  -> LLH11 : Circular shift among all teams\n"
                  << " 12  -> LLH12 : Swap half weakest team with pool\n"
                  << " 13  -> LLH13 : Ruin and Recreate\n"
                  << " 14  -> LLH14 : Random small perturbation\n"
                  << " 15  -> LLH15 : Swap subset between two teams\n"
                  << " 16  -> LLH16 : 3-team circular chain swap\n"
                  << "  0  -> EXIT PROGRAM\n\n";

        std::cout << "Enter your choice: ";

        int llh_choice;
        std::cin >> llh_choice;

        // ---------------- EXIT CONDITION ----------------
        if (llh_choice == 0) {
            std::cout << "\nExiting program...\n";
            break;
        }

        if (llh_choice < 1 || llh_choice > 17) {
            std::cout << "Invalid choice. Try again.\n";
            continue;   // return to menu
        }

        // ======================================================
        //     RUN ONLY THE FIRST INSTANCE (as before)
        // ======================================================
        std::string current_file = datasetFiles[0];

        fs::path p(current_file);
        std::string instanceName = p.filename().string();

        std::cout << "\nRunning on instance: " << instanceName << "\n";

        std::string resultsFilePath =
            resultsDir + "LLH_" + std::to_string(llh_choice) + "_" + instanceName + ".txt";

        std::ofstream resultsFile(resultsFilePath);
        if (!resultsFile) {
            std::cerr << "Cannot write to results file.\n";
            continue;
        }

        resultsFile << "LLH\tBest_Eff\tBest_Div\tTime(sec)\n";

        // ---------------------------------------------
        // Initialization
        // ---------------------------------------------
        H.Parameters();
        H.initialization(current_file);
        std::cout << "num_node= " << num_node << "  "
                  << "num_team= " << num_team
                  << ",num_each_t= " << num_each_t << "  "
                  << "min_div= " << min_div << std::endl;
        std::cout << "------------------------------------------------------------------" << std::endl;

        H.generate_initialrandom();
        std::cout << "------------------------------------------------------------------" << std::endl;
        H.objective_Function(team);

        std::cout << "\nInitial Solution:\n";
        H.display(team);

        auto start = std::chrono::high_resolution_clock::now();

        // ---------------------------------------------
        // Apply LLH
        // ---------------------------------------------
        switch (llh_choice) {
            case 1:  H.LLH1(team);  break;
            case 2:  H.LLH2(team);  break;
            case 3:  H.LLH3(team);  break;
            case 4:  H.LLH4(team);  break;
            case 5:  H.LLH5(team);  break;
            case 6:  H.LLH6(team);  break;
            case 7:  H.LLH7(team);  break;
            case 8:  H.LLH8(team);  break;
            case 9:  H.LLH9(team);  break;
            case 10: H.LLH10(team); break;
            case 11: H.LLH11(team); break;
            case 12: H.LLH12(team); break;
            case 13: H.LLH13(team); break;
            case 14: H.LLH14(team); break;
            case 15: H.LLH15(team); break;
            case 16: H.LLH16(team); break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_taken = std::chrono::duration<double>(end - start).count();

        // Final objective recomputation
        H.objective_Function(team);
        double best_eff = f_cur;
        double best_div = f_cur_div;

        // ---------------------------------------------
        // Print results
        // ---------------------------------------------
        std::cout << "\nAfter LLH " << llh_choice << ":\n";
        H.display(team);

        std::cout << "----------------------------------------\n"
                  << " LLH " << llh_choice << " Evaluation Results\n"
                  << " Best Efficiency : " << best_eff << "\n"
                  << " Best Diversity  : " << best_div << "\n"
                  << " Time Taken (s)  : " << time_taken << "\n"
                  << "----------------------------------------\n";

        resultsFile << "LLH" << llh_choice << "\t"
                    << best_eff   << "\t"
                    << best_div   << "\t"
                    << time_taken << "\n";

        resultsFile.close();

        std::cout << "Saved results → " << resultsFilePath << "\n";

        // ★★★★★ RETURN TO MENU ★★★★★
        std::cout << "\nPress ENTER to return to menu...";
        std::cin.ignore();
        std::cin.get();
    }

    free_memory();
    std::cout << "\nAll tests completed.\n";
    return 0;
}
*/
int main(int argc, char *argv[])
{
	srand((unsigned)time(NULL));
	if (argc < 1)	{
		cout << "usage: MA.exe input_file";
		exit(1);
	}

    std::string datasetDir   = "D:/Datasets/";
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
        std::cout << " 5: Multi-Agent HH\n";
        std::cout << " 6: Greedy Selection HH\n";
        std::cout << " 7: Compare Acceptance Criteria\n";
        std::cout << " 8: Sequence-based HH\n";
        std::cout << " 9: Adaptive Sequence-based HH\n";
        std::cout << "10: MultiStage HH Framework\n";
        std::cout << "11: Tri-Level HH (QL + MAB + QL)\n";
        std::cout << "12: Single Optimization Algorithm\n";
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
        if (choice == 12) {
            std::cout << "\nSelect ONE Optimization Algorithm:\n";
            std::cout << "12: Iterated Local Search (ILS)\n";
            std::cout << "13: Simulated Annealing (SA)\n";
            std::cout << "14: Tabu Search (TS)\n";
            std::cout << "15: Feasible Local Search (FLS)\n";
            std::cout << "16: Infeasible Local Search (IFLS)\n";
            std::cout << "17: Memetic Algorithm (MA)\n";
            std::cout << "18: Great Deluge Algorithm (GD)\n";
            std::cout << "19: Variable Neighborhood Search (VNS)\n";
            std::cout << "20: Late Acceptance Hill Climbing (LAHC)\n";
            std::cout << "21: Guided Local Search (GLS)\n";
            std::cout << "22: Harmony Search (HS)\n";
            std::cout << "Enter (12 to 22): ";
            std::cin >> single_choice;

            if (single_choice < 12 || single_choice > 22) {
                std::cout << "Invalid selection. Returning to menu...\n";
                continue;
            }
        }

        // ====================================================
        //           RUN SELECTED METHOD ON ALL INSTANCESdatasetFiles.size()
        // ====================================================
        for (size_t idx = 0; idx < 1 ; idx++) {

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
                    H.MAHH_Selection_CMCEE(max_time);
                    resultsFile << "MAHH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 6:
                    H.Greedy_Selection_Hyperheuristic_CMCEE(max_time);
                    resultsFile << "Greedy_HH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 7:
                    H.compare_acceptance_criteria(500.0);
                    resultsFile << "Compare_Accept\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 8:
                    H.SSHH_Selection_Hyperheuristic_CMCEE(max_time);
                    resultsFile << "SSHH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 9:
                    H.Adaptive_SSHH_Selection_Hyperheuristic_CMCEE(max_time);
                    resultsFile << "Adaptive_SSHH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 10: {
                    MultiStageHyperHeuristicFramework1 HH1;
                    HH1.run(max_time, 20);
                    resultsFile << "MultiStage_HH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;
                }

                case 11:
                    H.TriLevel_HH_Qlearning_CMCEE(max_time);
                    resultsFile << "TriLevel_HH\t" << best_eff << "\t" << best_div << "\t" << time_taken << "\n";
                    break;

                case 12:
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


