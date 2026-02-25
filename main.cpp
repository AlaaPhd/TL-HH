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
#include <filesystem>
#include <regex>
#define MINVALUE -99999999
#define MAXVALUE 99999999
#define MAXNUM 100
#define DEBUG
#define Pop_Num 10
#include <functional>
#include <mutex>

using namespace std;
namespace fs = std::filesystem;
std::mutex global_data_mutex;

std::mutex log_mutex; // for thread-safe console and file output
// Inside Hyper_heuristic.h or a relevant header
typedef std::function<int(int)> AlgorithmFunction;
// instance data
string file_name; // Global variable used in initializing()
// Global instanceName variable
std::string instanceName;
std::map<int, int> heuristic_usage_count; // Number of times each heuristic was used
int num_node;                             // number of practitioners
int num_each_t;                           // number of practitioners in each crew
int num_team;                             // number of crews
double min_div;                           // the diversity threshold of each crew
double **div_in;                          // the diversity between each pair of practitioners
int *eff;                                 // efficiency of practitioners
int f_cur_div;
double d_min = 0.8; // d_min = {0.8,1.0,1.05}
double beta = 0.4;  // beta = 0.4
int fbest_eff;
int fbest_div;
// for tabu search
int **tabu_list;  // tabu_list[num_node][num_team+1]
int tl;           // disabled
int tabu_tenure;  // tabu tenure
int fls_depth;    // search depth of feasible local search
int ils_depth;    // search depth of infeasible local search
int *num_t_cur;   // record the number of practitioners of each crew in current solution
int **team_check; // for the use of verifying solution
double start_time, end_time, time_limit;
int f_cur_eff;
// parameters for feasible and infeasible local search
double p_factor; // penalty factor
int p_count;
int lamba;
int u1, u2;
int tow;
double deg_cur;     // infeasible degree of the current solution
double *degree_inf; // infeasible degree of each crew
int best_cost_eff;
int best_cost_div;
double initial_temp = 100.0;
double cooling_rate = 0.995;
double x0 = 10.0; // Initial guess
int max_iterations = 1500;
double Result;
int best_eff = 0, best_div = 0;
double time_taken = 0.0;
double average_objective = 0.0; // average objective function
double average_diversity = 0.0;
int worst_objective = 0;       // Worst Objective Function Value
double average_cpu_time = 0.0; // Average CPU Time per Iteration
// variables that need to be updated
int *address; // record the address of each practitioner
int **team;   // record the practitioners for each crew

int **team1;
int **delta_div;
int *state;     // record which crew the practitioners are in
int *w_div;     // record the diversity of each crew
int *w_eff;     // record the efficiency of each crew
int f_cur;      // the objective value of current solution
int *best_inn;  // the best solution obtained during fits search
int f_best_inn; // the objective value of the f_best_inn
int S[100];     // Population size
int nm[100];
int m[100];
int fprod;
int l;
int *aa = new int[num_team + 1];

// for the memetic algorithm
typedef struct population
{
    int *p;
    int cost;
} population;
population *pop, *A; // population pop
int generations;     // the generations of evolutionary process

// Define the Solution struct
// preserve the best solution over 30 independent runs
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
double avg_time = 0.0;

// Enum to specify the MAB strategy
enum MABStrategy
{
    UCB,
    EPSILON_GREEDY,
    THOMPSON_SAMPLING
};

// Define a structure to hold heuristic performance metrics
struct Heuristic
{
    int id;                // Heuristic identifier
    std::string name;      // Heuristic name
    int usage_count;       // Number of times the heuristic was used
    double total_reward;   // Cumulative reward
    double average_reward; // Average reward

    Heuristic(int heuristic_id, const std::string &heuristic_name)
        : id(heuristic_id), name(heuristic_name), usage_count(0),
          total_reward(0.0), average_reward(0.0) {}
};



class Hyper_heuristic
{
    // Method to implement Algorithm 7: Selection Hyper-Heuristic Framework

private:
    // -------------------------------------------------------------------------
    // Variable Declarations
    // -------------------------------------------------------------------------
    typedef int **(Hyper_heuristic::*LLH_Function)(int **);
    vector<LLH_Function> LLH_set;

    int **Sbest;
    double fbest_efficiency;
    double fbest_diversity;

    std::vector<int> sbest;


    // -------------------------------------------------------------------------
    // Internal Helper Functions
    // -------------------------------------------------------------------------
    LLH_Function SelectLowLevelHeuristic();
    int **ApplyHeuristic(LLH_Function heuristic, int **Scurrent);
    bool Accept(int **Scurrent, int **Snew, double f_current, double f_new);
    void updateBestSolution(int **Scurrent, double f_current);
    bool TerminationCriterionSatisfied(int iter, int max_iter);

public:

    int *team_size;
    // -------------------------------------------------------------------------
    // Entry Point
    // -------------------------------------------------------------------------
    int selection_hyper_heuristic(int max_iterations);

    // -------------------------------------------------------------------------
    // State-vector structure (10D feature vector)
    struct StateFeatures
    {
        double f_eff_norm;     // Normalized efficiency
        double f_div_norm;     // Normalized diversity
        double delta_eff_norm; // Efficiency improvement
        double delta_div_norm; // Diversity improvement
        double iter_ratio;     // Current iteration progress
        double accept_ratio;   // Acceptance ratio
        double reward_avg;     // Average recent reward
        double div_std_norm;   // Normalized diversity std. deviation
        double temp_norm;      // Normalized temperature (SA)
        double flex_norm;      // Normalized flexibility threshold
    };

    // -------------------------------------------------------------------------
    // Objective & Evaluation Functions
    // -------------------------------------------------------------------------
    void compute_mindiv();
    void objective_Function(int **);
    void objective_Function1(int **);
    int  min_func(int *, int);
    int  sec_func(int *, int, int);
    int  th_func(int *, int, int, int);
    int  randomInt(int);
    int  max_func(int *, int); // <-- ✅ Add this
    int  rand_func(int *, int);

    int  select_max_multiple(int *, int *, int &);
    int **Apply_LS_OP(int, int, int **);
    // -------------------------------------------------------------------------
    // Meta-Heuristic Algorithms Declaration
    // -------------------------------------------------------------------------
    int **great_deluge_algorithm();
    int guided_local_search();
    int **late_acceptance_hill_climbing();
    int Hill_Late_Acceptance();
    int **iterated_local_search();
    int iterated_local_search12();
    int **simulated_annealing();
    int **memetic();

    // -------------------------------------------------------------------------
    // Local Search Operators
    // -------------------------------------------------------------------------
    int **local_search();
    int **infeasible_local_search();
    int **local_search1();
    int   feasible_local_search();

    // -------------------------------------------------------------------------
    // Initialization Functions
    // -------------------------------------------------------------------------
    void Parameters();
    void initialization(const string &);
    void initial_population();
    int  generate_initialrandom();
    int  generate_initial();


    // -------------------------------------------------------------------------
    // Repair Functions
    // -------------------------------------------------------------------------
     void repair_solution();

    // -------------------------------------------------------------------------
    // Swap and Mutation Operators
    // -------------------------------------------------------------------------

    int swap1(int, int &, int &);
    int swap_ils(int, int &, int &);
    int swap_min(int, int &, int &);


    // -------------------------------------------------------------------------
    // Crossover Operators
    // -------------------------------------------------------------------------
    void cross_over2();


    // -------------------------------------------------------------------------
    // Tabu & Utility Functions
    // -------------------------------------------------------------------------
    bool isTabu(const std::vector<std::vector<int>> &, int, int, int);
    void applySwap(int, int);
    void updateTabuList(std::vector<std::vector<int>> &, int, int, int, int);
    void update_delta(int, int, int);
    void update_populaion(int *, int);
    int **fits();

    // -------------------------------------------------------------------------
    // Function Selection (Hyper-Heuristic) and Application
    // -------------------------------------------------------------------------
    int **ApplyHeuristic(int, int **);
    int **ApplyMeta_Heuristic(int, int **);


    // -------------------------------------------------------------------------
    // Function Selection (TRI-LEVEL Hyper-Heuristic) and Application
    // -------------------------------------------------------------------------
    int **SimulatedAnnealing(int **Sstart, int OPj, double T0, double alpha);
    int **IteratedLocalSearch(int **Sstart, int OPj);
    int **LateAcceptance(int **Sstart, int OPj, int Lwindow);
    int **GreatDeluge(int **Sstart, int OPj, double level0, double delta);
    int **TabuSearch(int **Sstart, int OPj);
    // ---- Operator selection (Level 2) ----
    int **apply_LLHop(int op_id, int **sol);
    void apply_atomic_swap(int, int );


    // -------------------------------------------------------------------------
    // Low-Level Heuristics (LLH)
    // -------------------------------------------------------------------------
    int **LLH1(int **);
    int **LLH2(int **);
    int **LLH3(int **);
    int **LLH4(int **);
    int **LLH5(int **);
    int **LLH6(int **);
    int **LLH7(int **);
    int **LLH8(int **);
    int **LLH9(int **);
    int **LLH10(int **);
    int **LLH11(int **);
    int **LLH12(int **);
    int **LLH13(int **);
    int **LLH14(int **);
    int **LLH15(int **);


    // -------------------------------------------------------------------------
    // Selection Hyper-Heuristic Strategies
    // -------------------------------------------------------------------------

    void Random_Selection_Hyperheuristic_CMCEE(int);
    void HH_Choice_Function_Selection_CMCEE(int);
    void MAB_Selection_Hyperheuristic_CMCEE(int);
    void Greedy_Selection_Hyperheuristic_CMCEE(int);

    // Q-Learning Based Selection Function
    void Q_Learning_Selection_Hyperheuristic_CMCEE(int);
    void TriLevel_HH_Qlearning_CMCEE(int);
    void execute_algorithm(int, const std::string &);
    void runSingleOptimizationAlgorithm(int, int);

    // -------------------------------------------------------------------------
    // Utility and Display
    // -------------------------------------------------------------------------
    void display(int **);
    void displayResults();
    void check_best_solution();
    bool dominates(int, int, int, int);

    // -------------------------------------------------------------------------
    // Deep Copy / Free Utilities
    // -------------------------------------------------------------------------
    // ======================================================================
    // Deep Copy of a 2D Team Structure with Variable Team Sizes
    // ======================================================================
    int **clone_solution(int **src);
    void copy_solution(int **dest, int **src);
    void free_solution(int **sol);

    int **deep_copy_solution(int **source_team,
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
        int **new_team = new int *[num_team + 1];
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
    void free_solution(int **team_copy,
                       int num_node,
                       int num_team,
                       int num_each_t)
    {
        if (!team_copy)
            return;

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
        const std::vector<double> &reward_hist,
        const std::vector<int> &div_values,
        double temp, double temp0,
        double flex, double flex_max);
    StateFeatures compute_state_vector1(
        int, int, int, int, int, int, int, int, int, int,
        const std::vector<double> &,
        const std::vector<int> &,
        double, double);

    // ---- Utility functions ----
    double compute_delta(int f_curr, int f_prev);
    double reward_from_delta(double delta);
    std::string discretize_state(const StateFeatures &s);
    double rolling_average(const std::vector<double> &values, int window);

    // ---- Local search dispatch (Level 1) ----
    enum LSAlgo
    {
        LS_SA = 1,
        LS_ILS,
        LS_TS,
        LS_GD,
        LS_LAHC,
        LS_BasicLS
    };
    int **run_LS(LSAlgo algo, int **team);

    // ---- Feature and State Computation ----
    double compute_div_std_norm(const std::vector<int> &div_values, double div_max);

    // ---- Move acceptance (Level 3) ----
    enum MA_Strategy
    {
        MA_ONLY_IMPROVE = 1,
        MA_ACCEPT_ALL,
        MA_SA,
        MA_R2R,
        MA_THRESHOLD
    };
    bool accept_move(MA_Strategy rule, int cur_eff, int cur_div, int new_eff, int new_div, double min_div);
    // -------------------------------------------------------------------------
    // Destructor
    // -------------------------------------------------------------------------
    ~Hyper_heuristic()
    {
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
        for (int i = 0; i < num_node; ++i)
        {
            delete[] div_in[i];
            delete[] delta_div[i];
            delete[] tabu_list[i];
        }
        for (int i = 0; i <= num_team; ++i)
        {
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

int Hyper_heuristic::select_max_multiple(int *best_node, int *best_team, int &num_best)
{
    double max = -1000000;
    double product = 0;
    num_best = -1;
    double beta = 0.4;
    for (int i = 0; i < num_node; i++)
    {
        if (state[i] == 0)
        {
            for (int j = 1; j <= num_team; j++)
            {
                if (num_t_cur[j] < num_each_t) // if team have less  than Mt number of practitionar
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
                    else if ((fabs(product - max) < 0.01) && num_best < 50)
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
                            // d1 = i;
                            // d2 = k;
                            // num1 = 1;
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
                        // delta_eff = (eff[i] - eff[k]) + alpha*(cha2);
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
    // aspiration creterion
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
                delta_eff = delta_eff - p_factor * delta_inf_deg;
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
                delta_eff = delta_eff - p_factor * delta_inf_deg;
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
    // aspiration criterion
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
    dd = dd / ((num_node - 1) * num_node);
    min_div = d_min * num_each_t * (num_each_t - 1) / 2 * dd;
    cout << "dd=" << dd << " MinDiv= " << min_div << endl;
}

int Hyper_heuristic::randomInt(int n)
{
    return rand() % n;
}

// identify the index of the minimum number in the array aa
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

// identify the index of the second minimum number in the array aa
int Hyper_heuristic::sec_func(int *aa, int idx_min, int len)
{
    int min_value = MAXVALUE;
    int idx;
    for (int i = 1; i <= len; i++)
    { // cout<<aa[i];
        if (aa[i] < min_value && i != idx_min)
        {
            min_value = aa[i];
            idx = i;
        }
    }
    return idx;
}

// identify the index of the third minimum number in the array aa
int Hyper_heuristic::th_func(int *aa, int idx_min1, int idx_min2, int len)
{
    int min_value = MAXVALUE;
    int idx;
    for (int i = 1; i <= len; i++)
    { // cout<<aa[i];
        if (aa[i] < min_value && i != idx_min1 && i != idx_min2)
        {
            min_value = aa[i];
            idx = i;
        }
    }
    return idx;
}

// identify the index of the maximum number in the array aa
int Hyper_heuristic::max_func(int *aa, int len)
{
    int pos;
    int max_value = MINVALUE;
    for (int i = 0; i < len; i++)
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
    for (int i = 0; i < len; i++)
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
    // cout<<"begin repair solution "<<endl;
    // for (int t=1; t<=num_each_t; t++){
    int idx = min_func(w_div, num_team);
    min = w_div[idx];
    // cout<<idx;
    for (int i = 0; i < num_node; i++)
        for (int j = 0; j <= num_team; j++)
            tabu_list[i][j] = 0; //
    while (min < min_div)
    {
        best = swap_min(iter, c1, c2);
        node1 = c1;           /* move in */
        node2 = c2;           /* move out */
        team_min = state[c2]; /*team with min div*/
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

    // cout<<f_cur;
}

// generate an initial solution Greedy Construction
int Hyper_heuristic::generate_initial()
{
    // srand((unsigned)time(NULL));
    int *sort_eff = new int[num_node];
    int *best_node = new int[num_node];
    int *best_team = new int[num_node];
    int *arr_t = new int[num_team + 1];
    int num_best = -1;
    memset(state, 0, sizeof(int) * num_node); // initial all members in state array to team zero
    memset(best_solution, 0, sizeof(int) * num_node);
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
    {
        sort_eff[i] = eff[i];
    }
    for (int i = 0; i < num_node; i++)
    {
        int v = max_func(sort_eff, num_node);
        // cout<<"sort_eff="<< v <<"  ";
    }
    //       //cout<<endl<<endl;
    //    cout<<"----------------------------------------------------"<<endl;
    //	cout<<"(3)Starting Greedy Construction initial solution:"<<endl;
    int m = 1;
    while (1)
    {
        // allocate a practitioner with the with any efficiency to each crew
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
    // cout<<f_cur;
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
    // cout<<endl;
    // cout<<"---------------------------------------------------"<<endl;
    delete[] sort_eff;
    delete[] best_node;
    delete[] best_team;
    delete[] arr_t;
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
    memset(state, 0, sizeof(int) * num_node);
    memset(best_solution, 0, sizeof(int) * num_node);
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

    // cout<<"----------------------------------------------------"<<endl;
    // cout<<"(4)Starting Random Construction initial solution:"<<endl;
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

    delete[] sort_eff;
    delete[] best_node;
    delete[] best_team;
    delete[] arr_t;
    return 0;
}

void Hyper_heuristic::initialization(const string &current_file_name)
{
    ifstream fic;
    file_name = current_file_name; // Set the global file_name
    fic.open(file_name.c_str());
    if (fic.fail())
    {
        cout << "### Error opening file: " << file_name << endl;
        exit(0);
    }
    if (fic.eof())
    {
        cout << "### Error reading file: " << file_name << endl;
        exit(0);
    }
    char str_reading[100];
    double nn[4];
    for (int i = 0; i < 4; i++)
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
    // cout << " num_node= " << num_node << "  " << " num_team= " << num_team << ",num_each_t=" << num_each_t << "  " << " min_div= " << min_div << endl;

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
    div_in = new double *[num_node];
    degree_inf = new double[num_team + 1];
    for (int i = 0; i < num_node; i++)
        div_in[i] = new double[num_node];
    delta_div = new int *[num_node];
    for (int i = 0; i < num_node; i++)
        delta_div[i] = new int[num_team + 1];
    tabu_list = new int *[num_node];
    for (int i = 0; i < num_node; i++)
        tabu_list[i] = new int[num_team + 1];
    team = new int *[num_team + 1];
    for (int i = 0; i <= num_team; i++)
        team[i] = new int[num_node];
    team_check = new int *[num_team + 1];
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

int **Hyper_heuristic::iterated_local_search()
{
    // double avg_time =0.0;
    int ls_depth = 100; // or 200
    int iter = 0;
    int f_Sintial, f_Sbest, F_perturb, f_S, f_Spar, f_Spar2;
    double best;
    int node1, node2, team_min, team_old;
    int a1, a2;
    int d1 = 0, d2 = 0;
    int **Sinitial, **Sbest, **S, **Spar, **Spar2;
    // begin of iterated local search
    // generate_initial();//generate initial solution s0
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
    if (conv_file.is_open())
    {
        conv_file << "Iteration,Fbest\n";
    }
    // --------------------------------------------
    // start_time = clock();
    // while (1.0*(clock()-start_time)/CLOCKS_PER_SEC<time_limit)
    while (iter < ls_depth)
    { // start_time = clock();
        Spar = simulated_annealing();
        f_Spar = f_cur;
        f_Spar2 = feasible_local_search();
        Spar2 = team;
        // accept (S,S'')---->S=S''
        if (f_Spar2 > f_S)
        {
            f_S = f_Spar2;
            S = Spar2;
            fbest = f_Spar2;
            team = S;
            end_time = clock();
        }

        if (f_Spar2 > f_best_inn)
        {
            f_best_inn = f_Spar2;
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

        if (conv_file.is_open())
        {
            conv_file << iter + 1 << "," << f_best_inn << "\n";
        }
        iter++;
    }

    if (conv_file.is_open())
        conv_file.close();

    // std::cout << "Convergence data saved to: " << conv_path << std::endl;
    repair_solution();

    return team;
}

int **Hyper_heuristic::simulated_annealing()
{
    double best;
    int node1, node2, team_min, team_old;
    int a1, a2;
    double current_temp = initial_temp;
    int iter = 0;
    double f_s0, f_s1;
    int fbest1;
    int d1 = 0, d2 = 0;
    float Tmin = 0.001;

    int **S0_current = team; // initial solution (greedy constructed)
    int **S1_solution = nullptr;
    int **S1 = S0_current;

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
    if (conv_file.is_open())
    {
        conv_file << "Iteration,Fbest,Temperature\n";
    }
    // --------------------------------------------

    while (current_temp > Tmin)
    {
        while (iter < 100)
        {
            best = swap1(iter, d1, d2);
            node1 = d1;           /* move in */
            node2 = d2;           /* move out */
            team_min = state[d2]; /* team with min eff */
            team_old = state[d1];
            a1 = address[node1];
            a2 = address[node2];

            if (team_old == 0)
            {
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
            }
            else
            {
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
            if (delta_fitness >= 0)
            {
                end_time = clock();
                S0_current = S1_solution;
                f_s0 = f_s1;
            }
            else if (exp(-delta_fitness / current_temp) > (rand() / (double)RAND_MAX))
            {
                end_time = clock();
                S0_current = S1_solution;
                f_s0 = f_s1;
            }

            // ---- Update Best Solution ----
            if (f_s1 > fbest1)
            {
                fbest1 = f_s1;
                fbest = fbest1;
                end_time = clock();

                for (int m = 0; m < num_node; m++)
                {
                    best_solution[m] = state[m];
                    best_inn[m] = state[m];
                }
                for (int m = 1; m <= num_team; m++)
                {
                    eff_best[m] = w_eff[m];
                    div_best[m] = w_div[m];
                }
                // ---- Save Convergence ----
                convergence.push_back(fbest1);
            }

            current_temp *= cooling_rate;
            if (conv_file.is_open())
            {
                conv_file << iter << "," << fbest1 << "," << current_temp << "\n";
            }
        }
    }
    if (conv_file.is_open())
        conv_file.close();

    repair_solution();
    // std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return team;
}

int **Hyper_heuristic::clone_solution(int **src)
{
    return deep_copy_solution(src, num_node, num_team, num_each_t);
}

void Hyper_heuristic::copy_solution(int **dest, int **src)
{
    for (int t = 0; t <= num_team; t++)
        for (int j = 0; j < num_each_t; j++)
            dest[t][j] = src[t][j];
}

void Hyper_heuristic::free_solution(int **sol)
{
    for (int t = 0; t <= num_team; t++)
        delete[] sol[t];

    delete[] sol;
}

int **Hyper_heuristic::IteratedLocalSearch(int **Sstart, int OPj)
{
    int max_iter = 100;

    int **S = clone_solution(Sstart);
    team = S;
    feasible_local_search();
    int f_S = f_cur;

    int **Sbest = clone_solution(S);
    int f_best = f_S;

    // -------- Convergence Setup --------
    std::filesystem::path folder_path =
        "D:/Datasets/TRI_LEVEL_HH_MODELS/TRI_LEVEL_MHS_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::ofstream conv_file(folder_path / "IteratedLocalSearch_Convergence.csv");
    conv_file << "Iteration,Fbest\n";

    for (int iter = 0; iter < max_iter; iter++)
    {
        int **Spert = clone_solution(S);

        team = Spert;
        apply_LLHop(OPj, Spert);
        feasible_local_search();

        int f_new = f_cur;
        int div_new = f_cur_div;

        if (f_new > f_best && div_new >= min_div)
        {
            copy_solution(Sbest, Spert);
            f_best = f_new;
        }

        if (f_new >= f_S && div_new >= min_div)
        {
            copy_solution(S, Spert);
            f_S = f_new;
        }

        conv_file << iter + 1 << "," << f_best << "\n";
    }

    conv_file.close();
    team = Sbest;
    objective_Function(team);
    return team;
}


int **Hyper_heuristic::SimulatedAnnealing(int **Sstart, int OPj,
                                          double T0, double alpha)
{
    double T = T0;
    double Tmin = 0.001;

    int **S = clone_solution(Sstart);
    team = S;
    objective_Function(S);
    int f_S = f_cur;

    int **Sbest = clone_solution(S);
    int f_best = f_S;

    // -------- Convergence Setup --------
    std::filesystem::path folder_path =
        "D:/Datasets/TRI_LEVEL_HH_MODELS/TRI_LEVEL_MHS_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::ofstream conv_file(folder_path / "SimulatedAnnealing_Convergence.csv");
    conv_file << "Iteration,Fbest\n";

    int iter = 0;

    while (T > Tmin)
    {
        int **backup = clone_solution(S);

        team = S;
        apply_LLHop(OPj, S);
        objective_Function(S);
        int f_new = f_cur;
        int div_new = f_cur_div;

        bool accept = false;

        if (f_new > f_S && div_new >= min_div)
            accept = true;
        else
        {
            double delta = f_new - f_S;
            double prob = exp(delta / T);
            if ((double)rand() / RAND_MAX < prob && div_new >= min_div)
                accept = true;
        }

        if (accept)
        {
            f_S = f_new;
            if (f_new > f_best && div_new >= min_div)
            {
                copy_solution(Sbest, S);
                f_best = f_new;
            }
        }
        else
        {
            copy_solution(S, backup);
        }

        conv_file << iter + 1 << "," << f_best << "\n";

        T *= alpha;
        iter++;
    }

    conv_file.close();
    team = Sbest;
    objective_Function(team);
    return team;
}

int **Hyper_heuristic::GreatDeluge(int **Sstart, int OPj,
                                   double level0, double rainSpeed)
{
    int max_iter = 100;

    int **S = clone_solution(Sstart);
    team = S;
    objective_Function(S);
    int f_S = f_cur;

    int **Sbest = clone_solution(S);
    int f_best = f_S;

    double level = level0;

    std::filesystem::path folder_path =
        "D:/Datasets/TRI_LEVEL_HH_MODELS/TRI_LEVEL_MHS_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::ofstream conv_file(folder_path / "GreatDeluge_Convergence.csv");
    conv_file << "Iteration,Fbest\n";

    for (int iter = 0; iter < max_iter; iter++)
    {
        int **backup = clone_solution(S);

        team = S;
        apply_LLHop(OPj, S);
        objective_Function(S);

        int f_new = f_cur;
        int div_new = f_cur_div;

        if (f_new > f_best && div_new >= min_div)
        {
            copy_solution(Sbest, S);
            f_best = f_new;
        }

        if ((f_new >= level) && div_new >= min_div)
            f_S = f_new;
        else
            copy_solution(S, backup);

        conv_file << iter + 1 << "," << f_best << "\n";

        level -= rainSpeed;
    }

    conv_file.close();
    team = Sbest;
    objective_Function(team);
    return team;
}

int **Hyper_heuristic::LateAcceptance(int **Sstart, int OPj, int Lwindow)
{
    int max_iter = 100;

    int **S = clone_solution(Sstart);
    team = S;
    objective_Function(S);
    int f_S = f_cur;

    int **Sbest = clone_solution(S);
    int f_best = f_S;

    std::vector<int> history(Lwindow, f_S);

    std::filesystem::path folder_path =
        "D:/Datasets/TRI_LEVEL_HH_MODELS/TRI_LEVEL_MHS_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::ofstream conv_file(folder_path / "LateAcceptance_Convergence.csv");
    conv_file << "Iteration,Fbest\n";

    for (int iter = 0; iter < max_iter; iter++)
    {
        int **backup = clone_solution(S);

        team = S;
        apply_LLHop(OPj, S);
        objective_Function(S);

        int f_new = f_cur;
        int div_new = f_cur_div;

        if (f_new > f_best && div_new >= min_div)
        {
            copy_solution(Sbest, S);
            f_best = f_new;
        }

        int idx = iter % Lwindow;

        if (f_new >= history[idx] && div_new >= min_div)
        {
            f_S = f_new;
        }
        else
        {
            copy_solution(S, backup);
        }

        history[idx] = f_S;

        conv_file << iter + 1 << "," << f_best << "\n";
    }

    conv_file.close();
    team = Sbest;
    objective_Function(team);
    return team;
}

// ======================================================================
//  FITS: Lightweight Tabu Search on CMCEE solution (NO deep copies)
//  - Uses ONLY team[][]
//  - Neighbors generated by apply_LLHop(OPj, team)
//  - Tabu list stores only forbidden operators
// ======================================================================
int **Hyper_heuristic::TabuSearch(int **Sstart, int OPj)
{
    int max_iter = 100;
    int tabu_tenure = 10;

    int **S = clone_solution(Sstart);
    team = S;
    objective_Function(S);
    int f_S = f_cur;

    int **Sbest = clone_solution(S);
    int f_best = f_S;

    std::deque<int> tabu_list;

    std::filesystem::path folder_path =
        "D:/Datasets/TRI_LEVEL_HH_MODELS/TRI_LEVEL_MHS_Convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::ofstream conv_file(folder_path / "TabuSearch_Convergence.csv");
    conv_file << "Iteration,Fbest\n";

    for (int iter = 0; iter < max_iter; iter++)
    {
        int **backup = clone_solution(S);

        if (std::find(tabu_list.begin(), tabu_list.end(), OPj) != tabu_list.end())
            continue;

        team = S;
        apply_LLHop(OPj, S);
        objective_Function(S);

        int f_new = f_cur;
        int div_new = f_cur_div;

        if (f_new > f_best && div_new >= min_div)
        {
            copy_solution(Sbest, S);
            f_best = f_new;
        }

        f_S = f_new;

        tabu_list.push_back(OPj);
        if ((int)tabu_list.size() > tabu_tenure)
            tabu_list.pop_front();

        conv_file << iter + 1 << "," << f_best << "\n";
    }

    conv_file.close();
    team = Sbest;
    objective_Function(team);
    return team;
}


void Hyper_heuristic::display(int **team)
{
    cout << endl;
    cout << endl;
    cout << "team 0:";
    for (int i = 0; i < num_node - (num_team * num_each_t); ++i)
    {
        cout << team[0][i] << " ";
    }
    cout << endl;
    cout << endl;
    for (int t = 1; t <= num_team; t++)
    {
        cout << "team " << t << ": ";
        for (int j = 0; j < num_each_t; j++)
        {
            cout << team[t][j] << "\t";
        }
        cout << "eff=" << w_eff[t] << "\t\t" << "div=" << w_div[t];
        cout << endl;
    }
    // check_best_solution();
    cout << endl;
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
    // cout<<"team_check[i][i]"<<team_check[i][i]<<" ";
    for (int i = 0; i <= num_team; i++)
    {
        aa[i] = 0;
        for (int j = 0; j < num_node; j++)
        {
            if (fbest_solution[j] == i)
            { // cout<<fbest_solution[j] ;
                team_check[i][aa[i]] = j;
                // cout<<aa[i]<<" ";
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
    // int sum_eff[num_team + 1], sum_div[num_team + 1];

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
            cout << " you are wrong with w_eff of team: " << i << endl;
            getchar();
        }
    }
    cout << endl;
    cout << endl;
    for (int i = 1; i <= num_team; i++)
    {
        if (sum_div[i] != div_fbest[i])
        {
            cout << " you are wrong with w_div of team: " << i << endl;
            getchar();
        }
    }
    int t = 0;
    cout << "team_check " << t << ":  ";
    for (int j = 0; j < aa[t]; j++)
    {
        cout << team_check[t][j] << "\t";
    }
    cout << endl;
    cout << endl;

    for (int i = 1; i <= num_team; i++)
    {
        cout << "team_check " << i << ":  ";
        for (int j = 0; j < aa[i]; j++)
        {

            cout << team_check[i][j] << "\t";
        }
        cout << "sum_eff=" << sum_eff[i] << " ";
        cout << "sum_div=" << sum_div[i];
        cout << endl
             << endl;
    }
    cout << endl;
    cout << " finish check best solution " << endl;
    cout << "---------------------------------" << endl;
    delete[] aa;
    delete[] sum_eff;
    delete[] sum_div;
}

int Hyper_heuristic::feasible_local_search()
{
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
    if (conv_file.is_open())
    {
        conv_file << "Iteration,Fbest\n";
    }
    for (int i = 0; i < num_node; i++)
        for (int j = 0; j <= num_team; j++)
            tabu_list[i][j] = 0;
    while (iter < fls_depth)
    {
        best = swap1(iter, d1, d2);
        node1 = d1; /* move in */
        node2 = d2; /* move out */
        // cout<<"d1="<<d1<<"  "<<"d2="<<d2<<"\n";
        team_min = state[d2]; /*team with min eff*/
        team_old = state[d1];
        a1 = address[node1];
        a2 = address[node2];
        // cout << "in tabu method, iter=" << iter << ",node1=" << node1 << ",node2=" << node2 << endl;
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
        if (f_cur > f_best_inn)
        {
            f_best_inn = f_cur;
            convergence.push_back(f_best_inn);
            for (int m = 0; m < num_node; m++)
                best_inn[m] = state[m];
        }

        // ---------- NEW: Save Convergence ----------
        if (conv_file.is_open())
        {
            conv_file << iter << "," << fbest << "\n";
        }
        // --------------------------------------------
    }

    int idx11 = min_func(w_div, num_team);
    f_cur_div = w_div[idx11];
    repair_solution();

    if (conv_file.is_open())
        conv_file.close();

    // std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return f_best_inn;
}

int **Hyper_heuristic::local_search()
{
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
    if (conv_file.is_open())
    {
        conv_file << "Iteration,Fbest\n";
    }
    for (int i = 0; i < num_node; i++)
        for (int j = 0; j <= num_team; j++)
            tabu_list[i][j] = 0;
    while (iter < fls_depth)
    {
        best = swap1(iter, d1, d2);
        node1 = d1; /* move in */
        node2 = d2; /* move out */
        // cout<<"d1="<<d1<<"  "<<"d2="<<d2<<"\n";
        team_min = state[d2]; /*team with min eff*/
        team_old = state[d1];
        a1 = address[node1];
        a2 = address[node2];
        // cout << "in tabu method, iter=" << iter << ",node1=" << node1 << ",node2=" << node2 << endl;
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
        if (f_cur > f_best_inn)
        {
            f_best_inn = f_cur;
            convergence.push_back(f_best_inn);
            for (int m = 0; m < num_node; m++)
                best_inn[m] = state[m];
        }

        // ---------- NEW: Save Convergence ----------
        if (conv_file.is_open())
        {
            conv_file << iter << "," << fbest << "\n";
        }
        // --------------------------------------------
    }

    int idx11 = min_func(w_div, num_team);
    f_cur_div = w_div[idx11];
    repair_solution();

    if (conv_file.is_open())
        conv_file.close();

    // std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return team;
}

int **Hyper_heuristic::infeasible_local_search()
{
    double best;
    int node1, node2, team_min, team_old;
    int a1, a2;
    int d1 = 0, d2 = 0, d3 = 0;
    int iter = 0;
    // ---------- NEW: Convergence Setup ----------
    std::vector<int> convergence;
    convergence.push_back(f_cur);

    std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "infeasible_local_search_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open())
    {
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
        node1 = d1;           /* move in */
        node2 = d2;           /* move out */
        team_min = state[d2]; /*team with min eff*/
        team_old = state[d1];
        a1 = address[node1];
        a2 = address[node2];
        // cout << "in tabu method, iter=" << iter << ",node1=" << node1 << ",node2=" << node2 << endl;
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

        if (deg_cur > 0) // infeasible solution
            p_count++;
        if (iter % lamba == 0)
        {
            if (p_count > u1)
                //	p_factor *= tow;
                p_factor += tow;
            else if (p_count < u2)
            // p_factor /= tow;
            {
                p_factor -= tow;
                if (p_factor < 0)
                    p_factor = 1;
            }
            p_count = 0;
        }
        // ---------- NEW: Save Convergence ----------
        convergence.push_back(fbest);
        if (conv_file.is_open())
        {
            conv_file << iter << "," << fbest << "\n";
        }
        // --------------------------------------------
    }

    repair_solution();

    if (conv_file.is_open())
        conv_file.close();

    // std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return team;
}

// Tabu Search based feasible
int **Hyper_heuristic::fits()
{
    f_best_inn = 0;
    feasible_local_search();
    infeasible_local_search();
    repair_solution();
    end_time = clock();
    return team;
}

// initialize the population
void Hyper_heuristic::initial_population()
{
    for (int i = 0; i < Pop_Num; i++)
    {
        for (int m = 0; m < num_node; m++)
            pop[i].p[m] = state[m];
        pop[i].cost = f_cur;
    }
}

// backbone crossover: based on maximum match grouping
void Hyper_heuristic::cross_over2()
{
    int **arr1 = new int *[num_team + 1];
    int **arr2 = new int *[num_team + 1];
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
        state[i] = 0; // crew 0 preserves the practitioners that are not allocated
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
        // cout<<"clu1="<<clu1<<endl;
        arr1[clu1][len1[clu1]++] = i;
        // cout<<"arr1="<<arr1[clu1][len1[clu1]++];
        int clu2 = pop[parent2].p[i];
        arr2[clu2][len2[clu2]++] = i;
    }

    // maximum match
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
                        // index = 0;
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
                                            // index = n;
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
                if (ver == ver2) // marked the practitioner is being allocated to the offspring solution
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
        while (len000 < num_each_t && (len111 > 0 || len222 > 0))
        {
            cont_best = 0;
            if (coin % 2 == 0) // selected from parent 1
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
                            if (cont * eff[ver] > cont_best)
                            {
                                cont_best = cont * eff[ver];
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
            else // selected from parent 2
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
                            if (cont * eff[ver] > cont_best)
                            {
                                cont_best = cont * eff[ver];
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

    // greedy allocate the unallocted practitioners£¬and repair infeasible solution
    int unass_len2 = unassLen;
    while (unass_len2 > (num_node - num_team * num_each_t))
    {
        // int ver = unassV[rand() % unassLen];
        // int k = rand() % num_team + 1;
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
        if (state[ver] == 0 && len1[crew] < num_each_t)
        {
            state[ver] = crew;
            len1[crew]++;
            unass_len2--;
        }
        // cout << "unass_len2=" << unass_len2 << endl;
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
    for (int i = 0; i < num_team + 1; i++)
    {
        delete[] arr1[i];
    }
    delete[] arr1;
    for (int i = 0; i < num_team + 1; i++)
    {
        delete[] arr2[i];
    }
    delete[] arr2;
    delete[] len1;
    delete[] len2;
    delete[] match;
    delete[] flagC1;
    delete[] flagC2;
    delete[] flagV;
    delete[] unassV;
    delete[] addressUnaV;
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
            // cout<<"min_cost="<<min_cost<<"  " <<pop[i].cost<<endl;
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

int **Hyper_heuristic::memetic()
{
    int gen = 0;
    fbest = 0;

    pop = new population[Pop_Num]; // population number 10
    for (int i = 0; i < Pop_Num; i++)
    {
        pop[i].p = new int[num_node];
    }
    initial_population();
    // while (1.0*(clock()-start_time)/CLOCKS_PER_SEC<time_limit)
    while (gen < generations) // generation 50
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
    // beta = 0.4
    tabu_tenure = 10;
    generations = 50;
    fls_depth = 4000;
    ils_depth = 1000;
    int f1, f2, f3, f4, f5, f6, f7, f8, f9;
    double f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23;
    double f24, f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39;
}

void free_memory()
{
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
    delete[] degree_inf;
    for (int i = 0; i < num_node; i++)
    {
        delete[] div_in[i];
    }
    delete[] div_in;
    for (int i = 0; i < num_node; i++)
    {
        delete[] delta_div[i];
    }
    delete[] delta_div;
    for (int i = 0; i < num_node; i++)
    {
        delete[] tabu_list[i];
    }
    delete[] tabu_list;
    for (int i = 0; i < num_team + 1; i++)
    {

        delete[] team[i];

        delete[] team_check[i];
    }
    delete[] team;
    delete[] team_check;
    for (int i = 0; i < Pop_Num; i++)
    {
        delete[] pop[i].p;
    }
    delete[] pop;
}

// Implementation of the Selection Hyper-Heuristic Strategies Framework
void Hyper_heuristic::objective_Function(int **team)
{

    // Always recompute team 0
    w_eff[0] = 0;
    w_div[0] = 0;

    // Compute efficiency and diversity for teams 1 .. num_team
    for (int t = 1; t <= num_team; t++)
    {
        w_eff[t] = 0.0;
        w_div[t] = 0.0;

        for (int j = 0; j < num_each_t; j++)
        {
            int node_j = team[t][j];
            w_eff[t] += eff[node_j];

            for (int k = j + 1; k < num_each_t; k++)
            {
                int node_k = team[t][k];
                w_div[t] += div_in[node_j][node_k];
            }
        }
    }

    // Determine current objective values
    int idx_eff = min_func(w_eff, num_team);
    int idx_div = min_func(w_div, num_team);

    f_cur = w_eff[idx_eff];
    f_cur_div = w_div[idx_div];

     // Fix membership mapping (ensures no corruption)
     for (int t = 0; t <= num_team; t++) {
         for (int j = 0; j < num_each_t; j++) {
             int node = team[t][j];
             state[node] = t;
             address[node] = j;
         }
     }

    // Save best solution
    for (int m = 0; m < num_node; m++) {
        best_solution[m] = state[m];
        best_inn[m] = state[m];
    }

    for (int m = 1; m <= num_team; m++) {
        eff_best[m] = w_eff[m];
        div_best[m] = w_div[m];
    }
    repair_solution();
}

void Hyper_heuristic::objective_Function1(int **team)
{

    // Determine the best solution and update f_cur
    int idx = min_func(w_eff, num_team);
    f_cur = w_eff[idx];

    // Determine the best solution and update f_cur_div
    int idx3 = min_func(w_div, num_team);
    f_cur_div = w_div[idx3];
    // Update best_solution and best_inn
    for (int m = 0; m < num_node; m++)
    {
        best_solution[m] = state[m];
        best_inn[m] = state[m];
    }
    for (int m = 1; m <= num_team; m++)
    {
        eff_best[m] = w_eff[m];
        div_best[m] = w_div[m];
    }
    // Update f_cur after repair or infeasible local search
}

int **Hyper_heuristic::great_deluge_algorithm()
{
    int max_iter = 100;
    double decay_rate = 0.98; // water level multiplier (< 1.0)
    int iter = 0;

    int f_s = f_cur;                         // current solution value
    fbest = f_cur;                           // global best
    double level = static_cast<double>(f_s); // water level (starts at current)

    // (optional) convergence log
    std::vector<int> convergence;
    convergence.push_back(f_s);

    // backups for rejection
    std::vector<int> state_bak(num_node);
    std::vector<int> address_bak(num_node);
    std::vector<double> w_eff_bak(num_team + 1), w_div_bak(num_team + 1);
    std::vector<std::vector<int>> team_bak(num_team + 1, std::vector<int>(num_node));

    // ---------- NEW: Convergence File ----------
    std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "GreatDeluge_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open())
    {
        conv_file << "Iteration,Fbest,WaterLevel\n";
    }
    // --------------------------------------------

    while (iter < max_iter)
    {
        // ---- backup current state ----
        for (int i = 0; i < num_node; i++)
        {
            state_bak[i] = state[i];
            address_bak[i] = address[i];
        }
        for (int t = 0; t <= num_team; t++)
        {
            for (int p = 0; p < num_node; p++)
                team_bak[t][p] = team[t][p];
        }
        for (int k = 1; k <= num_team; k++)
        {
            w_eff_bak[k] = w_eff[k];
            w_div_bak[k] = w_div[k];
        }

        // ---- propose a move (produces a new state/fitness) ----
        int f_new = feasible_local_search();

        // ---- acceptance test (maximize): accept if f_new >= level ----
        if (static_cast<double>(f_new) >= level)
        {
            f_s = f_new;
            if (f_s > fbest)
            {
                fbest = f_s;
                for (int m = 0; m < num_node; m++)
                {
                    best_solution[m] = state[m];
                    best_inn[m] = state[m];
                }
                for (int i = 1; i <= num_team; i++)
                {
                    eff_best[i] = w_eff[i];
                    div_best[i] = w_div[i];
                }
            }
        }
        else
        {
            // ---- reject: restore previous state ----
            for (int i = 0; i < num_node; i++)
            {
                state[i] = state_bak[i];
                address[i] = address_bak[i];
            }
            for (int t = 0; t <= num_team; t++)
            {
                for (int p = 0; p < num_node; p++)
                    team[t][p] = team_bak[t][p];
            }
            for (int k = 1; k <= num_team; k++)
            {
                w_eff[k] = w_eff_bak[k];
                w_div[k] = w_div_bak[k];
            }
            int idx = min_func(w_eff, num_team); // recompute current value
            f_cur = static_cast<int>(w_eff[idx]);
        }

        // ---- lower the water level ----
        level *= decay_rate; // decreases over time
        convergence.push_back(fbest);

        // ---- log convergence ----
        if (conv_file.is_open())
        {
            conv_file << iter << "," << fbest << "," << level << "\n";
        }

        iter++;
    }

    if (conv_file.is_open())
        conv_file.close();

    // std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return team;
}

int Hyper_heuristic::guided_local_search()
{
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
    if (conv_file.is_open())
    {
        conv_file << "Iteration,Fbest\n";
    }
    // --------------------------------------------

    auto augmented_cost = [&](int i, int j)
    {
        return -eff[i] + lambda_penalty * penalties[i][j];
    };

    for (int iter = 0; iter < max_iter; iter++)
    {
        double f_base = feasible_local_search();

        // ---- Update penalties ----
        for (int i = 0; i < num_node; i++)
        {
            for (int j = i + 1; j < num_node; j++)
            {
                if (augmented_cost(i, j) > 0)
                {
                    penalties[i][j] += 1;
                }
            }
        }

        // ---- Update best solution ----
        if (f_base > fbest)
        {
            fbest = f_base;
            for (int i = 1; i <= num_team; i++)
            {
                eff_best[i] = w_eff[i];
                div_best[i] = w_div[i];
            }
            for (int m = 0; m < num_node; m++)
            {
                best_solution[m] = state[m];
                best_inn[m] = state[m];
            }
        }

        // ---- Record convergence ----
        convergence.push_back(fbest);
        if (conv_file.is_open())
        {
            conv_file << iter << "," << fbest << "\n";
        }
    }

    if (conv_file.is_open())
        conv_file.close();

    // std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return fbest;
}

int **Hyper_heuristic::late_acceptance_hill_climbing()
{
    int L = 50;
    int max_iter = 100;
    std::vector<int> history(L, f_cur);
    int f_s = f_cur;
    std::vector<int> convergence = {f_s};
    int iter = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // ---------- NEW: Convergence File ----------
    std::filesystem::path folder_path = "D:/Datasets/MHS_Algorithms_convergence/";
    if (!std::filesystem::exists(folder_path))
        std::filesystem::create_directories(folder_path);

    std::filesystem::path conv_path = folder_path / "LateAcceptance_Convergence.csv";
    std::ofstream conv_file(conv_path);
    if (conv_file.is_open())
    {
        conv_file << "Iteration,Fbest\n";
    }
    // --------------------------------------------

    while (iter < max_iter)
    {
        int f_new = feasible_local_search();

        // ---- Acceptance condition ----
        if (f_new >= history[iter % L])
        {
            f_s = f_new;

            // ---- Update best solution ----
            if (f_s > fbest)
            {
                fbest = f_s;
                for (int i = 1; i <= num_team; i++)
                {
                    eff_best[i] = w_eff[i];
                    div_best[i] = w_div[i];
                }
                for (int m = 0; m < num_node; m++)
                {
                    best_solution[m] = state[m];
                    best_inn[m] = state[m];
                }
            }
        }

        history[iter % L] = f_s;
        convergence.push_back(fbest);

        // ---- Log convergence ----
        if (conv_file.is_open())
        {
            conv_file << iter << "," << fbest << "\n";
        }

        iter++;
    }

    if (conv_file.is_open())
        conv_file.close();

    // std::cout << "Convergence data saved to: " << conv_path << std::endl;
    return team;
}

// ---------------- Random Generator ----------------
inline std::mt19937 &hh_rng()
{
    static thread_local std::mt19937 gen(
        static_cast<unsigned>(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch().count()));
    return gen;
}


// ---------------- Pool Size (STATIC — safer) ----------------
inline int pool_size_unallocated()
{
    // Since team sizes are fixed
    return num_node - (num_team * num_each_t);
}


// ---------------- Random Allocated Team ----------------
inline int rand_alloc_team()
{
    if (num_team <= 0)
        return 1;   // safe fallback

    std::uniform_int_distribution<int> dist(1, num_team);
    return dist(hh_rng());
}


// ---------------- Two Distinct Allocated Teams ----------------
inline std::pair<int, int> rand_two_distinct_alloc_teams()
{
    if (num_team <= 1)
        return {1, 1};

    std::uniform_int_distribution<int> dist(1, num_team);

    int a = dist(hh_rng());
    int b = dist(hh_rng());

    if (num_team == 2)
    {
        b = (a == 1 ? 2 : 1);
    }
    else
    {
        while (b == a)
            b = dist(hh_rng());
    }

    return {a, b};
}


// ---------------- Random Member Index ----------------
inline int rand_member_idx()
{
    if (num_each_t <= 0)
        return 0;   // safe fallback

    std::uniform_int_distribution<int> dist(0, num_each_t - 1);
    return dist(hh_rng());
}


// ---------------- Two Distinct Member Indices ----------------
inline std::pair<int, int> rand_two_distinct_member_idx()
{
    if (num_each_t <= 1)
        return {0, 0};

    std::uniform_int_distribution<int> dist(0, num_each_t - 1);

    int i = dist(hh_rng());
    int j = dist(hh_rng());

    if (num_each_t == 2)
    {
        j = (i == 0 ? 1 : 0);
    }
    else
    {
        while (j == i)
            j = dist(hh_rng());
    }

    return {i, j};
}


// ---------------- Random Pool Index ----------------
inline int rand_pool_idx(int poolSize)
{
    if (poolSize <= 0)
        return 0;   // safe fallback

    std::uniform_int_distribution<int> dist(0, poolSize - 1);
    return dist(hh_rng());
}


// ---------------- Two Distinct Pool Indices ----------------
inline std::pair<int, int> rand_two_distinct_pool_idx(int poolSize)
{
    if (poolSize <= 1)
        return {0, 0};

    std::uniform_int_distribution<int> dist(0, poolSize - 1);

    int a = dist(hh_rng());
    int b = dist(hh_rng());

    if (poolSize == 2)
    {
        b = (a == 0 ? 1 : 0);
    }
    else
    {
        while (b == a)
            b = dist(hh_rng());
    }

    return {a, b};
}


void Hyper_heuristic::apply_atomic_swap(int node_in, int node_out)
{
    int team_new = state[node_out];
    int team_old = state[node_in];

    int pos_out = address[node_out];
    int pos_in  = address[node_in];

    // ---- Efficiency ----
    w_eff[team_new] += eff[node_in] - eff[node_out];

    if (team_old != 0)
        w_eff[team_old] -= eff[node_in] - eff[node_out];

    // ---- Diversity ----
    w_div[team_new] += delta_div[node_in][team_new]
                     - delta_div[node_out][team_new]
                     - div_in[node_in][node_out];

    if (team_old != 0)
        w_div[team_old] += delta_div[node_out][team_old]
                         - delta_div[node_in][team_old]
                         - div_in[node_in][node_out];

    // ---- Apply move ----
    team[team_new][pos_out] = node_in;
    team[team_old][pos_in]  = node_out;

    // ---- Update delta structure ----
    update_delta(node_in, team_new, team_old);
    update_delta(node_out, team_old, team_new);

    // ---- Update state ----
    state[node_in]  = team_new;
    state[node_out] = team_old;

    address[node_in]  = pos_out;
    address[node_out] = pos_in;
}


int **Hyper_heuristic::LLH1(int **team)
{
    // LLH1:
    // Random team ↔ pool swap (atomic, literature-consistent)

    int team0 = 0;
    if (pool_size_unallocated() == 0) return team;

    int t = rand_alloc_team();
    int idx_team = rand_member_idx();
    int idx_pool = rand_pool_idx(pool_size_unallocated());

    int node1 = team[team0][idx_pool];   // move in
    int node2 = team[t][idx_team];       // move out

    int team_old = team0;
    int team_new = t;

    int a1 = idx_pool;
    int a2 = idx_team;

    // ---- Efficiency ----
    w_eff[team_new] += eff[node1] - eff[node2];

    // ---- Diversity ----
    w_div[team_new] += delta_div[node1][team_new]
                     - delta_div[node2][team_new]
                     - div_in[node1][node2];

    // ---- Apply ----
    team[team_new][a2] = node1;
    team[team_old][a1] = node2;

    // ---- Delta ----
    update_delta(node1, team_new, team_old);
    update_delta(node2, team_old, team_new);

    // ---- State ----
    state[node1] = team_new;
    state[node2] = team_old;

    address[node1] = a2;
    address[node2] = a1;

    return team;
}


int **Hyper_heuristic::LLH2(int **team)
{
    int team0 = 0;
    int poolSize = pool_size_unallocated();

    if (poolSize < 2)
        return team;

    // =========================
    // ---- SWAP 1 ------------
    // =========================

    int t1 = rand_alloc_team();
    int idx1 = rand_member_idx();
    int p1 = rand_pool_idx(poolSize);

    int node_pool1 = team[team0][p1];
    int node_team1 = team[t1][idx1];

    // Update efficiency
    w_eff[t1] += eff[node_pool1] - eff[node_team1];

    // Update diversity
    w_div[t1] += delta_div[node_pool1][t1]
               - delta_div[node_team1][t1]
               - div_in[node_pool1][node_team1];

    // Apply swap
    team[t1][idx1] = node_pool1;
    team[team0][p1] = node_team1;

    update_delta(node_pool1, t1, team0);
    update_delta(node_team1, team0, t1);

    state[node_pool1] = t1;
    state[node_team1] = team0;

    address[node_pool1] = idx1;
    address[node_team1] = p1;


    // =========================
    // ---- SWAP 2 ------------
    // =========================

    poolSize = pool_size_unallocated();   // recompute (safe)

    if (poolSize < 1)
        return team;

    int t2;
    do {
        t2 = rand_alloc_team();
    } while (t2 == t1);   // ensure different team

    int idx2 = rand_member_idx();

    int p2;
    do {
        p2 = rand_pool_idx(poolSize);
    } while (p2 == p1);   // ensure different pool index

    int node_pool2 = team[team0][p2];
    int node_team2 = team[t2][idx2];

    // Update efficiency
    w_eff[t2] += eff[node_pool2] - eff[node_team2];

    // Update diversity
    w_div[t2] += delta_div[node_pool2][t2]
               - delta_div[node_team2][t2]
               - div_in[node_pool2][node_team2];

    // Apply swap
    team[t2][idx2] = node_pool2;
    team[team0][p2] = node_team2;

    update_delta(node_pool2, t2, team0);
    update_delta(node_team2, team0, t2);

    state[node_pool2] = t2;
    state[node_team2] = team0;

    address[node_pool2] = idx2;
    address[node_team2] = p2;

    return team;
}


int **Hyper_heuristic::LLH3(int **team)
{
    int team0 = 0;
    int poolSize = pool_size_unallocated();

    if (poolSize < 2)
        return team;

    int t = rand_alloc_team();

    // =========================
    // ---- SWAP 1 ------------
    // =========================

    int idx1 = rand_member_idx();
    int p1 = rand_pool_idx(poolSize);

    int node_pool1 = team[team0][p1];
    int node_team1 = team[t][idx1];

    w_eff[t] += eff[node_pool1] - eff[node_team1];

    w_div[t] += delta_div[node_pool1][t]
              - delta_div[node_team1][t]
              - div_in[node_pool1][node_team1];

    team[t][idx1] = node_pool1;
    team[team0][p1] = node_team1;

    update_delta(node_pool1, t, team0);
    update_delta(node_team1, team0, t);

    state[node_pool1] = t;
    state[node_team1] = team0;

    address[node_pool1] = idx1;
    address[node_team1] = p1;


    // =========================
    // ---- SWAP 2 ------------
    // =========================

    poolSize = pool_size_unallocated();

    int idx2;
    do {
        idx2 = rand_member_idx();
    } while (idx2 == idx1);

    int p2;
    do {
        p2 = rand_pool_idx(poolSize);
    } while (p2 == p1);

    int node_pool2 = team[team0][p2];
    int node_team2 = team[t][idx2];

    w_eff[t] += eff[node_pool2] - eff[node_team2];

    w_div[t] += delta_div[node_pool2][t]
              - delta_div[node_team2][t]
              - div_in[node_pool2][node_team2];

    team[t][idx2] = node_pool2;
    team[team0][p2] = node_team2;

    update_delta(node_pool2, t, team0);
    update_delta(node_team2, team0, t);

    state[node_pool2] = t;
    state[node_team2] = team0;

    address[node_pool2] = idx2;
    address[node_team2] = p2;

    return team;
}

int **Hyper_heuristic::LLH4(int **team)
{
    // LLH4:
    // Atomic team-to-team swap (literature style)

    auto [t1, t2] = rand_two_distinct_alloc_teams();

    int idx1 = rand_member_idx();
    int idx2 = rand_member_idx();

    int node1 = team[t1][idx1];
    int node2 = team[t2][idx2];

    int team_min = t1;
    int team_old = t2;

    int a1 = idx1;
    int a2 = idx2;

    // ---- Efficiency ----
    w_eff[team_min] += eff[node2] - eff[node1];
    w_eff[team_old] -= eff[node2] - eff[node1];

    // ---- Diversity ----
    w_div[team_min] += delta_div[node2][team_min]
                     - delta_div[node1][team_min]
                     - div_in[node1][node2];

    w_div[team_old] += delta_div[node1][team_old]
                     - delta_div[node2][team_old]
                     - div_in[node1][node2];

    // ---- Apply ----
    team[team_min][a1] = node2;
    team[team_old][a2] = node1;

    update_delta(node1, team_old, team_min);
    update_delta(node2, team_min, team_old);

    state[node1] = team_old;
    state[node2] = team_min;

    address[node1] = a2;
    address[node2] = a1;

    return team;
}


int **Hyper_heuristic::LLH5(int **team)
{
    // LLH5:
    // Best improving pool-to-min-eff team atomic swap

    int team0 = 0;
    if (pool_size_unallocated() == 0) return team;

    int tmin = min_func(w_eff, num_team);

    int best_in = -1, best_out = -1;
    double best_delta = MINVALUE;

    for (int i = 0; i < num_node; i++)
    {
        if (state[i] == 0)
        {
            for (int j = 0; j < num_each_t; j++)
            {
                int k = team[tmin][j];
                double delta = eff[i] - eff[k];

                if (delta > best_delta)
                {
                    best_delta = delta;
                    best_in = i;
                    best_out = k;
                }
            }
        }
    }

    if (best_in == -1) return team;

    int node1 = best_in;
    int node2 = best_out;

    int team_new = tmin;
    int team_old = 0;

    int a1 = address[node1];
    int a2 = address[node2];

    w_eff[team_new] += eff[node1] - eff[node2];

    w_div[team_new] += delta_div[node1][team_new]
                     - delta_div[node2][team_new]
                     - div_in[node1][node2];

    team[team_new][a2] = node1;
    team[team_old][a1] = node2;

    update_delta(node1, team_new, team_old);
    update_delta(node2, team_old, team_new);

    state[node1] = team_new;
    state[node2] = team_old;

    address[node1] = a2;
    address[node2] = a1;

    return team;
}

int **Hyper_heuristic::LLH6(int **team)
{
    int team0 = 0;
    if (pool_size_unallocated() == 0) return team;

    int tmin = min_func(w_eff, num_team);

    int idx_team = rand_member_idx();
    int idx_pool = rand_pool_idx(pool_size_unallocated());

    int node1 = team[team0][idx_pool];
    int node2 = team[tmin][idx_team];

    int team_new = tmin;
    int team_old = 0;

    int a1 = idx_pool;
    int a2 = idx_team;

    w_eff[team_new] += eff[node1] - eff[node2];

    w_div[team_new] += delta_div[node1][team_new]
                     - delta_div[node2][team_new]
                     - div_in[node1][node2];

    team[team_new][a2] = node1;
    team[team_old][a1] = node2;

    update_delta(node1, team_new, team_old);
    update_delta(node2, team_old, team_new);

    state[node1] = team_new;
    state[node2] = team_old;

    address[node1] = a2;
    address[node2] = a1;

    return team;
}


int **Hyper_heuristic::LLH7(int **team)
{
    int tmin = min_func(w_eff, num_team);
    int tmax = max_func(w_eff, num_team);

    if (tmin == tmax) return team;

    int idx1 = rand_member_idx();
    int idx2 = rand_member_idx();

    int node1 = team[tmax][idx2];
    int node2 = team[tmin][idx1];

    int team_new = tmin;
    int team_old = tmax;

    int a1 = idx2;
    int a2 = idx1;

    w_eff[team_new] += eff[node1] - eff[node2];
    w_eff[team_old] -= eff[node1] - eff[node2];

    w_div[team_new] += delta_div[node1][team_new]
                     - delta_div[node2][team_new]
                     - div_in[node1][node2];

    w_div[team_old] += delta_div[node2][team_old]
                     - delta_div[node1][team_old]
                     - div_in[node1][node2];

    team[team_new][a2] = node1;
    team[team_old][a1] = node2;

    update_delta(node1, team_new, team_old);
    update_delta(node2, team_old, team_new);

    state[node1] = team_new;
    state[node2] = team_old;

    address[node1] = a2;
    address[node2] = a1;

    return team;
}

int **Hyper_heuristic::LLH8(int **team)
{
    auto [t1, t2] = rand_two_distinct_alloc_teams();

    int idx1 = rand_member_idx();
    int idx2 = rand_member_idx();

    int node1 = team[t2][idx2];
    int node2 = team[t1][idx1];

    int team_new = t1;
    int team_old = t2;

    w_eff[team_new] += eff[node1] - eff[node2];
    w_eff[team_old] -= eff[node1] - eff[node2];

    w_div[team_new] += delta_div[node1][team_new]
                     - delta_div[node2][team_new]
                     - div_in[node1][node2];

    w_div[team_old] += delta_div[node2][team_old]
                     - delta_div[node1][team_old]
                     - div_in[node1][node2];

    team[t1][idx1] = node1;
    team[t2][idx2] = node2;

    update_delta(node1, team_new, team_old);
    update_delta(node2, team_old, team_new);

    state[node1] = team_new;
    state[node2] = team_old;

    address[node1] = idx1;
    address[node2] = idx2;

    return team;
}


int **Hyper_heuristic::LLH9(int **team)
{
    int tmin = min_func(w_div, num_team);

    int idx1 = rand_member_idx();
    int idx2 = rand_member_idx();

    auto [t2, _] = rand_two_distinct_alloc_teams();

    int node1 = team[t2][idx2];
    int node2 = team[tmin][idx1];

    int team_new = tmin;
    int team_old = t2;

    w_eff[team_new] += eff[node1] - eff[node2];
    w_eff[team_old] -= eff[node1] - eff[node2];

    w_div[team_new] += delta_div[node1][team_new]
                     - delta_div[node2][team_new]
                     - div_in[node1][node2];

    w_div[team_old] += delta_div[node2][team_old]
                     - delta_div[node1][team_old]
                     - div_in[node1][node2];

    team[team_new][idx1] = node1;
    team[team_old][idx2] = node2;

    update_delta(node1, team_new, team_old);
    update_delta(node2, team_old, team_new);

    state[node1] = team_new;
    state[node2] = team_old;

    address[node1] = idx1;
    address[node2] = idx2;

    return team;
}

int **Hyper_heuristic::LLH10(int **team)
{
    // LLH10:
    // Forward circular shift across all allocated teams
    // at the same position (atomic multi-team operator)

    if (num_team <= 1) return team;

    int idx = rand_member_idx();

    // Store first node
    int first_node = team[1][idx];
    int prev_node  = first_node;

    // Rotate from team 1 → team 2 → ... → team K
    for (int t = 1; t < num_team; t++)
    {
        int next_node = team[t + 1][idx];

        // ---- Efficiency update ----
        w_eff[t] += eff[next_node] - eff[prev_node];

        // ---- Diversity update ----
        w_div[t] += delta_div[next_node][t]
                  - delta_div[prev_node][t]
                  - div_in[next_node][prev_node];

        // ---- Apply ----
        team[t][idx] = next_node;

        // ---- Update delta structure ----
        update_delta(next_node, t, state[next_node]);
        update_delta(prev_node, state[prev_node], t);

        // ---- Update state ----
        state[next_node] = t;
        address[next_node] = idx;

        prev_node = next_node;
    }

    // Last team receives first node
    w_eff[num_team] += eff[first_node] - eff[prev_node];

    w_div[num_team] += delta_div[first_node][num_team]
                     - delta_div[prev_node][num_team]
                     - div_in[first_node][prev_node];

    team[num_team][idx] = first_node;

    update_delta(first_node, num_team, state[first_node]);
    update_delta(prev_node, state[prev_node], num_team);

    state[first_node] = num_team;
    address[first_node] = idx;

    return team;
}
int **Hyper_heuristic::LLH11(int **team)
{
    auto [t1, t2] = rand_two_distinct_alloc_teams();

    int node1 = team[t2][0];
    int node2 = team[t1][0];

    int team_new = t1;
    int team_old = t2;

    w_eff[team_new] += eff[node1] - eff[node2];
    w_eff[team_old] -= eff[node1] - eff[node2];

    w_div[team_new] += delta_div[node1][team_new]
                     - delta_div[node2][team_new]
                     - div_in[node1][node2];

    w_div[team_old] += delta_div[node2][team_old]
                     - delta_div[node1][team_old]
                     - div_in[node1][node2];

    team[t1][0] = node1;
    team[t2][0] = node2;

    update_delta(node1, team_new, team_old);
    update_delta(node2, team_old, team_new);

    state[node1] = team_new;
    state[node2] = team_old;

    address[node1] = 0;
    address[node2] = 0;

    return team;
}

int **Hyper_heuristic::LLH12(int **team)
{
    // LLH12:
    // Swap middle-position individual between two teams (atomic)

    auto [t1, t2] = rand_two_distinct_alloc_teams();

    int idx = num_each_t / 2;

    int node1 = team[t2][idx];  // move in
    int node2 = team[t1][idx];  // move out

    int team_new = t1;
    int team_old = t2;

    // ---- Efficiency ----
    w_eff[team_new] += eff[node1] - eff[node2];
    w_eff[team_old] -= eff[node1] - eff[node2];

    // ---- Diversity ----
    w_div[team_new] += delta_div[node1][team_new]
                     - delta_div[node2][team_new]
                     - div_in[node1][node2];

    w_div[team_old] += delta_div[node2][team_old]
                     - delta_div[node1][team_old]
                     - div_in[node1][node2];

    // ---- Apply ----
    team[t1][idx] = node1;
    team[t2][idx] = node2;

    update_delta(node1, team_new, team_old);
    update_delta(node2, team_old, team_new);

    state[node1] = team_new;
    state[node2] = team_old;

    address[node1] = idx;
    address[node2] = idx;

    return team;
}

int **Hyper_heuristic::LLH13(int **team)
{
    // LLH13:
    // Swap random-position individual between two teams (atomic)

    auto [t1, t2] = rand_two_distinct_alloc_teams();

    int idx1 = rand_member_idx();
    int idx2 = rand_member_idx();

    int node1 = team[t2][idx2];
    int node2 = team[t1][idx1];

    int team_new = t1;
    int team_old = t2;

    // ---- Efficiency ----
    w_eff[team_new] += eff[node1] - eff[node2];
    w_eff[team_old] -= eff[node1] - eff[node2];

    // ---- Diversity ----
    w_div[team_new] += delta_div[node1][team_new]
                     - delta_div[node2][team_new]
                     - div_in[node1][node2];

    w_div[team_old] += delta_div[node2][team_old]
                     - delta_div[node1][team_old]
                     - div_in[node1][node2];

    // ---- Apply ----
    team[t1][idx1] = node1;
    team[t2][idx2] = node2;

    update_delta(node1, team_new, team_old);
    update_delta(node2, team_old, team_new);

    state[node1] = team_new;
    state[node2] = team_old;

    address[node1] = idx1;
    address[node2] = idx2;

    return team;
}

int **Hyper_heuristic::LLH14(int **team)
{
    // LLH14:
    // Ruin 50% of allocated members (random atomic pool swaps)
    // Then greedy recreate

    int team0 = 0;

    int ruin_count = (num_each_t * num_team) / 2;

    // =========================
    // RUIN PHASE (atomic swaps)
    // =========================
    for (int r = 0; r < ruin_count; r++)
    {
        if (pool_size_unallocated() == 0)
            break;

        int t = rand_alloc_team();
        int idx_team = rand_member_idx();
        int idx_pool = rand_pool_idx(pool_size_unallocated());

        int node_in  = team[team0][idx_pool];   // from pool
        int node_out = team[t][idx_team];       // from team

        // ---- Efficiency update ----
        w_eff[t] += eff[node_in] - eff[node_out];

        // ---- Diversity update ----
        w_div[t] += delta_div[node_in][t]
                  - delta_div[node_out][t]
                  - div_in[node_in][node_out];

        // ---- Apply swap ----
        team[t][idx_team] = node_in;
        team[team0][idx_pool] = node_out;

        update_delta(node_in, t, team0);
        update_delta(node_out, team0, t);

        state[node_in]  = t;
        state[node_out] = team0;

        address[node_in]  = idx_team;
        address[node_out] = idx_pool;
    }

    // =========================
    // RECREATE PHASE (greedy)
    // =========================
    int pool_size = pool_size_unallocated();

    for (int p = 0; p < pool_size; p++)
    {
        int node = team[team0][p];

        int best_team = -1;
        double best_gain = MINVALUE;

        for (int t = 1; t <= num_team; t++)
        {
            double gain = eff[node];

            if (gain > best_gain)
            {
                best_gain = gain;
                best_team = t;
            }
        }

        if (best_team == -1)
            continue;

        int idx = rand_member_idx();
        int node_out = team[best_team][idx];

        // ---- Efficiency ----
        w_eff[best_team] += eff[node] - eff[node_out];

        // ---- Diversity ----
        w_div[best_team] += delta_div[node][best_team]
                          - delta_div[node_out][best_team]
                          - div_in[node][node_out];

        // ---- Apply ----
        team[best_team][idx] = node;
        team[team0][p] = node_out;

        update_delta(node, best_team, team0);
        update_delta(node_out, team0, best_team);

        state[node] = best_team;
        state[node_out] = team0;

        address[node] = idx;
        address[node_out] = p;
    }

    return team;
}

int **Hyper_heuristic::LLH15(int **team)
{
    // LLH15:
    // Ruin 50% of worst-efficiency team
    // Recreate using pool individuals

    int team0 = 0;
    int tmin = min_func(w_eff, num_team);

    int k = num_each_t / 2;

    if (pool_size_unallocated() < k)
        k = pool_size_unallocated();

    for (int i = 0; i < k; i++)
    {
        int idx_team = rand_member_idx();
        int idx_pool = rand_pool_idx(pool_size_unallocated());

        int node_out = team[tmin][idx_team];
        int node_in  = team[team0][idx_pool];

        // ---- Efficiency ----
        w_eff[tmin] += eff[node_in] - eff[node_out];

        // ---- Diversity ----
        w_div[tmin] += delta_div[node_in][tmin]
                     - delta_div[node_out][tmin]
                     - div_in[node_in][node_out];

        // ---- Apply ----
        team[tmin][idx_team] = node_in;
        team[team0][idx_pool] = node_out;

        update_delta(node_in, tmin, team0);
        update_delta(node_out, team0, tmin);

        state[node_in]  = tmin;
        state[node_out] = team0;

        address[node_in]  = idx_team;
        address[node_out] = idx_pool;
    }

    return team;
}



int **Hyper_heuristic::ApplyHeuristic(int h, int **solution)
{
    int **Scurrent = solution;
    int fbest12 = 0;

    switch (h)
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

    default:
        std::cout << "Invalid heuristic index\n";
    }

    return Scurrent;
}

int **Hyper_heuristic::ApplyMeta_Heuristic(int h, int **solution)
{
    int **Scurrent = solution;
    int fbest12 = 0;

    switch (h)
    {
        // ---------------------------------------------------------
        //                    METAHEURISTICS (High-level)
        // ---------------------------------------------------------

    case 1:
        Scurrent = simulated_annealing();
        // Metaheuristic: Simulated Annealing (SA)
        break;

    case 2:
        Scurrent = iterated_local_search();
        // Metaheuristic: Iterated Local Search (ILS)
        break;

    case 3:
        Scurrent = fits();
        // Metaheuristic: Tabu Search (FITS)
        break;

    case 4:
        Scurrent = great_deluge_algorithm();
        // Metaheuristic: Great Deluge
        break;

    case 5:
        Scurrent = late_acceptance_hill_climbing();
        // Metaheuristic: LAHC
        break;

    case 6:
        f_cur = feasible_local_search();
        // Metaheuristic: infeasible local search: Apply local hill-climbing / feasible_local_search() on current solution.
        break;

    case 7:
        Scurrent = infeasible_local_search();
        // Metaheuristic: infeasible local search
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

void Hyper_heuristic::Q_Learning_Selection_Hyperheuristic_CMCEE(int max_time)
{
    std::cout << "=============================================================================\n"
                 "Q-Learning Selection Hyper-heuristic Framework Start its Processes.\n"
                 "=============================================================================\n";

    // ------------------------------------------------------------
    // RNG SEEDING
    // ------------------------------------------------------------
    static bool seeded = false;

    // ------------------------------------------------------------
    // Q-LEARNING PARAMETERS
    // ------------------------------------------------------------
    const double alpha = 0.60;
    const double gamma = 0.70;
    double epsilon = 0.30;
    const double eps_decay = 0.99;
    const double eps_min = 0.05;

    const int topK = 5;
    const std::vector<int> heuristics = {1, 2, 3, 4, 5};

    // Q-table: Q[state][action] = value
    std::map<std::string, std::map<int, double>> Q_table;

    auto reward_from_delta = [](int d) -> double
    {
        if (d > 0)
            return 1.0;
        if (d < 0)
            return -1.0;
        return 0.0;
    };

    // ------------------------------------------------------------
    // INITIAL SOLUTION
    // ------------------------------------------------------------
    // generate_initialrandom();
    // display(team);
    objective_Function1(team);

    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    //int **previous_solution = team;
    //int **best_solution1 = team;
    int** previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** best_solution1    = deep_copy_solution(team, num_node, num_team, num_each_t);

    best_eff = cost_eff;
    best_div = cost_div;
    // Allocate team_size array globally
    team_size = new int[num_team + 1];

    // Assign team sizes
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;

    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;


    std::cout << "Initial objectives: Eff=" << cost_eff
              << "  Div=" << cost_div << "\n";

    std::vector<int> Selected;
    std::vector<int> objective_values;
    std::vector<int> diversity_values;
    std::vector<double> iteration_times;

    std::map<int, double> heuristic_total_time;
    std::map<int, int> heuristic_improvement_count;
    std::map<int, double> heuristic_rewards;
    std::map<int, int> heuristic_usage_count;

    // ------------------------------------------------------------
    // STATE TRACKING
    // ------------------------------------------------------------
    int f_eff_max = 10000;
    int f_div_max = 50000;

    int accepted_moves = 0;
    int total_moves = 0;

    std::vector<double> reward_hist;
    std::vector<int> div_values;
    std::vector<int> div_dummy;
    std::vector<int> eff_values;
    std::vector<int> rewards_record;

    double flex = 0.0;
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
    int max_iter = 1000;
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

        if (!Q_table.count(key_state))
        {
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
        else
        {
            double maxQ = -1e18;
            std::vector<int> ties;
            for (int h : heuristics)
            {
                double qv = Q_table[key_state][h];
                if (qv > maxQ)
                {
                    maxQ = qv;
                    ties = {h};
                }
                else if (fabs(qv - maxQ) < 1e-12)
                {
                    ties.push_back(h);
                }
            }
            i_next = ties[rand() % ties.size()];
        }

        // ------------------------------------------------------------
        // APPLY HEURISTIC
        // ------------------------------------------------------------
        auto t_hstart = std::chrono::steady_clock::now();
        ApplyMeta_Heuristic(i_next, team);
        // ApplyHeuristic(i_next, team);
        double heuristic_time = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - t_hstart)
                                    .count();

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
        if (!Q_table.count(key_next))
        {
            for (int h : heuristics)
                Q_table[key_next][h] = 0.0;
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
        if ((new_eff > cost_eff) && (new_div >= min_div))
        {
            // Accept the new solution
            cost_eff = new_eff;
            cost_div = new_div;
            previous_solution = team;
            accepted_moves++;
            //free_solution(previous_solution, num_node, num_team, num_each_t);
            //previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
            //  Optionally, deep copy team to previous_solution if needed
            Selected.push_back(i_next);
            if (cost_eff > best_cost_eff)
            {
                // Update best solution
                best_eff = cost_eff;
                best_div = cost_div;
                best_solution1 = team;
                time_taken = heuristic_time;
                //free_solution(best_solution1, num_node, num_team, num_each_t);
                //best_solution1 = deep_copy_solution(team, num_node, num_team, num_each_t);

                // Optionally, deep copy team to best_solution1 if needed
                // Update best solution arrays
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++)
                {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        }
        else
        {
            // Revert to previous_solution
            //team = previous_solution;
        for (int i = 0; i <= num_team; ++i)
            for (int j = 0; j < team_size[i]; ++j)
              team[i][j] = previous_solution[i][j];
        }

        total_moves++;

        // ------------------------------------------------------------
        // ITER TIME
        // ------------------------------------------------------------
        double iteration_time = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - t_iter_start)
                                    .count();

        eff_values.push_back(new_eff);
        div_values.push_back(new_div);
        rewards_record.push_back(reward);
        iteration_times.push_back(iteration_time);

        // ------------------------------------------------------------
        // FILE OUTPUT
        // ------------------------------------------------------------
        if (outfile)
        {
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
              << best_eff << "," << best_div << ","
              << reward << "," << avg_Q_value << "," << runtime << "\n";
        // ------------------------------------------------------------
        // CONSOLE OUTPUT (your preferred form)
        // ------------------------------------------------------------
        std::cout << "Iter: " << iteration
                  << " | Selected Heuristic: " << i_next
                  << " | Eff: " << new_eff
                  << " | Div: " << new_div
                  << " | Best Eff: " << best_eff
                  << " | Best Div: " << best_div
                  << " | Time Taken: " << heuristic_time
                  << " | Delta: " << delta
                  << " | Reward: " << reward
                  << " | Qsa[" << i_next << "]: " << Qsa
                  << std::endl;

        if (iteration % 50 == 0)
            // std::cout << "Epsilon decayed to: " << epsilon << std::endl;
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
                                  objective_values.end(), 0.0) /
                  objective_values.size();

    if (!diversity_values.empty())
        avg_div = std::accumulate(diversity_values.begin(),
                                  diversity_values.end(), 0.0) /
                  diversity_values.size();

    if (!iteration_times.empty())
        avg_cpu = std::accumulate(iteration_times.begin(),
                                  iteration_times.end(), 0.0) /
                  iteration_times.size();

    // ------------------------------------------------------------
    // FINAL PRINT
    // ------------------------------------------------------------
    std::cout << "\nFinal Results\n"
                 "--------------------------\n";

    std::cout << "Best Efficiency: " << best_eff << "\n";
    std::cout << "Best Diversity : " << best_div << "\n";
    std::cout << "Avg Efficiency : " << avg_eff << "\n";
    std::cout << "Avg Diversity  : " << avg_div << "\n";
    std::cout << "Avg Iter Time  : " << avg_cpu << "\n";

    std::cout << "Used LLH sequence: ";
    for (int h : Selected)
        std::cout << h << " ";
    std::cout << "\n";
    std::cout << "Total Iterations: " << total_moves << "\n";
    std::cout << "Convergence Trace Saved: " << trace_file << "\n";
    std::cout << "=============================================================================\n";

    check_best_solution();

    std::cout << "=============================================================================\n"
                 "Q-Learning Selection Hyper-heuristic Framework Finished.\n"
                 "=============================================================================\n";
}

void Hyper_heuristic::Greedy_Selection_Hyperheuristic_CMCEE(int max_time)
{
    // ==========================================================================
    // 1) INITIAL SOLUTION SETUP
    // ==========================================================================
    // Seed the random number generator
    // std::srand(static_cast<unsigned int>(std::time(0)));
    std::cout << "Greedy Hyper-heuristic Framework Start its Processes." << std::endl;

    // Evaluate the current 'team' solution (could be random or pre-generated)
    // generate_initialrandom();
    objective_Function1(team);
    int cost_eff = f_cur;     // Current solution efficiency
    int cost_div = f_cur_div; // Current solution diversity

    std::cout << "\tInitial objectives: Efficiency = " << cost_eff
              << ", Diversity = " << cost_div << "\n";

    // Track the best solution found so far
    best_eff = cost_eff;
    best_div = cost_div;
    int best_found_at = 0;

    // Keep pointers to solutions for acceptance/reversion
    // int** best_solution1   = team;  // stores the best solution
    // int** current_solution = team;  // currently accepted solution
    // Store deep copies of current and best solutions
    int **current_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int **best_solution1 = deep_copy_solution(team, num_node, num_team, num_each_t);
    int *team_size = new int[num_team + 1];

    // Assign team sizes
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;

    // Team 0 stores remaining unallocated individuals
    team_size[0] = num_node - num_each_t * num_team;

    // ==========================================================================
    // 2) HEURISTICS & PERFORMANCE TRACKING
    // ==========================================================================
    // Pool of low-level heuristics
    std::vector<int> heuristics = {1, 2, 3, 4, 5};
    // std::vector<int> heuristics = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 17, 18, 19, 20};
    std::vector<double> rewards_record;

    // "heuristic_scores" tracks the total improvement each heuristic has achieved
    std::map<int, double> heuristic_scores;
    for (auto h : heuristics)
    {
        heuristic_scores[h] = 0.0;
    }

    // Data structures for analysis
    std::vector<int> objective_values;
    std::vector<int> diversity_values;
    std::vector<double> iteration_times;

    double total_elapsed_time = 0.0;

    // ==========================================================================
    // 3) FILE I/O SETUP
    // ==========================================================================
    // Create/log directory
    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";
    if (!std::filesystem::exists(folder_path))
    {
        try
        {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return;
        }
    }

    // Define file path for iteration logs
    std::filesystem::path results_file = folder_path / ("Greedy_Selection_HH_CMCEE_" + instanceName + "_results.txt");
    std::ofstream iteration_outfile(results_file);
    if (!iteration_outfile)
    {
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
    int iter = 0; // an additional iteration counter

    // Track how many consecutive iterations the global best solution fails to improve
    int no_improvement_count = 0;
    const int NO_IMPROVEMENT_LIMIT = 100; // Terminate if no improvement for 100 consecutive iterations

    // The loop runs as long as EITHER time < max_time OR iter < 1000
    while ((static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC < max_time))
    {
        iteration++;
        iter++;
        clock_t iteration_start_time = clock();

        // ------------------------------------------------
        // 4a) Pick a heuristic
        // ------------------------------------------------
        if (iteration <= static_cast<int>(heuristics.size()))
        {
            // First N iterations: apply each heuristic once in sequence
            selected_heuristic = heuristics[iteration - 1];
        }
        else
        {
            // After that, pick the best-performing heuristic so far
            selected_heuristic = std::max_element(
                                     heuristic_scores.begin(), heuristic_scores.end(),
                                     [](const std::pair<int, double> &a, const std::pair<int, double> &b)
                                     {
                                         return a.second < b.second;
                                     })
                                     ->first;
        }

        // ------------------------------------------------
        // 4b) Apply the selected heuristic & measure CPU time
        // ------------------------------------------------
        clock_t heuristic_start_time = clock();
        ApplyMeta_Heuristic(selected_heuristic, team);
        double heuristic_time = static_cast<double>(clock() - heuristic_start_time) / CLOCKS_PER_SEC;
        total_elapsed_time += heuristic_time;

        // ------------------------------------------------
        // 4c) Evaluate the new solution
        // ------------------------------------------------
        objective_Function1(team);
        int new_cost_eff = f_cur;
        int new_cost_div = f_cur_div;

        // Compute delta = new_eff - old_eff for acceptance
        // double delta = new_cost_eff - cost_eff;
        double delta = new_cost_eff - cost_eff;
        double reward = (delta > 0 ? 1.0 : (delta < 0 ? -1.0 : 0.0));
        rewards_record.push_back(reward);
        // ------------------------------------------------
        // 4d) Accept/Reject the new solution
        // ------------------------------------------------
        bool improved = false;
        if (delta > 0)
        {
            // Accept => Update the current solution
            cost_eff = new_cost_eff;
            cost_div = new_cost_div;
            current_solution = team; // newly improved solution

            // free_solution(current_solution , num_node, num_team, num_each_t);
            // current_solution  = deep_copy_solution(team, num_node, num_team, num_each_t);
            //  Accumulate improvement in heuristic_scores
            heuristic_scores[selected_heuristic] += delta;
            improved = true;

            // If it's the best so far & meets min_div
            if ((cost_eff > best_eff) && (cost_div >= min_div))
            {
                best_eff = cost_eff;
                best_div = cost_div;
                // free_solution(best_solution1 , num_node, num_team, num_each_t);
                // best_solution1  = deep_copy_solution(team, num_node, num_team, num_each_t);
                best_solution1 = team;
                best_eff = cost_eff;
                best_div = cost_div;
                time_taken = heuristic_time;
                best_found_at = iteration;

                // Copy solution arrays if needed
                for (int m = 0; m < num_node; m++)
                {
                    fbest_solution[m] = best_solution[m];
                }
                for (int m = 1; m <= num_team; m++)
                {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        }
        else
        {
            // No improvement => revert
            delta = 0.0;
            // team = current_solution;
            for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = current_solution[i][j];
            // team = current_solution;
        }

        // ------------------------------------------------
        // 4e) Check if best solution improved
        // ------------------------------------------------
        if (improved && (cost_eff >= best_eff))
        {
            // There's a global improvement => reset no_improvement_count
            no_improvement_count = 0;
        }
        else
        {
            // No global improvement => increment
            no_improvement_count++;
        }
        // Terminate early if no improvement for 100 consecutive iterations
        if (no_improvement_count >= NO_IMPROVEMENT_LIMIT)
        {
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
                  << " | Best Efficiency: " << best_eff
                  << " | Best Diversity: " << best_div
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
    if (!objective_values.empty())
    {
        double total_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0);
        average_objective = total_objective / objective_values.size();
    }

    if (!diversity_values.empty())
    {
        double total_diversity = std::accumulate(diversity_values.begin(), diversity_values.end(), 0.0);
        average_diversity = total_diversity / diversity_values.size();
    }

    if (!objective_values.empty())
    {
        worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    }

    double average_cpu_time = 0.0;
    if (!iteration_times.empty())
    {
        double total_iteration_time = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
        average_cpu_time = total_iteration_time / iteration_times.size();
    }

    // Close iteration log file
    iteration_outfile.close();

    // Save best solution summary
    std::filesystem::path summary_path = folder_path / ("Greedy_Selection_HH_CMCEE_" + instanceName + "_Summary.txt");
    {
        std::ofstream summary_file(summary_path);
        if (summary_file)
        {
            summary_file << "Best selected heuristic: " << selected_heuristic << "\n";
            summary_file << "Best Efficiency: " << best_eff << "\n";
            summary_file << "Best Diversity: " << best_div << "\n";
            summary_file << "Found at Iteration: " << best_found_at << "\n";
            summary_file << "Total Time: " << total_time << " seconds\n";
            summary_file << "Average Objective Function Value: " << average_objective << "\n";
            summary_file << "Average Diversity Value: " << average_diversity << "\n";
            summary_file << "Worst Objective Function Value: " << worst_objective << "\n";
            summary_file << "Average CPU Time per Iteration: " << average_cpu_time << " seconds\n";
            summary_file.close();
        }
        else
        {
            std::cerr << "Error: Unable to save summary file.\n";
        }
    }

    // Print summary to console
    std::cout << "Best selected heuristic: " << selected_heuristic << "\n"
              << "Best Efficiency: " << best_eff << "\n"
              << "Best Diversity: " << best_div << "\n"
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

void Hyper_heuristic::Random_Selection_Hyperheuristic_CMCEE(int max_time)
{

    std::cout << "=============================================================================\n"
                 "Random Selection Hyper-Heuristic Framework Start its Processes.\n"
                 "=============================================================================\n";

    // ------------------------------------------------------------
    // INITIALIZATION
    // ------------------------------------------------------------
    int **previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int **best_solution112 = deep_copy_solution(team, num_node, num_team, num_each_t);

    int *team_size = new int[num_team + 1];
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    objective_Function1(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    best_eff = cost_eff;
    best_div = cost_div;

    std::cout << "Initial objectives - Efficiency: "
              << cost_eff << ", Diversity: " << cost_div << std::endl;

    // ------------------------------------------------------------
    // HEURISTICS AND TRACKING
    // ------------------------------------------------------------
    std::vector<int> heuristics = {1, 2, 3, 4, 5};
    int max_iterations = 1000;

    std::map<int, std::vector<int>> heuristic_costs;
    std::map<int, std::vector<double>> heuristic_times;
    std::vector<int> objective_values;
    std::vector<int> diversity_values;
    std::vector<double> iteration_times;
    std::vector<double> rewards_record;
    std::map<int, double> heuristic_total_time;
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

    std::filesystem::path trace_file =
        folder / ("Random_Selection_HH_CMCEE_" + instanceName + "_Convergence_Trace.csv");

    std::ofstream outfile(results_file);
    outfile << "Iteration\tHeuristic\tOldEff\tOldDiv\tNewEff\tNewDiv\tTime\n";

    std::ofstream trace(trace_file);
    trace << "Iteration,Efficiency,Diversity,BestEfficiency,BestDiversity,Reward,Runtime\n";

    // ------------------------------------------------------------
    // MAIN LOOP
    // ------------------------------------------------------------
    clock_t total_start_time = clock();
    int iteration = 0;

    while ((double)(clock() - total_start_time) / CLOCKS_PER_SEC < max_time && iteration < max_iterations)
    {

        iteration++;

        int old_eff = cost_eff;
        int old_div = cost_div;

        int selected_heuristic =
            heuristics[std::rand() % heuristics.size()];

        // Apply heuristic
        clock_t h_start = clock();
        ApplyMeta_Heuristic(selected_heuristic, team);
        double elapsed_time =
            (double)(clock() - h_start) / CLOCKS_PER_SEC;

        total_elapsed_time += elapsed_time;

        // Evaluate
        objective_Function1(team);
        int new_eff = f_cur;
        int new_div = f_cur_div;

        double reward =
            (new_eff > old_eff ? 1.0 : (new_eff < old_eff ? -1.0 : 0.0));

        rewards_record.push_back(reward);
        heuristic_total_time[selected_heuristic] += elapsed_time;
        fitness_history.push_back(new_eff);

        // ACCEPTANCE
        if ((new_eff > cost_eff) && (new_div >= min_div))
        {

            cost_eff = new_eff;
            cost_div = new_div;
            previous_solution = team;

            if (new_eff > best_eff)
            {
                best_eff = new_eff;
                best_div = new_div;
                best_solution112 = team;
                time_taken = elapsed_time;
            }
        }
        else
        {
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

        outfile << iteration << "\t"
                << selected_heuristic << "\t"
                << old_eff << "\t" << old_div << "\t"
                << new_eff << "\t" << new_div << "\t"
                << elapsed_time << "\n";

        double runtime =
            (double)(clock() - total_start_time) / CLOCKS_PER_SEC;

        trace << iteration << "," << new_eff << "," << new_div << ","
              << best_eff << "," << best_div << ","
              << reward << "," << runtime << "\n";

        std::cout << "Iteration: " << iteration
                  << " | Heuristic: " << selected_heuristic
                  << " | Efficiency: " << new_eff
                  << " | Diversity: " << new_div
                  << " | Best Eff: " << best_eff
                  << " | Best Div: " << best_div
                  << " | Reward=" << reward
                  << " | Time=" << elapsed_time << "s\n";
    }

    // ------------------------------------------------------------
    // CLOSE FILES AND SUMMARY
    // ------------------------------------------------------------
    outfile.close();
    trace.close();

    double average_objective = 0.0;
    double average_diversity = 0.0;
    double average_cpu_time = 0.0;

    if (!objective_values.empty())
        average_objective =
            std::accumulate(objective_values.begin(),
                            objective_values.end(), 0.0) /
            objective_values.size();

    if (!diversity_values.empty())
        average_diversity =
            std::accumulate(diversity_values.begin(),
                            diversity_values.end(), 0.0) /
            diversity_values.size();

    if (!iteration_times.empty())
        average_cpu_time =
            std::accumulate(iteration_times.begin(),
                            iteration_times.end(), 0.0) /
            iteration_times.size();

    std::cout << "\nFinal Summary\n"
                 "---------------------------\n"
                 "Best Efficiency : "
              << best_eff << "\n"
                             "Best Diversity  : "
              << best_div << "\n"
                             "Average Eff     : "
              << average_objective << "\n"
                                      "Average Div     : "
              << average_diversity << "\n"
                                      "Average IterTime: "
              << average_cpu_time << "s\n"
                                     "Convergence Trace Saved: "
              << trace_file << "\n"
                               "===========================\n";

    check_best_solution();

    std::cout << "=============================================================================\n"
                 "Random Selection Hyper-Heuristic Framework Finished.\n"
                 "=============================================================================\n";
}

// Implementation of Hyper - heuristic Based Multi - Armed Bandit(UCB) Selection Strategy for CMCEE
void Hyper_heuristic::MAB_Selection_Hyperheuristic_CMCEE(int max_time)
{
    std::cout << "=============================================================================\n"
                 "Multi-Armed Bandit (UCB1) with Credit Assignment Selection Hyper-Heuristic Framework Start its Processes.\n"
                 "=============================================================================\n";

    // ------------------------------------------------------------
    // INITIALIZATION
    // ------------------------------------------------------------
    // generate_initialrandom();
    int **previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
    int **best_solution1 = deep_copy_solution(team, num_node, num_team, num_each_t);

    objective_Function1(team);
    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    int best_eff = cost_eff;
    int best_div = cost_div;

    int *team_size = new int[num_team + 1];
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    // ------------------------------------------------------------
    // HEURISTICS AND MAB STRUCTURES
    // ------------------------------------------------------------
    std::vector<int> heuristics = {1, 2, 3, 4, 5, 6, 7, 8};
    std::map<int, int> heuristic_selections;
    std::map<int, double> heuristic_credits;

    for (int h : heuristics)
    {
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
    while ((double)(clock() - total_start_time) / CLOCKS_PER_SEC < max_time)
    {
        iteration++;

        int old_eff = cost_eff;
        int old_div = cost_div;

        // ----- UCB1 SELECTION -----
        int selected_heuristic = -1;
        double best_ucb = -1.0;

        for (int h : heuristics)
        {
            double avg_credit = (heuristic_selections[h] > 0)
                                    ? (heuristic_credits[h] / heuristic_selections[h])
                                    : 0.0;
            double explore = std::sqrt(2.0 * std::log(iteration + 1) / (heuristic_selections[h] + 1e-6));
            double ucb_value = avg_credit + explore;
            if (ucb_value > best_ucb)
            {
                best_ucb = ucb_value;
                selected_heuristic = h;
            }
        }

        // ----- APPLY HEURISTIC -----
        clock_t h_start = clock();
        ApplyMeta_Heuristic(selected_heuristic, team);
        double elapsed_time = (double)(clock() - h_start) / CLOCKS_PER_SEC;
        total_elapsed_time += elapsed_time;

        // ----- EVALUATE -----
        objective_Function1(team);
        int new_eff = f_cur;
        int new_div = f_cur_div;

        int delta = new_eff - old_eff;

        double reward = (delta > 0) ? 1.0 : (delta < 0) ? -1.0
                                                        : 0.0;

        rewards_record.push_back(reward);

        // ----- CREDIT ASSIGNMENT -----
        if (reward > 0.0)
        {
            // Selected heuristic rewarded
            heuristic_credits[selected_heuristic] += 1.0;

            if (heuristics.size() > 1)
            {
                double penalty = 1.0 / (heuristics.size() - 1);
                for (int h : heuristics)
                    if (h != selected_heuristic)
                        heuristic_credits[h] -= penalty;
            }
        }
        else if (reward < 0.0)
        {
            // Selected heuristic penalized
            heuristic_credits[selected_heuristic] -= 1.0;

            if (heuristics.size() > 1)
            {
                double bonus = 1.0 / (heuristics.size() - 1);
                for (int h : heuristics)
                    if (h != selected_heuristic)
                        heuristic_credits[h] += bonus;
            }
        }
        // reward == 0 → no credit update

        heuristic_selections[selected_heuristic]++;

        // ----- ACCEPT/REJECT -----
        if ((new_eff > old_eff) && (new_div >= min_div))
        {
            cost_eff = new_eff;
            cost_div = new_div;
            // free_solution(previous_solution, num_node, num_team, num_each_t);
            // previous_solution = deep_copy_solution(team, num_node, num_team, num_each_t);
            previous_solution = team;

            if (new_eff > best_eff)
            {
                best_eff = new_eff;
                best_div = new_div;

                // free_solution(best_solution1, num_node, num_team, num_each_t);
                // best_solution1 = deep_copy_solution(team, num_node, num_team, num_each_t);
                time_taken = elapsed_time;
                best_solution1 = team;
                // Copy best arrays
                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++)
                {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        }
        else
        {
            // Revert
            for (int i = 0; i <= num_team; ++i)
                for (int j = 0; j < team_size[i]; ++j)
                    team[i][j] = previous_solution[i][j];
            // team = previous_solution;
        }
        if (iteration > 50 && reward == -1)
        {
            // force diversification
            selected_heuristic = heuristics[rand() % heuristics.size()];
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
                << elapsed_time << "\t" << delta << "\n";

        double runtime = (double)(clock() - total_start_time) / CLOCKS_PER_SEC;
        trace << iteration << "," << new_eff << "," << new_div << ","
              << best_eff << "," << best_div << ","
              << reward << "," << runtime << "\n";

        // Console progress
        std::cout << "Iter: " << iteration
                  << " | LLH: " << selected_heuristic
                  << " | Eff: " << new_eff
                  << " | Div: " << new_div
                  << " | BestEff: " << best_eff
                  << " | BestDiv: " << best_div
                  << " | Reward=" << reward
                  << " | Delat_Credit=" << delta
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
              << "Best Efficiency : " << best_eff << "\n"
              << "Best Diversity  : " << best_div << "\n"
              << "Average Eff     : " << avg_eff << "\n"
              << "Average Div     : " << avg_div << "\n"
              << "Average IterTime: " << avg_cpu << "s\n"
              << "Convergence Trace Saved: " << trace_file << "\n"
              << "------------------------------------\n";

    check_best_solution();

    std::cout << "=============================================================================\n"
                 "Multi-Armed Bandit (UCB1) Hyper-Heuristic Framework Finished.\n"
                 "=============================================================================\n";
}

// Implementation of Hyper-heuristic Based Multi-Armed bandit Selection Strategies For Solving CMCEE
int fbest_solution_eff;

void Hyper_heuristic::HH_Choice_Function_Selection_CMCEE(int max_time)
{
    // ==========================================================================
    // 1) PARAMETER & DATA STRUCTURE SETUP
    std::cout << "=============================================================================" << std::endl;
    std::cout << "Choice Function Selection Hyper-heuristic Framework Start its Processes." << std::endl;
    std::cout << "=============================================================================" << std::endl;

    double alpha = 1.0;
    double beta = 0.5;
    double gamma = 0.5;

    std::map<int, double> heuristic_recent_perf;
    std::map<int, double> heuristic_total_perf;
    std::map<int, int> heuristic_last_used_iter;
    std::map<int, int> heuristic_usage_count;
    std::map<int, double> heuristic_total_time;
    std::map<int, int> heuristic_improvement_count;
    std::map<int, double> LLH_score;

    std::vector<int> objective_values;
    std::vector<double> iteration_times;
    std::vector<int> selected_heuristics;

    // ==========================================================================
    // 2) INITIALIZE CURRENT SOLUTION & HEURISTICS
    // ==========================================================================
    // generate_initialrandom();
    objective_Function1(team);

    int cost_eff = f_cur;
    int cost_div = f_cur_div;
    int best_iteration = 0;

    // int** s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
    // int** s_best    = deep_copy_solution(team, num_node, num_team, num_each_t);
    int **s_current = team;
    int **s_best = team;
    int *team_size = new int[num_team + 1];

    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    int fbest_solution_eff = cost_eff;
    int fbest_solution_div = cost_div;

    std::vector<int> heuristics = {1, 2, 3, 4, 5};
    for (int h : heuristics)
    {
        heuristic_last_used_iter[h] = 0;
        heuristic_recent_perf[h] = 0.0;
        heuristic_total_perf[h] = 0.0;
        heuristic_usage_count[h] = 0;
        heuristic_total_time[h] = 0.0;
        heuristic_improvement_count[h] = 0;
        LLH_score[h] = 0.0;
    }

    // ==========================================================================
    // 3) FILE & TIMER SETUP
    // ==========================================================================
    clock_t total_start_time = clock();
    int iteration = 0;

    std::filesystem::path folder_path = "D:/Datasets/RESULTS_OF_HH_MODELS/InstanceSeparateHH_Results/";
    if (!std::filesystem::exists(folder_path))
    {
        try
        {
            std::filesystem::create_directories(folder_path);
            std::cout << "Directory created: " << folder_path << std::endl;
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return;
        }
    }

    // Main result file
    std::filesystem::path results_file = folder_path / ("HH_Choice_Function_Selection_CMCEE_" + instanceName + "_results.txt");
    std::ofstream outfile(results_file);
    if (!outfile)
    {
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
    while (static_cast<double>(clock() - total_start_time) / CLOCKS_PER_SEC < max_time)
    {
        iteration++;
        clock_t iteration_start_time = clock();

        // ---------------------- 4.1 Choice Function ----------------------
        std::map<int, double> choice_function_values;
        for (int h : heuristics)
        {
            double f1 = heuristic_recent_perf[h];
            double f2 = heuristic_total_perf[h];
            double f3 = static_cast<double>(iteration - heuristic_last_used_iter[h]);
            choice_function_values[h] = alpha * f1 + beta * f2 + gamma * f3 + LLH_score[h];
        }

        // ---------------------- 4.2 Select Heuristic ---------------------
        int selected_heuristic = 1;
        double max_cf_value = -std::numeric_limits<double>::infinity();
        for (int h : heuristics)
        {
            if (choice_function_values[h] > max_cf_value)
            {
                max_cf_value = choice_function_values[h];
                selected_heuristic = h;
            }
        }
        heuristic_last_used_iter[selected_heuristic] = iteration;

        // ---------------------- 4.3 Apply Heuristic ----------------------
        // int** s_temp = team;
        clock_t heuristic_start_time = clock();
        ApplyMeta_Heuristic(selected_heuristic, team);
        double heuristic_time = static_cast<double>(clock() - heuristic_start_time) / CLOCKS_PER_SEC;
        heuristic_total_time[selected_heuristic] += heuristic_time;
        heuristic_usage_count[selected_heuristic]++;

        objective_Function1(team);
        int cost_eff_temp = f_cur;
        int cost_div_temp = f_cur_div;

        // ---------------------- 4.4 Delta Calculation --------------------
        double f_c = static_cast<double>(cost_eff);
        double f_n = static_cast<double>(cost_eff_temp);

        double eps = 1e-9;
        double Delta = f_n - f_c;

        // Binary reward from Delta sign
        double reward = 0.0;
        if (Delta > 0.0)
            reward = 1.0;
        else if (Delta < 0.0)
            reward = -1.0;
        else
            reward = 0.0;

        // ---------------------- 4.5 Credit Assignment --------------------
        if (reward > 0.0)
        {
            // Selected heuristic gets reward
            LLH_score[selected_heuristic] += reward;

            double punish_value = reward / (heuristics.size() - 1);
            for (int h : heuristics)
                if (h != selected_heuristic)
                    LLH_score[h] -= punish_value;

            heuristic_improvement_count[selected_heuristic]++;
            heuristic_recent_perf[selected_heuristic] = reward;
            heuristic_total_perf[selected_heuristic] += reward;
        }
        else if (reward < 0.0)
        {
            // Selected heuristic gets penalty
            LLH_score[selected_heuristic] += reward; // reward is negative

            double reward_others_value = (-reward) / (heuristics.size() - 1);
            for (int h : heuristics)
                if (h != selected_heuristic)
                    LLH_score[h] += reward_others_value;

            heuristic_recent_perf[selected_heuristic] = reward;
        }
        else
        {
            // No change
            heuristic_recent_perf[selected_heuristic] = 0.0;
        }

        // ---------------------- 4.6 Accept/Reject ------------------------
        if ((cost_eff_temp > cost_eff) && (cost_div_temp >= min_div))
        {
            // free_solution(s_current, num_node, num_team, num_each_t);
            // s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
            s_current = team;
            cost_eff = cost_eff_temp;
            cost_div = cost_div_temp;

            if ((cost_eff_temp > fbest_solution_eff) && (cost_div_temp >= min_div))
            {
                // free_solution(s_best, num_node, num_team, num_each_t);
                // s_best = deep_copy_solution(team, num_node, num_team, num_each_t);
                s_best = team;

                fbest_solution_eff = cost_eff_temp;
                fbest_solution_div = cost_div_temp;
                best_iteration = iteration;
                best_eff = cost_eff_temp;
                best_div = cost_div_temp;
                time_taken = heuristic_time;
                selected_heuristics.push_back(selected_heuristic);

                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++)
                {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        }
        else
        {
            // for (int i = 0; i <= num_team; ++i)
            //  for (int j = 0; j < team_size[i]; ++j)
            //    team[i][j] = s_current[i][j];
            team = s_current;
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

    if (!objective_values.empty())
    {
        average_objective = std::accumulate(objective_values.begin(), objective_values.end(), 0.0) / objective_values.size();
        worst_objective = *std::min_element(objective_values.begin(), objective_values.end());
    }

    double average_iteration_time = (!iteration_times.empty())
                                        ? std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0) / iteration_times.size()
                                        : 0.0;

    int best_heuristic = -1, worst_heuristic = -1;
    double max_total_perf = -std::numeric_limits<double>::infinity();
    double min_total_perf = std::numeric_limits<double>::infinity();

    for (int h : heuristics)
    {
        if (heuristic_total_perf[h] > max_total_perf)
        {
            max_total_perf = heuristic_total_perf[h];
            best_heuristic = h;
        }
        if (heuristic_total_perf[h] < min_total_perf)
        {
            min_total_perf = heuristic_total_perf[h];
            worst_heuristic = h;
        }
    }

    std::filesystem::path summary_file_path = folder_path / ("HH_Choice_Function_Selection_CMCEE_" + instanceName + "_summary.txt");
    std::ofstream summary_file(summary_file_path);
    if (summary_file)
    {
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

    std::cout << "=============================================================================" << std::endl;
    std::cout << "Choice Function Selection Hyper-heuristic Framework Finished its Processes." << std::endl;
    std::cout << "=============================================================================" << std::endl;
}

double calculate_mean(const std::vector<double>& values)
{
    if (values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

double calculate_stddev(const std::vector<double> &values, double mean)
{
    double sq_sum = std::accumulate(values.begin(), values.end(), 0.0,
                                    [mean](double acc, double val)
                                    {
                                        return acc + (val - mean) * (val - mean);
                                    });
    return std::sqrt(sq_sum / values.size());
}

// Implementation of Accept
bool Hyper_heuristic::Accept(int **Scurrent, int **Snew, double f_current, double f_new)
{
    // Example acceptance criteria:
    // - If the new solution is better, accept it
    // - If worse, accept with a probability (simulated annealing-like)
    if (f_new > f_current)
    {
        return true;
    }
    else
    {
        double probability = exp((f_new - f_current) / 100.0); // Temperature-like factor
        double rand_val = (double)rand() / RAND_MAX;
        return rand_val < probability;
    }
}

// Implementation of updateBestSolution
void Hyper_heuristic::updateBestSolution(int **Scurrent, double f_current)
{
    if (f_current > fbest_efficiency)
    {
        fbest_efficiency = f_current;
        fbest = fbest_efficiency;
        fbest_diversity = f_cur_div; // Update diversity as well

        // Update Sbest
        for (int i = 0; i <= num_team; i++)
        {
            for (int j = 0; j < num_node; j++)
            {
                Sbest[i][j] = Scurrent[i][j];
            }
        }
        team = Sbest;
        cout << "New Best Found: Efficiency = " << fbest_efficiency
             << ", Diversity = " << fbest_diversity << endl;
    }
}

// Implementation of TerminationCriterionSatisfied
bool Hyper_heuristic::TerminationCriterionSatisfied(int iter, int max_iter)
{
    // Example criterion: maximum number of iterations
    return iter >= max_iter;
}

// Helper function to perform a deep copy of a 2D array
int **DeepCopySolution(int **original, int rows, int cols)
{
    int **copy = new int *[rows];
    for (int i = 0; i < rows; ++i)
    {
        copy[i] = new int[cols];
        for (int j = 0; j < cols; ++j)
        {
            copy[i][j] = original[i][j];
        }
    }
    return copy;
}


// ==========================================================
//  SECTION 1: STATE FEATURE VECTOR AND DISCRETIZATION
// ==========================================================
// Table 3.5 implementation for s_t = [f_eff^norm, f_div^norm, Δeff, Δdiv, iter_ratio,
//                                    accept_ratio, reward_avg, div_std_norm, temp_norm, flex_norm]

// Rolling average helper
double Hyper_heuristic::rolling_average(const std::vector<double> &history, int window = 20)
{
    if (history.empty())
        return 0.0;
    int start = std::max(0, (int)history.size() - window);
    double sum = 0.0;
    for (int i = start; i < (int)history.size(); ++i)
        sum += history[i];
    return sum / (history.size() - start);
}

Hyper_heuristic::StateFeatures Hyper_heuristic::compute_state_vector1(
    int f_eff, int f_div,
    int prev_eff, int prev_div,
    int f_eff_max, int f_div_max,
    int iter, int max_iter,
    int accepted_moves, int total_moves,
    const std::vector<double> &reward_hist,
    const std::vector<int> &div_values,
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
    const std::vector<double> &reward_hist,
    const std::vector<int> &div_values,
    double temp, double temp0,
    double flex, double flex_max)
{
    StateFeatures s;
    s.f_eff_norm = (f_eff_max > 0) ? (double)f_eff / f_eff_max : 0.0;
    s.f_div_norm = (f_div_max > 0) ? (double)f_div / f_div_max : 0.0;
    s.delta_eff_norm = ((double)f_eff - prev_eff) / std::max(1.0, (double)f_eff_max);
    s.delta_div_norm = ((double)f_div - prev_div) / std::max(1.0, (double)f_div_max);
    s.iter_ratio = (double)iter / std::max(1, max_iter);
    s.accept_ratio = (total_moves > 0) ? (double)accepted_moves / total_moves : 0.0;
    s.reward_avg = rolling_average(reward_hist, 20);
    s.div_std_norm = compute_div_std_norm(div_values, (double)f_div_max);
    s.temp_norm = (temp0 > 0) ? temp / temp0 : 0.0;
    s.flex_norm = (flex_max > 0) ? flex / flex_max : 0.0;
    return s;
}

// Compute normalized standard deviation of diversity values
double Hyper_heuristic::compute_div_std_norm(const std::vector<int> &div_values, double div_max)
{
    if (div_values.empty() || div_max <= 0.0)
        return 0.0;

    double mean = 0.0;
    for (int v : div_values)
        mean += v;
    mean /= div_values.size();

    double variance = 0.0;
    for (int v : div_values)
        variance += (v - mean) * (v - mean);
    variance /= div_values.size();

    double std_dev = std::sqrt(variance);
    return std_dev / div_max; // Normalize by maximum diversity
}

// Convert the continuous feature vector into a discrete state key (Option A)
std::string Hyper_heuristic::discretize_state(const StateFeatures &s)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "S";
    std::vector<double> feats = {
        s.f_eff_norm, s.f_div_norm, s.delta_eff_norm, s.delta_div_norm,
        s.iter_ratio, s.accept_ratio, s.reward_avg,
        s.div_std_norm, s.temp_norm, s.flex_norm};
    for (double f : feats)
    {
        int bin = static_cast<int>(std::round(f * 10)); // discretize into 0–10 bins
        oss << "_" << std::clamp(bin, 0, 10);
    }
    return oss.str();
}

// ==========================================================
//  SECTION 2: LS / OP / MA ADAPTERS + REWARD & ACCEPTANCE UTILITIES
// ==========================================================

// ----------  LS Algorithm Adapters (Level 1)  ----------
enum LSAlgo
{
    LS_SA = 1,
    LS_ILS,
    LS_TS,
    LS_GD,
    LS_LAHC,
    LS_BASIC
};


int **Hyper_heuristic::Apply_LS_OP(int LSi, int OPj, int **Sstart)
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
    double GD_level0 = f_cur;
    double GD_decayRate = 0.98;

    // ---- Selection of LS Algorithm ----
    if (LSi == 1)
    {
        // Simulated Annealing
        return SimulatedAnnealing(Sstart, OPj, T0, alpha);
    }
    else if (LSi == 2)
    {
        // Iterated Local Search
        return IteratedLocalSearch(Sstart, OPj);
    }
    else if (LSi == 3)
    {
        // Late Acceptance Hill Climbing
        return LateAcceptance(Sstart, OPj, Lwindow);
    }
    else if (LSi == 4)
    {
        // Great Deluge
        return GreatDeluge(Sstart, OPj, GD_level0, GD_decayRate);
    }
    else if (LSi == 5)
    {
        // Tabu-based Fits
        return TabuSearch(Sstart, OPj);
    }
    else
    {
        // Fallback: return Sstart unchanged
        return Sstart;
    }
}
// ----------  Operator Adapters (Level 2)  ----------
// LLH 1–15 → atomic neighborhood structures
int **Hyper_heuristic::apply_LLHop(int op_id, int **sol)
{
    switch (op_id)
    {
    case 1:
        return LLH1(sol);
    case 2:
        return LLH2(sol);
    case 3:
        return LLH3(sol);
    case 4:
        return LLH4(sol);
    case 5:
        return LLH5(sol);
    case 6:
        return LLH6(sol);
    case 7:
        return LLH7(sol);
    case 8:
        return LLH8(sol);
    case 9:
        return LLH9(sol);
    case 10:
        return LLH10(sol);
    case 11:
        return LLH11(sol);
    case 12:
        return LLH12(sol);
    case 13:
        return LLH13(sol);
    case 14:
        return LLH14(sol);
    case 15:
        return LLH15(sol);
    default:
        return sol;
    }
}

// ----------  Move-Acceptance (Level 3)  ----------
enum MA_Strategy
{
    MA_ONLY_IMPROVE = 1,
    MA_ACCEPT_ALL,
    MA_SA,
    MA_R2R,
    MA_THRESHOLD
};

// Hyper-parameters (can be adjusted dynamically in the main loop)
double SA_Temp = 100.0; // starting temperature
double SA_Cooling = 0.995;
double R2R_Threshold = 5.0;
double TH_Threshold = 3.0;
double decay_R2R = 0.999;
double decay_TH = 0.999;
int R2R_record_eff = std::numeric_limits<int>::lowest();

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

    if (ma == MA_SA)
    {
        if (new_div < min_div)
            return false;
        int diff = new_eff - cur_eff;
        if (diff >= 0)
            return true;
        double prob = std::exp((double)diff / std::max(1e-9, SA_Temp));
        return ((double)rand() / RAND_MAX) < prob;
    }

    if (ma == MA_R2R)
    {
        if (new_div < min_div)
            return false;
        if (R2R_record_eff == std::numeric_limits<int>::lowest())
            R2R_record_eff = cur_eff;
        return (new_eff >= R2R_record_eff - (int)std::floor(R2R_Threshold));
    }

    if (ma == MA_THRESHOLD)
    {
        if (new_div < min_div)
            return false;
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
    if (delta > 0)
        return 1.0;
    if (delta == 0)
        return 0.0;
    return -1.0;
}

void Hyper_heuristic::TriLevel_HH_Qlearning_CMCEE(int max_time)
{

    std::cout << "===================================================================\n";
    std::cout << "Tri-Level Hyper-Heuristic Framework (LS->OP->MA)\n";
    std::cout << "===================================================================\n";

    // ---------------- Initialization ----------------
    std::cout << "Initial Solution:\n";
    objective_Function1(team);
    std::cout << "\n===================================================================\n\n";
    std::cout << "Initial objective functions: Eff=" << f_cur << " | Div=" << f_cur_div << "\n\n";
    std::cout << "===================================================================\n\n";

    int cost_eff = f_cur, cost_div = f_cur_div;
    best_eff = cost_eff;
    best_div = cost_div;
    int prev_eff = cost_eff, prev_div = cost_div;
    int f_eff_max = cost_eff, f_div_max = cost_div;

    //int** s_current = team;
    //int** best_sol  =  team;
    int** s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
    int** best_sol  = deep_copy_solution(team, num_node, num_team, num_each_t);

    team_size = new int[num_team + 1];
    for (int i = 1; i <= num_team; ++i)
        team_size[i] = num_each_t;
    team_size[0] = num_node - num_each_t * num_team;

    int best_LS = -1, best_OP = -1;
    MA_Strategy best_MA = MA_ONLY_IMPROVE;
    double time_best_found = 0.0;

    // Q-tables
    std::map<std::string, std::map<int, double>> Q_LS;
    std::map<std::string, std::map<int, double>> Q_MA;

    std::vector<double> reward_hist;
    std::vector<int> div_values;
    int accepted_moves = 0, total_moves = 0;

    // Parameters
    double alpha = 0.7, gamma = 0.6;
    double eps_LS = 1.0, eps_MA = 1.00;
    double eps_decay = 0.99, eps_min = 0.05;
    double SA_temp = 1.0, SA_init = 1.0;
    double flex = 1.0, flex_max = 1.0;
    // -------------------------------------------------
    // LS / OP / MA STATISTICS (SUMMARY ONLY)
    // -------------------------------------------------
    std::map<int, int> LS_usage, OP_usage, MA_usage;
    std::map<int, int> LS_improve, OP_improve, MA_improve;
    std::map<int, double> LS_reward, OP_reward, MA_reward;
    std::map<int, double> LS_time, OP_time, MA_time;

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

    static std::map<int, int> op_counts;
    static std::map<int, double> op_rewards;
    static std::map<int, double> op_time_total;
    static std::map<int, double> op_time_avg;

    // ====================== MAIN LOOP ======================
    while (true)
    {
        iteration++;
        total_time = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
        if (total_time >= max_time) //|| iteration >= max_iter)
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
                                   [](auto &x, auto &y)
                                   { return x.second < y.second; })
                      ->first;
        if (LSi < 1 || LSi > 5)
            LSi = 1;

        // ---------------- Level 2: OP selection (UCB) ----------------
        int selected_op = -1;

        // Force each operator to be tried once
        for (int op = 1; op <= 15; op++)
        {
            if (op_counts[op] == 0)
            {
                selected_op = op;
                break;
            }
        }

        // After all tried, use normal UCB
        if (selected_op == -1)
        {
            double total_counts = 0;
            for (int op = 1; op <= 15; op++)
                total_counts += op_counts[op];

            double best_ucb = -1e18;
            double c = 1.0;

            for (int op = 1; op <= 15; op++)
            {
                double cnt = op_counts[op];

                double avg_reward = op_rewards[op] / cnt;

                double explore = c * std::sqrt((2.0 * std::log(total_counts)) / cnt);

                double ucb = avg_reward + explore;

                if (ucb > best_ucb)
                {
                    best_ucb = ucb;
                    selected_op = op;
                }
            }
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
                                                Q_MA[key_state].end(), [](auto &x, auto &y)
                                                { return x.second < y.second; })
                      ->first;
        if (aMA < 1 || aMA > 5)
            aMA = MA_ONLY_IMPROVE;

        // ---------------- Apply LS + OP ----------------
        auto step_start = std::chrono::steady_clock::now();
        team = Apply_LS_OP(LSi, selected_op, team);
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

        if (accepted)
        {
            prev_eff = cost_eff;
            prev_div = cost_div;
            cost_eff = new_eff;
            cost_div = new_div;
            accepted_moves++;
            LS_improve[LSi]++;
            OP_improve[selected_op]++;
            MA_improve[aMA]++;

            if (new_eff > best_eff && new_div >= min_div)
            {
                best_eff = new_eff;
                best_div = new_div;
                //s_current = team;
                //best_sol = team;
                free_solution(s_current, num_node, num_team, num_each_t);
                s_current = deep_copy_solution(team, num_node, num_team, num_each_t);
                free_solution(best_sol, num_node, num_team, num_each_t);
                best_sol = deep_copy_solution(team, num_node, num_team, num_each_t);
                best_LS = LSi;
                best_OP = selected_op;
                best_MA = aMA;
                time_best_found = total_time;

                for (int m = 0; m < num_node; m++)
                    fbest_solution[m] = best_solution[m];
                for (int m = 1; m <= num_team; m++)
                {
                    eff_fbest[m] = eff_best[m];
                    div_fbest[m] = div_best[m];
                }
            }
        }
        else
        {
            // Restore previous solution
            //team = s_current;
            free_solution(team, num_node, num_team, num_each_t);
            team = deep_copy_solution(s_current, num_node, num_team, num_each_t);

        }

        double step_sec = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - step_start)
                              .count();
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
                  << " | t=" << total_time << "ms\n";
    }

    // ---------------- Close Files ----------------
    outfile.close();
    conv_file.close();
    // ====================== SUMMARY FILE ======================
    double runtime = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count();

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
enum MenuOptions
{
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

void Hyper_heuristic::execute_algorithm(int runs, const std::string &algorithm_name)
{
    d_min = 1.05;
    time_limit = num_node / 2;

    ffbest = std::numeric_limits<int>::min();
    double stat_best = std::numeric_limits<double>::lowest();
    double stat_avg = 0.0;
    double stat_worst = std::numeric_limits<double>::max();
    double avg_time = 0.0;

    // ---------------------------------------------------------------------
    // Create results directory if not exists
    // ---------------------------------------------------------------------
    std::filesystem::path base_folder = "D:/Datasets/MHs_Algorithm_Results/";
    if (!std::filesystem::exists(base_folder))
        std::filesystem::create_directories(base_folder);

    // File paths
    std::filesystem::path results_path = base_folder / ("Results_" + algorithm_name + ".csv");
    std::filesystem::path bestsol_path = base_folder / ("Best_Solution_" + algorithm_name + ".txt");

    std::ofstream results_file(results_path, std::ios::app);
    std::ofstream best_sol_file(bestsol_path, std::ios::app);

    if (!results_file.is_open() || !best_sol_file.is_open())
    {
        std::cerr << "Error opening output files in: " << base_folder << std::endl;
        return;
    }

    // Header for CSV (only if file empty)
    if (results_file.tellp() == 0)
        results_file << "Run,Algorithm,Best_Fitness,Avg_Fitness,Worst_Fitness,Avg_Time(sec)\n";

    int **solution = nullptr;

    // ---------------------------------------------------------------------
    // Main execution over runs
    // ---------------------------------------------------------------------
    for (int run = 0; run < runs; run++)
    {
        fbest = std::numeric_limits<int>::min();
        f_best_inn = std::numeric_limits<int>::min();

        generate_initial();

        double t0 = static_cast<double>(clock());
        int fitness = std::numeric_limits<int>::min();

        if (algorithm_name == "ILS")
        {
            solution = iterated_local_search();
            fitness = fbest;
        }
        else if (algorithm_name == "SA")
        {
            solution = simulated_annealing();
            fitness = fbest;
        }
        else if (algorithm_name == "TS")
        {
            solution = fits();
            fitness = f_best_inn;
        }
        else if (algorithm_name == "FLS")
        {
            fitness = feasible_local_search();
        }
        else if (algorithm_name == "IFLS")
        {
            solution = infeasible_local_search();
            fitness = f_best_inn;
        }
        else if (algorithm_name == "MA")
        {
            solution = memetic();
            fitness = fbest;
        }
        else if (algorithm_name == "GD")
        {
            solution = great_deluge_algorithm();
            fitness = fbest;
        }
        else if (algorithm_name == "LAHC")
        {
            solution = late_acceptance_hill_climbing();
            fitness = fbest;
        }
        else if (algorithm_name == "GLS")
        {
            fitness = guided_local_search();
        }
        else
        {
            std::cerr << "Unknown algorithm: " << algorithm_name << ". Skipping run.\n";
            continue;
        }

        double t1 = static_cast<double>(clock());
        double elapsed_time = (t1 - t0) / CLOCKS_PER_SEC;
        avg_time += elapsed_time;

        // Track global best
        if (fitness > ffbest)
        {
            ffbest = fitness;

            for (int m = 0; m < num_node; m++)
                fbest_solution[m] = best_solution[m];
            for (int m = 1; m <= num_team; m++)
            {
                eff_fbest[m] = eff_best[m];
                div_fbest[m] = div_best[m];
            }
        }

        if (fitness > stat_best)
            stat_best = fitness;
        if (fitness < stat_worst)
            stat_worst = fitness;
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
    std::cout << "\n";
    std::cout << "------------------------------------------------\n";
    std::cout << "The best solution:\n";
    for (int t = 1; t <= num_team; t++)
        std::cout << "eff[" << t << "] = " << eff_fbest[t] << " ";
    std::cout << "\n";
    for (int t = 1; t <= num_team; t++)
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
    for (int t = 1; t <= num_team; t++)
        best_sol_file << eff_fbest[t] << " ";
    best_sol_file << "\nDiversity per Team: ";
    for (int t = 1; t <= num_team; t++)
        best_sol_file << div_fbest[t] << " ";
    best_sol_file << "\nBest Solution Vector:\n";
    for (int i = 0; i < num_node; i++)
        best_sol_file << fbest_solution[i] << " ";
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
    switch (choice)
    {
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
    if (instance_str.length() < 2)
    {
        instance_str = "0" + instance_str;
    }

    // Construct the full file path
    return dataset_dir + instance_str + pattern;
}

std::string parseInstanceName(const std::string &filename)
{
    // Match and extract parts from the filename
    std::regex pattern(R"((\d{2})test-n(\d+)m(\d+)t(\d+)\.dat)");
    std::smatch match;

    if (std::regex_match(filename, match, pattern))
    {
        std::string instance = match[1].str() + "-P" + match[2].str() + "T" + match[3].str() + "M" + match[4].str();
        return instance;
    }
    else
    {
        return ""; // Return an empty string if parsing fails
    }
}

int main(int argc, char *argv[])
{
    // srand((unsigned)time(NULL));
    if (argc < 1)
    {
        cout << "usage: input_file";
        exit(1);
    }

    std::string datasetDir = "D:/Datasets/";
    std::string resultsDir = "D:\\Datasets\\RESULTS_OF_HH_MODELS_ML_CMCEE\\";

    std::vector<std::string> datasetFiles;
    for (const auto &entry : fs::directory_iterator(datasetDir))
    {
        if (entry.is_regular_file())
        {
            datasetFiles.push_back(entry.path().string());
        }
    }

    Hyper_heuristic H;
    // ====================================================
    //              MAIN MENU LOOP
    // ====================================================
    while (true)
    {

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

        if (choice == 0)
        {
            std::cout << "Program terminated.\n";
            break;
        }

        // ---------------------------------------------
        // SELECT SINGLE OPTIMIZATION (OPTION 12)
        // ---------------------------------------------
        int single_choice = -1;
        if (choice == 7)
        {
            std::cout << "\nSelect ONE Optimization Algorithm:\n";
            std::cout << "7:  Iterated Local Search (ILS)\n";
            std::cout << "8:  Simulated Annealing (SA)\n";
            std::cout << "9:  Tabu Search (TS)\n";
            std::cout << "10: Feasible Local Search (FLS)\n";
            std::cout << "11: Infeasible Local Search (IFLS)\n";
            std::cout << "12: Memetic Algorithm (MA)\n";
            std::cout << "13: Great Deluge Algorithm (GD)\n";
            std::cout << "14: Late Acceptance Hill Climbing (LAHC)\n";
            std::cout << "Enter (7 to 14): ";
            std::cin >> single_choice;

            if (single_choice < 7 || single_choice > 14)
            {
                std::cout << "Invalid selection. Returning to menu...\n";
                continue;
            }
        }

        // ====================================================
        //           RUN SELECTED METHOD ON ALL INSTANCESdatasetFiles.size()
        // ====================================================
        const int INDEPENDENT_RUNS = 31;
        const int max_time = 600; // 10 minutes per run

        for (size_t idx = 0; idx < datasetFiles.size(); idx++)
        {
            std::string current_file = datasetFiles[idx];
            fs::path p(current_file);
            std::string rawName = p.filename().string();
            std::string instanceName = parseInstanceName(rawName);

            if (instanceName.empty())
                continue;

            std::cout << "\n=====================================\n";
            std::cout << "Instance: " << instanceName << "\n";
            std::cout << "=====================================\n";

            std::string resultsPath =
                resultsDir + "Results_" + instanceName + ".csv";

            std::ofstream resultsFile(resultsPath);
            resultsFile << "Run,BestEff,BestDiv,Time(sec)\n";

            std::vector<double> eff_values;
            std::vector<double> div_values;
            std::vector<double> time_values;

            double global_best_eff = -1e9;
            double global_best_div = 0;
            int best_run_id = 0;

            for (int run = 1; run <= INDEPENDENT_RUNS; run++)
            {
                std::cout << "\n---- Run " << run
                          << " / " << INDEPENDENT_RUNS << " ----\n";
                H.Parameters();
                H.initialization(current_file);
                H.generate_initialrandom();
                //H.compute_mindiv();
                H.display(team);

                // Display problem info
                std::cout << "num_node="  << num_node
                          << "  num_team=" << num_team
                          << "  num_each_t=" << num_each_t
                          << "  min_div="   << min_div << "\n";

                auto start = std::chrono::steady_clock::now();

                switch (choice)
                {
                case 1:
                    H.Q_Learning_Selection_Hyperheuristic_CMCEE(max_time);
                    break;
                case 2:
                    H.HH_Choice_Function_Selection_CMCEE(max_time);
                    break;
                case 3:
                    H.Random_Selection_Hyperheuristic_CMCEE(max_time);
                    break;
                case 4:
                    H.MAB_Selection_Hyperheuristic_CMCEE(max_time);
                    break;
                case 5:
                    H.Greedy_Selection_Hyperheuristic_CMCEE(max_time);
                    break;
                case 6:
                    H.TriLevel_HH_Qlearning_CMCEE(max_time);
                    break;
                case 7:
                    H.runSingleOptimizationAlgorithm(single_choice, max_time);
                    break;
                default:
                    break;
                }

                auto end = std::chrono::steady_clock::now();
                double time_taken =
                    std::chrono::duration<double>(end - start).count();

                eff_values.push_back(best_eff);
                div_values.push_back(best_div);
                time_values.push_back(time_taken);

                if (best_eff > global_best_eff)
                {
                    global_best_eff = best_eff;
                    global_best_div = best_div;
                    best_run_id = run;
                }

                resultsFile << run << ","
                            << best_eff << ","
                            << best_div << ","
                            << time_taken << "\n";

                std::cout << "Run " << run << " finished.\n";
            }

            // ===== Statistics =====
            auto mean = [](const std::vector<double> &v)
            {
                return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            };

            auto stddev = [&](const std::vector<double> &v)
            {
                double m = mean(v);
                double sum = 0.0;
                for (double x : v)
                    sum += (x - m) * (x - m);
                return std::sqrt(sum / (v.size() - 1));
            };

            double mean_eff = mean(eff_values);
            double std_eff = stddev(eff_values);
            double min_eff = *std::min_element(eff_values.begin(), eff_values.end());
            double max_eff = *std::max_element(eff_values.begin(), eff_values.end());
            double mean_time = mean(time_values);
            double std_time = stddev(time_values);

            resultsFile << "\nSUMMARY\n";
            resultsFile << "MeanEff," << mean_eff << "\n";
            resultsFile << "StdEff," << std_eff << "\n";
            resultsFile << "MinEff," << min_eff << "\n";
            resultsFile << "MaxEff," << max_eff << "\n";
            resultsFile << "MeanTime," << mean_time << "\n";
            resultsFile << "StdTime," << std_time << "\n";
            resultsFile << "BestOverallRun," << best_run_id << "\n";
            resultsFile << "BestOverallEff," << global_best_eff << "\n";
            resultsFile << "BestOverallDiv," << global_best_div << "\n";

            resultsFile.close();

            std::cout << "Saved: " << resultsPath << "\n";
        }
    }

    free_memory();
    return 0;
}

