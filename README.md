How to Run
🖥 OPTION 1 — Windows Local Execution
Step 1 — Download Dataset

Extract dataset files into:

D:\Datasets\

Expected structure:

D:\Datasets\
    01-P100T10M6.txt
    02-P200T10M12.txt
    ...
Step 2 — Verify Dataset Path

Ensure main() contains:

std::string datasetDir = "D:/Datasets/";
Step 3 — Compile

Using MinGW:

g++ -std=c++17 -O2 main.cpp -o CMCEE.exe
CMCEE.exe

Or build in Release mode using Visual Studio.

Step 4 — Select Method

Program menu:

1–7 : Hyper-Heuristic Methods
8   : Single Optimization Algorithms
0   : EXIT

If selecting option 8, the following menu appears:

1  : ILS
2  : SA
3  : TS
4  : FLS
5  : IFLS
6  : MA
7  : GD
8  : VNS
9  : LAHC
10 : GLS
11 : HSA
Step 5 — Results

Results are automatically saved in:

D:\Datasets\MHs_Algorithm_Results\

Generated files:

Results_ALGO.csv

Best_Solution_ALGO.txt

☁ OPTION 2 — GitHub Codespaces Execution
Step 1 — Upload Dataset

Place dataset inside repository:

/Datasets/
Step 2 — Update Dataset Path

Modify in main():

std::string datasetDir = "Datasets/";
Step 3 — Compile and Run

Inside Codespace terminal:

g++ -std=c++17 -O2 main.cpp -o CMCEE
./CMCEE
