**Execution Options**
**Option 1 — Windows Local Execution**

		Download and extract the dataset files into the folder:

		D:\Datasets\

		Ensure the program path in main() is set to:

		std::string datasetDir = "D:/Datasets/";

		Compile the program using MinGW or Visual Studio in Release mode, then run the executable.
Results will be automatically saved inside:

	D:\Datasets\

**Option 2 — GitHub Codespaces Execution**

		Upload the dataset folder into the repository under:
		/Datasets/

		Modify the dataset path in main() to:

		std::string datasetDir   = "/workspaces/TL-HH/Datasets";

		Compile and run the program using the Codespace terminal:

		g++ -std=c++17 -O2 main.cpp -o CMCEE
		./CMCEE

Results will be saved inside the repository directory.

Citation 

If you use this code in your research, please cite:

A. K. Abbas and E. Taha Yassen, 
“Machine Learning-Driven Tri-Level Hyper-Heuristic Selection With Adaptive Move Acceptance for Composing Medical Crew Scheduling Problem,” 
IEEE Access, vol. 14, pp. 37206–37232, 2026. 
https://doi.org/10.1109/ACCESS.2026.3671676
