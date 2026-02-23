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
