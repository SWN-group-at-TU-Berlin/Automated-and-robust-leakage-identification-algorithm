# Automated-and-robust-leakage-identification-algorithm
This is a Python-based algorithm for robust and automated leakage detection in water distribution networks. It analyzes pressure observations from water distribution systems to identify leakages.

# Code and Dataset based on publication
E. Steins, N. Langer, J. Kowalski, A. Cominola. “From theory to practice: towards robust and automated data-driven leakage detection in water distribution networks”, …

# Abstract
Reliable leak detection in water distribution networks is essential for reducing water losses, costs, and operational risks. We here present an improved semi-supervised pressure-based algorithm that enables near real-time detection through two key features: (i) automation of sensor selection and retraining times, avoiding manual tuning and prior knowledge of leak conditions, and (ii) an adaptive, nonparametric cumulative sum (CUSUM) with a self-starting scheme, robust to noisy and non-normal data. Tested on a real-world network dataset with synthetic non-overlapping leaks, the algorithm detects all leakages without false alarms, demonstrating strong potential for practical deployment. 


# Key Features
1.	Automated sensor selection: Training of regression modules on pressure data for automated sensor combinations.
2.	Iterative training times: Automated re-training times, including a verification module that adjusts training periods to avoid them containing leakages.
3.	Adaptive Detection: Uses an adaptive nonparametric CUSUM method for robust leak detection.
4.	Configurable Parameters: Centralized configuration system for easy parameter setting.

	
# Algorithm Workflow
The workflow of the algorithm is summarized in E. Steins, N. Langer, J. Kowalski, A. Cominola. “From theory to practice: towards robust and automated data-driven leakage detection in water distribution networks”, … (see Figure 1).


# Requirements
Python >= 3.8

numpy >= 1.21.0

pandas >= 1.3.0

matplotlib >= 3.4.0


# Project Structure

1. LILA2_config.py: Configuration class with all parameters 

2. LILA2_main.py: Main execution script

3. LILA2_utils.py: Core algorithm functions

4. data/ pressure.csv and flows.csv: Pressure measurements from sensors and flow (pump) data

5. README.md: This file


# Case study
The case study is described in the publication. The dataset consists of two files, i.e., pressure.csv and flow.csv . The default configuration parameters, including CUSUM threshold, ground truth, sensor combinations, and initial start date, are specified for this dataset. Custom configurations for other datasets are possible by modifying the parameters programmatically:

from LILA2_config import LILAConfig 

#Create custom configuration  

config = LILAConfig(cusum_threshold=60.0, training_days=10, detection_days=21, data_dir="./my_data/")

#Validate configuration

config.validate()

#Run detection

from LILA2_main import main

main()

# Run script

Run with default configuration: python LILA2_main.py


# References

R. Liu, J. Tang, Z. Zhang (2014). Adaptive nonparametric CUSUM scheme for detecting unknown shifts in location. International Journal of Production Research, 52(6), 1592-1606.

E. Steins, A. Cominola (2025). Addressing Practical Challenges of Stochastic Process Control for Leakage Detection in Water Distribution Networks: A Comparative Analysis. In: Journal of Water Resources Planning and Management 151.10, p. 04025056.

E. Steins, N. Langer, J. Kowalski, A. Cominola. From theory to practice: towards robust and automated data-driven leakage detection in water distribution networks, …

I. Daniel, J. Pesantez, S. Letzgus, M. Khaksar Fasael, F. Alghamdi, E. Berglund, G. Mahinthakumar, A. Cominola (2022). A sequential pressure-based algorithm for data-driven leakage identification and model-based localization in 169 water distribution networks. In: Journal of Water Resources Planning and Management 148.6, p. 04022025.

I. Daniel, S. Letzgus, A. Cominola (2021). A high-resolution pressure- driven leakage identification and localization algorithm [Code]. Accessed April 10, 2025. https://github.com/SWN-group-at-TU-Berlin/LILA.

# Citation
If you use LILA2 in your research, please cite:

@software{lila2_2025,
  author = {E. Steins, N. Langer, J. Koslowski, A. Cominola},
  title = {Automated and robust leak identification algorithm},
  year = {2025},
  url = {https://github.com/Ella-Steins/Automated-and-robust-leak-identification-algorithm}
}


# Contact
For questions or issues: steins@tu-berlin.de

# Acknowledgments

1.	Original LILA algorithm development: I. Daniel et al. (2022). “A sequential pressure-based algorithm for data-driven leakage identification and model-based localization in 169 water distribution networks”. In: Journal of Water Resources Planning and Management 148.6, p. 04022025.
   
2.	Funding: This work is supported by the iOLE project, which receives funding from the Federal Ministry of Education and Research (BMBF) within the funding measure “Digital GreenTech—Environmental Engineering meets Digitalisation” as part of the “Research for Sustainability (FONA) Strategy” (funding code: 02WDG1689A).
   
3.	Data providers: The authors thank Gelsenwasser AG for providing the hydraulic model used in this study and for sharing the corresponding measurement data, which was crucial to achieve the results.
   
4.	After implementation, the project code was cleaned using the suggestions of Claude AI, which were then reviewed and integrated into the code by the authors. 
