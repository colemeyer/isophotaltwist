# isophotaltwist

1.1 – Automated input and output of galaxies in ImageFIT.py, with 3 models for each: Sersic, Double Sersic, and Sersic+Exponential. Removed individual galaxy folders from main directory.

2.0 - Converted all .py scripts to .ipynb files. Separated automated steps into cutout, masking, and fitting. Generalized piping to work on multiple systems with minimal modification. Attempted a fix of masking functions.

2.1 - Simplified code package; code now creates its own folder and files if they don't already exist, and modifies existing folders and files if present.

2.2 – Adjusted masking and fitting parameters to ensure better starting guesses:
          Threshold –> 5.0
          Contrast –> 0.001
          Solver –> 'NM'
          Center of galaxy bounds –> [center of image] +/- 20
      Implemented try/except function to prevent halted program. Separated output into 3 separate CSV files, according to model.
      
2.3 – Removed ring from masks and updated outputs.
