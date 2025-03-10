# coding=utf-8
import pandas as pd

### Change this to your file location ###
# Open Input Sounding file
file = open('C:/Users/jmayhall/Downloads/aes740_hw3/input_sounding')

# Separate first line (sfc pres (mb)    sfc theta (K)    sfc qv (g/kg))
first_line = file.readline()

# Create DataFrame from the file without the first line
df = pd.read_csv(file, sep=' ', header=None, names=['z (m)', 'theta (K)', 'qv (g/kg)', 'u (m/s)', 'v (m/s)'])

# Define variables to change (just u)
u = df['u (m/s)'].values

# Apply slight change in u values to change wind shear
df['u (m/s)'] = df['u (m/s)'] * (1 + 0.1 * df.index)

### Change this to your file location ###
# Specify the file name
filename = "C:/Users/jmayhall/Downloads/aes740_hw3/input_sounding"

# Write the header line first
with open(filename, "w") as f:
    f.write(first_line)

# Append the DataFrame
df.to_csv(filename, mode = "a", sep = "\t", index = False, header = False)