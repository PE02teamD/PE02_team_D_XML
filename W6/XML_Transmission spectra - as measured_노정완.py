import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

# Approach the path
rootDir = "../data set/"  # Input your path
fname = 'HY202103_D07_(0,0)_LION1_DCM_LMZC.xml'  # Input file name
WorkingDir = rootDir + fname

# Parse XML file
tree = elemTree.parse(WorkingDir)
root = tree.getroot()  # 해당 트리의 root를 반환

# Handle subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Graph 1
# Initialize data containers
iv_data = {'voltage': [], 'current': []}

# Extract current and voltage data
for iv_measurement in root.iter('IVMeasurement'):
    current = list(map(float, iv_measurement.find('Current').text.split(',')))
    voltage = list(map(float, iv_measurement.find('Voltage').text.split(',')))
    current_abs = [abs(i) for i in current]
    iv_data['voltage'].extend(voltage)
    iv_data['current'].extend(current_abs)

# Poly1d, numpy (np.poly1d)로 근사치 매기는 fitting
fp = np.polyfit(iv_data['voltage'], iv_data['current'], 12)
f = np.poly1d(fp)


# R-squared 구하기
def calc_R_squared():
    current_predicted_poly = np.polyval(fp, current_abs)
    residuals = current_abs - current_predicted_poly
    SSR = np.sum(residuals ** 2)
    SST = np.sum((current_abs - np.mean(current_abs) ** 2))
    return 1 - (SSR / SST)


# Plot data using matplotlib
ax1.scatter('voltage', 'current', data=iv_data, color='mediumseagreen', label='data')
ax1.plot(iv_data['voltage'], f(iv_data['voltage']), linestyle='--', lw=2, color='r', label='best-fit')

# Add annotations for current values and R-squared value
for x, y in zip(iv_data['voltage'], iv_data['current']):
    if x in [-2.0, -1.0, 1.0]:
        ax1.annotate(f"{y:.2e}A", xy=(x, y), xytext=(3, 10), textcoords='offset points', ha='center', fontsize=10)
ax1.annotate(f"R² = {calc_R_squared()}", xy=(-2.1, 10 ** -6), ha='left', fontsize=15)

# Handle graph details
ax1.set_yscale('log', base=10)
ax1.set_xlabel('Voltage [V]', size=16, fontweight='bold')
ax1.set_ylabel('Current [A]', size=16, fontweight='bold')
ax1.set_title('IV - analysis', size=20, fontweight='bold', style='italic')
ax1.tick_params(axis='both', which='major', size=14)  # tick 크기 설정
ax1.grid()
ax1.legend(fontsize=16)

# Graph 2
# Handle label color
cmap = plt.cm.get_cmap('jet')
a = 0
# Extract Wavelength and dB data
for wavelength_sweep in root.iter('WavelengthSweep'):
    # Choose a color for the scatter plot based on the iteration index
    color = cmap(a / 7)
    a += 1
    # Make it a dict for easier handling
    wavelength_data = {'wavelength': [], 'measured_transmission': []}
    # Get data from each element
    wavelength = list(map(float, wavelength_sweep.find('L').text.split(',')))
    measured_transmission = list(map(float, wavelength_sweep.find('IL').text.split(',')))
    wavelength_data['wavelength'].extend(wavelength)
    wavelength_data['measured_transmission'].extend(measured_transmission)
    # Create a scatter plot using the data
    ax2.plot('wavelength', 'measured_transmission', data=wavelength_data, color=color,
             label=wavelength_sweep.get('DCBias') + ' V'
             if wavelength_sweep != list(root.iter('WavelengthSweep'))[-1] else '')

# Handle graph details
ax2.set_xlabel('Wavelength [nm]', size=16, fontweight='bold')
ax2.set_ylabel('Measured_transmission [dB]', size=16, fontweight='bold')
ax2.set_title('Transmission spectra - as measured', size=20, fontweight='bold', style='italic')
ax2.tick_params(axis='both', which='major', size=14)  # tick 크기 설정
ax2.grid()
ax2.legend(loc='lower center', ncol=3, fontsize=10)

# Output graph
plt.show()