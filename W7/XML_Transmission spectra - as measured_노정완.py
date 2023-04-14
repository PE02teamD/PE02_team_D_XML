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
fig, axs = plt.subplots(2, 3, figsize=(18, 8))
# Increase the horizontal and vertical spacing between subplots
fig.subplots_adjust(hspace=0.5, wspace=0.3)
# Numbering each subplot
ax1, ax2, ax3 = axs[1][0], axs[0][0], axs[0][1]
# Hide other graph
for axs in axs.flatten():
    if axs not in [ax1, ax2, ax3]:
        axs.axis('off')

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
def calc_R_squared(x_set, y_set):
    y_predicted_poly = np.polyval(fp, x_set)
    residuals = y_set - y_predicted_poly
    SSR = np.sum(residuals ** 2)
    SST = np.sum((y_set - np.mean(y_set)) ** 2)
    return 1 - (SSR / SST)


# Plot data using matplotlib
ax1.scatter('voltage', 'current', data=iv_data, color='mediumseagreen', label='data')
ax1.plot(iv_data['voltage'], f(iv_data['voltage']), linestyle='--', lw=2, color='r', label='best-fit')
# Add annotations for current values and R-squared value
for x, y in zip(iv_data['voltage'], iv_data['current']):
    if x in [-2.0, -1.0, 1.0]:
        ax1.annotate(f"{y:.2e}A", xy=(x, y), xytext=(3, 10), textcoords='offset points', ha='center', fontsize=10)
ax1.annotate(f"R² = {calc_R_squared(voltage, current_abs):2f}", xy=(-2.1, 10 ** -6), ha='left', fontsize=12)

# Graph 2
# Handle label color
cmap, a = plt.cm.get_cmap('jet'), 0
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

# Graph 3
# Ignore RankWarning from numpy.polyfit()
import warnings
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned', category=np.RankWarning)

ax3.plot('wavelength', 'measured_transmission', data=wavelength_data, label='')
for i in range(1, 9):
    color = cmap(i / 9)
    fp = np.polyfit(wavelength_data['wavelength'], wavelength_data['measured_transmission'], i)
    f = np.poly1d(fp)
    ax3.plot(wavelength_data['wavelength'], f(wavelength_data['wavelength']), color=color, lw=0.8, label=f'{i}th')
    ax3.annotate(f"R² = {calc_R_squared(wavelength_data['wavelength'], wavelength_data['measured_transmission'])}",
                 xy=(1580.7, -17 + i), ha='left', fontsize=8)

detail_list = [
    {'ax1_title': 'IV - analysis', 'ax1_titlesize': 15,
     'ax1_xlabel': 'Voltage [V]', 'ax1_ylabel': 'Current [A]', 'ax1_size': 13, 'ax1_ticksize': 14,
     'ax1_legendloc': 'best', 'ax1_legendncol': 1, 'ax1_legendsize': 10},

    {'ax2_title': 'Transmission spectra - as measured', 'ax2_titlesize': 15,
     'ax2_xlabel': 'Wavelength [nm]', 'ax2_ylabel': 'Measured_transmission [dB]', 'ax2_size': 13, 'ax2_ticksize': 14,
     'ax2_legendloc': 'lower center', 'ax2_legendncol': 3, 'ax2_legendsize': 8},

    {'ax3_title': 'Transmission spectra - as measured', 'ax3_titlesize': 15,
     'ax3_xlabel': 'Wavelength [nm]', 'ax3_ylabel': 'Measured_transmission [dB]', 'ax3_size': 13, 'ax3_ticksize': 14,
     'ax3_legendloc': 'lower center', 'ax3_legendncol': 3, 'ax3_legendsize': 10}
]

for i, axs in enumerate([ax1, ax2, ax3]):
    details = detail_list[i]
    axs.set_xlabel(details[f'ax{i + 1}_xlabel'], size=details[f'ax{i + 1}_size'], fontweight='bold')
    axs.set_ylabel(details[f'ax{i + 1}_ylabel'], size=details[f'ax{i + 1}_size'], fontweight='bold')
    axs.set_title(details[f'ax{i + 1}_title'], size=details[f'ax{i + 1}_titlesize'], fontweight='bold', style='italic')
    axs.tick_params(axis='both', which='major', size=details[f'ax{i + 1}_legendsize'])  # tick 크기 설정
    axs.legend(loc=details[f'ax{i + 1}_legendloc'], ncol=details[f'ax{i + 1}_legendncol'],
               fontsize=details[f'ax{i + 1}_legendsize'])
    axs.grid()
ax1.set_yscale('log', base=10)

# Output graph
plt.show()

# test 3:40
