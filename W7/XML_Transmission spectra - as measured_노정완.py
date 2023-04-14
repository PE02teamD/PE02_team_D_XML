import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from sklearn.metrics import r2_score

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

# Graph 1: I-V curve
# Extract current and voltage data
iv_data = {'voltage': [], 'current': []}
for iv_measurement in root.iter('IVMeasurement'):
    current = list(map(float, iv_measurement.find('Current').text.split(',')))
    voltage = list(map(float, iv_measurement.find('Voltage').text.split(',')))
    current_abs = [abs(i) for i in current]
    iv_data['voltage'].extend(voltage)
    iv_data['current'].extend(current_abs)

I_s=iv_data['current'][0]
# Threshold voltage 기준으로 구간 설정
x_bef_Vth, x_aft_Vth = iv_data['voltage'][:10], iv_data['voltage'][9:]
y_bef_Vth, y_aft_Vth = iv_data['current'][:10], iv_data['current'][9:]
# Threshold voltage 이전
fp = np.polyfit(x_bef_Vth, y_bef_Vth, 7) # Data point가 9개 이므로 7차까지 해야 근사에 의미가 있음
f = np.poly1d(fp) # Equation으로 만듬

# IV data 근사하는 함수
def fit_IV(voltage, I_s, q, nkT):
    return I_s * (np.exp(q * voltage / nkT ) - 1)

# Threshold voltage 이후
model = Model(fit_IV)
params = model.make_params(I_s=I_s, q=1, nkT=1)
# fit the model to the data
result = model.fit(y_aft_Vth, params, voltage=x_aft_Vth)
y_fit = list(f(x_bef_Vth))
y_fit.extend(result.best_fit[1:])
residuals = np.subtract(current_abs, y_fit)

# Plot data using matplotlib
ax1.scatter('voltage', 'current', data=iv_data, color='mediumseagreen', label='data')
ax1.plot(voltage, y_fit, linestyle='--', lw=2, color='r', label='best-fit')
# Add annotations for current values and R-squared value
for x, y in zip(iv_data['voltage'], iv_data['current']):
    if x in [-2.0, -1.0, 1.0]:
        ax1.annotate(f"{y:.2e}A", xy=(x, y), xytext=(3, 10), textcoords='offset points', ha='center', fontsize=10)
ax1.annotate(f"R² = {r2_score(current_abs, y_fit)}", xy=(-2.1, 10 ** -6), ha='left', fontsize=12)

# Graph 2
# Handle label color
cmap, a = plt.colormaps.get_cmap('jet'), 0
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

r2_list = []
max_r2 = 0
ax3.plot('wavelength', 'measured_transmission', data=wavelength_data, label='')
for i in range(1, 9):
    color = cmap(i / 9)
    fp = np.polyfit(wavelength_data['wavelength'], wavelength_data['measured_transmission'], i)
    f = np.poly1d(fp)
    r2 = r2_score(wavelength_data['measured_transmission'], f(wavelength_data['wavelength']))
    r2_list.append(r2)
    if r2_list[i-1] > max_r2:
        max_r2 = r2
    ax3.plot(wavelength_data['wavelength'], f(wavelength_data['wavelength']), color=color, lw=0.8, label=f'{i}th')

for i in range(8):
    ax3.annotate(f"R² = {r2_list[i]}", xy=(1580.7, -16 + i), ha='left', fontsize=8, color='r' if r2_list[i] == max_r2 else None)

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
