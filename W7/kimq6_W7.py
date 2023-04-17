import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

# Approach the path
rootDir = ""  # Input your path
fname = 'HY202103_D07_(0,0)_LION1_DCM_LMZC.xml'  # Input file name
WorkingDir = rootDir + fname

# Parse XML file
tree = elemTree.parse(WorkingDir)
root = tree.getroot()  # 해당 트리의 root를 반환

# Handle subplot
fig, axes = plt.subplots(2, 3, figsize=(12, 5))

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
print(type(fp))
print(type(iv_data['current']))


# R-squared 구하기
def calc_R_squared(y_pred, y_test):
    current_predicted_poly = np.polyval(y_pred, y_test)
    residuals = y_test - current_predicted_poly
    SSR = np.sum(residuals ** 2)
    SST = np.sum((y_test - np.mean(y_test) ** 2))
    return 1 - (SSR / SST)


# Plot data using matplotlib
axes[1][0].scatter('voltage', 'current', data=iv_data, color='mediumseagreen', label='data')
axes[1][0].plot(iv_data['voltage'], f(iv_data['voltage']), linestyle='--', lw=2, color='r', label='best-fit')

# Add annotations for current values and R-squared value
for x, y in zip(iv_data['voltage'], iv_data['current']):
    if x in [-2.0, -1.0, 1.0]:
        axes[1][0].annotate(f"{y:.2e}A", xy=(x, y), xytext=(3, 10), textcoords='offset points', ha='center',
                            fontsize=10)
axes[1][0].annotate(f"R² = {calc_R_squared(fp, iv_data['current'])}", xy=(-2.1, 10 ** -6), ha='left', fontsize=15)

# Handle graph details
axes[1][0].set_yscale('log', base=10)
axes[1][0].set_xlabel('Voltage [V]', size=16, fontweight='bold')
axes[1][0].set_ylabel('Current [A]', size=16, fontweight='bold')
axes[1][0].set_title('IV - analysis', size=20, fontweight='bold', style='italic')
axes[1][0].tick_params(axis='both', which='major', size=14)  # tick 크기 설정
axes[1][0].grid()
axes[1][0].legend(fontsize=16)

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
    axes[0][0].plot('wavelength', 'measured_transmission', data=wavelength_data, color=color,
                    label=wavelength_sweep.get('DCBias') + ' V'
                    if wavelength_sweep != list(root.iter('WavelengthSweep'))[-1] else '')

# Handle graph details
axes[0][0].set_xlabel('Wavelength [nm]', size=16, fontweight='bold')
axes[0][0].set_ylabel('Measured_transmission [dB]', size=16, fontweight='bold')
axes[0][0].set_title('Transmission spectra - as measured', size=20, fontweight='bold', style='italic')
axes[0][0].tick_params(axis='both', which='major', size=14)  # tick 크기 설정
axes[0][0].grid()
axes[0][0].legend(loc='lower center', ncol=3, fontsize=10)

# 축 지우기
axes[0][2].axis('off')
axes[1][1].axis('off')
axes[1][2].axis('off')

# 데이터 가져오기
asdf = list(root.iter('WavelengthSweep'))[-1]
print(asdf)

# dic 생성, 값을 그래프에
data = {'wavelength': [], 'measured_transmission': []}
wavelength = list(map(float, asdf.find('L').text.split(',')))
measured_transmission = list(map(float, asdf.find('IL').text.split(',')))
data['wavelength'].extend(wavelength)
data['measured_transmission'].extend(measured_transmission)
axes[0][1].plot(data['wavelength'], data['measured_transmission'], 'r')

# polyfit 이용해 근사
from sklearn.metrics import r2_score
poly_list = []
poly_color = ['b', 'orange', 'lime', 'r', 'purple', 'brown', 'pink', 'cyan']
for i in range(8):
    fp = np.polyfit(data['wavelength'], data['measured_transmission'], i + 1)
    f = np.poly1d(fp)
    poly_list.append(f)
    R2 = r2_score(data['measured_transmission'], f(data['measured_transmission']))
    axes[0][1].plot(data['wavelength'], poly_list[i](data['wavelength']), color=poly_color[i], lw=0.8, label=f'{i + 1}th')
    axes[0][1].annotate(f"R² = {R2}", xy=(1580, -8 - i))
    axes[0][1].legend(loc='lower center', ncol=3, fontsize=10)

# Output graph
plt.show()
