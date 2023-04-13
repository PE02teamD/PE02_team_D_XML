import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

tree = elemTree.parse('../data set/HY202103_D07_(0,0)_LION1_DCM_LMZC.xml')

voltage = tree.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Voltage')
current = tree.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Current')


voltage_list = list(map(float, voltage.text.split(',')))
current_list = list(map(float, current.text.split(',')))
current_list = list(map(abs, current_list))

x = voltage_list
y = current_list

# fitting
pf = np.polyfit(x, y, 12)
f = np.poly1d(pf)

# R2
from sklearn.metrics import r2_score
R2 = r2_score(y, f(x))
print(R2)
print(R2 == 1.0)

# graph
plt.plot(x, y, '--ok', x, f(x), 'r')
plt.xlabel('Voltage[V]')
plt.ylabel('Current[C]')
plt.title('IV-analysis')
plt.yscale('log')



<<<<<<< HEAD
plt.show()

=======
plt.show()
>>>>>>> 4d00831d2e036541d649fc2fa6661f0e25d3c3a7
