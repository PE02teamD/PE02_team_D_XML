import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from lmfit import Model, Parameters
import time
import warnings


##파싱하기, 빈리스트만들기
root = ET.parse('HY202103_D07_(0,0)_LION1_DCM_LMZC.xml').getroot()
v = []
for iv_ in root.find('.//IVMeasurement'):
    v.append(list(map(float, iv_.text.split(','))))


##다항식, 지수 근사
def poly_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def i_d_function(x, a, b, c, d):
    # b * (np.exp((d * x) / (a * c)) - 1) 출력
    return (a * np.exp(x / b) + c) * d


## v[0], v[1]에 배열
x = np.asarray(v[0][0:9])
y = np.asarray(v[1][0:9])

x2 = np.asarray(v[0][8:13])
y2 = np.asarray(v[1][8:13])

## 변수abcd값 선언 후 1로 초기화
params = Parameters()
params.add("a", value=1)
params.add("b", value=1)
params.add("c", value=1)
params.add("d", value=1)

## 근사
poly_model = Model(poly_function)
poly_result = poly_model.fit(y, x=x, params=params)
print('R_Square value for poly-function: ', r2_score(y, poly_result.best_fit))

x_values = np.linspace(-2, 1, 50)
y_values = []
for i in x_values:
    y_values.append(poly_function(i, poly_result.values['a'], poly_result.values['b'], poly_result.values['c'],
                                  poly_result.values['d']))

i_d_model = Model(i_d_function)
i_d_result = i_d_model.fit(y2, x=x2, params=params)
print('R_Square value for id-function: ', r2_score(y2, i_d_result.best_fit))
print()

x_values2 = np.linspace(0.1, 1, 50)
y_values2 = []
for i in x_values2:
    y_values2.append(i_d_function(i, i_d_result.values['a'], i_d_result.values['b'], i_d_result.values['c'],
                                  i_d_result.values['d']))


# IV (Raw data)
plt.subplot(2,3,1)
plt.plot(v[0], abs(np.asarray(v[1])), 'k-o')
plt.title('IV-Analysis')
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A]')
plt.yscale('log')

# Spectrum (Raw data)
v = []
for waveLengthSweep in root.findall('.//WavelengthSweep'):
    waveValues = []
    for iv_ in waveLengthSweep:
        waveValues.append(list(map(float, iv_.text.split(','))))
    waveValues.append(waveLengthSweep.attrib['DCBias'])
    v.append(waveValues)

plt.subplot(2,3,2)
plots = []
for i in range(len(v) - 1):
    line, = plt.plot(v[i][0], v[i][1], label="DCBias=\"" + str(v[i][2]) + "\"")
    plots.append(line)

line, = plt.plot(v[6][0], v[6][1], color='black', label="REF")
plt.gca().add_artist(plt.legend(handles=[line], loc='upper right'))
plt.legend(handles=plots, ncol=2, loc="lower center")
plt.title("Transmission spectra - as measured")
plt.xlabel('Wavelength [nm]')
plt.ylabel('Measured transmission [dB]')


# Spectrum (Ref. - Raw data & fitting data)
plt.subplot(2,3,3)
plt.plot(v[6][0], v[6][1], color='black', label="Fit ref polynomial O3")

handle = []
color = ['red', 'green', 'blue', 'limegreen', 'orange', 'purple', 'blue']
for i in range(2, 8):
        model = np.poly1d(np.polyfit(v[6][0], v[6][1], i))
        polyline = np.linspace(1530, 1580, 6065)
        l, = plt.plot(polyline, model(polyline), '--', color=color[i - 2], label='degree ' + str(i))
        handle.append(l)
        print('R_Square value for degree ', i, ': ', r2_score(v[6][1], model(v[6][0])))

# min, max
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    y_values = np.asarray(np.poly1d(np.polyfit(v[6][0], v[6][1], 6))(v[6][0]))
print()
print('Max value: ', v[6][0][y_values.argmax()], v[6][1][y_values.argmax()])
print('Min value: ', v[6][0][y_values.argmin()], v[6][1][y_values.argmin()])
plt.plot([v[6][0][y_values.argmax()], v[6][0][y_values.argmin()]],
         [v[6][1][y_values.argmax()], v[6][1][y_values.argmin()]], 'o')

plt.legend(handles=handle, ncol=2, loc="lower center")
plt.title("REF fitting")
plt.xlabel('Wavelength [nm]')
plt.ylabel('Measured transmission [dB]')

plt.show()
