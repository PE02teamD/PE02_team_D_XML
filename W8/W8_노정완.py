"""
Lot - Batch
Wafer - Wafer
Mask - Maskset
TestSite - TestSite
Name - filename
Script ID - process LMZ or process P1N1 (TestSite의 뒷 단어 적기)
Script version 처음올리면 0.1 두번째 올리면 0.2
Script Owner - D
Operator - ykim
Row - DieRow
Column - DieColumn
R^2 > 0.95 이면, 제대로 근사함, ErrorFlag 0, R^2 < 0.95, ErrorFlag 1
Error description - ErrorFlag 0이면 No Error, 1이면 Ref. spec. Error

Analysis Wavelength - AlignWavelength
Rsq of Ref. spectrum - R^2
Max transmission of Ref. spec. - fitting에서 가져옴
나머지 정보들도 fitting에서 가져오기
----------------------------------------------------------------------------------
위의 내용들을 pandas로 정리해서 csv파일로 extract하기
출력 그래프들을 폴더에 저장할 수 있게
"""