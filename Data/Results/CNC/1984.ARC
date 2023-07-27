;%_N_000263_MPF
;$PATH=/_N_MPF_DIR
;(Date:06/22/22 Time:13:32:41)
;(Sheet material:Al99,thickness:0.8)
;(Sheet size(LxW):266.178x122.063)
;piercing parameter
R1=2000.0      ;(2000.0)piering power unit:kw
R2=30.0      ;(30.0)piering pressure unit:bar
R3=3      ;(3)piering gas type: 1=oxy; 2=N2;3=air
R4=3.0      ;(3.0)piering time unit:S
R5=2      ;(2)piering mode: 1=PE;2=SP;3=HP 4=CW
R6=100      ;(100)piering frequcecy unit:Hz
R7=34      ;(34)piering duty unit:%
R8=3.0      ;(3.0)piering height unit:mm
R9=2.0      ;(2.0)piering focus postion unit:mm
;
;Layer1   cutting parameter
R11=12300.0      ;(12300.0)cutting  power unit:kw
R12=40.0      ;(40.0)cutting  pressure unit:bar
R13=0      ;(0)cutting gas type: 1=oxy; 2=N2;3=air
R14=4.0      ;(4.0)after piering time unit:S
R15=0      ;(0)cutting mode: 1=PE;2=SP;3=HP 4=CW
R16=10      ;(10)cutting frequcecy unit:Hz
R17=80      ;(80)cutting duty unit:%
R18=2.0      ;(2.0)cutting height unit:mm
R19=2      ;(2)line type
R20=2.0      ;(2.0)cutting focus postion unit:mm
R21=6200      ;(6200)cutting feed speed unit:mm/min
R22=100      ;(100)lead_in feed speed unit:mm/min
R23=5.0      ;(5.0)z axis lift height unit:mm
;
FFWON
G64
SOFT
M1=0 M2=1 M3=1
G12(2)
G04F1
L2
G71
G90
G00 X253.160 Y94.190
L1100
L1001
G01 X246.180 Y93.700 F=R22
G01 X229.830 Y92.550 F=R21
G01 X229.250 Y100.800
G01 X221.230 Y91.590
G02 X205.510 Y96.400 I-6.780 J5.900
G01 X204.850 Y101.830
G01 X193.620 Y101.050
G01 X192.990 Y88.980
G01 X183.010 Y88.280
G01 X180.710 Y100.150
G01 X170.200 Y99.410
G01 X153.770 Y80.030
G02 X107.330 Y82.790 I-19.620 J61.970
G01 X64.760 Y102.060
G01 X52.790 Y101.230
G01 X53.070 Y97.240
G01 X31.120 Y95.700
G01 X30.850 Y99.690
G01 X20.000 Y98.930
G03 X57.870 Y46.780 I114.150 J43.070
G02 X70.910 Y37.660 I7.020 J-3.850
G03 X140.780 Y20.180 I63.240 J104.340
G02 X172.230 Y26.090 I15.700 J3.090
G03 X246.180 Y93.700 I-38.080 J115.910
G01 X248.160 Y98.290
L1007
L1008
M30
