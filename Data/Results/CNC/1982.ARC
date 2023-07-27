;%_N_000261_MPF
;$PATH=/_N_MPF_DIR
;(Date:06/22/22 Time:13:32:41)
;(Sheet material:Al99,thickness:0.8)
;(Sheet size(LxW):140x140)
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
G00 X70.000 Y88.000
L1100
L1001
G01 X70.000 Y95.000 F=R22
G03 X70.000 Y45.000 I0.000 J-25.000 F=R21
G03 X70.000 Y95.000 I0.000 J25.000
G01 X70.000 Y90.000
L1007
G00 X52.320 Y87.680
L1100
L1001
G01 X20.000 Y120.000 F=R21
L1007
G00 X127.000 Y20.000
L1100
L1001
G01 X120.000 Y20.000 F=R22
G01 X30.000 Y20.000 F=R21
G02 X20.000 Y30.000 I0.000 J10.000
G01 X20.000 Y120.000
G01 X110.000 Y120.000
G02 X120.000 Y110.000 I0.000 J-10.000
G01 X120.000 Y20.000
G01 X120.000 Y15.000
L1007
L1008
M30
