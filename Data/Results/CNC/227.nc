%
O2001(227)
(MATERIAL SELECT)
M100[SPCO7,2.0]
M352
G90 G28 Z0.
G28 X0. Y0.
G253 A180. B115. I20. J20. T1
M66
G90
G92 X0. Y0.
/GOTO 1
N1G52 X0. Y0.
M98 P9061
(PIECE 1)
M98 H1011
G01G40
G90 G28 Z0.
G52 X0. Y0.
G28 X0. Y0.
M30

N1011( SUB 11 OF 2001 )
G90
G00G40X32.5Y25.
#501=107
M98P9010
G01G41X35.Y25.
G03X35.Y25.I-10.J0.
M121
M199
G00G40X32.5Y90.
#501=107
M98P9010
G01G41X35.Y90.
G03X35.Y90.I-10.J0.
M121
M199
G00G40X124.43Y48.23
#501=107
M98P9010
G01G41X126.43Y48.23
G01X155.29Y48.23
G03X155.29Y68.23I0.J10.
G01X126.43Y68.23
G03X100.29Y94.37I-36.14J-10.
G01X100.29Y104.37
G01X80.29Y104.37
G01X80.29Y94.37
G03X54.15Y68.23I10.J-36.14
G01X25.29Y68.23
G03X25.29Y48.23I0.J-10.
G01X54.15Y48.23
G03X80.29Y22.08I36.14J10.
G01X80.29Y12.08
G01X100.29Y12.08
G01X100.29Y22.08
G03X126.43Y48.23I-10.J36.15
M121
M199
G00G40X162.5Y90.
#501=107
M98P9010
G01G41X165.Y90.
G03X165.Y90.I-10.J0.
M121
M199
G00G40X162.5Y25.
#501=107
M98P9010
G01G41X165.Y25.
G03X165.Y25.I-10.J0.
M121
M199
G00G40X174.09Y5.91
#501=107
M98P9010
G01G42X172.68Y7.32
G03X180.Y25.I-17.68J17.68
G01X180.Y90.
G03X155.Y115.I-25.J0.
G01X25.Y115.
G03X0.Y90.I0.J-25.
G01X0.Y25.
G03X25.Y0.I25.J0.
G01X155.Y0.
G03X172.68Y7.32I0.J25.
M121
M199
G01 G40
M99
%