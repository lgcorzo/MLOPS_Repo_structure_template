<?xml version="1.0" encoding="utf-8"?>
<CuttingControlProgram MeasurementSystem="Metric" Thickness="0.8">
    <Technology />
    <ApplicationFlat PlanWidth="226.261848291" PlanHeight="87.9900125" SheetWidth="225.261584999" SheetHeight="86.989483" />
    <SubPrograms>
      <SubProgram Id="1" Name="1" PartNumber="1" PartWidth="225.261584999" PartHeight="86.989483">
        <ControlCommands>
          <SetOperationMode Id="2" Value="0" />
          <SetSensingSensor Id="3" Value="On" />
          <SetToolCorrection Id="4" Value="Right" />
          <MoveLineTo Id="5" X="0.84" Y="-0.54" />
          <BeginLeadIn Id="6" />
          <SetPiercingMode Id="7" Value="Fine" />
          <ExecPiercing Id="8" />
          <LineTo Id="9" X="4.63" Y="5.345" />
          <EndLeadIn Id="10" />
          <BeginCutOut Id="11" Contour="Exterior" />
          <ArcTo Id="12" X="15.455" Y="-3.405" SweepAngle="-192.397086712" />
          <ArcTo Id="13" X="15.58" Y="-4.925" SweepAngle="99.562198538" />
          <ArcTo Id="14" X="84.62" Y="-27.425" SweepAngle="34.33650644" />
          <ArcTo Id="15" X="85.635" Y="-26.305" SweepAngle="98.311968621" />
          <ArcTo Id="16" X="115.43" Y="-22.829999999" SweepAngle="-180.983033261" />
          <ArcTo Id="17" X="116.67" Y="-23.684999999" SweepAngle="97.582269347" />
          <ArcTo Id="18" X="195.794999999" Y="39.1" SweepAngle="48.485070792" />
          <ArcTo Id="19" X="194.905" Y="40.56" SweepAngle="117.332359491" />
          <LineTo Id="20" X="179.514999999" Y="40.56" />
          <LineTo Id="21" X="179.514999999" Y="47.829999999" />
          <ArcTo Id="22" X="177.81" Y="48.534999999" SweepAngle="135" />
          <LineTo Id="23" X="169.175" Y="39.905" />
          <ArcTo Id="24" X="155.525" Y="45.14" SweepAngle="-132.016294764" />
          <LineTo Id="25" X="155.245" Y="50.61" />
          <ArcTo Id="26" X="154.245" Y="51.56" SweepAngle="87.137594773" />
          <LineTo Id="27" X="142.985" Y="51.56" />
          <ArcTo Id="28" X="141.995" Y="50.68" SweepAngle="83.08877288" />
          <LineTo Id="29" X="140.629999999" Y="39.56" />
          <LineTo Id="30" X="132.4" Y="39.56" />
          <LineTo Id="31" X="131.039999999" Y="50.68" />
          <ArcTo Id="32" X="130.044999999" Y="51.56" SweepAngle="83.123169262" />
          <LineTo Id="33" X="119.515" Y="51.56" />
          <ArcTo Id="34" X="118.799999999" Y="51.259999999" SweepAngle="45.607353005" />
          <LineTo Id="35" X="101.254999999" Y="33.28" />
          <ArcTo Id="36" X="56.115" Y="39.24" SweepAngle="-41.674324816" />
          <LineTo Id="37" X="14.99" Y="61.439999999" />
          <ArcTo Id="38" X="14.515" Y="61.56" SweepAngle="28.358963446" />
          <LineTo Id="39" X="2.515" Y="61.56" />
          <ArcTo Id="40" X="1.515" Y="60.56" SweepAngle="90" />
          <LineTo Id="41" X="1.515" Y="57.56" />
          <LineTo Id="42" X="-18.485" Y="57.56" />
          <LineTo Id="43" X="-18.485" Y="60.56" />
          <ArcTo Id="44" X="-19.485" Y="61.56" SweepAngle="90" />
          <LineTo Id="45" X="-30.355" Y="61.56" />
          <ArcTo Id="46" X="-31.315" Y="60.27" SweepAngle="106.808691625" />
          <ArcTo Id="47" X="3.11" Y="5.155" SweepAngle="30.633355735" />
          <BeginLeadOut Id="48" />
          <LineTo Id="49" X="6.785" Y="1.765" />
          <EndLeadOut Id="50" />
          <EndCutOut Id="51" />
          <SetToolCorrection Id="52" Value="None" />
        </ControlCommands>
      </SubProgram>
    </SubPrograms>
    <ControlCommands>
      <SetClamps Id="53" Value="Off" />
      <CallSubProgram Id="54" X="30" Y="26" Rotation="0" SubProgramName="1" />
    </ControlCommands>
</CuttingControlProgram>
