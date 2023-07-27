<?xml version="1.0" encoding="utf-8"?>
<CuttingControlProgram MeasurementSystem="Metric" Thickness="0.8">
    <Technology />
    <ApplicationFlat PlanWidth="101" PlanHeight="105" SheetWidth="100" SheetHeight="100" />
    <SubPrograms>
      <SubProgram Id="1" Name="1" PartNumber="1" PartWidth="100" PartHeight="100">
        <ControlCommands>
          <SetOperationMode Id="2" Value="0" />
          <SetSensingSensor Id="3" Value="On" />
          <SetToolCorrection Id="4" Value="Left" />
          <MoveLineTo Id="5" X="0.635" Y="-0.77" />
          <BeginLeadIn Id="6" />
          <SetPiercingMode Id="7" Value="Fine" />
          <ExecPiercing Id="8" />
          <LineTo Id="9" X="-4.77" Y="-5.22" />
          <ArcTo Id="10" X="-4.85" Y="-6.69" SweepAngle="94.881086302" />
          <EndLeadIn Id="11" />
          <BeginCutOut Id="12" Contour="Interior" />
          <ArcTo Id="13" X="-4.85" Y="-6.69" SweepAngle="360" />
          <ArcTo Id="14" X="-4.51" Y="-7.03" SweepAngle="1.147902971" />
          <BeginLeadOut Id="15" />
          <ArcTo Id="16" X="-3.04" Y="-6.95" SweepAngle="94.881086302" />
          <LineTo Id="17" X="0.14" Y="-3.09" />
          <EndLeadOut Id="18" />
          <EndCutOut Id="19" />
          <SetToolCorrection Id="20" Value="None" />
          <MoveLineTo Id="21" X="-5.39" Y="27.79" />
          <BeginCutOut Id="22" Contour="Interior" />
          <LineTo Id="23" X="-37.71" Y="60.11" />
          <EndCutOut Id="24" />
          <SetSensingSensor Id="25" Value="Off" />
          <SetToolCorrection Id="26" Value="Left" />
          <MoveLineTo Id="27" X="-44.71" Y="61.11" />
          <BeginLeadIn Id="28" />
          <SetPiercingMode Id="29" Value="Fine" />
          <ExecPiercing Id="30" />
          <LineTo Id="31" X="-37.71" Y="61.11" />
          <EndLeadIn Id="32" />
          <BeginCutOut Id="33" Contour="Exterior" />
          <LineTo Id="34" X="52.29" Y="61.11" />
          <ArcTo Id="35" X="63.29" Y="50.11" SweepAngle="-90" />
          <LineTo Id="36" X="63.29" Y="-39.89" />
          <ArcTo Id="37" X="62.29" Y="-40.89" SweepAngle="-90" />
          <LineTo Id="38" X="-27.71" Y="-40.89" />
          <ArcTo Id="39" X="-38.71" Y="-29.89" SweepAngle="-90" />
          <LineTo Id="40" X="-38.71" Y="60.11" />
          <BeginLeadOut Id="41" />
          <LineTo Id="42" X="-38.71" Y="65.11" />
          <EndLeadOut Id="43" />
          <EndCutOut Id="44" />
          <SetToolCorrection Id="45" Value="None" />
        </ControlCommands>
      </SubProgram>
    </SubPrograms>
    <ControlCommands>
      <SetClamps Id="46" Value="Off" />
      <CallSubProgram Id="47" X="38" Y="40" Rotation="0" SubProgramName="1" />
    </ControlCommands>
</CuttingControlProgram>
