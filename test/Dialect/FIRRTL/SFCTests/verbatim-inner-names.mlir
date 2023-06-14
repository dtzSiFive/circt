// RUN: firtool --verilog %s | FileCheck %s

firrtl.circuit "Foo" {
  extmodule @Bar(
    in extClockIn: !firrtl.clock sym @symExtClockIn,
    out extClockOut: !firrtl.clock sym @symExtClockOut
  )
  module @Foo(
    in %value: !firrtl.uint<42> sym @symValue,
    in %clock: !firrtl.clock sym @symClock,
    in %reset: !firrtl.uint<1> sym @symReset
  ) {
    %instName_clockIn, %instName_clockOut = instance instName sym @instSym @Bar(in extClockIn: !firrtl.clock, out extClockOut: !firrtl.clock)
    %nodeName = node sym @nodeSym %value : !firrtl.uint<42>
    %wireName = wire sym @wireSym : !firrtl.uint<42>
    %regName = reg sym @regSym %clock : !firrtl.clock, !firrtl.uint<42>
    %regResetName = regreset sym @regResetSym %clock, %reset, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<42>, !firrtl.uint<42>

    %invalid_ui42 = invalidvalue : !firrtl.uint<42>
    connect %instName_clockIn, %clock : !firrtl.clock, !firrtl.clock
    connect %wireName, %invalid_ui42 : !firrtl.uint<42>, !firrtl.uint<42>
  }
}

// CHECK: ----- 8< -----
sv.verbatim "----- 8< -----"
sv.verbatim "VERB symExtClockIn = `{{0}}`" {symbols = [#hw.innerNameRef<@Bar::@symExtClockIn>]}
sv.verbatim "VERB symExtClockOut = `{{0}}`" {symbols = [#hw.innerNameRef<@Bar::@symExtClockOut>]}
// CHECK-NEXT: VERB symExtClockIn = `extClockIn`
// CHECK-NEXT: VERB symExtClockOut = `extClockOut`
sv.verbatim "VERB symValue = `{{0}}`" {symbols = [#hw.innerNameRef<@Foo::@symValue>]}
sv.verbatim "VERB symClock = `{{0}}`" {symbols = [#hw.innerNameRef<@Foo::@symClock>]}
sv.verbatim "VERB symReset = `{{0}}`" {symbols = [#hw.innerNameRef<@Foo::@symReset>]}
// CHECK-NEXT: VERB symValue = `value`
// CHECK-NEXT: VERB symClock = `clock`
// CHECK-NEXT: VERB symReset = `reset`
sv.verbatim "VERB instSym = `{{0}}`" {symbols = [#hw.innerNameRef<@Foo::@instSym>]}
sv.verbatim "VERB nodeSym = `{{0}}`" {symbols = [#hw.innerNameRef<@Foo::@nodeSym>]}
sv.verbatim "VERB wireSym = `{{0}}`" {symbols = [#hw.innerNameRef<@Foo::@wireSym>]}
sv.verbatim "VERB regSym = `{{0}}`" {symbols = [#hw.innerNameRef<@Foo::@regSym>]}
sv.verbatim "VERB regResetSym = `{{0}}`" {symbols = [#hw.innerNameRef<@Foo::@regResetSym>]}
// CHECK-NEXT: VERB instSym = `instName`
// CHECK-NEXT: VERB nodeSym = `nodeName`
// CHECK-NEXT: VERB wireSym = `wireName`
// CHECK-NEXT: VERB regSym = `regName`
// CHECK-NEXT: VERB regResetSym = `regResetName`
