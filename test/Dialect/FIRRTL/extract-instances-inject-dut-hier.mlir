// RUN: circt-opt --firrtl-inject-dut-hier --firrtl-extract-instances --verify-diagnostics %s | FileCheck %s

// Tests extracted from:
// - test/scala/firrtl/ExtractClockGates.scala

//===----------------------------------------------------------------------===//
// ExtractClockGates Multigrouping
//===----------------------------------------------------------------------===//

// CHECK: circuit "ExtractClockGatesMultigrouping"
firrtl.circuit "ExtractClockGatesMultigrouping" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt", group = "ClockGatesGroup"}, {class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "InjectedSubmodule"}]} {
  extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  // CHECK-LABEL: module private @SomeModule
  module private @SomeModule(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK-NOT: instance gate @EICG_wrapper
    %gate_in, %gate_en, %gate_out = instance gate @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: module private @InjectedSubmodule
  // CHECK: instance inst0 sym [[INST0_SYM:@.+]] @SomeModule
  // CHECK: instance inst1 sym [[INST1_SYM:@.+]] @SomeModule

  // CHECK-LABEL: module private @ClockGatesGroup
  // CHECK: instance gate @EICG_wrapper
  // CHECK: instance gate @EICG_wrapper

  // CHECK-LABEL: module private @DUTModule
  module private @DUTModule(in %clock: !firrtl.clock, in %foo_en: !firrtl.uint<1>, in %bar_en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK-NOT: instance gate @EICG_wrapper
    // CHECK-NOT: instance gate @EICG_wrapper
    %inst0_clock, %inst0_en = instance inst0 @SomeModule(in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    %inst1_clock, %inst1_en = instance inst1 @SomeModule(in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    // CHECK: instance ClockGatesGroup sym [[CLKGRP_SYM:@.+]] @ClockGatesGroup
    // CHECK: instance InjectedSubmodule sym [[INJMOD_SYM:@.+]] @InjectedSubmodule
    connect %inst0_clock, %clock : !firrtl.clock, !firrtl.clock
    connect %inst1_clock, %clock : !firrtl.clock, !firrtl.clock
    connect %inst0_en, %foo_en : !firrtl.uint<1>, !firrtl.uint<1>
    connect %inst1_en, %bar_en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: module @ExtractClockGatesMultigrouping
  module @ExtractClockGatesMultigrouping(in %clock: !firrtl.clock, in %foo_en: !firrtl.uint<1>, in %bar_en: !firrtl.uint<1>) {
    %dut_clock, %dut_foo_en, %dut_bar_en = instance dut  @DUTModule(in clock: !firrtl.clock, in foo_en: !firrtl.uint<1>, in bar_en: !firrtl.uint<1>)
    connect %dut_clock, %clock : !firrtl.clock, !firrtl.clock
    connect %dut_bar_en, %bar_en : !firrtl.uint<1>, !firrtl.uint<1>
    connect %dut_foo_en, %foo_en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: clock_gate_1 -> {{0}}.{{1}}.{{2}}\0A
  // CHECK-SAME{LITERAL}: clock_gate_0 -> {{0}}.{{1}}.{{3}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"ClockGates.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@DUTModule::[[INJMOD_SYM]]>
  // CHECK-SAME: #hw.innerNameRef<@InjectedSubmodule::[[INST0_SYM]]>
  // CHECK-SAME: #hw.innerNameRef<@InjectedSubmodule::[[INST1_SYM]]>
  // CHECK-SAME: ]
}
