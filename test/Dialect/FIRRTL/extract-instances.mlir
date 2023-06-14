// RUN: circt-opt --firrtl-extract-instances %s | FileCheck %s

// Tests extracted from:
// - test/scala/firrtl/ExtractBlackBoxes.scala
// - test/scala/firrtl/ExtractClockGates.scala
// - test/scala/firrtl/ExtractSeqMems.scala

//===----------------------------------------------------------------------===//
// ExtractBlackBoxes Simple
//===----------------------------------------------------------------------===//

// CHECK: circuit "ExtractBlackBoxesSimple"
firrtl.circuit "ExtractBlackBoxesSimple" attributes {annotations = [{class = "firrtl.transforms.BlackBoxTargetDirAnno", targetDir = "BlackBoxes"}]} {
  // CHECK-LABEL: extmodule private @MyBlackBox
  extmodule private @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", filename = "BlackBoxes.txt", prefix = "bb"}], defname = "MyBlackBox"}
  // CHECK-LABEL: module private @BBWrapper
  // CHECK-SAME: out %bb_0_in: !firrtl.uint<8>
  // CHECK-SAME: in %bb_0_out: !firrtl.uint<8>
  module private @BBWrapper(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK-NOT: instance bb @MyBlackBox
    %bb_in, %bb_out = instance bb @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    %invalid_ui8 = invalidvalue : !firrtl.uint<8>
    strictconnect %bb_in, %invalid_ui8 : !firrtl.uint<8>
    // CHECK: connect %out, %bb_0_out
    // CHECK: connect %bb_0_in, %in
    connect %out, %bb_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %bb_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: module private @DUTModule
  // CHECK-SAME: out %bb_0_in: !firrtl.uint<8>
  // CHECK-SAME: in %bb_0_out: !firrtl.uint<8>
  module private @DUTModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK-NOT: instance bb @MyBlackBox
    // CHECK: %mod_in, %mod_out, %mod_bb_0_in, %mod_bb_0_out = instance mod sym [[WRAPPER_SYM:@.+]] @BBWrapper
    // CHECK-NEXT: strictconnect %bb_0_in, %mod_bb_0_in
    // CHECK-NEXT: strictconnect %mod_bb_0_out, %bb_0_out
    %mod_in, %mod_out = instance mod @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    connect %out, %mod_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %mod_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: module @ExtractBlackBoxesSimple
  module @ExtractBlackBoxesSimple(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK: %dut_in, %dut_out, %dut_bb_0_in, %dut_bb_0_out = instance dut sym {{@.+}} @DUTModule
    // CHECK-NEXT: %bb_in, %bb_out = instance bb @MyBlackBox
    // CHECK-NEXT: strictconnect %bb_in, %dut_bb_0_in
    // CHECK-NEXT: strictconnect %dut_bb_0_out, %bb_out
    %dut_in, %dut_out = instance dut @DUTModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    connect %out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %dut_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: bb_0 -> {{0}}.{{1}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"BlackBoxes.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@DUTModule::[[WRAPPER_SYM]]>
  // CHECK-SAME: ]
}

//===----------------------------------------------------------------------===//
// ExtractBlackBoxes Simple (modified)
// ExtractBlackBoxes RenameTargets (modified)
//===----------------------------------------------------------------------===//

// CHECK: circuit "ExtractBlackBoxesSimple2"
firrtl.circuit "ExtractBlackBoxesSimple2" attributes {annotations = [{class = "firrtl.transforms.BlackBoxTargetDirAnno", targetDir = "BlackBoxes"}]} {
  // Old style NLAs
  hw.hierpath private @nla_old1 [@DUTModule::@mod, @BBWrapper::@bb]
  hw.hierpath private @nla_old2 [@ExtractBlackBoxesSimple2::@dut, @DUTModule::@mod, @BBWrapper::@bb]
  // New style NLAs on extracted instance
  hw.hierpath private @nla_on1 [@DUTModule::@mod, @BBWrapper]
  hw.hierpath private @nla_on2 [@ExtractBlackBoxesSimple2::@dut, @DUTModule::@mod, @BBWrapper]
  // New style NLAs through extracted instance
  hw.hierpath private @nla_thru1 [@BBWrapper::@bb, @MyBlackBox]
  hw.hierpath private @nla_thru2 [@DUTModule::@mod, @BBWrapper::@bb, @MyBlackBox]
  hw.hierpath private @nla_thru3 [@ExtractBlackBoxesSimple2::@dut, @DUTModule::@mod, @BBWrapper::@bb, @MyBlackBox]
  // CHECK: hw.hierpath private [[THRU1:@nla_thru1]] [@ExtractBlackBoxesSimple2::@bb, @MyBlackBox]
  // CHECK: hw.hierpath private [[THRU2:@nla_thru2]] [@ExtractBlackBoxesSimple2::@bb, @MyBlackBox]
  // CHECK: hw.hierpath private [[THRU3:@nla_thru3]] [@ExtractBlackBoxesSimple2::@bb, @MyBlackBox]

  // Annotation on the extmodule itself
  // CHECK-LABEL: extmodule private @MyBlackBox
  extmodule private @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {annotations = [
      {class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", filename = "BlackBoxes.txt", prefix = "prefix"},
      {circt.nonlocal = @nla_thru1, class = "Thru1"},
      {circt.nonlocal = @nla_thru2, class = "Thru2"},
      {circt.nonlocal = @nla_thru3, class = "Thru3"}
    ], defname = "MyBlackBox"}
  // Annotation will be on the instance
  // CHECK-LABEL: extmodule private @MyBlackBox2
  extmodule private @MyBlackBox2(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {defname = "MyBlackBox"}

  // CHECK-LABEL: module private @BBWrapper
  // CHECK-SAME: out %prefix_0_in: !firrtl.uint<8>
  // CHECK-SAME: in %prefix_0_out: !firrtl.uint<8>
  // CHECK-SAME: out %prefix_1_in: !firrtl.uint<8>
  // CHECK-SAME: in %prefix_1_out: !firrtl.uint<8>
  module private @BBWrapper(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK-NOT: instance bb @MyBlackBox
    // CHECK-NOT: instance bb2 @MyBlackBox2
    %bb_in, %bb_out = instance bb sym @bb {annotations = [
        {circt.nonlocal = @nla_old1, class = "Old1"},
        {circt.nonlocal = @nla_old2, class = "Old2"},
        {circt.nonlocal = @nla_on1, class = "On1"},
        {circt.nonlocal = @nla_on2, class = "On2"}
      ]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    %bb2_in, %bb2_out = instance bb2 {annotations = [{class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", filename = "BlackBoxes.txt", prefix = "prefix"}]} @MyBlackBox2(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    // CHECK: connect %out, %prefix_0_out
    // CHECK: connect %prefix_0_in, %prefix_1_out
    // CHECK: connect %prefix_1_in, %in
    connect %out, %bb2_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %bb2_in, %bb_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %bb_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: module private @DUTModule
  // CHECK-SAME: out %prefix_0_in: !firrtl.uint<8>
  // CHECK-SAME: in %prefix_0_out: !firrtl.uint<8>
  // CHECK-SAME: out %prefix_1_in: !firrtl.uint<8>
  // CHECK-SAME: in %prefix_1_out: !firrtl.uint<8>
  module private @DUTModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK-NOT: instance bb @MyBlackBox
    // CHECK-NOT: instance bb2 @MyBlackBox2
    // CHECK: %mod_in, %mod_out, %mod_prefix_0_in, %mod_prefix_0_out, %mod_prefix_1_in, %mod_prefix_1_out = instance mod
    // CHECK-SAME: sym [[WRAPPER_SYM:@.+]] @BBWrapper
    // CHECK-NOT: annotations =
    // CHECK-NEXT: strictconnect %prefix_1_in, %mod_prefix_1_in
    // CHECK-NEXT: strictconnect %mod_prefix_1_out, %prefix_1_out
    // CHECK-NEXT: strictconnect %prefix_0_in, %mod_prefix_0_in
    // CHECK-NEXT: strictconnect %mod_prefix_0_out, %prefix_0_out
    %mod_in, %mod_out = instance mod sym @mod @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    connect %out, %mod_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %mod_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: module @ExtractBlackBoxesSimple2
  module @ExtractBlackBoxesSimple2(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK: %dut_in, %dut_out, %dut_prefix_0_in, %dut_prefix_0_out, %dut_prefix_1_in, %dut_prefix_1_out = instance dut
    // CHECK-NOT: annotations =
    // CHECK-SAME: sym {{@.+}} @DUTModule
    // CHECK-NEXT: %bb_in, %bb_out = instance bb sym [[BB_SYM:@.+]] {annotations = [{class = "Old1"}, {class = "On1"}, {class = "Old2"}, {class = "On2"}]} @MyBlackBox
    // CHECK-NEXT: strictconnect %bb_in, %dut_prefix_1_in
    // CHECK-NEXT: strictconnect %dut_prefix_1_out, %bb_out
    // CHECK-NEXT: %bb2_in, %bb2_out = instance bb2 @MyBlackBox2
    // CHECK-NEXT: strictconnect %bb2_in, %dut_prefix_0_in
    // CHECK-NEXT: strictconnect %dut_prefix_0_out, %bb2_out
    %dut_in, %dut_out = instance dut sym @dut @DUTModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    connect %out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %dut_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: prefix_0 -> {{0}}.{{1}}\0A
  // CHECK-SAME{LITERAL}: prefix_1 -> {{0}}.{{1}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"BlackBoxes.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: @DUTModule::[[WRAPPER_SYM]]
  // CHECK-SAME: ]
}

//===----------------------------------------------------------------------===//
// ExtractBlackBoxes IntoDUTSubmodule
//===----------------------------------------------------------------------===//

// CHECK: circuit "ExtractBlackBoxesIntoDUTSubmodule"
firrtl.circuit "ExtractBlackBoxesIntoDUTSubmodule"  {
  // CHECK-LABEL: hw.hierpath private @nla_new_0 [
  // CHECK-SAME:    @ExtractBlackBoxesIntoDUTSubmodule::@tb
  // CHECK-SAME:    @TestHarness::@dut
  // CHECK-SAME:    @DUTModule::@BlackBoxes
  // CHECK-SAME:    @BlackBoxes
  // CHECK-SAME:  ]
  // CHECK-LABEL: hw.hierpath private @nla_new_1 [
  // CHECK-SAME:    @ExtractBlackBoxesIntoDUTSubmodule::@tb
  // CHECK-SAME:    @TestHarness::@dut
  // CHECK-SAME:    @DUTModule::@BlackBoxes
  // CHECK-SAME:    @BlackBoxes
  // CHECK-SAME:  ]
  hw.hierpath private @nla_new [
    @ExtractBlackBoxesIntoDUTSubmodule::@tb,
    @TestHarness::@dut,
    @DUTModule::@mod,
    @BBWrapper
  ]
  // CHECK-LABEL: hw.hierpath private @nla_old1 [
  // CHECK-SAME:    @ExtractBlackBoxesIntoDUTSubmodule::@tb
  // CHECK-SAME:    @TestHarness::@dut
  // CHECK-SAME:    @DUTModule::@BlackBoxes
  // CHECK-SAME:    @BlackBoxes::@bb1
  // CHECK-SAME:  ]
  hw.hierpath private @nla_old1 [
    @ExtractBlackBoxesIntoDUTSubmodule::@tb,
    @TestHarness::@dut,
    @DUTModule::@mod,
    @BBWrapper::@bb1
  ]
  // CHECK-LABEL: hw.hierpath private @nla_old2 [
  // CHECK-SAME:    @ExtractBlackBoxesIntoDUTSubmodule::@tb
  // CHECK-SAME:    @TestHarness::@dut
  // CHECK-SAME:    @DUTModule::@BlackBoxes
  // CHECK-SAME:    @BlackBoxes::@bb2
  // CHECK-SAME:  ]
  hw.hierpath private @nla_old2 [
    @ExtractBlackBoxesIntoDUTSubmodule::@tb,
    @TestHarness::@dut,
    @DUTModule::@mod,
    @BBWrapper::@bb2
  ]
  extmodule private @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", dest = "BlackBoxes", filename = "BlackBoxes.txt", prefix = "bb"}], defname = "MyBlackBox"}
  module private @BBWrapper(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %bb1_in, %bb1_out = instance bb1 sym @bb1 {annotations = [{circt.nonlocal = @nla_old1, class = "Dummy1"}, {circt.nonlocal = @nla_new, class = "Dummy3"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    %bb2_in, %bb2_out = instance bb2 sym @bb2 {annotations = [{circt.nonlocal = @nla_old2, class = "Dummy2"}, {circt.nonlocal = @nla_new, class = "Dummy4"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    connect %out, %bb2_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %bb2_in, %bb1_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %bb1_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: module private @BlackBoxes(
  // CHECK-SAME:    in %bb_0_in: !firrtl.uint<8>
  // CHECK-SAME:    out %bb_0_out: !firrtl.uint<8>
  // CHECK-SAME:    in %bb_1_in: !firrtl.uint<8>
  // CHECK-SAME:    out %bb_1_out: !firrtl.uint<8>
  // CHECK-SAME:  ) {
  // CHECK-NEXT:    %bb2_in, %bb2_out = instance bb2 sym [[BB2_SYM:@.+]] {annotations = [{circt.nonlocal = @nla_new_0, class = "Dummy4"}, {circt.nonlocal = @nla_old2, class = "Dummy2"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
  // CHECK-NEXT:    strictconnect %bb2_in, %bb_0_in : !firrtl.uint<8>
  // CHECK-NEXT:    strictconnect %bb_0_out, %bb2_out : !firrtl.uint<8>
  // CHECK-NEXT:    %bb1_in, %bb1_out = instance bb1 sym [[BB1_SYM:@.+]] {annotations = [{circt.nonlocal = @nla_new_1, class = "Dummy3"}, {circt.nonlocal = @nla_old1, class = "Dummy1"}]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
  // CHECK-NEXT:    strictconnect %bb1_in, %bb_1_in : !firrtl.uint<8>
  // CHECK-NEXT:    strictconnect %bb_1_out, %bb1_out : !firrtl.uint<8>
  // CHECK-NEXT:  }
  // CHECK-LABEL: module private @DUTModule
  module private @DUTModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK: %BlackBoxes_bb_0_in, %BlackBoxes_bb_0_out, %BlackBoxes_bb_1_in, %BlackBoxes_bb_1_out = instance BlackBoxes sym @BlackBoxes
    // CHECK-SAME: @BlackBoxes
    // CHECK-NEXT: %mod_in, %mod_out, %mod_bb_0_in, %mod_bb_0_out, %mod_bb_1_in, %mod_bb_1_out = instance mod
    // CHECK-NOT: annotations =
    // CHECK-SAME: sym [[WRAPPER_SYM:@.+]] @BBWrapper
    // CHECK-NEXT: strictconnect %BlackBoxes_bb_1_in, %mod_bb_1_in
    // CHECK-NEXT: strictconnect %mod_bb_1_out, %BlackBoxes_bb_1_out
    // CHECK-NEXT: strictconnect %BlackBoxes_bb_0_in, %mod_bb_0_in
    // CHECK-NEXT: strictconnect %mod_bb_0_out, %BlackBoxes_bb_0_out
    %mod_in, %mod_out = instance mod sym @mod @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    connect %out, %mod_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %mod_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  module @TestHarness(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %dut_in, %dut_out = instance dut sym @dut @DUTModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    connect %out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %dut_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  module @ExtractBlackBoxesIntoDUTSubmodule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %tb_in, %tb_out = instance tb sym @tb @TestHarness(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    connect %out, %tb_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %tb_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: bb_0 -> {{0}}.{{1}}\0A
  // CHECK-SAME{LITERAL}: bb_1 -> {{0}}.{{1}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"BlackBoxes.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: @DUTModule::[[WRAPPER_SYM]]
  // CHECK-SAME: ]
}

//===----------------------------------------------------------------------===//
// ExtractClockGates Simple
//===----------------------------------------------------------------------===//

// CHECK: circuit "ExtractClockGatesSimple"
firrtl.circuit "ExtractClockGatesSimple" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"}]} {
  extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  module private @DUTModule(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %gate_in, %gate_en, %gate_out = instance gate @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: module @ExtractClockGatesSimple
  module @ExtractClockGatesSimple(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK: instance gate @EICG_wrapper
    %dut_clock, %dut_en = instance dut @DUTModule(in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    connect %dut_clock, %clock : !firrtl.clock, !firrtl.clock
    connect %dut_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: clock_gate_0 -> {{0}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"ClockGates.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: ]
}

//===----------------------------------------------------------------------===//
// ExtractClockGates TestHarnessOnly
//===----------------------------------------------------------------------===//

// CHECK: circuit "ExtractClockGatesTestHarnessOnly"
firrtl.circuit "ExtractClockGatesTestHarnessOnly" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"}]} {
  extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  module private @DUTModule(in %clock: !firrtl.clock, in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>, in %en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %0 = add %in, %en : (!firrtl.uint<8>, !firrtl.uint<1>) -> !firrtl.uint<9>
    %_io_out_T = node %0 : !firrtl.uint<9>
    %1 = tail %_io_out_T, 1 : (!firrtl.uint<9>) -> !firrtl.uint<8>
    %_io_out_T_1 = node %1 : !firrtl.uint<8>
    connect %out, %_io_out_T_1 : !firrtl.uint<8>, !firrtl.uint<8>
  }
  module @ExtractClockGatesTestHarnessOnly(in %clock: !firrtl.clock, in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>, in %en: !firrtl.uint<1>) {
    %gate_in, %gate_en, %gate_out = instance gate @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    %dut_clock, %dut_in, %dut_out, %dut_en = instance dut @DUTModule(in clock: !firrtl.clock, in in: !firrtl.uint<8>, out out: !firrtl.uint<8>, in en: !firrtl.uint<1>)
    connect %dut_clock, %gate_out : !firrtl.clock, !firrtl.clock
    connect %dut_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    connect %out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %dut_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-NOT: sv.verbatim "clock_gate
}

//===----------------------------------------------------------------------===//
// ExtractClockGates Mixed
//===----------------------------------------------------------------------===//

// Mixed ClockGate extraction should only extract clock gates in the DUT
// CHECK: circuit "ExtractClockGatesMixed"
firrtl.circuit "ExtractClockGatesMixed" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"}]} {
  extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  // CHECK-LABEL: module private @Child
  module private @Child(in %clock: !firrtl.clock, in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>, in %en: !firrtl.uint<1>) {
    // CHECK-NOT: instance gate sym @ckg1 @EICG_wrapper
    %gate_in, %gate_en, %gate_out = instance gate sym @ckg1 @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: module private @DUTModule
  module private @DUTModule(in %clock: !firrtl.clock, in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>, in %en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %inst_clock, %inst_in, %inst_out, %inst_en = instance inst sym @inst @Child(in clock: !firrtl.clock, in in: !firrtl.uint<8>, out out: !firrtl.uint<8>, in en: !firrtl.uint<1>)
    connect %inst_clock, %clock : !firrtl.clock, !firrtl.clock
    connect %inst_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
    connect %inst_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NOT: instance gate sym @ckg2 @EICG_wrapper
    %gate_in, %gate_en, %gate_out = instance gate sym @ckg2 @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK-LABEL: module @ExtractClockGatesMixed
  module @ExtractClockGatesMixed(in %clock: !firrtl.clock, in %intf_in: !firrtl.uint<8>, out %intf_out: !firrtl.uint<8>, in %intf_en: !firrtl.uint<1>, in %en: !firrtl.uint<1>) {
    // CHECK: instance gate sym @ckg3 @EICG_wrapper
    %gate_in, %gate_en, %gate_out = instance gate sym @ckg3 @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
    %dut_clock, %dut_in, %dut_out, %dut_en = instance dut sym @dut @DUTModule(in clock: !firrtl.clock, in in: !firrtl.uint<8>, out out: !firrtl.uint<8>, in en: !firrtl.uint<1>)
    // CHECK: instance gate sym @ckg2 @EICG_wrapper
    // CHECK: instance gate sym @ckg1 @EICG_wrapper
    connect %dut_clock, %gate_out : !firrtl.clock, !firrtl.clock
    connect %dut_en, %intf_en : !firrtl.uint<1>, !firrtl.uint<1>
    connect %intf_out, %dut_out : !firrtl.uint<8>, !firrtl.uint<8>
    connect %dut_in, %intf_in : !firrtl.uint<8>, !firrtl.uint<8>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: clock_gate_0 -> {{0}}.{{1}}\0A
  // CHECK-SAME{LITERAL}: clock_gate_1 -> {{0}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"ClockGates.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: @DUTModule::@inst
  // CHECK-SAME: ]
}

//===----------------------------------------------------------------------===//
// ExtractClockGates Composed
//===----------------------------------------------------------------------===//

// CHECK: circuit "ExtractClockGatesComposed"
firrtl.circuit "ExtractClockGatesComposed" attributes {annotations = [
  {class = "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation", filename = "ClockGates.txt"},
  {class = "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation", filename = "SeqMems.txt"}
]} {
  extmodule private @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  memmodule @mem_ext() attributes {dataWidth = 8 : ui32, depth = 8 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  module private @DUTModule(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    instance mem_ext @mem_ext()
    %gate_in, %gate_en, %gate_out = instance gate @EICG_wrapper(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    connect %gate_in, %clock : !firrtl.clock, !firrtl.clock
    connect %gate_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: module @ExtractClockGatesComposed
  module @ExtractClockGatesComposed(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK: instance gate @EICG_wrapper
    // CHECK: instance mem_ext @mem_ext
    %dut_clock, %dut_en = instance dut @DUTModule(in clock: !firrtl.clock, in en: !firrtl.uint<1>)
    connect %dut_clock, %clock : !firrtl.clock, !firrtl.clock
    connect %dut_en, %en : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: sv.verbatim ""
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: clock_gate_0 -> {{0}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"ClockGates.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: ]
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: mem_wiring_0 -> {{0}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"SeqMems.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: ]
}

//===----------------------------------------------------------------------===//
// ExtractSeqMems Simple2
//===----------------------------------------------------------------------===//

// CHECK: circuit "ExtractSeqMemsSimple2"
firrtl.circuit "ExtractSeqMemsSimple2" attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation", filename = "SeqMems.txt"}]} {
  memmodule @mem_ext() attributes {dataWidth = 8 : ui32, depth = 8 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  // CHECK-LABEL: module @mem
  module @mem() {
    // CHECK-NOT: instance mem_ext @mem_ext
    instance mem_ext @mem_ext()
  }
  // CHECK-LABEL: module private @DUTModule
  module private @DUTModule() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK-NEXT: instance mem sym [[MEM_SYM:@.+]] @mem
    instance mem @mem()
  }
  // CHECK-LABEL: module @ExtractSeqMemsSimple2
  module @ExtractSeqMemsSimple2() {
    instance dut @DUTModule()
    // CHECK-NEXT: instance dut sym [[DUT_SYM:@.+]] @DUTModule
    // CHECK-NEXT: instance mem_ext @mem_ext
  }
  // CHECK: sv.verbatim ""
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: mem_wiring_0 -> {{0}}.{{1}}\0A
  // CHECK-SAME: output_file = #hw.output_file<"SeqMems.txt", excludeFromFileList>
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: @DUTModule::[[MEM_SYM]]
  // CHECK-SAME: ]
}

//===----------------------------------------------------------------------===//
// ExtractSeqMems NoExtraction
//===----------------------------------------------------------------------===//

// CHECK: circuit "ExtractSeqMemsNoExtraction"
firrtl.circuit "ExtractSeqMemsNoExtraction"  attributes {annotations = [{class = "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation", filename = "SeqMems.txt"}]} {
  module @ExtractSeqMemsNoExtraction() {}
  // CHECK: sv.verbatim ""
  // CHECK-SAME: output_file = #hw.output_file<"SeqMems.txt", excludeFromFileList>
}

//===----------------------------------------------------------------------===//
// Conflicting Instance Symbols
// https://github.com/llvm/circt/issues/3089
//===----------------------------------------------------------------------===//

// CHECK: circuit "InstSymConflict"
firrtl.circuit "InstSymConflict" {
  // CHECK-NOT: hw.hierpath private @nla_1
  // CHECK-NOT: hw.hierpath private @nla_2
  hw.hierpath private @nla_1 [
    @InstSymConflict::@dut,
    @DUTModule::@mod1,
    @BBWrapper::@bb
  ]
  hw.hierpath private @nla_2 [
    @InstSymConflict::@dut,
    @DUTModule::@mod2,
    @BBWrapper::@bb
  ]
  extmodule private @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>) attributes {defname = "MyBlackBox"}
  module private @BBWrapper(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %bb_in, %bb_out = instance bb sym @bb {annotations = [
        {class = "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation", filename = "BlackBoxes.txt", prefix = "bb"},
        {circt.nonlocal = @nla_1, class = "DummyA"},
        {circt.nonlocal = @nla_2, class = "DummyB"}
      ]} @MyBlackBox(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    strictconnect %bb_in, %in : !firrtl.uint<8>
    strictconnect %out, %bb_out : !firrtl.uint<8>
  }
  module private @DUTModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %mod1_in, %mod1_out = instance mod1 sym @mod1 @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    %mod2_in, %mod2_out = instance mod2 sym @mod2 @BBWrapper(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    strictconnect %mod1_in, %in : !firrtl.uint<8>
    strictconnect %mod2_in, %mod1_out : !firrtl.uint<8>
    strictconnect %out, %mod2_out : !firrtl.uint<8>
  }
  // CHECK-LABEL: module @InstSymConflict
  module @InstSymConflict(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK-NEXT: instance dut sym @dut @DUTModule
    // CHECK: instance bb sym @bb {annotations = [{class = "DummyB"}]} @MyBlackBox
    // CHECK: instance bb sym @bb_0 {annotations = [{class = "DummyA"}]} @MyBlackBox
    %dut_in, %dut_out = instance dut sym @dut @DUTModule(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    strictconnect %dut_in, %in : !firrtl.uint<8>
    strictconnect %out, %dut_out : !firrtl.uint<8>
  }
  // CHECK: sv.verbatim "
  // CHECK-SAME{LITERAL}: bb_1 -> {{0}}.{{1}}\0A
  // CHECK-SAME{LITERAL}: bb_0 -> {{0}}.{{2}}\0A
  // CHECK-SAME: symbols = [
  // CHECK-SAME: @DUTModule
  // CHECK-SAME: #hw.innerNameRef<@DUTModule::@mod1>
  // CHECK-SAME: #hw.innerNameRef<@DUTModule::@mod2>
  // CHECK-SAME: ]
}
