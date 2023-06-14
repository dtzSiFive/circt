// RUN: circt-opt -firrtl-dft %s --verify-diagnostics | FileCheck %s

// The clock gate corresponds to this FIRRTL external module:
// ```firrtl
// extmodule EICG_wrapper :
//   input in : Clock
//   input test_en : UInt<1>
//   input en : UInt<1>
//   (input dft_clk_div_bypass : UInt<1>)?
//   output out : Clock
//   defname = EICG_wrapper
// ```

// Should not error when there is no enable.
// CHECK-LABEL: circuit "NoDuts"
firrtl.circuit "NoDuts" {
  module @NoDuts() {}
}

// Should be fine when there are no clock gates.
firrtl.circuit "NoClockGates" {
  module @A() { }
  module @NoClockGates() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK: instance a @A()
    instance a @A()
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }
}

// Simple example with the enable signal in the top level DUT module.
// CHECK-LABEL: circuit "Simple"
firrtl.circuit "Simple" {
  extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  
  // CHECK: module @A(in %test_en: !firrtl.uint<1>)
  module @A() {
    %in, %test_en, %en, %out = instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK: strictconnect %eicg_test_en, %test_en : !firrtl.uint<1>
  }

  module @Simple() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK: %a_test_en = instance a  @A(in test_en: !firrtl.uint<1>)
    instance a @A()
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
    // CHECK: strictconnect %a_test_en, %test_en
  }
}

// Complex example. The enable signal should flow using output ports up to the
// LCA, and downward to the leafs using input ports.  
// CHECK-LABEL: circuit "TestHarness"
firrtl.circuit "TestHarness" {

  extmodule @EICG_wrapper1(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  extmodule @EICG_wrapper2(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  extmodule @EICG_wrapper3(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper_with_bypass"}

  // CHECK: module @C(in %test_en: !firrtl.uint<1>, in %dft_clk_div_bypass: !firrtl.uint<1>)
  module @C() {
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = instance eicg @EICG_wrapper1(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)

    %eicg3_in, %eicg3_test_en, %eicg3_en, %eicg3_dft_clk_div_bypass, %eicg3_out = instance eicg3 @EICG_wrapper3(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock)

    // CHECK: strictconnect %eicg_test_en, %test_en
    // CHECK: strictconnect %eicg3_test_en, %test_en
    // CHECK: strictconnect %eicg3_dft_clk_div_bypass, %dft_clk_div_bypass
  }

  // CHECK: module @B(in %test_en: !firrtl.uint<1>)
  module @B() {
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = instance eicg @EICG_wrapper1(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)

    // CHECK: strictconnect %eicg_test_en, %test_en
  }

  // CHECK: module @A(in %test_en: !firrtl.uint<1>, in %dft_clk_div_bypass: !firrtl.uint<1>)
  module @A() {
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = instance eicg @EICG_wrapper2(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK: %b_test_en = instance b  @B(in test_en: !firrtl.uint<1>)
    instance b @B()
    // CHECK: %c_test_en, %c_dft_clk_div_bypass = instance c @C(in test_en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>)
    instance c @C()

    // CHECK: strictconnect %eicg_test_en, %test_en : !firrtl.uint<1>
    // CHECK: strictconnect %b_test_en, %test_en : !firrtl.uint<1>
    // CHECK: strictconnect %c_test_en, %test_en : !firrtl.uint<1>
    // CHECK: strictconnect %c_dft_clk_div_bypass, %dft_clk_div_bypass : !firrtl.uint<1>
  }

  // CHECK: module @TestEn0(out %test_en: !firrtl.uint<1>)
  module @TestEn0() {
    // A bundle type should be work for the enable signal using annotations with fieldIDs.
    %test_en = wire {annotations = [{circt.fieldID = 3 : i32, class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    // CHECK: %0 = subindex %test_en_0[0] : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    // CHECK: %1 = subfield %0[qux] : !firrtl.bundle<baz: uint<1>, qux: uint<1>>
    // CHECK: strictconnect %test_en, %1 : !firrtl.uint<1>
    instance b @B()

    // CHECK: strictconnect %b_test_en, %1 : !firrtl.uint<1>
  }

  // CHECK: module @TestEn1(out %test_en: !firrtl.uint<1>, out %dft_clk_div_bypass: !firrtl.uint<1>)
  module @TestEn1() {
    // CHECK: %test_en0_test_en = instance test_en0  @TestEn0(out test_en: !firrtl.uint<1>)
    instance test_en0 @TestEn0()

    instance c @C()
    // CHECK: %c_test_en, %c_dft_clk_div_bypass = instance c @C(in test_en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>)

    // A bundle type should be work for the bypass signal using annotations with fieldIDs.
    %dft_clk_div_bypass = wire {annotations = [{circt.fieldID = 3 : i32, class = "sifive.enterprise.firrtl.DFTClockDividerBypassAnnotation"}]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    // CHECK: %0 = subindex %dft_clk_div_bypass_0[0] : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
    // CHECK: %1 = subfield %0[qux] : !firrtl.bundle<baz: uint<1>, qux: uint<1>>


    // CHECK: strictconnect %test_en, %test_en0_test_en : !firrtl.uint<1>
    // CHECK: strictconnect %c_test_en, %test_en0_test_en : !firrtl.uint<1>
    // CHECK: strictconnect %dft_clk_div_bypass, %1 : !firrtl.uint<1>
    // CHECK: strictconnect %c_dft_clk_div_bypass, %1 : !firrtl.uint<1>
  }

  // CHECK: module @DUT()
  module @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // CHECK: %a_test_en, %a_dft_clk_div_bypass = instance a  @A(in test_en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>)
    instance a @A()
    // CHECK: %b_test_en = instance b  @B(in test_en: !firrtl.uint<1>)
    instance b @B()
    // CHECK: %test_en1_test_en, %test_en1_dft_clk_div_bypass = instance test_en1  @TestEn1(out test_en: !firrtl.uint<1>, out dft_clk_div_bypass: !firrtl.uint<1>)
    instance test_en1 @TestEn1()

    // CHECK: strictconnect %a_test_en, %test_en1_test_en : !firrtl.uint<1>
    // CHECK: strictconnect %b_test_en, %test_en1_test_en : !firrtl.uint<1>
    // CHECK: strictconnect %a_dft_clk_div_bypass, %test_en1_dft_clk_div_bypass : !firrtl.uint<1>
  }

  // CHECK: module @TestHarness()
  module @TestHarness() {
    // CHECK: instance dut  @DUT()
    instance dut @DUT()

    // The clock gate outside of the DUT should not be wired.
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = instance eicg @EICG_wrapper2(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK-NOT: connect
  }
}

// Test enable signal as input to top module, and outside of DUT (issue #3784).
// CHECK-LABEL: circuit "EnableOutsideDUT"
firrtl.circuit "EnableOutsideDUT" {
  extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  // CHECK: module @A(in %test_en: !firrtl.uint<1>)
  module @A() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %in, %test_en, %en, %out = instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK: strictconnect %eicg_test_en, %test_en : !firrtl.uint<1>
  }

  // Regardless of enable signal origin, leave clocks outside DUT alone.
  module @OutsideDUT() {
    %in, %test_en, %en, %out = instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK-NOT: connect
  }

  module @EnableOutsideDUT(in %port_en: !firrtl.uint<1>) attributes {
    portAnnotations = [[{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]]
  } {
    // CHECK: instance o @OutsideDUT
    instance o @OutsideDUT()

    // CHECK: %a_test_en = instance a  @A(in test_en: !firrtl.uint<1>)
    instance a @A()
    // CHECK: strictconnect %a_test_en, %port_en
  }
}

// Test enable signal outside DUT but not top.
// CHECK-LABEL: circuit "EnableOutsideDUT2"
firrtl.circuit "EnableOutsideDUT2" {
  extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  // CHECK: module @A(in %test_en: !firrtl.uint<1>)
  module @A() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %in, %test_en, %en, %out = instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK: strictconnect %eicg_test_en, %test_en : !firrtl.uint<1>
  }

  // Regardless of enable signal origin, leave clocks outside DUT alone.
  // CHECK: @OutsideDUT()
  module @OutsideDUT() {
    %in, %test_en, %en, %out = instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    // CHECK-NOT: strictconnect
  }

  // CHECK-LABEL: @enableSignal
  module @enableSignal() {
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }

  // CHECK-LABEL: @EnableOutsideDUT2
  module @EnableOutsideDUT2() {
    // CHECK: %[[en:.+]] = instance en @enableSignal
    instance en @enableSignal()
    // CHECK: instance o @OutsideDUT
    instance o @OutsideDUT()

    // CHECK: %[[a_en:.+]] = instance a  @A(in test_en: !firrtl.uint<1>)
    instance a @A()
    // CHECK: strictconnect %[[a_en]], %[[en]]
  }
}

// Test ignore bypass with wrong port name or at different direction/type.
firrtl.circuit "BypassWrong" {
  // Warning if port is compatible but name doesn't match, conservatively skipping.
  // expected-warning @below {{compatible port in bypass position has wrong name, skipping}}
  extmodule @EICG_wrapper_wrongName(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in wrong_name: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper_name"}
  extmodule @EICG_wrapper_wrongType(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<3>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper_type"}
  extmodule @EICG_wrapper_wrongDirection(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper_dir"}

  extmodule @EICG_wrapper_right(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper_right"}

  // No bypass wiring.
  // CHECK-LABEL: @Gates(in %test_en: !firrtl.uint<1>)
  module private @Gates() {
    %name_in, %name_test_en, %name_en, %name_wrong_name, %name_out = instance name @EICG_wrapper_wrongName(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in wrong_name: !firrtl.uint<1>, out out: !firrtl.clock)
    %type_in, %type_test_en, %type_en, %type_dft_clk_div_bypass, %type_out = instance type @EICG_wrapper_wrongType(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<3>, out out: !firrtl.clock)

    %dir_in, %dir_test_en, %dir_en, %dir_dft_clk_div_bypass, %dir_out = instance dir @EICG_wrapper_wrongDirection(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock)
  }

  module @BypassWrong() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
    %dft_clk_div_bypass = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTClockDividerBypassAnnotation"}]} : !firrtl.uint<1>

    instance g @Gates()
  }
}

