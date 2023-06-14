// RUN: circt-opt -firrtl-dft -verify-diagnostics -split-input-file %s

// expected-error @+1 {{no DUT module found}}
firrtl.circuit "NoDuts" {
  module @NoDuts() {
    %test_en1 = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]}: !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "TwoDuts" {
  module @TwoDuts() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {}
  // expected-error @+2 {{more than one module marked DUT}}
  // expected-note  @-2 {{first module here}}
  module @TwoDuts0() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {}
}

// -----

firrtl.circuit "TwoSignals" {
  module @TwoSignals(in %test_en0: !firrtl.uint<1>) attributes {portAnnotations = [[{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]]} {
    // expected-error @+2 {{more than one thing marked as sifive.enterprise.firrtl.DFTTestModeEnableAnnotation}}
    // expected-note  @-2 {{first thing defined here}}
    %test_en1 = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]}: !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "TwoEnables" {

  extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  module @TestEn() {
    // expected-error @+1 {{multiple instantiations of the DFT enable signal}}
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }
  
  module @TwoEnables() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // expected-note @+1 {{second instance here}}
    instance test_en0 @TestEn()
    // expected-note @+1 {{first instance here}}
    instance test_en1 @TestEn()
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
  }
}


// -----

// Test enable signal that isn't reachable from DUT.
// expected-error @below {{unable to connect enable signal and DUT, may not be reachable from top-level module}}
firrtl.circuit "EnableNotReachable" {
  extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  module @TestEn() {
    // expected-note @below {{enable signal here}}
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }
  // expected-note @below {{DUT here}}
  // expected-note @below {{top-level module here}}
  module @EnableNotReachable() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
  }
}

// -----

// Test bypass signal without enable.
firrtl.circuit "BypassWithoutEnable" {
  extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  module @BypassWithoutEnable() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // expected-error @below {{bypass signal specified without enable signal}}
    %dft_clk_div_bypass = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTClockDividerBypassAnnotation"}]} : !firrtl.uint<1>
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_dft_clk_div_bypass, %eicg_out = instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock)
  }
}

// -----

// Test bypass signal unreachable.
// expected-error @below {{unable to connect bypass signal and DUT (and enable), may not be reachable from top-level module}}
firrtl.circuit "BypassNotReachable" {
  extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  module @Bypass() {
    // expected-note @below {{bypass signal here}}
    %dft_clk_div_bypass = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTClockDividerBypassAnnotation"}]} : !firrtl.uint<1>
  }
  // expected-note @below {{DUT here}}
  // expected-note @below {{top-level module here}}
  module @BypassNotReachable() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // expected-note @below {{enable signal here}}
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_dft_clk_div_bypass, %eicg_out = instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, in dft_clk_div_bypass: !firrtl.uint<1>, out out: !firrtl.clock)
  }
}

// -----

// Test clock gate instantiated both in and outside DUT
firrtl.circuit "InAndOutOfDUT" {
  extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  // expected-error @below {{clock gates within DUT must not be instantiated outside the DUT}}
  module @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    instance a @A()
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }

  module @A() {
    %in, %test_en, %en, %out = instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
  }

  module @InAndOutOfDUT() {
    instance a @A()
    instance d @DUT()
  }
}

// -----
// Not an extmodule.

firrtl.circuit "NotExtModule" {
  // expected-error @below {{clock gate module must be an extmodule}}
  module @EICG_wrapper_mod(in %in: !firrtl.clock, in %test_en: !firrtl.clock, in %en: !firrtl.uint<1>, out %out: !firrtl.clock) attributes {defname = "EICG_wrapper"} {}

  module @NotExtModule() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = instance eicg @EICG_wrapper_mod(in in: !firrtl.clock, in test_en: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }
}

// -----
// Type mismatch between port and signal, must be UInt<1>.

firrtl.circuit "WrongType" {
  // expected-error @below {{clock gate module must have second port with type UInt<1>}}
  // expected-note @below {{Second port ("foo") has type '!firrtl.clock', expected '!firrtl.uint<1>'}}
  extmodule @EICG_wrapper_type(in in: !firrtl.clock, in foo: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  
  module @WrongType() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = instance eicg @EICG_wrapper_type(in in: !firrtl.clock, in foo: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }
}

// -----
// Port is missing.

firrtl.circuit "MissingPort" {
  // expected-error @below {{clock gate module must have at least two ports}}
  extmodule @EICG_wrapper_noports() attributes {defname = "EICG_wrapper"}

  module @MissingPort() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    instance eicg @EICG_wrapper_noports()
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }
}

// -----
// Port direction is wrong.

firrtl.circuit "WrongDirection" {
  // expected-error @below {{clock gate module must have second port with input direction}}
  extmodule @EICG_wrapper_direction(in in: !firrtl.clock, out bar: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  
  module @WrongDirection() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = instance eicg @EICG_wrapper_direction(in in: !firrtl.clock, out bar: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    %test_en = wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }
}
