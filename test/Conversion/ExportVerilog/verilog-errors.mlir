// RUN: circt-opt -pass-pipeline='builtin.module(hw.design(export-verilog))' -verify-diagnostics --split-input-file -mlir-print-op-on-diagnostic=false %s

hw.design {
// expected-error @+1 {{value has an unsupported verilog type 'f32'}}
hw.module @Top(%out: f32) {
}
}

// -----

// expected-error @+2 {{unknown style option 'badOption'}}
// expected-error @+1 {{unknown style option 'anotherOne'}}
module attributes {circt.loweringOptions = "badOption,anotherOne"} { hw.design {} }

// -----

hw.design {
hw.module.extern @A<width: none> ()

hw.module @B() {
  // expected-error @+1 {{should have a typed value; has value @Foo}}
  hw.instance "foo" @A<width: none = @Foo>() -> ()
}
}

// -----

hw.design {
// expected-error @+1 {{name "parameter" is not allowed in Verilog output}}
hw.module.extern @parameter ()
}
