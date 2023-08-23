// RUN: circt-opt --pass-pipeline='builtin.module(hw.design(export-verilog))' -verify-diagnostics --split-input-file %s

hw.design {
// expected-error @+1 {{value has an unsupported verilog type 'vector<3xi1>'}}
hw.module @A(%a: vector<3 x i1>) -> () { }
}
