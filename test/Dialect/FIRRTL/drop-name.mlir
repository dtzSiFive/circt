// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-drop-names{preserve-values=all})))' %s   | FileCheck %s --check-prefix=ALL
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-drop-names{preserve-values=named})))' %s | FileCheck %s --check-prefix=NAMED
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-drop-names{preserve-values=none})))' %s  | FileCheck %s --check-prefix=NONE

firrtl.circuit "Foo" {
  module @Foo() {
    // ALL:   %a = wire  interesting_name : !firrtl.uint<1>
    // NAMED: %a = wire  interesting_name : !firrtl.uint<1>
    // NONE:  %a = wire  : !firrtl.uint<1>
    %a = wire interesting_name : !firrtl.uint<1>

    // ALL:   %_a = wire  interesting_name : !firrtl.uint<1>
    // NAMED: %_a = wire  : !firrtl.uint<1>
    // NONE:  %_a = wire  : !firrtl.uint<1>
    %_a = wire interesting_name : !firrtl.uint<1>

    // ALL:   %_T = wire : !firrtl.uint<1>
    // NAMED: %0 = wire  : !firrtl.uint<1>
    // NONE:  %0 = wire  : !firrtl.uint<1>
    %_T = wire interesting_name : !firrtl.uint<1>

  }
}
