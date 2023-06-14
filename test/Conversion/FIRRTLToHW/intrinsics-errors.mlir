// RUN: circt-opt --lower-firrtl-to-hw --verify-diagnostics --split-input-file %s

firrtl.circuit "Foo" {
  module @Foo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) {
    %0 = int.ltl.delay %a, 42 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // expected-error @below {{operand of type '!ltl.sequence' cannot be used as an integer}}
    // expected-error @below {{couldn't handle this operation}}
    %1 = and %0, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo" {
  module @Foo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) {
    %0 = wire : !firrtl.uint<1>
    // expected-note @below {{leaking outside verification context here}}
    %1 = and %0, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // expected-error @below {{verification operation used in a non-verification context}}
    %2 = int.ltl.delay %a, 42 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %0, %2 : !firrtl.uint<1>
  }
}
