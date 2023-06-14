// RUN: circt-opt %s  --firrtl-lower-xmr -split-input-file -verify-diagnostics

// Test for same module lowering
// CHECK-LABEL: circuit "xmr"
firrtl.circuit "xmr" {
  // expected-error @+1 {{reference dataflow cannot be traced back to the remote read op for module port 'a'}}
  module @xmr(in %a: !firrtl.probe<uint<2>>) {
    %x = ref.resolve %a : !firrtl.probe<uint<2>>
  }
}

// -----

firrtl.circuit "Top" {
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %xmr_a = instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a = instance child @Child1(in _a: !firrtl.probe<uint<1>>)
    %c_b = instance child @Child2(in _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %xmr_a : !firrtl.probe<uint<1>>
    ref.define %c_b, %xmr_a : !firrtl.probe<uint<1>>
  }
  module @Child1(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    %c_b = instance child @Child2(in _a: !firrtl.probe<uint<1>>)
    ref.define %c_b, %_a : !firrtl.probe<uint<1>>
  }
  // expected-error @+1 {{op multiply instantiated module with input RefType port '_a'}}
  module @Child2(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
  }
}

// -----

firrtl.circuit "Top" {
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %xmr_a = instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a = instance child @Child1(in _a: !firrtl.probe<uint<1>>)
    %c_b = instance child @Child2(in _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %xmr_a : !firrtl.probe<uint<1>>
  }
  module @Child1(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
  }
  // expected-error @+1 {{reference dataflow cannot be traced back to the remote read op for module port '_a'}}
  module @Child2(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
  }
}

// -----
// Check handling of unexpected ref.sub.

firrtl.circuit "RefSubNotFromMemory" {
  module @RefSubNotFromMemory(in %in : !firrtl.bundle<a: uint<1>, b: uint<2>>) {
    // expected-note @below {{input here}}
    %ref = ref.send %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
    // expected-error @below {{can only lower RefSubOp of Memory}}
    %sub = ref.sub %ref[1] : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
    %res = ref.resolve %sub : !firrtl.probe<uint<2>>
  }
}

// -----
// Check handling of unexpected ref.sub, from port.

firrtl.circuit "RefSubNotFromOp" {
  // expected-note @below {{input here}}
  module private @Child(in %ref : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>) {
    // expected-error @below {{can only lower RefSubOp of Memory}}
    %sub = ref.sub %ref[1] : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
    %res = ref.resolve %sub : !firrtl.probe<uint<2>>
  }
  module @RefSubNotFromOp(in %in : !firrtl.bundle<a: uint<1>, b: uint<2>>) {
    %ref = ref.send %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
    %child_ref = instance child @Child(in ref : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>)
    ref.define %child_ref, %ref : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
  }
}
