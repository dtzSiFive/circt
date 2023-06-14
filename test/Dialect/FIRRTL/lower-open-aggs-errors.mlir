// RUN: circt-opt --pass-pipeline="builtin.module(firrtl.circuit(firrtl-lower-open-aggs))" %s --split-input-file --verify-diagnostics

firrtl.circuit "Symbol" {
  // expected-error @below {{symbol found on aggregate with no HW}}
  module @Symbol(out %r : !firrtl.openbundle<p: probe<uint<1>>> sym @bad) {
    %zero = constant 0 : !firrtl.uint<1>
    %ref = ref.send %zero : !firrtl.uint<1>
    %r_p = opensubfield %r[p] : !firrtl.openbundle<p: probe<uint<1>>>
    ref.define %r_p, %ref : !firrtl.probe<uint<1>>
  }
}

// -----

firrtl.circuit "Annotation" {
  // expected-error @below {{annotations found on aggregate with no HW}}
  module @Annotation(out %r : !firrtl.openbundle<p: probe<uint<1>>>) attributes {portAnnotations = [[{class = "circt.test"}]]} {
    %zero = constant 0 : !firrtl.uint<1>
    %ref = ref.send %zero : !firrtl.uint<1>
    %r_p = opensubfield %r[p] : !firrtl.openbundle<p: probe<uint<1>>>
    ref.define %r_p, %ref : !firrtl.probe<uint<1>>
  }
}

// -----
// Open aggregates are expected to be removed before annotations,
// but check this is detected and an appropriate diagnostic is presented.

firrtl.circuit "MixedAnnotation" {
  // expected-error @below {{annotations on open aggregates not handled yet}}
  module @MixedAnnotation(out %r : !firrtl.openbundle<a: uint<1>, p: probe<uint<1>>>) attributes {portAnnotations = [[{class = "circt.test"}]]} {
    %zero = constant 0 : !firrtl.uint<1>
    %ref = ref.send %zero : !firrtl.uint<1>
    %r_p = opensubfield %r[p] : !firrtl.openbundle<a: uint<1>, p: probe<uint<1>>>
    ref.define %r_p, %ref : !firrtl.probe<uint<1>>
    %r_a = opensubfield %r[a] : !firrtl.openbundle<a: uint<1>, p: probe<uint<1>>>
    strictconnect %r_a, %zero : !firrtl.uint<1>
  }
}

// -----
// Reject unhandled ops w/open types in them.

firrtl.circuit "UnhandledOp" {
  module @UnhandledOp(out %r : !firrtl.openbundle<p: probe<uint<1>>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %ref = ref.send %zero : !firrtl.uint<1>
    %r_p = opensubfield %r[p] : !firrtl.openbundle<p: probe<uint<1>>>
    ref.define %r_p, %ref : !firrtl.probe<uint<1>>

    // expected-error @below {{unhandled use or producer of types containing references}}
    %x = wire : !firrtl.openbundle<p : probe<uint<1>>>
  }
}
