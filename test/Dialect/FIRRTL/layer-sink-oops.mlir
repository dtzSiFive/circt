// Demonstrate known busted examples.
// Setup to check the errors, but ensure this fails by not making these "expected"

// RUN: circt-opt -pass-pipeline="builtin.module(firrtl.circuit(firrtl.module(firrtl-layer-sink)))" %s --split-input-file --verify-diagnostics

firrtl.circuit "Sub" {
  firrtl.layer @Subfield bind {}
  firrtl.module @Sub() {
    %w = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.bundle<a: uint<1>>
    %w_a = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>>
    %z = firrtl.constant 0 : !firrtl.uint<1>
    // {{op connects to a destination which is defined outside its enclosing layer block}}
    firrtl.matchingconnect %w_a, %z: !firrtl.uint<1>
    firrtl.layerblock @Subfield {
      firrtl.node %w_a : !firrtl.uint<1>
    }
  }
}

// -----

firrtl.circuit "Sink" {
  firrtl.layer @L bind {}
  firrtl.module @Sink() {
    %z = firrtl.constant 0 : !firrtl.uint<1>
    %w = firrtl.wire : !firrtl.uint<1>
    firrtl.layerblock @L {
      firrtl.node %w : !firrtl.uint<1>
    }
    %n = firrtl.node %z : !firrtl.uint<1>
    // {{operand #1 does not dominate this use}}
    firrtl.matchingconnect %w, %n : !firrtl.uint<1>
    %n2 = firrtl.node %n : !firrtl.uint<1>
  }
}
