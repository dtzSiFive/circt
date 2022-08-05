// RUN: circt-opt %s -split-input-file

firrtl.circuit "xmr" {
  firrtl.module private @Test(out %x: !firrtl.ref<uint<2>>) {
    %w = firrtl.wire : !firrtl.uint<2>
    firrtl.ref.send %x, %w : !firrtl.ref<uint<2>>
  }
  firrtl.module @xmr() {
    %test_x = firrtl.instance test @Test(out x: !firrtl.ref<uint<2>>)
    %x = firrtl.ref.resolve %test_x : !firrtl.ref<uint<2>>
  }
}

// -----

firrtl.circuit "DUT" {
  firrtl.module private @Submodule (out %ref_out1: !firrtl.ref<uint<1>>, out %ref_out2: !firrtl.ref<uint<4>>) {
    %w_data1 = firrtl.wire : !firrtl.uint<1>
    firrtl.ref.send %ref_out1, %w_data1 : !firrtl.ref<uint<1>>
    %w_data2 = firrtl.wire : !firrtl.uint<4>
    firrtl.ref.send %ref_out2, %w_data2 : !firrtl.ref<uint<4>>
  }
  firrtl.module @DUT() {
    %w = firrtl.wire sym @w : !firrtl.uint<1>
    %view_out1, %view_out2 = firrtl.instance sub @Submodule(out ref_out1: !firrtl.ref<uint<1>>, out ref_out2: !firrtl.ref<uint<4>>)
    %view_in1, %view_in2 = firrtl.instance MyView_companion @MyView_companion(in ref_in1: !firrtl.uint<1>, in ref_in2: !firrtl.uint<4>)

    %1 = firrtl.ref.resolve %view_out1 : !firrtl.ref<uint<1>>
    %2 = firrtl.ref.resolve %view_out2 : !firrtl.ref<uint<4>>
    firrtl.strictconnect %view_in1, %1 : !firrtl.uint<1>
    firrtl.strictconnect %view_in2, %2 : !firrtl.uint<4>
  }

  firrtl.module private @MyView_companion (in %ref_in1: !firrtl.uint<1>, in %ref_in2: !firrtl.uint<4>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %_WIRE = firrtl.wire sym @_WIRE : !firrtl.uint<1>
    firrtl.strictconnect %_WIRE, %c0_ui1 : !firrtl.uint<1>
    %iface = sv.interface.instance sym @__MyView_MyInterface__  : !sv.interface<@MyInterface>
  }

  sv.interface @MyInterface {
    sv.verbatim "// a wire called 'bool'" {symbols = []}
    sv.interface.signal @bool : i1
  }
}

// -----

//// "Example 1"
//firrtl.circuit "SimpleWrite" {
//  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
//    %zero = firrtl.constant 0 : !firrtl.uint<1>
//    firrtl.ref.send %_a, %zero : !firrtl.ref<uint<1>>
//  }
//  firrtl.module @SimpleWrite() {
//    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
//
//    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
//    %a = firrtl.wire : !firrtl.uint<1>
//    firrtl.strictconnect %0, %a    : !firrtl.uint<1>
//  }
//}

// -----

// "Example 2"
firrtl.circuit "SimpleRead" {
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.ref.send %_a, %zero : !firrtl.ref<uint<1>>
  }
  firrtl.module @SimpleRead() {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

//firrtl.circuit "UnconnectedRef" {
//  firrtl.module @Bar(in %_a: !firrtl.ref<uint<1>>) {
//    %a = firrtl.wire : !firrtl.uint<1>
//    firrtl.ref.recv %a, %_a : !firrtl.ref<uint<1>>
//  }
//  firrtl.module @UnconnectedRef() {
//    %bar_a = firrtl.instance bar1 @Bar(in _a: !firrtl.ref<uint<1>>)
//    // bar_b is unconnected.
//    %bar_b = firrtl.instance bar2 @Bar(in _a: !firrtl.ref<uint<1>>)
//
//    %zero = firrtl.constant 0 : !firrtl.uint<1>
//    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
//    firrtl.strictconnect %0, %zero : !firrtl.uint<1>
//  }
//}

// -----

firrtl.circuit "ForwardToInstance" {
  firrtl.module @Bar2(out %_a: !firrtl.ref<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.ref.send %_a, %zero : !firrtl.ref<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.ref<uint<1>>) {
    %bar_2 = firrtl.instance bar @Bar2(out _a: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %_a, %bar_2 : !firrtl.ref<uint<1>>
  }
  firrtl.module @ForwardToInstance() {
    %bar_a = firrtl.instance bar @Bar(out _a: !firrtl.ref<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.ref<uint<1>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}
