// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-remove-unused-ports))' %s -split-input-file | FileCheck %s
firrtl.circuit "Top"   {
  // CHECK-LABEL: module @Top(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
  // CHECK-SAME :                    out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>)
  module @Top(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                     out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c, %A_d_unused, %A_d_invalid, %A_d_constant = instance A  @UseBar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d_unused: !firrtl.uint<1>, out d_invalid: !firrtl.uint<1>, out d_constant: !firrtl.uint<1>)
    // CHECK: %A_b, %A_c = instance A @UseBar(in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    // CHECK-NEXT: connect %A_b, %b
    // CHECK-NEXT: connect %c, %A_c
    // CHECK-NEXT: connect %d_unused, %{{invalid_ui1.*}}
    // CHECK-NEXT: connect %d_invalid, %{{invalid_ui1.*}}
    // CHECK-NEXT: connect %d_constant, %{{c1_ui1.*}}
    connect %A_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    connect %A_b, %b : !firrtl.uint<1>, !firrtl.uint<1>
    connect %c, %A_c : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d_unused, %A_d_unused : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d_invalid, %A_d_invalid : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d_constant, %A_d_constant : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Check that %a, %d_unused, %d_invalid and %d_constant are removed.
  // CHECK-LABEL: module private @Bar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>)
  // CHECK-NEXT:    connect %c, %b
  // CHECK-NEXT:  }
  module private @Bar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                     out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>

    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    connect %d_invalid, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %c1_i1 = constant 1 : !firrtl.uint<1>
    connect %d_constant, %c1_i1 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Check that %a, %d_unused, %d_invalid and %d_constant are removed.
  // CHECK-LABEL: module private @UseBar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
  module private @UseBar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                        out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c, %A_d_unused, %A_d_invalid, %A_d_constant = instance A  @Bar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d_unused: !firrtl.uint<1>, out d_invalid: !firrtl.uint<1>, out d_constant: !firrtl.uint<1>)
    connect %A_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: %A_b, %A_c = instance A  @Bar(in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    connect %A_b, %b : !firrtl.uint<1>, !firrtl.uint<1>
    connect %c, %A_c : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d_unused, %A_d_unused : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d_invalid, %A_d_invalid : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d_constant, %A_d_constant : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Make sure that %a, %b and %c are not erased because they have an annotation or a symbol.
  // CHECK-LABEL: module private @Foo(in %a: !firrtl.uint<1> sym @dntSym, in %b: !firrtl.uint<1> [{a = "a"}], out %c: !firrtl.uint<1> sym @dntSym2)
  module private @Foo(in %a: !firrtl.uint<1> sym @dntSym, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1> sym @dntSym2) attributes {
    portAnnotations = [[], [{a = "a"}], []]}
  {
    // CHECK: connect %c, %{{invalid_ui1.*}}
    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    connect %c, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: module private @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>)
  module private @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c = instance A  @Foo(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    // CHECK: %A_a, %A_b, %A_c = instance A @Foo(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    connect %A_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    connect %A_b, %b : !firrtl.uint<1>, !firrtl.uint<1>
    connect %c, %A_c : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Strict connect version.
firrtl.circuit "Top"   {
  // CHECK-LABEL: module @Top(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
  // CHECK-SAME :                    out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>)
  module @Top(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                     out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c, %A_d_unused, %A_d_invalid, %A_d_constant = instance A  @UseBar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d_unused: !firrtl.uint<1>, out d_invalid: !firrtl.uint<1>, out d_constant: !firrtl.uint<1>)
    // CHECK: %A_b, %A_c = instance A @UseBar(in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    // CHECK-NEXT: strictconnect %A_b, %b
    // CHECK-NEXT: strictconnect %c, %A_c
    // CHECK-NEXT: strictconnect %d_unused, %{{invalid_ui1.*}}
    // CHECK-NEXT: strictconnect %d_invalid, %{{invalid_ui1.*}}
    // CHECK-NEXT: strictconnect %d_constant, %{{c1_ui1.*}}
    strictconnect %A_a, %a : !firrtl.uint<1>
    strictconnect %A_b, %b : !firrtl.uint<1>
    strictconnect %c, %A_c : !firrtl.uint<1>
    strictconnect %d_unused, %A_d_unused : !firrtl.uint<1>
    strictconnect %d_invalid, %A_d_invalid : !firrtl.uint<1>
    strictconnect %d_constant, %A_d_constant : !firrtl.uint<1>
  }

  // Check that %a, %d_unused, %d_invalid and %d_constant are removed.
  // CHECK-LABEL: module private @Bar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>)
  // CHECK-NEXT:    strictconnect %c, %b
  // CHECK-NEXT:  }
  module private @Bar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                     out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    strictconnect %c, %b : !firrtl.uint<1>

    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    strictconnect %d_invalid, %invalid_ui1 : !firrtl.uint<1>
    %c1_i1 = constant 1 : !firrtl.uint<1>
    strictconnect %d_constant, %c1_i1 : !firrtl.uint<1>
  }

  // Check that %a, %d_unused, %d_invalid and %d_constant are removed.
  // CHECK-LABEL: module private @UseBar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
  module private @UseBar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>,
                        out %d_unused: !firrtl.uint<1>, out %d_invalid: !firrtl.uint<1>, out %d_constant: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c, %A_d_unused, %A_d_invalid, %A_d_constant = instance A  @Bar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d_unused: !firrtl.uint<1>, out d_invalid: !firrtl.uint<1>, out d_constant: !firrtl.uint<1>)
    strictconnect %A_a, %a : !firrtl.uint<1>
    // CHECK: %A_b, %A_c = instance A  @Bar(in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    strictconnect %A_b, %b : !firrtl.uint<1>
    strictconnect %c, %A_c : !firrtl.uint<1>
    strictconnect %d_unused, %A_d_unused : !firrtl.uint<1>
    strictconnect %d_invalid, %A_d_invalid : !firrtl.uint<1>
    strictconnect %d_constant, %A_d_constant : !firrtl.uint<1>
  }

  // Make sure that %a, %b and %c are not erased because they have an annotation or a symbol.
  // CHECK-LABEL: module private @Foo(in %a: !firrtl.uint<1> sym @dntSym, in %b: !firrtl.uint<1> [{a = "a"}], out %c: !firrtl.uint<1> sym @dntSym2)
  module private @Foo(in %a: !firrtl.uint<1> sym @dntSym, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1> sym @dntSym2) attributes {
    portAnnotations = [[], [{a = "a"}], []]}
  {
    // CHECK: strictconnect %c, %{{invalid_ui1.*}}
    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    strictconnect %c, %invalid_ui1 : !firrtl.uint<1>
  }

  // CHECK-LABEL: module private @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>)
  module private @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c = instance A  @Foo(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    // CHECK: %A_a, %A_b, %A_c = instance A @Foo(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    strictconnect %A_a, %a : !firrtl.uint<1>
    strictconnect %A_b, %b : !firrtl.uint<1>
    strictconnect %c, %A_c : !firrtl.uint<1>
  }
}

// -----

// Ensure that the "output_file" attribute isn't destroyed by RemoveUnusedPorts.
// This matters for interactions between Grand Central (which sets these) and
// RemoveUnusedPorts which may clone modules with stripped ports.
//
// CHECK-LABEL: "PreserveOutputFile"
firrtl.circuit "PreserveOutputFile" {
  // CHECK-NEXT: module {{.+}}@Sub
  // CHECK-SAME:   output_file
  module private @Sub(in %a: !firrtl.uint<1>) attributes {output_file = #hw.output_file<"hello">} {}
  // CHECK: module @PreserveOutputFile
  module @PreserveOutputFile() {
    // CHECK-NEXT: instance sub
    // CHECK-SAME: output_file
    instance sub {output_file = #hw.output_file<"hello">} @Sub(in a: !firrtl.uint<1>)
  }
}

// -----

// CHECK-LABEL: "UnusedOutput"
firrtl.circuit "UnusedOutput"  {
  // CHECK: module {{.+}}@SingleDriver
  // CHECK-NOT:     out %c
  module private @SingleDriver(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    // CHECK-NEXT: %[[c_wire:.+]] = wire
    // CHECK-NEXT: strictconnect %b, %[[c_wire]]
    strictconnect %b, %c : !firrtl.uint<1>
    // CHECK-NEXT: %[[not_a:.+]] = not %a
    %0 = not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %[[c_wire]], %[[not_a]]
    strictconnect %c, %0 : !firrtl.uint<1>
  }
  // CHECK-LABEL: @UnusedOutput
  module @UnusedOutput(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // CHECK: %singleDriver_a, %singleDriver_b = instance singleDriver
    %singleDriver_a, %singleDriver_b, %singleDriver_c = instance singleDriver @SingleDriver(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    strictconnect %singleDriver_a, %a : !firrtl.uint<1>
    strictconnect %b, %singleDriver_b : !firrtl.uint<1>
  }
}
