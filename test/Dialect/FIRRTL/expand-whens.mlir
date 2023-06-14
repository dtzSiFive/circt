// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-expand-whens)))' %s | FileCheck %s
firrtl.circuit "ExpandWhens" {
firrtl.module @ExpandWhens () {}

// Test that last connect semantics are resolved for connects.
firrtl.module @shadow_connects(out %out : !firrtl.uint<1>) {
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  connect %out, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %out, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
}
// CHECK-LABEL: module @shadow_connects(out %out: !firrtl.uint<1>) {
// CHECK-NEXT:   %c0_ui1 = constant 0 : !firrtl.uint<1>
// CHECK-NEXT:   %c1_ui1 = constant 1 : !firrtl.uint<1>
// CHECK-NEXT:   connect %out, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT: }


// Test that last connect semantics are resolved in a WhenOp
firrtl.module @shadow_when(in %p : !firrtl.uint<1>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    %w = wire : !firrtl.uint<2>
    connect %w, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    connect %w, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}
// CHECK-LABEL: module @shadow_when(in %p: !firrtl.uint<1>) {
// CHECK-NEXT:   %c0_ui2 = constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %w = wire  : !firrtl.uint<2>
// CHECK-NEXT:   connect %w, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test all simulation constructs
firrtl.module @simulation(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, in %enable : !firrtl.uint<1>, in %reset : !firrtl.uint<1>) {
  when %p : !firrtl.uint<1> {
    printf %clock, %enable, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
    stop %clock, %enable, 0 : !firrtl.clock, !firrtl.uint<1>
    assert %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    assume %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    cover %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  } else {
    printf %clock, %reset, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
    stop %clock, %enable, 1 : !firrtl.clock, !firrtl.uint<1>
    assert %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    assume %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    cover %clock, %p, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  }
}
// CHECK-LABEL: module @simulation(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, in %enable: !firrtl.uint<1>, in %reset: !firrtl.uint<1>) {
// CHECK-NEXT:   %0 = and %p, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   printf %clock, %0, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
// CHECK-NEXT:   %1 = and %p, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   stop %clock, %1, 0 : !firrtl.clock, !firrtl.uint<1>
// CHECK-NEXT:   %2 = and %p, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   assert %clock, %p, %2, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
// CHECK-NEXT:   %3 = and %p, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   assume %clock, %p, %3, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
// CHECK-NEXT:   %4 = and %p, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   cover %clock, %p, %4, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
// CHECK-NEXT:   %5 = not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %6 = and %5, %reset : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   printf %clock, %6, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
// CHECK-NEXT:   %7 = and %5, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   stop %clock, %7, 1 : !firrtl.clock, !firrtl.uint<1>
// CHECK-NEXT:   %8 = and %5, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   assert %clock, %p, %8, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
// CHECK-NEXT:   %9 = and %5, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   assume %clock, %p, %9, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
// CHECK-NEXT:   %10 = and %5, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   cover %clock, %p, %10, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
// CHECK-NEXT: }


// Test nested when operations work correctly.
firrtl.module @nested_whens(in %clock : !firrtl.clock, in %p0 : !firrtl.uint<1>, in %p1 : !firrtl.uint<1>, in %enable : !firrtl.uint<1>, in %reset : !firrtl.uint<1>) {
  when %p0 : !firrtl.uint<1> {
    when %p1 : !firrtl.uint<1> {
      printf %clock, %enable, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
    }
  }
}
// CHECK-LABEL: module @nested_whens(in %clock: !firrtl.clock, in %p0: !firrtl.uint<1>, in %p1: !firrtl.uint<1>, in %enable: !firrtl.uint<1>, in %reset: !firrtl.uint<1>) {
// CHECK-NEXT:   %0 = and %p0, %p1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = and %0, %enable : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   printf %clock, %1, "CIRCT Rocks!" : !firrtl.clock, !firrtl.uint<1>
// CHECK-NEXT: }


// Test that a parameter set in both sides of the connect is resolved. The value
// used is local to each region.
firrtl.module @set_in_both(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  when %p : !firrtl.uint<1> {
    %c0_ui2 = constant 0 : !firrtl.uint<2>
    connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
    %c1_ui2 = constant 1 : !firrtl.uint<2>
    connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}
// CHECK-LABEL: module @set_in_both(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %c1_ui2 = constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %1 = mux(%p, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   connect %out, %1 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that a parameter set before a WhenOp, and then in both sides of the
// WhenOp is resolved.
firrtl.module @set_before_and_in_both(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  %c2_ui2 = constant 2 : !firrtl.uint<2>
  connect %out, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
     connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}
// CHECK-LABEL: module @set_before_and_in_both(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %c2_ui2 = constant 2 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = mux(%p, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   connect %out, %1 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that a parameter set in a WhenOp is not the last connect.
firrtl.module @set_after(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  %c2_ui2 = constant 2 : !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
    connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  connect %out, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
}
// CHECK-LABEL: module @set_after(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %c2_ui2 = constant 2 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = mux(%p, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   connect %out, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that wire written to in only the thenblock is resolved.
firrtl.module @set_in_then0(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}
// CHECK-LABEL: module @set_in_then0(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = mux(%p, %c1_ui2, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   connect %out, %0 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that wire written to in only the then block is resolved.
firrtl.module @set_in_then1(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
}
// CHECK-LABEL: module @set_in_then1(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that wire written to in only the else is resolved.
firrtl.module @set_in_else0(in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
  } else {
    connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}
// CHECK-LABEL: module @set_in_else0(in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = mux(%p, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   connect %out, %1 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Test that when there is implicit extension, the mux infers the correct type.
firrtl.module @check_mux_return_type(in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  connect %out, %c0_ui1 : !firrtl.uint<2>, !firrtl.uint<1>
  when %p : !firrtl.uint<1> {
  } else {
    connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: mux(%p, %c0_ui1, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<2>
}

// Test that wire written to in only the else block is resolved.
firrtl.module @set_in_else1(in %clock : !firrtl.clock, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
  } else {
    connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
}
// CHECK-LABEL: module @set_in_else1(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = not %p : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }

// Check that nested WhenOps work.
firrtl.module @nested(in %clock : !firrtl.clock, in %p0 : !firrtl.uint<1>, in %p1 : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  %c2_ui2 = constant 2 : !firrtl.uint<2>

  connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  when %p0 : !firrtl.uint<1> {
    when %p1 : !firrtl.uint<1> {
      connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    }
  }
}
// CHECK-LABEL: module @nested(in %clock: !firrtl.clock, in %p0: !firrtl.uint<1>, in %p1: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
// CHECK-NEXT:   %c0_ui2 = constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   %c1_ui2 = constant 1 : !firrtl.uint<2>
// CHECK-NEXT:   %c2_ui2 = constant 2 : !firrtl.uint<2>
// CHECK-NEXT:   %0 = and %p0, %p1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK-NEXT:   %1 = mux(%p1, %c1_ui2, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   %2 = mux(%p0, %1, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
// CHECK-NEXT:   connect %out, %2 : !firrtl.uint<2>, !firrtl.uint<2>
// CHECK-NEXT: }


// Check that nested WhenOps work.
firrtl.module @nested2(in %clock : !firrtl.clock, in %p0 : !firrtl.uint<1>, in %p1 : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  %c2_ui2 = constant 2 : !firrtl.uint<2>
  %c3_ui2 = constant 3 : !firrtl.uint<2>

  when %p0 : !firrtl.uint<1> {
    when %p1 : !firrtl.uint<1> {
      connect %out, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    } else {
      connect %out, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    }
  } else {
    when %p1 : !firrtl.uint<1> {
      connect %out, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    } else {
      connect %out, %c3_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    }
  }
}
//CHECK-LABEL: module @nested2(in %clock: !firrtl.clock, in %p0: !firrtl.uint<1>, in %p1: !firrtl.uint<1>, out %out: !firrtl.uint<2>) {
//CHECK-NEXT:   %c0_ui2 = constant 0 : !firrtl.uint<2>
//CHECK-NEXT:   %c1_ui2 = constant 1 : !firrtl.uint<2>
//CHECK-NEXT:   %c2_ui2 = constant 2 : !firrtl.uint<2>
//CHECK-NEXT:   %c3_ui2 = constant 3 : !firrtl.uint<2>
//CHECK-NEXT:   %0 = and %p0, %p1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %1 = not %p1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %2 = and %p0, %1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %3 = mux(%p1, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
//CHECK-NEXT:   %4 = not %p0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %5 = and %4, %p1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %6 = not %p1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %7 = and %4, %6 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
//CHECK-NEXT:   %8 = mux(%p1, %c2_ui2, %c3_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
//CHECK-NEXT:   %9 = mux(%p0, %3, %8) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
//CHECK-NEXT:   connect %out, %9 : !firrtl.uint<2>, !firrtl.uint<2>
//CHECK-NEXT: }

// Test invalid value optimization
// CHECK-LABEL: module @InvalidValues
firrtl.module @InvalidValues(in %p: !firrtl.uint<1>, out %out0: !firrtl.uint<2>, out %out1: !firrtl.uint<2>, out %out2: !firrtl.uint<2>, out %out3: !firrtl.uint<2>, out %out4: !firrtl.uint<2>, out %out5: !firrtl.uint<2>) {
  %c2_ui2 = constant 2 : !firrtl.uint<2>
  %invalid_ui2 = invalidvalue : !firrtl.uint<2>

  when %p : !firrtl.uint<1> {
    connect %out0, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else  {
    connect %out0, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: connect %out0, %c2_ui2

  when %p : !firrtl.uint<1> {
    connect %out1, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else  {
    connect %out1, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: connect %out1, %c2_ui2

  connect %out2, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    connect %out2, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: connect %out2, %c2_ui2

  connect %out3, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    skip
  } else  {
    connect %out3, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: connect %out3, %c2_ui2

  connect %out4, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    connect %out4, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: connect %out4, %c2_ui2

  connect %out5, %c2_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    skip
  } else  {
    connect %out5, %invalid_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK: connect %out5, %c2_ui2
}
    
// Test that registers are multiplexed with themselves.
firrtl.module @register_mux(in %p : !firrtl.uint<1>, in %clock: !firrtl.clock) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>

  // CHECK: %reg0 = reg %clock
  // CHECK: connect %reg0, %reg0
  %reg0 = reg %clock : !firrtl.clock, !firrtl.uint<2>

  // CHECK: %reg1 = reg %clock
  // CHECK: connect %reg1, %c0_ui2
  %reg1 = reg %clock : !firrtl.clock, !firrtl.uint<2>
  connect %reg1, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>

  // CHECK: %reg2 = reg %clock
  // CHECK: [[MUX:%.+]] = mux(%p, %c0_ui2, %reg2)
  // CHECK: connect %reg2, [[MUX]]
  %reg2 = reg %clock : !firrtl.clock, !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    connect %reg2, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }

  // CHECK: %reg3 = reg %clock
  // CHECK: [[MUX:%.+]] = mux(%p, %c0_ui2, %c1_ui2)
  // CHECK: connect %reg3, [[MUX]]
  %reg3 = reg %clock : !firrtl.clock, !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    connect %reg3, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
    connect %reg3, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}


// Test that bundle types are supported.
firrtl.module @bundle_types(in %p : !firrtl.uint<1>, in %clock: !firrtl.clock) {

  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  %w = wire  : !firrtl.bundle<a: uint<2>, b flip: uint<2>>

  // CHECK: [[W_A:%.*]] = subfield %w[a]
  // CHECK: [[MUX:%.*]] = mux(%p, %c1_ui2, %c0_ui2)
  // CHECK: connect [[W_A]], [[MUX]]
  when %p : !firrtl.uint<1> {
    %w_a = subfield %w[a] : !firrtl.bundle<a : uint<2>, b flip: uint<2>>
    connect %w_a, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
    %w_a = subfield %w[a] : !firrtl.bundle<a : uint<2>, b flip: uint<2>>
    connect %w_a, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }

  // CHECK: [[W_B:%.*]] = subfield %w[b]
  // CHECK: [[MUX:%.*]] = mux(%p, %c1_ui2, %c0_ui2)
  // CHECK: connect [[W_B]], [[MUX]]
  %w_b0 = subfield %w[b] : !firrtl.bundle<a : uint<2>, b flip: uint<2>>
  connect %w_b0, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
  } else {
    %w_b1 = subfield %w[b] : !firrtl.bundle<a : uint<2>, b flip: uint<2>>
    connect %w_b1, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}


// This is exercising a bug in field reference creation when the bundle is
// wrapped in an outer flip. See https://github.com/llvm/circt/issues/1172.
firrtl.module @simple(in %in : !firrtl.bundle<a: uint<1>>) { }
firrtl.module @bundle_ports() {
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %simple_in = instance test0 @simple(in in : !firrtl.bundle<a: uint<1>>)
  %0 = subfield %simple_in[a] : !firrtl.bundle<a: uint<1>>
  connect %0, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// This that types are converted to passive when they are muxed together.
firrtl.module @simple2(in %in : !firrtl.uint<3>) { }
firrtl.module @as_passive(in %p : !firrtl.uint<1>) {
  %c2_ui3 = constant 2 : !firrtl.uint<3>
  %c3_ui3 = constant 3 : !firrtl.uint<3>
  %simple0_in = instance test0 @simple2(in in : !firrtl.uint<3>)
  connect %simple0_in, %c2_ui3 : !firrtl.uint<3>, !firrtl.uint<3>

  %simple1_in = instance test0 @simple2(in in : !firrtl.uint<3>)
  when %p : !firrtl.uint<1> {
    // This is the tricky part, connect the input ports together.
    connect %simple1_in, %simple0_in : !firrtl.uint<3>, !firrtl.uint<3>
  } else {
    connect %simple1_in, %c3_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
  }
  // CHECK: [[MUX:%.*]] = mux(%p, %test0_in, %c3_ui3) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
  // CHECK: connect %test0_in_0, [[MUX]] : !firrtl.uint<3>, !firrtl.uint<3>
}


// Test that analog types are not tracked by ExpandWhens
firrtl.module @analog(out %analog : !firrtl.analog<1>) {
  // Should not complain about the output

  // Should not complain about the embeded analog.
  %c1 = constant 0 : !firrtl.uint<1>
  %w = wire : !firrtl.bundle<a: uint<1>, b: analog<1>>
  %w_a = subfield %w[a] : !firrtl.bundle<a : uint<1>, b : analog<1>>
  connect %w_a, %c1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @vector_simple
firrtl.module @vector_simple(in %clock: !firrtl.clock, out %ret: !firrtl.vector<uint<1>, 1>) {
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %0 = subindex %ret[0] : !firrtl.vector<uint<1>, 1>
  connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %0, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:      %0 = subindex %ret[0] : !firrtl.vector<uint<1>, 1>
  // CHECK-NEXT: connect %0, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1
}

// CHECK-LABEL: @shadow_when_vector
firrtl.module @shadow_when_vector(in %p : !firrtl.uint<1>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  when %p : !firrtl.uint<1> {
    %w = wire : !firrtl.vector<uint<2>, 1>
    %0 = subindex %w[0] : !firrtl.vector<uint<2>, 1>
    connect %0, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    connect %0, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK:      %w = wire  : !firrtl.vector<uint<2>, 1>
  // CHECK-NEXT: %0 = subindex %w[0] : !firrtl.vector<uint<2>, 1>
  // CHECK-NEXT: connect %0, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
}

// CHECK-LABEL: @multi_dim_vector
firrtl.module @multi_dim_vector(in %p : !firrtl.uint<1>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  %w = wire : !firrtl.vector<vector<uint<2>, 2>, 1>
  %0 = subindex %w[0] : !firrtl.vector<vector<uint<2>, 2>, 1>
  %1 = subindex %0[0] : !firrtl.vector<uint<2>, 2>
  %2 = subindex %0[1] : !firrtl.vector<uint<2>, 2>
  when %p : !firrtl.uint<1> {
    connect %1, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    connect %2, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  } else {
    connect %1, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
    connect %2, %c0_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK:      [[MUX1:%.*]] = mux(%p, %c0_ui2, %c1_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
  // CHECK-NEXT: connect %1, [[MUX1]] : !firrtl.uint<2>, !firrtl.uint<2>
  // CHECK-NEXT: [[MUX2:%.*]] = mux(%p, %c1_ui2, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
  // CHECK-NEXT: connect %2, [[MUX2]] : !firrtl.uint<2>, !firrtl.uint<2>
}

// CHECK-LABEL: @vector_of_bundle
firrtl.module @vector_of_bundle(in %p : !firrtl.uint<1>, out %ret: !firrtl.vector<bundle<a:uint<1>>, 1>) {
  %0 = subindex %ret[0] : !firrtl.vector<bundle<a:uint<1>>, 1>
  %1 = subfield %0[a] : !firrtl.bundle<a:uint<1>>
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:     connect %1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @aggregate_register
firrtl.module @aggregate_register(in %clock: !firrtl.clock) {
  %0 = reg %clock : !firrtl.clock, !firrtl.bundle<a : uint<1>, b : uint<1>>
  // CHECK:      %1 = subfield %0[a]
  // CHECK-NEXT: connect %1, %1
  // CHECK-NEXT: %2 = subfield %0[b]
  // CHECK-NEXT: connect %2, %2
}

// CHECK-LABEL: @aggregate_regreset
firrtl.module @aggregate_regreset(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %resetval: !firrtl.vector<uint<1>, 2>) {
  %0 = regreset %clock, %reset, %resetval : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>
  // CHECK:      %1 = subindex %0[0]
  // CHECK-NEXT: connect %1, %1
  // CHECK-NEXT: %2 = subindex %0[1]
  // CHECK-NEXT: connect %2, %2
}

// CHECK-LABEL: @refdefine
firrtl.module @refdefine(in %x : !firrtl.uint<1>, out %out : !firrtl.probe<uint<1>>) {
  // CHECK-NEXT: %[[REF:.+]] = ref.send %x
  // CHECK-NEXT: ref.define %out, %[[REF]]
  // CHECK-NEXT: }
  when %x : !firrtl.uint<1> {
    %ref = ref.send %x : !firrtl.uint<1>
    ref.define %out, %ref : !firrtl.probe<uint<1>>
  }
}

// CHECK-LABEL: @WhenCForce
firrtl.module @WhenCForce(in %c: !firrtl.uint<1>, in %clock : !firrtl.clock, in %x: !firrtl.uint<4>) {
  // CHECK-NOT: when
  %n, %n_ref = node %x forceable : !firrtl.uint<4>
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  when %c : !firrtl.uint<1> {
    // CHECK: ref.force %clock, %c, %n_ref, %x :
    ref.force %clock, %c1_ui1, %n_ref, %x : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>
    // CHECK: ref.force_initial %c, %n_ref, %x :
    ref.force_initial %c1_ui1, %n_ref, %x : !firrtl.uint<1>, !firrtl.uint<4>
    // CHECK: ref.release %clock, %c, %n_ref :
    ref.release %clock, %c1_ui1, %n_ref : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    // CHECK: ref.release_initial %c, %n_ref :
    ref.release_initial %c1_ui1, %n_ref : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
  }
}

// Check that propassign initialized output ports.
// CHECK-LABEL: module @PropInitOut(out %out: !firrtl.string)
firrtl.module @PropInitOut(out %out : !firrtl.string) {
  %0 = string "hello"
  // CHECK: propassign %out, %0 : !firrtl.string
  propassign %out, %0 : !firrtl.string
}

}
