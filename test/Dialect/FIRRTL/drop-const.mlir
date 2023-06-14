// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-drop-const)))' %s | FileCheck %s --implicit-check-not=const.
firrtl.circuit "DropConst" {
firrtl.module @DropConst() {}

// Const is dropped from extmodule signature
// CHECK-LABEL: extmodule @ConstPortExtModule(
// CHECK-SAME: in a: !firrtl.uint<1>
// CHECK-SAME: in b: !firrtl.bundle<a: uint<1>>
// CHECK-SAME: in c: !firrtl.bundle<a: uint<1>>,
// CHECK-SAME: in d: !firrtl.vector<uint<1>, 3>,
// CHECK-SAME: in e: !firrtl.vector<uint<1>, 3>,
// CHECK-SAME: in f: !firrtl.enum<a: uint<2>, b: uint<1>>,
// CHECK-SAME: out g: !firrtl.probe<uint<1>>)
firrtl.extmodule @ConstPortExtModule(
  in a: !firrtl.const.uint<1>, 
  in b: !firrtl.const.bundle<a: uint<1>>,
  in c: !firrtl.bundle<a: const.uint<1>>,
  in d: !firrtl.const.vector<uint<1>, 3>,
  in e: !firrtl.vector<const.uint<1>, 3>,
  in f: !firrtl.const.enum<a: uint<2>, b: uint<1>>,
  out g: !firrtl.probe<const.uint<1>>
)

// Const is dropped from module signature and ops
// CHECK-LABEL: module @ConstPortModule(
// CHECK-SAME: in %a: !firrtl.uint<1>
// CHECK-SAME: in %b: !firrtl.bundle<a: uint<1>>
// CHECK-SAME: in %c: !firrtl.bundle<a: uint<1>>,
// CHECK-SAME: in %d: !firrtl.vector<uint<1>, 3>,
// CHECK-SAME: in %e: !firrtl.vector<uint<1>, 3>,
// CHECK-SAME: in %f: !firrtl.enum<a: uint<2>, b: uint<1>>,
// CHECK-SAME: out %g: !firrtl.probe<uint<1>>)
firrtl.module @ConstPortModule(
  in %a: !firrtl.const.uint<1>, 
  in %b: !firrtl.const.bundle<a: uint<1>>,
  in %c: !firrtl.bundle<a: const.uint<1>>,
  in %d: !firrtl.const.vector<uint<1>, 3>,
  in %e: !firrtl.vector<const.uint<1>, 3>,
  in %f: !firrtl.const.enum<a: uint<2>, b: uint<1>>,
  out %g: !firrtl.probe<const.uint<1>>
) {
  // CHECK-NEXT: instance inst @ConstPortExtModule(
  // CHECK-SAME: in a: !firrtl.uint<1>
  // CHECK-SAME: in b: !firrtl.bundle<a: uint<1>>
  // CHECK-SAME: in c: !firrtl.bundle<a: uint<1>>,
  // CHECK-SAME: in d: !firrtl.vector<uint<1>, 3>,
  // CHECK-SAME: in e: !firrtl.vector<uint<1>, 3>,
  // CHECK-SAME: in f: !firrtl.enum<a: uint<2>, b: uint<1>>,
  // CHECK-SAME: out g: !firrtl.probe<uint<1>>)
  %a2, %b2, %c2, %d2, %e2, %f2, %g2 = instance inst @ConstPortExtModule(
    in a: !firrtl.const.uint<1>, 
    in b: !firrtl.const.bundle<a: uint<1>>,
    in c: !firrtl.bundle<a: const.uint<1>>,
    in d: !firrtl.const.vector<uint<1>, 3>,
    in e: !firrtl.vector<const.uint<1>, 3>,
    in f: !firrtl.const.enum<a: uint<2>, b: uint<1>>,
    out g: !firrtl.probe<const.uint<1>>
  )

  strictconnect %a2, %a : !firrtl.const.uint<1>
  strictconnect %b2, %b : !firrtl.const.bundle<a: uint<1>>
  strictconnect %c2, %c : !firrtl.bundle<a: const.uint<1>>
  strictconnect %d2, %d : !firrtl.const.vector<uint<1>, 3>
  strictconnect %e2, %e : !firrtl.vector<const.uint<1>, 3>
  strictconnect %f2, %f : !firrtl.const.enum<a: uint<2>, b: uint<1>>
  ref.define %g, %g2 : !firrtl.probe<const.uint<1>>
}

// Const-cast ops are erased
// CHECK-LABEL: module @ConstCastErase
firrtl.module @ConstCastErase(in %in: !firrtl.const.uint<1>, out %out: !firrtl.uint<1>) {
  // CHECK-NOT: constCast
  // CHECK-NEXT: strictconnect %out, %in : !firrtl.uint<1>
  %0 = constCast %in : (!firrtl.const.uint<1>) -> !firrtl.uint<1>
  strictconnect %out, %0 : !firrtl.uint<1> 
}

// Const is dropped within when blocks
// CHECK-LABEL: module @ConstDropInWhenBlock
firrtl.module @ConstDropInWhenBlock(in %cond: !firrtl.const.uint<1>, in %in1: !firrtl.const.sint<2>, in %in2: !firrtl.const.sint<2>, out %out: !firrtl.const.sint<2>) {
  // CHECK: when %cond : !firrtl.uint<1>
  when %cond : !firrtl.const.uint<1> {
    // CHECK: strictconnect %out, %in1 : !firrtl.sint<2>
    strictconnect %out, %in1 : !firrtl.const.sint<2>
  } else {
    // CHECK: strictconnect %out, %in2 : !firrtl.sint<2>
    strictconnect %out, %in2 : !firrtl.const.sint<2>
  }
}
}