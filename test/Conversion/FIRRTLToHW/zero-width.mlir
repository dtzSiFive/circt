// RUN: circt-opt -lower-firrtl-to-hw %s | FileCheck %s

firrtl.circuit "Arithmetic" {
  // CHECK-LABEL: hw.module @Arithmetic
  module @Arithmetic(in %uin3c: !firrtl.uint<3>,
                            out %out0: !firrtl.uint<3>,
                            out %out1: !firrtl.uint<4>,
                            out %out2: !firrtl.uint<4>,
                            out %out3: !firrtl.uint<1>) {
  %uin0c = wire : !firrtl.uint<0>

    // CHECK-DAG: [[MULZERO:%.+]] = hw.constant 0 : i3
    %0 = mul %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<3>
    connect %out0, %0 : !firrtl.uint<3>, !firrtl.uint<3>

    // Lowers to nothing.
    %m0 = mul %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>

    // Lowers to nothing.
    %node = node %m0 : !firrtl.uint<0>

    // Lowers to nothing.  Issue #429.
    %div = div %node, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<0>

    // CHECK-DAG: %c0_i4 = hw.constant 0 : i4
    // CHECK-DAG: %false = hw.constant false
    // CHECK-NEXT: [[UIN3EXT:%.+]] = comb.concat %false, %uin3c : i1, i3
    // CHECK-NEXT: [[ADDRES:%.+]] = comb.add bin %c0_i4, [[UIN3EXT]] : i4
    %1 = add %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<4>
    connect %out1, %1 : !firrtl.uint<4>, !firrtl.uint<4>

    %2 = shl %node, 4 : (!firrtl.uint<0>) -> !firrtl.uint<4>
    connect %out2, %2 : !firrtl.uint<4>, !firrtl.uint<4>

    // Issue #436
    %3 = eq %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
    connect %out3, %3 : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: hw.output %c0_i3, [[ADDRES]], %c0_i4, %true
  }

  // CHECK-LABEL: hw.module private @Exotic
  module private @Exotic(in %uin3c: !firrtl.uint<3>,
                        out %out0: !firrtl.uint<3>,
                        out %out1: !firrtl.uint<3>) {
    %uin0c = wire : !firrtl.uint<0>

    // CHECK-DAG: = hw.constant true
    %0 = andr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // CHECK-DAG: = hw.constant false
    %1 = xorr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    %2 = orr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // Lowers to the uin3 value.
    %3 = cat %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<3>
    connect %out0, %3 : !firrtl.uint<3>, !firrtl.uint<3>

    // Lowers to the uin3 value.
    %4 = cat %uin3c, %uin0c : (!firrtl.uint<3>, !firrtl.uint<0>) -> !firrtl.uint<3>
    connect %out1, %4 : !firrtl.uint<3>, !firrtl.uint<3>

    // Lowers to nothing.
    %5 = cat %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>

    // CHECK: hw.output %uin3c, %uin3c : i3, i3
  }

  // CHECK-LABEL: hw.module private @Decls
  module private @Decls(in %uin3c: !firrtl.uint<3>) {
    %sin0c = wire : !firrtl.sint<0>
    %uin0c = wire : !firrtl.uint<0>

    // Lowers to nothing.
    %wire = wire : !firrtl.sint<0>
    connect %wire, %sin0c : !firrtl.sint<0>, !firrtl.sint<0>

    // CHECK-NEXT: hw.output
  }

}
