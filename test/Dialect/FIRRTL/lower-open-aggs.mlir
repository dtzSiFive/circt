// RUN: circt-opt --pass-pipeline="builtin.module(firrtl.circuit(firrtl-lower-open-aggs))" %s --split-input-file | FileCheck %s --implicit-check-not=openvector --implicit-check-not=openbundle --implicit-check-not=opensub

// CHECK-LABEL: circuit "Bundle"
firrtl.circuit "Bundle" {
// CHECK-LABEL: module private @Child
  module private @Child(in %in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>,
                               out %r: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>) {
    %0 = ref.send %in : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    ref.define %r, %0 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
  }
// CHECK-LABEL: module private @Probe
// CHECK-SAME: in %in
// All probes
// CHECK-NOT:  out %r:
// CHECK-SAME: out %r_a: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>,
// CHECK-SAME: out %r_b: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>,
// Mixed HW / nonHW.
// "mixed" has non-hw removed but preserved structure:
// CHECK-SAME: out %mixed: !firrtl.bundle<a: uint<1>, x flip: vector<bundle<data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>,
// CHECK-SAME: out %mixed_x_0_p: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>,
// CHECK-SAME: out %mixed_x_1_p: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>,
// All probes, but interior structure has no HW projection.
// CHECK-NOT:  out %nohw:
// CHECK-SAME: out %nohw_x_0_p: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>,
// CHECK-SAME: out %nohw_x_1_p: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>) {
  module private @Probe(in %in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>,
                               out %r: !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>,
                               out %mixed: !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>,
                               out %nohw: !firrtl.openbundle<x: openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>>) {
    %0 = opensubfield %nohw[x] : !firrtl.openbundle<x: openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>>
    %1 = opensubindex %0[1] : !firrtl.openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>
    %2 = opensubfield %1[p] : !firrtl.openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %3 = opensubindex %0[0] : !firrtl.openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>
    %4 = opensubfield %3[p] : !firrtl.openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %5 = opensubfield %mixed[x] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %6 = opensubindex %5[1] : !firrtl.openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>
    %7 = opensubfield %6[data] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %8 = opensubfield %6[p] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %9 = opensubindex %5[0] : !firrtl.openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>
    %10 = opensubfield %9[data] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %11 = opensubfield %9[p] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %12 = opensubfield %mixed[b] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %13 = opensubfield %mixed[a] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %14 = opensubfield %r[b] : !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %15 = opensubfield %r[a] : !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %c1_in, %c1_r = instance c1 interesting_name @Child(in in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out r: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>)
    %16 = ref.sub %c1_r[1] : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    %17 = ref.sub %c1_r[0] : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    %c2_in, %c2_r = instance c2 interesting_name @Child(in in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out r: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>)
    %18 = ref.sub %c2_r[0] : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    strictconnect %c1_in, %in : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    strictconnect %c2_in, %in : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    ref.define %15, %c1_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    ref.define %14, %c2_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    %19 = ref.resolve %17 : !firrtl.probe<uint<1>>
    strictconnect %13, %19 : !firrtl.uint<1>
    %20 = ref.resolve %16 : !firrtl.probe<vector<uint<1>, 2>>
    strictconnect %12, %20 : !firrtl.vector<uint<1>, 2>
    ref.define %11, %c1_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    ref.define %8, %c2_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    %21 = ref.resolve %17 : !firrtl.probe<uint<1>>
    strictconnect %10, %21 : !firrtl.uint<1>
    %22 = ref.resolve %18 : !firrtl.probe<uint<1>>
    strictconnect %7, %22 : !firrtl.uint<1>
    ref.define %4, %c1_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    ref.define %2, %c2_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
  }
// CHECK-LABEL: module @Bundle
  module @Bundle(in %in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out1: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out2: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out3: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out4: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out5: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out6: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out7: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>) attributes {convention = #firrtl<convention scalarized>} {
    %0 = subfield %out7[b] : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %1 = subfield %out7[a] : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %p_in, %p_r, %p_mixed, %p_nohw = instance p interesting_name @Probe(in in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out r: !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, out mixed: !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>, out nohw: !firrtl.openbundle<x: openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>>)
    %2 = opensubfield %p_mixed[b] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %3 = opensubfield %p_mixed[a] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %4 = opensubfield %p_nohw[x] : !firrtl.openbundle<x: openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>>
    %5 = opensubindex %4[1] : !firrtl.openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>
    %6 = opensubfield %5[p] : !firrtl.openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %7 = opensubindex %4[0] : !firrtl.openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>
    %8 = opensubfield %7[p] : !firrtl.openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %9 = opensubfield %p_mixed[x] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %10 = opensubindex %9[1] : !firrtl.openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>
    %11 = opensubfield %10[p] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %12 = opensubindex %9[0] : !firrtl.openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>
    %13 = opensubfield %12[p] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %14 = opensubfield %p_r[b] : !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %15 = opensubfield %p_r[a] : !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    strictconnect %p_in, %in : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %16 = ref.resolve %15 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    strictconnect %out1, %16 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %17 = ref.resolve %14 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    strictconnect %out2, %17 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %18 = ref.resolve %13 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    strictconnect %out3, %18 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %19 = ref.resolve %11 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    strictconnect %out4, %19 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %20 = ref.resolve %8 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    strictconnect %out5, %20 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %21 = ref.resolve %6 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    strictconnect %out6, %21 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    strictconnect %1, %3 : !firrtl.uint<1>
    strictconnect %0, %2 : !firrtl.vector<uint<1>, 2>
  }

// CHECK-LABEL: extmodule @ExtProbes
  extmodule @ExtProbes(
    out r: !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>,
    out mixed: !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>,
    out nohw: !firrtl.openbundle<x: openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>>) attributes {convention = #firrtl<convention scalarized>}
}
