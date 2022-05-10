; Check main circuit + subcircuit mappings flow

; Execute flow both normally and with altering hierarchy (via inline)

; Create some directories for output, cleanup if present
; RUN: rm -rf %t && mkdir -p %t/default %t/inline

; 1) Run on main circuit, with signal mapping annotations
; RUN: firtool %s --annotation-file %S/subCircuit.json --firrtl-grand-central --split-verilog -o %t/default
; RUN: firtool %s --annotation-file %S/subCircuit.json --annotation-file %S/inline.json --firrtl-grand-central --split-verilog -o %t/inline

; Check the generated file
; RUN: FileCheck %s --input-file=%t/default/sigdrive.json --check-prefix=JSONDEFAULT
; RUN: FileCheck %s --input-file=%t/inline/sigdrive.json --check-prefix=JSONINLINE

; JSONDEFAULT:        SignalDriverAnnotation
; JSONDEFAULT-LABEL:    "sinkTargets"
; JSONDEFAULT{LITERAL}:   "_1": "~Top|Bar>b",
; JSONDEFAULT{LITERAL}:   "_1": "~Top|Foo>clock",
; JSONDEFAULT{LITERAL}:   "_1": "~Top|Foo>dataIn_a_b_c",
; JSONDEFAULT{LITERAL}:   "_1": "~Top|Foo>dataIn_d",
; JSONDEFAULT-LABEL:    "sourceTargets"
; JSONDEFAULT{LITERAL}:   "_1": "~Top|Foo>dataOut_w",
; JSONDEFAULT{LITERAL}:   "_1": "~Top|Foo>dataOut_x_y_z",
; JSONDEFAULT{LITERAL}:   "_1": "~Top|Top>clock",
; JSONDEFAULT:          "circuit"
; JSONDEFAULT:          "annotations": []
; JSONDEFAULT:          "circuitPackage": "test"

; JSONINLINE:         SignalDriverAnnotation
; JSONINLINE-LABEL:     "sinkTargets"
; JSONINLINE{LITERAL}:    "_1": "~Top|Bar>b",
; JSONINLINE{LITERAL}:    "_1": "~Top|Top>foo_clock",
; JSONINLINE{LITERAL}:    "_1": "~Top|Top>foo_dataIn_a_b_c",
; JSONINLINE{LITERAL}:    "_1": "~Top|Top>foo_dataIn_d",
; JSONINLINE-LABEL:     "sourceTargets"
; JSONINLINE{LITERAL}:    "_1": "~Top|Top>clock",
; JSONINLINE{LITERAL}:    "_1": "~Top|Top>foo_dataOut_w",
; JSONINLINE{LITERAL}:    "_1": "~Top|Top>foo_dataOut_x_y_z",
; JSONINLINE:           "circuit"
; JSONINLINE:           "annotations": []
; JSONINLINE:           "circuitPackage": "test"


; 2) Run on sub circuit, with updated annotation file (sigdrive.json) from (1):
; RUN: firtool %S/subcircuit.fir --annotation-file %t/default/sigdrive.json --firrtl-grand-central | FileCheck %s --check-prefix=MAPPINGDEFAULT
; RUN: firtool %S/subcircuit.fir --annotation-file %t/inline/sigdrive.json --firrtl-grand-central | FileCheck %s --check-prefix=MAPPINGINLINE

; MAPPINGDEFAULT-LABEL: module Sub_signal_mappings(
; MAPPINGDEFAULT-NEXT:   input           clock_sink,
; MAPPINGDEFAULT-NEXT:   input  [41:0]   data_sink_u,
; MAPPINGDEFAULT-NEXT:   input  [9000:0] data_sink_v,
; MAPPINGDEFAULT-NEXT:   input           data_sink_w_0,
; MAPPINGDEFAULT-NEXT:                   data_sink_w_1,
; MAPPINGDEFAULT-NEXT:   output          clock_source,
; MAPPINGDEFAULT-NEXT:   output [41:0]   data_source_u,
; MAPPINGDEFAULT-NEXT:   output [9000:0] data_source_v);
; (Don't preserve mappings to values that don't exist)
; MAPPINGDEFAULT-NOT:                    data_source_w_0
; MAPPINGDEFAULT-NOT:                    data_source_w_1
; MAPPINGDEFAULT:       `ifndef VERILATOR
; MAPPINGDEFAULT-NEXT:   initial begin
; MAPPINGDEFAULT-NEXT:     force Foo.clock = clock_sink;
; MAPPINGDEFAULT-NEXT:     force Foo.dataIn_a_b_c = data_sink_u;
; MAPPINGDEFAULT-NEXT:     force Foo.dataIn_d = data_sink_v;
; TODO: This is wrong but expected for now
; MAPPINGDEFAULT-NEXT:     force Bar.b = data_sink_w_0;
; MAPPINGDEFAULT-NEXT:     force Bar.b = data_sink_w_1;
; (cont)
; MAPPINGDEFAULT-NEXT:   end
; MAPPINGDEFAULT-NEXT:   `endif
; MAPPINGDEFAULT-NEXT:   assign clock_source = Top.clock;
; MAPPINGDEFAULT-NEXT:   assign data_source_u = Foo.dataOut_x_y_z;
; MAPPINGDEFAULT-NEXT:   assign data_source_v = Foo.dataOut_w;
; MAPPINGDEFAULT-NOT:           data_source_w_0
; MAPPINGDEFAULT-NOT:           data_source_w_1
; MAPPINGDEFAULT-NEXT: endmodule

; MAPPINGINLINE-LABEL: module Sub_signal_mappings(
; MAPPINGINLINE-NEXT:   input           clock_sink,
; MAPPINGINLINE-NEXT:   input  [41:0]   data_sink_u,
; MAPPINGINLINE-NEXT:   input  [9000:0] data_sink_v,
; MAPPINGINLINE-NEXT:   input           data_sink_w_0,
; MAPPINGINLINE-NEXT:                   data_sink_w_1,
; MAPPINGINLINE-NEXT:   output          clock_source,
; MAPPINGINLINE-NEXT:   output [41:0]   data_source_u,
; MAPPINGINLINE-NEXT:   output [9000:0] data_source_v);
; (Don't preserve mappings to values that don't exist)
; MAPPINGINLINE-NOT:                    data_source_w_0
; MAPPINGINLINE-NOT:                    data_source_w_1
; MAPPINGINLINE:       `ifndef VERILATOR
; MAPPINGINLINE-NEXT:   initial begin
; MAPPINGINLINE-NEXT:     force Top.foo_clock = clock_sink;
; MAPPINGINLINE-NEXT:     force Top.foo_dataIn_a_b_c = data_sink_u;
; MAPPINGINLINE-NEXT:     force Top.foo_dataIn_d = data_sink_v;
; TODO: This is wrong but expected for now
; MAPPINGINLINE-NEXT:     force Bar.b = data_sink_w_0;
; MAPPINGINLINE-NEXT:     force Bar.b = data_sink_w_1;
; (cont)
; MAPPINGINLINE-NEXT:   end
; MAPPINGINLINE-NEXT:   `endif
; MAPPINGINLINE-NEXT:   assign clock_source = Top.clock;
; MAPPINGINLINE-NEXT:   assign data_source_u = Top.foo_dataOut_x_y_z;
; MAPPINGINLINE-NEXT:   assign data_source_v = Top.foo_dataOut_w;
; MAPPINGINLINE-NOT:           data_source_w_0
; MAPPINGINLINE-NOT:           data_source_w_1
; MAPPINGINLINE-NEXT: endmodule

circuit Top :
  module Bar :
    input a: UInt
    input b: UInt
    output o: UInt
    o <= add(a, not(b))

  module Foo :
    input clock : Clock
    input reset : Reset
    input dataIn : {a: {b: {c: UInt<42>}}, d: UInt<9001>, e: UInt<42>}
    output dataOut : {x: {y: {z: UInt<42>}}, w: UInt<9001>, u: UInt<42>}
    reg stuff : UInt<42>, clock
    reg stuff2 : UInt<9001>, clock

    stuff <= dataIn.a.b.c
    stuff2 <= dataIn.d
    dataOut.x.y.z <= stuff
    dataOut.w <= stuff2

    inst bar of Bar
    bar.a <= dataIn.a.b.c
    bar.b <= dataIn.e
    dataOut.u <= bar.o
    

  module Top :
    input clock : Clock
    input reset : UInt<1>
    input dataIn : {a: {b: {c: UInt<42>}}, d: UInt<9001>, e: UInt<42>}
    output zu: UInt

    inst foo of Foo
    foo.clock <= asClock(UInt<1>(0))
    foo.reset <= reset
    foo.dataIn <- dataIn
    zu <= and(foo.dataOut.x.y.z, foo.dataOut.u)
