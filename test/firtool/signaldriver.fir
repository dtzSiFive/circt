; RUN: firtool --firrtl-grand-central %s | FileCheck %s
; CHECK-LABEL: FILE "sigdrive.json"
; CHECK: "class":
; CHECK-SAME: SignalDriverAnnotation

; CHECK-LABEL: "sinkTargets"
; CHECK: _1
; CHECK-SAME{LITERAL}: ~Foo|Foo>i
; CHECK-NEXT: _2
; CHECK-SAME{LITERAL}: ~signal_driver|signal_driver>Foo_i
; COM: CHECK: _1
; COM: CHECK-SAME{LITERAL}: ~Foo|Foo>i
; COM: CHECK-NEXT: _2
; COM: CHECK-SAME{LITERAL}: ~signal_driver|signal_driver>Foo_i

; CHECK-LABEL: "sourceTargets"
; CHECK: _1
; CHECK-SAME{LITERAL}: "~Foo|Foo>o"
; CHECK-NEXT: _2
; CHECK-SAME{LITERAL}: ~signal_driver|signal_driver>Foo_o
; 
; CHECK: "circuit"
; CHECK: "annotations"
; CHECK: "circuitPackage"

; TODO: Add inliner annotations, check things are updated appropriately


; SINK TARGET
; NLA not supported yet :(
;      {
;        "_1":"~Foo|Foo/b:Bar>i",
;        "_2":"~signal_driver|signal_driver>Bar_i"
;      }
circuit Foo: %[[
  {
  "class":"sifive.enterprise.grandcentral.SignalDriverAnnotation",
    "sinkTargets":[
      {
        "_1":"~Foo|Foo>i",
        "_2":"~signal_driver|signal_driver>Foo_i"
      }
    ],
    "sourceTargets":[
      {
        "_1":"~Foo|Foo>o",
        "_2":"~signal_driver|signal_driver>Foo_o"
      },
      {
        "_1":"~Bar|Bar>o",
        "_2":"~signal_driver|signal_driver>Bar_o"
      }
    ],
    "circuit":"circuit empty :\n  module empty :\n\n    skip\n",
    "annotations":[
    ],
    "circuitPackage":"foodriver"
}
]]
  module Foo:
    input i: UInt<1>
    output o: UInt<1>
    inst b of Bar
    b.i <- i
    o <- b.o

  module Bar:
    input i: UInt<1>
    output o: UInt<1>
    o <- not(i)
